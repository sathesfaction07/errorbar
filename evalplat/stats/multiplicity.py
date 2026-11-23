"""Multiplicity control across models x tasks (CAP-M7/S5).

The situation this exists for: a leaderboard compares 8 candidate models on 12
tasks. That is 96 comparisons. At alpha=0.05, roughly 5 of them come back
"significant" even when every single model is identical to the baseline. Someone
picks the best-looking cell and ships it.

`benjamini_hochberg` controls the false discovery rate -- the expected PROPORTION of
rejections that are false. That is the right target for leaderboard screening.
Bonferroni controls the family-wise error rate (the probability of ANY false
rejection), which is far stricter and costs most of the power; it is provided for
comparison, not as the default, and `run_capstone.py` (S5) measures what each
one costs.

An important asymmetry, and the reason `adjust()` returns q-values rather than a
bare boolean mask: a q-value is interpretable and auditable after the fact, whereas
a mask silently bakes in the alpha that happened to be in scope at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as _st

from evalplat.numbers.estimate import Estimate, EstimateError


class MultiplicityError(ValueError):
    pass


def _check_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(pvalues), dtype=float)
    if arr.size == 0:
        raise MultiplicityError(
            "no p-values supplied. An empty family is not 'nothing significant', it "
            "is 'nothing was tested' -- the caller must surface COULD_NOT_RUN."
        )
    if not np.all(np.isfinite(arr)):
        raise MultiplicityError(
            f"{int((~np.isfinite(arr)).sum())} non-finite p-value(s). A test that "
            "failed to produce a p-value must be reported, not dropped -- dropping it "
            "shrinks the family size and makes every other q-value look better."
        )
    if np.any(arr < 0) or np.any(arr > 1):
        bad = arr[(arr < 0) | (arr > 1)][:3]
        raise MultiplicityError(f"p-values outside [0,1], e.g. {bad.tolist()}")
    return arr


@dataclass(frozen=True, slots=True)
class MultiplicityResult:
    method: str
    alpha: float
    pvalues: tuple[float, ...]
    qvalues: tuple[float, ...]
    rejected: tuple[bool, ...]
    labels: tuple[str, ...]

    @property
    def n_tests(self) -> int:
        return len(self.pvalues)

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected)

    def render(self) -> str:
        lines = [
            f"{self.method} at alpha={self.alpha}: {self.n_rejected}/{self.n_tests} rejected"
        ]
        for lab, p, q, r in zip(self.labels, self.pvalues, self.qvalues, self.rejected):
            lines.append(
                f"  {'REJECT' if r else '  keep':>6}  {lab:<28} p={p:.4g}  q={q:.4g}"
            )
        return "\n".join(lines)


def benjamini_hochberg(
    pvalues: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultiplicityResult:
    """BH step-up. Returns adjusted p-values (q-values) as well as the mask.

    The q-value is computed by the standard monotone enforcement: sort ascending,
    scale each by n/rank, then take a running minimum from the largest p downward so
    the sequence is non-decreasing. Skipping the running-minimum step is the usual
    implementation bug -- it yields q-values that can DECREASE as p increases, which
    lets a larger p-value be rejected while a smaller one is not.
    """
    p = _check_pvalues(pvalues)
    if not (0.0 < alpha < 1.0):
        raise MultiplicityError(f"alpha must lie in (0,1), got {alpha}")
    labs = tuple(labels) if labels is not None else tuple(f"test[{i}]" for i in range(p.size))
    if len(labs) != p.size:
        raise MultiplicityError(f"{len(labs)} labels for {p.size} p-values")

    n = p.size
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, n + 1)
    scaled = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return MultiplicityResult(
        method="benjamini-hochberg",
        alpha=alpha,
        pvalues=tuple(float(x) for x in p),
        qvalues=tuple(float(x) for x in q),
        rejected=tuple(bool(x) for x in (q <= alpha)),
        labels=labs,
    )


def bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultiplicityResult:
    """Family-wise error control. Much stricter than BH; included so the power cost
    is measurable rather than asserted."""
    p = _check_pvalues(pvalues)
    labs = tuple(labels) if labels is not None else tuple(f"test[{i}]" for i in range(p.size))
    q = np.clip(p * p.size, 0.0, 1.0)
    return MultiplicityResult(
        method="bonferroni",
        alpha=alpha,
        pvalues=tuple(float(x) for x in p),
        qvalues=tuple(float(x) for x in q),
        rejected=tuple(bool(x) for x in (q <= alpha)),
        labels=labs,
    )


def unadjusted_DO_NOT_USE(
    pvalues: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultiplicityResult:
    """THE PLANTED DEFECT, retained as a counter-example.

    This is what a leaderboard does by default: threshold every raw p-value at
    alpha independently. It is not a silly mistake -- it is the natural thing to
    write, and it is wrong in a way that gets worse as the leaderboard grows.
    `run_capstone.py` (S5) measures the false-discovery count it produces, and
    `tests/test_evalplat.py` asserts it.
    """
    p = _check_pvalues(pvalues)
    labs = tuple(labels) if labels is not None else tuple(f"test[{i}]" for i in range(p.size))
    return MultiplicityResult(
        method="UNADJUSTED(defect)",
        alpha=alpha,
        pvalues=tuple(float(x) for x in p),
        qvalues=tuple(float(x) for x in p),
        rejected=tuple(bool(x) for x in (p <= alpha)),
        labels=labs,
    )


def paired_pvalue(baseline: Estimate, candidate: Estimate) -> float:
    """Two-sided paired t-test p-value on per-item differences.

    Requires `per_item` on both sides. It raises rather than falling back to an
    unpaired test, because silently switching test type changes the p-value by a
    large factor on correlated data and nothing in the output would reveal it.
    """
    if baseline.per_item is None or candidate.per_item is None:
        raise EstimateError(
            "paired_pvalue needs per_item scores on both estimates. Falling back to an "
            "unpaired test would silently change the p-value on correlated data."
        )
    if len(baseline.per_item) != len(candidate.per_item):
        raise EstimateError(
            f"per_item lengths differ ({len(baseline.per_item)} vs "
            f"{len(candidate.per_item)}) -- these are not paired observations"
        )
    ok, why = baseline.provenance.comparable_with(candidate.provenance)
    if not ok:
        raise EstimateError(f"cannot test incomparable estimates: {why}")
    d = np.asarray(candidate.per_item, float) - np.asarray(baseline.per_item, float)
    if d.size < 2:
        raise EstimateError("need at least 2 paired observations")
    if float(d.std(ddof=1)) == 0.0:
        return 0.0 if float(d.mean()) != 0.0 else 1.0
    return float(_st.ttest_rel(candidate.per_item, baseline.per_item).pvalue)
