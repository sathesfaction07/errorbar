"""Judge-human agreement (CAP-M7/S3). The gate that makes a judge number mean
something.

The situation: a team has 200 human labels and an LLM judge. The judge's scores are
quoted in every review. Nobody ever computed whether the judge agrees with the
humans. If agreement is at chance, every judge-derived number on the leaderboard is
noise with a decimal point.

Two statistics, and the choice is not arbitrary:

  `cohen_kappa`         two raters, nominal categories. Chance-corrected.
  `krippendorff_alpha`  any number of raters, MISSING DATA ALLOWED, nominal or
                        interval measurement levels.

Krippendorff's alpha is the right default for judge auditing specifically because
of the missing-data tolerance: real human label sets are ragged (three annotators
on some items, one on others), and a statistic that requires a complete matrix
forces you to throw away the partially-labelled items -- which are usually the hard,
informative ones.

ORDINAL IS NOT IMPLEMENTED, AND IS NOT OFFERED. A 1-to-5 rubric is genuinely ordinal
and computing it as interval asserts that the 1-to-2 gap equals the 4-to-5 gap,
which is usually false for rubrics. The honest handling is to omit the option rather
than expose an enum member that raises at runtime, so a caller cannot select a level
that looks supported and is not. Recorded as a deliberate gap.

WHY RAW AGREEMENT IS NOT ACCEPTABLE. "The judge agrees with humans 85% of the time"
is meaningless without the base rate. If 85% of items are class A, a judge that
always answers A scores 85% raw agreement and has learned nothing. Both statistics
here are chance-corrected, so that judge scores ~0. `percent_agreement` is provided
ONLY so the gap can be measured, and it is named to discourage its use.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from evalplat.core import Verdict


class MeasurementLevel(str, Enum):
    """Only the two levels that are actually implemented. See the module docstring
    for why ORDINAL is absent rather than present-and-raising."""

    NOMINAL = "nominal"
    INTERVAL = "interval"


class AgreementError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AgreementReport:
    statistic: str
    value: float
    n_units: int
    n_raters: int
    n_ratings: int
    level: str
    verdict: Verdict
    reason: str
    floor: float | None = None

    def render(self) -> str:
        f = f" (floor {self.floor:.2f})" if self.floor is not None else ""
        return (
            f"[{self.verdict.value:>13}] {self.statistic}={self.value:.4f}{f} "
            f"over {self.n_units} units x {self.n_raters} raters "
            f"({self.n_ratings} ratings, {self.level}) -- {self.reason}"
        )


def _as_matrix(reliability_data: Sequence[Sequence[float | None]]) -> np.ndarray:
    """raters x units, NaN for missing -- matching the `krippendorff` package so the
    crosscheck compares like with like."""
    rows = [list(r) for r in reliability_data]
    if not rows:
        raise AgreementError("no raters supplied")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise AgreementError(
            f"ragged rater rows {sorted(widths)}. Pad with None for missing ratings "
            "rather than truncating -- truncation silently drops units."
        )
    m = np.array(
        [[np.nan if v is None else float(v) for v in r] for r in rows], dtype=float
    )
    if m.shape[0] < 2:
        raise AgreementError(
            f"agreement needs at least 2 raters, got {m.shape[0]}. A single rater "
            "agreeing with itself is not evidence."
        )
    return m


def _delta_sq(level: MeasurementLevel, values: np.ndarray) -> np.ndarray:
    """Squared difference function. Nominal treats every distinct pair as maximally
    different; interval uses the squared numeric gap."""
    v = values.reshape(-1, 1)
    if level is MeasurementLevel.NOMINAL:
        return (v != v.T).astype(float)
    if level is MeasurementLevel.INTERVAL:
        return (v - v.T) ** 2
    raise AgreementError(f"unsupported measurement level {level!r}")


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[float | None]],
    level: MeasurementLevel = MeasurementLevel.NOMINAL,
    floor: float | None = None,
) -> AgreementReport:
    """Krippendorff's alpha via the coincidence matrix.

    Units rated by fewer than 2 raters are EXCLUDED, because a single rating carries
    no information about agreement -- but the count of excluded units is reported, so
    a reader can see how much of the label set was unusable. Silently including them
    would inflate alpha; silently dropping them without saying so would hide that
    most of the data was single-rated.
    """
    m = _as_matrix(reliability_data)
    n_raters, n_units = m.shape

    usable = [u for u in range(n_units) if np.count_nonzero(~np.isnan(m[:, u])) >= 2]
    dropped = n_units - len(usable)
    if not usable:
        return AgreementReport(
            "krippendorff_alpha", float("nan"), 0, n_raters, 0, level.value,
            Verdict.COULD_NOT_RUN,
            f"every one of {n_units} units has fewer than 2 ratings -- agreement is "
            "not computable, which is different from agreement being poor",
            floor,
        )

    values = np.unique(m[~np.isnan(m)])
    if values.size < 2:
        return AgreementReport(
            "krippendorff_alpha", float("nan"), len(usable), n_raters,
            int(np.count_nonzero(~np.isnan(m))), level.value, Verdict.COULD_NOT_RUN,
            f"all ratings share a single value ({values.tolist()}); alpha is undefined "
            "because expected disagreement is zero -- not a high-agreement result",
            floor,
        )

    v_index = {float(v): i for i, v in enumerate(values)}
    coincidence = np.zeros((values.size, values.size), dtype=float)
    for u in usable:
        col = m[:, u]
        present = col[~np.isnan(col)]
        mu = present.size
        for a, b in itertools.permutations(range(mu), 2):
            coincidence[v_index[float(present[a])], v_index[float(present[b])]] += 1.0 / (mu - 1)

    n_c = coincidence.sum(axis=1)
    n_total = n_c.sum()
    d2 = _delta_sq(level, values)

    observed = float((coincidence * d2).sum() / n_total)
    # The diagonal of d2 is zero by construction, so the usual (n_k - [c==k])
    # correction only touches entries that are multiplied by zero. Written as a plain
    # outer product for clarity; the two forms are algebraically identical here.
    expected = float((np.outer(n_c, n_c) * d2).sum() / (n_total * (n_total - 1)))

    if expected == 0.0:
        return AgreementReport(
            "krippendorff_alpha", float("nan"), len(usable), n_raters, int(n_total),
            level.value, Verdict.COULD_NOT_RUN,
            "expected disagreement is zero, so alpha is undefined", floor,
        )

    alpha = 1.0 - observed / expected
    note = f"{dropped} unit(s) excluded for having <2 ratings" if dropped else "all units usable"
    if floor is None:
        return AgreementReport(
            "krippendorff_alpha", alpha, len(usable), n_raters, int(n_total),
            level.value, Verdict.PASS, f"computed; {note}", floor,
        )
    ok = alpha >= floor
    return AgreementReport(
        "krippendorff_alpha", alpha, len(usable), n_raters, int(n_total), level.value,
        Verdict.PASS if ok else Verdict.FAIL,
        (f"alpha {alpha:.4f} {'meets' if ok else 'is below'} the floor {floor:.2f}"
         + ("" if ok else " -- this judge is not usable as a measuring instrument, and "
            "any number derived from it must not be quoted")
         + f"; {note}"),
        floor,
    )


def cohen_kappa(rater_a: Sequence[float], rater_b: Sequence[float]) -> AgreementReport:
    """Cohen's kappa for two raters on nominal categories, complete data only."""
    a = np.asarray(list(rater_a)); b = np.asarray(list(rater_b))
    if a.size != b.size:
        raise AgreementError(f"rater lengths differ: {a.size} vs {b.size}")
    if a.size == 0:
        raise AgreementError("no ratings")
    cats = np.unique(np.concatenate([a, b]))
    if cats.size < 2:
        return AgreementReport(
            "cohen_kappa", float("nan"), a.size, 2, a.size * 2, "nominal",
            Verdict.COULD_NOT_RUN,
            "only one category present; kappa is undefined (expected agreement = 1)",
        )
    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((cats.size, cats.size))
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(0) / n) @ (cm.sum(1) / n))
    if pe == 1.0:
        return AgreementReport(
            "cohen_kappa", float("nan"), a.size, 2, a.size * 2, "nominal",
            Verdict.COULD_NOT_RUN, "expected agreement is 1; kappa undefined",
        )
    k = (po - pe) / (1 - pe)
    return AgreementReport(
        "cohen_kappa", float(k), a.size, 2, a.size * 2, "nominal", Verdict.PASS,
        f"observed={po:.4f} expected={pe:.4f}",
    )


def percent_agreement(rater_a: Sequence[float], rater_b: Sequence[float]) -> float:
    """RAW agreement. Deliberately returns a bare float with no verdict, so it can
    never be mistaken for a gateable statistic. Present only so tests can measure
    the gap between it and kappa on imbalanced data -- see the module docstring."""
    a = np.asarray(list(rater_a)); b = np.asarray(list(rater_b))
    if a.size != b.size or a.size == 0:
        raise AgreementError("percent_agreement needs equal-length non-empty inputs")
    return float((a == b).mean())
