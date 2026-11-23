"""Disaggregated slice gate (CAP-M7/S9).

The situation: a candidate improves overall accuracy by 2 points and is promoted. Six
weeks later someone notices German-language requests got materially worse. The
aggregate was true the whole time; it was also the wrong number to look at, because
an aggregate is a weighted average and a large majority slice can absorb a severe
minority regression without moving it.

So: **a slice regression blocks promotion even when the aggregate improves.**

Three design decisions that are not obvious.

1. **Multiplicity applies here.** Testing 8 slices is 8 tests, and at alpha=0.05 one
   of them looks "significantly worse" roughly 34% of the time even when nothing
   changed. A slice gate without adjustment is a gate that blocks good changes at a
   high rate, which gets it switched off within a month. This module runs BH across
   the slice family, so it composes `stats.multiplicity` rather than re-deriving it.
   That composition is the seam S9 shares with S5.

2. **A slice too small to evaluate BLOCKS, it does not pass.** "We could not check
   the German customers" is not "the German customers are fine". Under-powered
   slices return COULD_NOT_RUN and the gate fails closed. The alternative -- dropping
   them -- means the gate is silently blind to exactly the smallest, most
   at-risk populations.

3. **Slice assignment is positional and validated.** `assignments[i]` labels
   `per_item[i]`. A length mismatch raises rather than truncating, because a
   silently truncated assignment array mislabels every item after the first missing
   one, which produces confident nonsense.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from evalplat.core import Result, Verdict
from evalplat.numbers.ci import MIN_N, analytic_mean_ci
from evalplat.numbers.estimate import (
    Comparison,
    ComparisonVerdict,
    Estimate,
    EstimateError,
    Metric,
    compare,
)
from evalplat.stats.multiplicity import benjamini_hochberg, paired_pvalue


class SliceError(ValueError):
    pass


MIN_SLICE_N = 20
"""Below this a slice comparison is not worth making. Chosen, not derived -- and
that is the point: it is a declared threshold a reviewer can argue with, rather than
an implicit consequence of whatever the data happened to contain."""


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """A slicing of the eval set. `assignments[i]` is the slice label of item i."""

    name: str
    assignments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SliceError("SliceSpec.name must be non-empty")
        if not self.assignments:
            raise SliceError(f"SliceSpec {self.name!r} has no assignments")
        if any(not isinstance(a, str) or not a.strip() for a in self.assignments):
            raise SliceError(
                f"SliceSpec {self.name!r}: every assignment must be a non-empty string. "
                "An unlabelled item cannot be attributed to a population, and "
                "bucketing it as 'unknown' silently invents a slice."
            )

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.assignments)))

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.assignments))


@dataclass(frozen=True, slots=True)
class SliceOutcome:
    level: str
    n: int
    comparison: Comparison | None
    pvalue: float | None
    qvalue: float | None
    verdict: Verdict
    reason: str

    def render(self) -> str:
        head = f"    {self.level:<14} n={self.n:<5} [{self.verdict.value:>13}]"
        if self.comparison is None or self.comparison.diff_interval is None:
            return f"{head} {self.reason}"
        ci = self.comparison.diff_interval
        q = f" q={self.qvalue:.4g}" if self.qvalue is not None else ""
        return (
            f"{head} {self.comparison.verdict.value:<18} "
            f"diff={self.comparison.difference:+.4g} "
            f"[{ci.lo:+.4g}, {ci.hi:+.4g}]{q}"
        )


@dataclass(frozen=True, slots=True)
class SliceReport:
    metric: Metric
    slice_name: str
    aggregate: Comparison
    outcomes: tuple[SliceOutcome, ...]
    verdict: Verdict
    reason: str

    @property
    def regressed(self) -> tuple[str, ...]:
        return tuple(
            o.level for o in self.outcomes
            if o.comparison is not None and o.comparison.verdict is ComparisonVerdict.WORSE
        )

    @property
    def unevaluable(self) -> tuple[str, ...]:
        return tuple(o.level for o in self.outcomes if not o.verdict.is_conclusive)

    def render(self) -> str:
        lines = [
            f"[{self.verdict.value:>13}] slice gate on {self.slice_name!r} "
            f"({self.metric.name}) -- {self.reason}",
            f"    AGGREGATE      {self.aggregate.verdict.value:<18} "
            f"diff={self.aggregate.difference:+.4g}",
        ]
        lines += [o.render() for o in self.outcomes]
        return "\n".join(lines)

    def as_result(self) -> Result:
        return Result(
            f"s9.slice[{self.slice_name}]",
            self.verdict,
            self.reason,
            {
                "aggregate": self.aggregate.verdict.value,
                "regressed": list(self.regressed),
                "unevaluable": list(self.unevaluable),
                "levels": {o.level: o.n for o in self.outcomes},
            },
        )


def slice_gate(
    baseline: Estimate,
    candidate: Estimate,
    spec: SliceSpec,
    *,
    alpha: float = 0.05,
    min_slice_n: int = MIN_SLICE_N,
) -> SliceReport:
    """Compare per slice, adjust across the slice family, and fail closed.

    Requires `per_item` on both estimates: a slice breakdown is not derivable from
    aggregates, and approximating one would be fabrication.
    """
    if baseline.per_item is None or candidate.per_item is None:
        raise SliceError(
            "slice_gate requires per_item scores on both estimates. A slice "
            "breakdown cannot be reconstructed from aggregates -- there is no "
            "approximation here, only invention."
        )
    if len(baseline.per_item) != len(candidate.per_item):
        raise SliceError(
            f"per_item lengths differ ({len(baseline.per_item)} vs "
            f"{len(candidate.per_item)}); these are not paired observations"
        )
    if len(spec.assignments) != len(baseline.per_item):
        raise SliceError(
            f"SliceSpec {spec.name!r} has {len(spec.assignments)} assignments but the "
            f"estimates have {len(baseline.per_item)} items. Refusing to truncate: a "
            "misaligned assignment array mislabels every item after the gap."
        )

    aggregate = compare(baseline, candidate)
    b = np.asarray(baseline.per_item, float)
    c = np.asarray(candidate.per_item, float)
    assign = np.asarray(spec.assignments, dtype=object)

    pending: list[tuple[str, int, Comparison, float]] = []
    skipped: list[SliceOutcome] = []

    for level in spec.levels:
        mask = assign == level
        n = int(mask.sum())
        if n < max(min_slice_n, MIN_N):
            skipped.append(SliceOutcome(
                level, n, None, None, None, Verdict.COULD_NOT_RUN,
                f"n={n} is below the declared minimum {max(min_slice_n, MIN_N)}; this "
                "slice was not evaluated, which is not the same as it being unharmed",
            ))
            continue
        sb, sc = b[mask], c[mask]
        try:
            eb = Estimate(float(sb.mean()), analytic_mean_ci(sb), baseline.provenance,
                          baseline.metric, tuple(sb))
            ec = Estimate(float(sc.mean()), analytic_mean_ci(sc), candidate.provenance,
                          candidate.metric, tuple(sc))
            cmp_ = compare(eb, ec)
            p = paired_pvalue(eb, ec)
        except (EstimateError, ValueError) as exc:
            skipped.append(SliceOutcome(
                level, n, None, None, None, Verdict.COULD_NOT_RUN,
                f"slice comparison failed: {type(exc).__name__}: {exc}",
            ))
            continue
        pending.append((level, n, cmp_, p))

    outcomes: list[SliceOutcome] = []
    if pending:
        adj = benjamini_hochberg(
            [p for *_, p in pending], alpha, [lv for lv, *_ in pending]
        )
        for (level, n, cmp_, p), q, rejected in zip(pending, adj.qvalues, adj.rejected):
            worse = cmp_.verdict is ComparisonVerdict.WORSE and rejected
            better = cmp_.verdict is ComparisonVerdict.BETTER and rejected
            if worse:
                v, why = Verdict.FAIL, (
                    f"regression survives BH adjustment (q={q:.4g}) -- this slice is "
                    "materially worse and blocks promotion"
                )
            else:
                v, why = Verdict.PASS, (
                    f"no adjusted regression (q={q:.4g}"
                    + (", improved" if better else "") + ")"
                )
            outcomes.append(SliceOutcome(
                level, n, cmp_, p, float(q), v,
                why if cmp_.verdict is not ComparisonVerdict.INCOMPARABLE
                else f"INCOMPARABLE: {cmp_.reason}",
            ))
    outcomes.extend(skipped)
    outcomes.sort(key=lambda o: o.level)

    regressed = tuple(o.level for o in outcomes if o.verdict is Verdict.FAIL)
    unevaluable = tuple(o.level for o in outcomes if not o.verdict.is_conclusive)

    if regressed:
        verdict = Verdict.FAIL
        reason = (
            f"{len(regressed)} slice(s) regressed after BH adjustment "
            f"({', '.join(regressed)}) while the aggregate reads "
            f"{aggregate.verdict.value} -- the aggregate is a weighted average and "
            "absorbed it"
        )
    elif unevaluable:
        verdict = Verdict.COULD_NOT_RUN
        reason = (
            f"{len(unevaluable)} slice(s) could not be evaluated "
            f"({', '.join(unevaluable)}); the gate fails closed because an unmeasured "
            "population is not a safe one"
        )
    elif not outcomes:
        verdict = Verdict.COULD_NOT_RUN
        reason = (
            f"slicing {spec.name!r} produced no evaluable levels at all -- a gate with "
            "nothing to check passes vacuously, so it reports COULD_NOT_RUN instead"
        )
    else:
        verdict = Verdict.PASS
        reason = (
            f"all {len(outcomes)} slice(s) evaluated with no adjusted regression; "
            f"aggregate reads {aggregate.verdict.value}"
        )

    return SliceReport(candidate.metric, spec.name, aggregate, tuple(outcomes), verdict, reason)


def aggregate_only_gate_DO_NOT_USE(baseline: Estimate, candidate: Estimate) -> Result:
    """THE PLANTED DEFECT, retained as a counter-example.

    Looks only at the aggregate. This is what almost every promotion check actually
    does, and it is not stupid -- the aggregate is the number everyone agreed to
    optimise. It is simply blind to distribution, and `run_capstone.py` (S9) measures
    a case where it approves a change that severely harms one slice.
    """
    c = compare(baseline, candidate)
    ok = c.verdict is ComparisonVerdict.BETTER
    return Result(
        "s9.aggregate_only(defect)",
        Verdict.PASS if ok else Verdict.FAIL,
        f"aggregate {c.verdict.value} (diff={c.difference:+.4g}) -- no slice breakdown "
        "was consulted",
        {"verdict": c.verdict.value},
    )
