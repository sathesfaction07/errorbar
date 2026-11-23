"""The Estimate type -- the teaching core of CAP-M7.

An `Estimate` cannot be constructed without an interval and a provenance triple.
That is the whole point: "promoted on a 2-point gain that was inside noise" is not
prevented by a policy or a code review, it is prevented by there being no way to
express a bare number in this system.

Four refusals are built in, and each corresponds to a real incident shape:

  1. No interval             -> cannot construct.  (a point estimate is not a result)
  2. Mismatched provenance   -> cannot compare.    (judge upgrade invalidates history)
  3. Difference spans zero   -> cannot rank.       (a gain inside noise is not a gain)
  4. Unknown metric direction-> cannot construct.  (lower-is-better compared upward)

A NOTE ON THE THIRD, because it is the one usually implemented wrongly.

The intuitive test for "is A better than B" is: do their confidence intervals
overlap? That test is WRONG, and wrong in both directions:

  - Two 95% CIs can OVERLAP while the difference IS significant at 95%. Interval
    overlap is a much stricter test than it looks, so it silently costs power.
  - Under pairing, marginal intervals can look nearly identical while the paired
    difference is large and tight, because per-item variation cancels.

So this module never compares intervals to each other. It requires an interval on
the DIFFERENCE, computed paired when both sides share an eval set and carry per-item
scores, and from combined standard errors otherwise -- recording which happened on
the `Comparison`, because the two are not equally trustworthy.

A NOTE ON THE FOURTH. `Metric` carries `higher_is_better` and construction fails
without it. Latency and cost are first-class eval axes (S10), and on those axes an
improvement is a DECREASE. A system that stores metric direction in the caller's
head eventually promotes a regression; storing it in the type makes that impossible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats as _st

EXACT_JUDGE = "EXACT"
"""The reserved `judge_bundle_hash` value for deterministic metrics (accuracy, latency, cost)
that involve no judge. It is an assertion -- "no judge participated" -- and it must
match on both sides of a comparison like any other provenance component, so a
judge-scored number can never be compared against an exactly-measured one."""


class EstimateError(ValueError):
    """A refusal to construct or compare. `ValueError` because the input was
    invalid, not because the system failed -- callers should not retry."""


class ComparisonVerdict(str, Enum):
    """Four states, never three.

    `INDISTINGUISHABLE` and `INCOMPARABLE` are different facts, and collapsing them
    is precisely how noise-promotion happens. "Not significantly better" is a
    completed measurement. "Cannot be compared at all" is a MISSING measurement and
    a pipeline defect. Only the second one means someone has to go fix something.
    """

    BETTER = "BETTER"
    WORSE = "WORSE"
    INDISTINGUISHABLE = "INDISTINGUISHABLE"
    INCOMPARABLE = "INCOMPARABLE"

    @property
    def is_decisive(self) -> bool:
        return self in (ComparisonVerdict.BETTER, ComparisonVerdict.WORSE)

    @property
    def permits_promotion(self) -> bool:
        """Only BETTER permits promotion. Note that INDISTINGUISHABLE does not --
        'no worse' is a different claim from 'better' and must be argued separately."""
        return self is ComparisonVerdict.BETTER


@dataclass(frozen=True, slots=True)
class Metric:
    """What was measured, and which direction counts as an improvement.

    Direction is not optional and has no default. A default would be wrong half the
    time on the cost/latency axes, and wrong silently.
    """

    name: str
    higher_is_better: bool
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EstimateError("Metric.name must be a non-empty string")
        if not isinstance(self.higher_is_better, bool):
            raise EstimateError(
                f"Metric.higher_is_better must be an explicit bool, got "
                f"{self.higher_is_better!r}. There is no safe default: on accuracy an "
                "improvement is an increase, on latency and cost it is a decrease."
            )

    def improvement_sign(self) -> int:
        return 1 if self.higher_is_better else -1


@dataclass(frozen=True, slots=True)
class Provenance:
    """The triple that makes two numbers comparable, or proves they are not.

    `system_version` is EXPECTED to differ across a comparison -- that is the thing
    being compared. The other two must match, or the two numbers were produced by
    different instruments on different data and their difference means nothing.
    """

    system_version: str
    judge_bundle_hash: str
    eval_set_hash: str

    def __post_init__(self) -> None:
        for name in ("system_version", "judge_bundle_hash", "eval_set_hash"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise EstimateError(
                    f"Provenance.{name} must be a non-empty string, got {v!r}. An "
                    "unknown provenance is not a provenance -- a number whose origin "
                    "you cannot state cannot legitimately enter a comparison."
                )

    def comparable_with(self, other: "Provenance") -> tuple[bool, str]:
        """Comparability concerns the instrument and the data, not the subject."""
        if self.judge_bundle_hash != other.judge_bundle_hash:
            return False, (
                f"judge bundle differs ({self.judge_bundle_hash[:12]} vs "
                f"{other.judge_bundle_hash[:12]}): scores from different judges are "
                "not on a common scale, so their difference is not a quality change"
            )
        if self.eval_set_hash != other.eval_set_hash:
            return False, (
                f"eval set differs ({self.eval_set_hash[:12]} vs "
                f"{other.eval_set_hash[:12]}): different questions, different test"
            )
        return True, "same judge bundle and eval set"


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval that knows how it was built.

    `method` and `level` are not decoration. A 95% bootstrap BCa interval and a 95%
    normal approximation on the same data differ materially at small n or on skewed
    metrics, and a reader comparing two numbers needs to know whether they were
    built the same way before trusting the difference.
    """

    lo: float
    hi: float
    level: float
    method: str
    n: int

    def __post_init__(self) -> None:
        for name in ("lo", "hi"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise EstimateError(f"Interval.{name} must be a real number, got {v!r}")
            if not math.isfinite(float(v)):
                raise EstimateError(
                    f"Interval.{name} is {v!r}. A non-finite bound is not an interval: "
                    "if the estimator diverged that is a COULD_NOT_RUN, not a result."
                )
        if self.lo > self.hi:
            raise EstimateError(f"Interval inverted: lo={self.lo} > hi={self.hi}")
        if not (0.0 < self.level < 1.0):
            raise EstimateError(
                f"Interval.level must lie in (0,1), got {self.level}. A 95% interval "
                "has level=0.95, not 95 -- and silently accepting 95 would produce an "
                "absurdly wide interval that never blocks anything."
            )
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise EstimateError(
                f"Interval.n must be a positive int, got {self.n!r}. An interval over "
                "zero observations is not an interval."
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise EstimateError("Interval.method must record how the interval was built")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def overlaps(self, other: "Interval") -> bool:
        """Provided ONLY so `run_capstone.py` (scenario S1b) can measure how often
        the naive overlap test disagrees with the correct difference test. It is not
        used by `compare()` and must not be: see the module docstring."""
        return self.lo <= other.hi and other.lo <= self.hi

    def standard_error(self) -> float:
        """Back out an approximate SE from the interval half-width.

        Only valid for symmetric, normal-approximation-shaped intervals. Bootstrap
        BCa intervals are deliberately asymmetric, so this is an approximation --
        which is exactly why `compare()` prefers the paired path when per-item
        scores are available and labels the fallback as an approximation.
        """
        z = float(_st.norm.ppf(1.0 - (1.0 - self.level) / 2.0))
        if z <= 0:
            raise EstimateError(f"degenerate level {self.level}")
        return self.width / (2.0 * z)


@dataclass(frozen=True, slots=True)
class Estimate:
    """A number that cannot exist without its uncertainty and its origin.

    `per_item` is optional but load-bearing when present: two estimates over the
    same eval set with per-item scores support a PAIRED difference, which is both
    more powerful and more correct than combining marginal SEs. Without it,
    `compare()` must assume independence, and says so in the result.
    """

    point: float
    interval: Interval
    provenance: Provenance
    metric: Metric
    per_item: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.point, bool) or not isinstance(self.point, (int, float)):
            raise EstimateError(f"Estimate.point must be a real number, got {self.point!r}")
        if not math.isfinite(float(self.point)):
            raise EstimateError(f"Estimate.point is {self.point!r}, which is not a number")
        if not isinstance(self.interval, Interval):
            raise EstimateError(
                "Estimate requires an Interval. This is the central invariant of CAP-M7: "
                "a bare point estimate cannot be represented in this system. If you have "
                "no interval you do not have a result -- compute one with numbers.ci, or "
                "return COULD_NOT_RUN."
            )
        if not isinstance(self.provenance, Provenance):
            raise EstimateError(
                "Estimate requires a Provenance triple (system_version, "
                "judge_bundle_hash, eval_set_hash). A number of unknown origin cannot "
                "be compared with anything, which makes it useless for promotion."
            )
        if not isinstance(self.metric, Metric):
            raise EstimateError(
                "Estimate requires a Metric (name + higher_is_better), not a bare "
                "string. Direction must travel with the number, or a latency "
                "regression eventually reads as an improvement."
            )
        if not self.interval.contains(float(self.point)):
            raise EstimateError(
                f"point {self.point} lies outside its own interval "
                f"[{self.interval.lo}, {self.interval.hi}]. This is an estimator bug; "
                "shipping it would corrupt every downstream comparison."
            )
        if self.per_item is not None:
            if len(self.per_item) != self.interval.n:
                raise EstimateError(
                    f"per_item has {len(self.per_item)} values but interval.n is "
                    f"{self.interval.n}. One is wrong, and the paired comparison path "
                    "would silently use the wrong denominator."
                )
            if not all(
                not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                for v in self.per_item
            ):
                raise EstimateError("per_item must contain only finite real numbers")

    def __gt__(self, other: object) -> bool:
        raise EstimateError(
            "Refusing `>` on Estimate. Ordering estimates by their point values is the "
            "exact defect this type exists to prevent -- it is how a gain inside noise "
            "becomes a promotion. Use numbers.estimate.compare(), which returns a "
            "four-state verdict including INDISTINGUISHABLE."
        )

    __lt__ = __ge__ = __le__ = __gt__

    def render(self) -> str:
        """There is no code path that renders a point without its interval. The
        asymmetry is deliberate: the easy path has to be the correct one."""
        unit = f" {self.metric.unit}" if self.metric.unit else ""
        arrow = "higher-better" if self.metric.higher_is_better else "lower-better"
        return (
            f"{self.metric.name}={self.point:.4g}{unit} "
            f"[{self.interval.lo:.4g}, {self.interval.hi:.4g}] "
            f"({self.interval.level:.0%} {self.interval.method}, n={self.interval.n}, {arrow})"
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """The result of comparing two estimates. Carries the difference interval and
    the method, so a reader can audit the decision rather than trust it."""

    verdict: ComparisonVerdict
    metric: Metric
    baseline: str
    candidate: str
    difference: float
    diff_interval: Interval | None
    method: str
    reason: str

    def render(self) -> str:
        if self.diff_interval is None:
            return (
                f"{self.metric.name}: {self.verdict.value} "
                f"({self.candidate} vs {self.baseline}) -- {self.reason}"
            )
        return (
            f"{self.metric.name}: {self.verdict.value} "
            f"({self.candidate} vs {self.baseline}) "
            f"diff={self.difference:+.4g} "
            f"[{self.diff_interval.lo:+.4g}, {self.diff_interval.hi:+.4g}] "
            f"via {self.method} -- {self.reason}"
        )


def compare(
    baseline: Estimate,
    candidate: Estimate,
    *,
    n_boot: int = 10_000,
    seed: int = 20260920,
) -> Comparison:
    """Compare two estimates on the difference, never on interval overlap.

    Returns INCOMPARABLE (not False, not an exception) when provenance or metric
    disagree, because "these cannot be compared" is a finding the pipeline must
    surface and act on, not an error to be caught and dropped.
    """
    if not isinstance(baseline, Estimate) or not isinstance(candidate, Estimate):
        raise EstimateError("compare() requires two Estimate instances")

    if baseline.metric != candidate.metric:
        return Comparison(
            verdict=ComparisonVerdict.INCOMPARABLE,
            metric=candidate.metric,
            baseline=baseline.provenance.system_version,
            candidate=candidate.provenance.system_version,
            difference=float("nan"),
            diff_interval=None,
            method="none",
            reason=(
                f"different metrics ({baseline.metric.name} vs {candidate.metric.name}) "
                "-- or the same name with opposite direction, which is worse"
            ),
        )

    ok, why = baseline.provenance.comparable_with(candidate.provenance)
    if not ok:
        return Comparison(
            verdict=ComparisonVerdict.INCOMPARABLE,
            metric=candidate.metric,
            baseline=baseline.provenance.system_version,
            candidate=candidate.provenance.system_version,
            difference=float("nan"),
            diff_interval=None,
            method="none",
            reason=why,
        )

    raw_diff = float(candidate.point) - float(baseline.point)
    level = min(baseline.interval.level, candidate.interval.level)

    paired = (
        baseline.per_item is not None
        and candidate.per_item is not None
        and len(baseline.per_item) == len(candidate.per_item)
    )

    if paired:
        assert baseline.per_item is not None and candidate.per_item is not None
        d = np.asarray(candidate.per_item, float) - np.asarray(baseline.per_item, float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        boot = d[idx].mean(axis=1)
        alpha = (1.0 - level) / 2.0
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
        method = f"paired-bootstrap({n_boot})"
        diff_iv = Interval(float(lo), float(hi), level, method, int(d.size))
        raw_diff = float(d.mean())
    else:
        se = math.sqrt(
            baseline.interval.standard_error() ** 2 + candidate.interval.standard_error() ** 2
        )
        z = float(_st.norm.ppf(1.0 - (1.0 - level) / 2.0))
        method = "independent-normal-approx"
        diff_iv = Interval(
            raw_diff - z * se,
            raw_diff + z * se,
            level,
            method,
            min(baseline.interval.n, candidate.interval.n),
        )

    if not diff_iv.excludes_zero():
        return Comparison(
            verdict=ComparisonVerdict.INDISTINGUISHABLE,
            metric=candidate.metric,
            baseline=baseline.provenance.system_version,
            candidate=candidate.provenance.system_version,
            difference=raw_diff,
            diff_interval=diff_iv,
            method=method,
            reason=(
                f"the {level:.0%} interval on the difference includes zero, so the "
                f"observed {raw_diff:+.4g} is not distinguishable from noise"
            ),
        )

    improved = (raw_diff * candidate.metric.improvement_sign()) > 0
    return Comparison(
        verdict=ComparisonVerdict.BETTER if improved else ComparisonVerdict.WORSE,
        metric=candidate.metric,
        baseline=baseline.provenance.system_version,
        candidate=candidate.provenance.system_version,
        difference=raw_diff,
        diff_interval=diff_iv,
        method=method,
        reason=(
            f"the {level:.0%} interval on the difference excludes zero and the change "
            f"is in the {'improving' if improved else 'regressing'} direction for a "
            f"{'higher' if candidate.metric.higher_is_better else 'lower'}-is-better metric"
        ),
    )
