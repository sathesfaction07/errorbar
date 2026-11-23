"""Interval constructors. The only legitimate way to make an `Estimate`.

Three methods, and the choice between them is not stylistic:

  `analytic_mean_ci`  Student-t on the mean. Correct for continuous scores with
                      roughly symmetric sampling distributions.
  `wilson_ci`         For PROPORTIONS (accuracy, pass rate, win rate). Use this
                      rather than a normal approximation -- see the note below.
  `bootstrap_ci`      Percentile bootstrap. Needed for statistics with no closed
                      form (medians, p95 latency, judge-score aggregates).

WHY WILSON AND NOT THE NORMAL APPROXIMATION.

The textbook interval `p +/- z*sqrt(p(1-p)/n)` fails in exactly the region eval
work lives in -- high accuracy, modest n:

  - At p=0.98, n=50 it produces an upper bound of 1.019. An accuracy above 1 is not
    a wide interval, it is a wrong one, and it silently means "no upper constraint".
  - At p=1.0 it produces a ZERO-WIDTH interval [1.0, 1.0], claiming perfect
    certainty from 50 observations. That is the most dangerous output in the module:
    a gate comparing against it can never fail.

`wilson_ci` is correct at the boundaries and never leaves [0,1]. The normal
approximation is retained as `_normal_approx_ci_DO_NOT_USE` because it is a
counter-example: it is what most implementations do, it is plausible, and
`tests/test_evalplat.py` measures exactly how it fails, and `run_capstone.py` S1
prints the two side by side.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats as _st

from evalplat.numbers.estimate import Estimate, EstimateError, Interval, Metric, Provenance

MIN_N = 2
"""Below this, no interval is defensible. Callers must surface COULD_NOT_RUN rather
than invent one -- a single observation has no sampling distribution."""


def _check(values: Sequence[float], what: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise EstimateError(f"{what}: expected a 1-D sequence, got shape {arr.shape}")
    if arr.size < MIN_N:
        raise EstimateError(
            f"{what}: n={arr.size} is below MIN_N={MIN_N}. This is a COULD_NOT_RUN, "
            "not a result -- do not substitute a point estimate."
        )
    if not np.all(np.isfinite(arr)):
        bad = int((~np.isfinite(arr)).sum())
        raise EstimateError(
            f"{what}: {bad} of {arr.size} values are non-finite. Dropping them "
            "silently would change the denominator without saying so."
        )
    return arr


def analytic_mean_ci(values: Sequence[float], level: float = 0.95) -> Interval:
    """Student-t interval on the mean. Uses t, not z: at n=30 the difference is ~4%
    of the half-width, which is enough to flip a marginal gate.

    The degenerate case is detected with `ptp` (max - min), NOT with
    `std(ddof=1) == 0.0`. On identical inputs numpy's variance is a tiny non-zero
    float (~1e-17 for 0.8 repeated), so the equality check silently missed the case
    and returned a ZERO-WIDTH interval labelled `analytic-t` -- indistinguishable
    from a genuine, infinitely precise measurement, and impossible for any gate to
    fail against. `ptp == 0` is exact for identical values and cannot drift.
    """
    arr = _check(values, "analytic_mean_ci")
    n = arr.size
    if float(np.ptp(arr)) == 0.0:
        m = float(arr[0])
        return Interval(m, m, level, "degenerate-zero-variance", n)
    se = float(arr.std(ddof=1) / math.sqrt(n))
    t = float(_st.t.ppf(1.0 - (1.0 - level) / 2.0, df=n - 1))
    m = float(arr.mean())
    iv = Interval(m - t * se, m + t * se, level, "analytic-t", n)
    if iv.width == 0.0:
        raise EstimateError(
            f"analytic_mean_ci produced a zero-width interval at n={n} despite "
            f"non-zero spread (ptp={float(np.ptp(arr)):.3e}). The half-width "
            "underflowed relative to the mean's magnitude. Refusing to return it: a "
            "zero-width interval claims perfect certainty and no gate can ever fail "
            "against it. Rescale the metric or use bootstrap_ci."
        )
    return iv


def wilson_ci(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a proportion. Correct at p=0 and p=1, and never
    escapes [0,1]."""
    if isinstance(successes, bool) or not isinstance(successes, int) or successes < 0:
        raise EstimateError(f"wilson_ci: successes must be a non-negative int, got {successes!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < MIN_N:
        raise EstimateError(
            f"wilson_ci: n must be an int >= {MIN_N}, got {n!r}; fewer observations "
            "than that is a COULD_NOT_RUN"
        )
    if successes > n:
        raise EstimateError(f"wilson_ci: successes={successes} exceeds n={n}")
    z = float(_st.norm.ppf(1.0 - (1.0 - level) / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    if not (lo <= p <= hi):
        raise EstimateError(
            f"wilson_ci produced [{lo}, {hi}] which excludes its own point {p} -- "
            "this is an implementation bug, refusing to return it"
        )
    return Interval(lo, hi, level, "wilson", n)


def _normal_approx_ci_DO_NOT_USE(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """RETAINED AS A COUNTER-EXAMPLE. Not exported, not used, deliberately ugly.

    This is the interval most eval code actually computes. `tests/test_evalplat.py`
    measures its two failure modes (bounds outside [0,1]; zero width at p=1) so the
    reason Wilson is mandatory is a measured fact in this repo rather than folklore.
    """
    z = float(_st.norm.ppf(1.0 - (1.0 - level) / 2.0))
    p = successes / n
    half = z * math.sqrt(p * (1 - p) / n)
    return (p - half, p + half)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] | None = None,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 20260920,
    support: tuple[float, float] | None = None,
) -> Interval:
    """Percentile bootstrap. Seeded, so the same data gives the same interval --
    an unseeded interval makes a gate nondeterministic, which reads as flakiness.

    `statistic=None` means the mean, and takes a vectorised fast path. It is a
    an explicit `None` rather than `lambda a: a.mean()` on purpose: an earlier version of this
    function dispatched on `statistic.__name__ != "<lambda>"`, which meant any
    caller passing a non-mean lambda silently got the MEAN computed instead of their
    statistic. Wrong answer, no error, no warning. Retained in the history as an
    example of why behaviour must never be keyed on a function's name.

    `support` clamps to a metric's valid range when one exists (e.g. (0,1) for a
    proportion). Clamping is honest here: the bound is a quantile of a resampling
    distribution, and a resample cannot legitimately fall outside the support.
    """
    arr = _check(values, "bootstrap_ci")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    if statistic is None:
        boot = arr[idx].mean(axis=1)
        label = "mean"
    else:
        boot = np.array([float(statistic(arr[row])) for row in idx])
        label = getattr(statistic, "__name__", "custom")
    if not np.all(np.isfinite(boot)):
        raise EstimateError(
            f"bootstrap_ci: statistic produced non-finite values on "
            f"{int((~np.isfinite(boot)).sum())} of {n_boot} resamples -- that is a "
            "COULD_NOT_RUN, not an interval"
        )
    alpha = (1.0 - level) / 2.0
    lo, hi = (float(x) for x in np.quantile(boot, [alpha, 1.0 - alpha]))
    if support is not None:
        lo, hi = max(support[0], lo), min(support[1], hi)
    return Interval(lo, hi, level, f"percentile-bootstrap({n_boot},{label})", arr.size)


def estimate_mean(
    values: Sequence[float],
    provenance: Provenance,
    metric: Metric,
    level: float = 0.95,
    method: str = "analytic",
    keep_per_item: bool = True,
) -> Estimate:
    """Build an Estimate for a mean. `keep_per_item` defaults True because it
    enables the paired comparison path, which is both correct and far more
    powerful than the independent fallback."""
    arr = _check(values, "estimate_mean")
    iv = analytic_mean_ci(arr, level) if method == "analytic" else bootstrap_ci(arr, level=level)
    return Estimate(
        point=float(arr.mean()),
        interval=iv,
        provenance=provenance,
        metric=metric,
        per_item=tuple(float(v) for v in arr) if keep_per_item else None,
    )


def estimate_proportion(
    successes: int,
    n: int,
    provenance: Provenance,
    metric: Metric,
    level: float = 0.95,
) -> Estimate:
    """Build an Estimate for a proportion using Wilson. There is deliberately no
    `method=` switch: the normal approximation is never the right answer here."""
    iv = wilson_ci(successes, n, level)
    per_item = tuple([1.0] * successes + [0.0] * (n - successes))
    return Estimate(
        point=successes / n,
        interval=iv,
        provenance=provenance,
        metric=metric,
        per_item=per_item,
    )
