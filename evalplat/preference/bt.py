"""Bradley-Terry preference modelling (CAP-M7/S4).

The defect this replaces: ranking models by raw win rate. Win rate is only a valid
ranking when every model faces the same distribution of opponents. In every real
arena it does not -- new models are matched against the current leader, retired
models accumulated easy matchups, and human preference collection is opportunistic.

Bradley-Terry models P(i beats j) = sigmoid(theta_i - theta_j), so a win against a
strong opponent counts for more than a win against a weak one. `run_capstone.py` (S4)
measures the rank error under deliberately unbalanced pairings.

TWO HONESTY PROBLEMS THAT MOST IMPLEMENTATIONS GET WRONG.

1. IDENTIFIABILITY. The likelihood only depends on differences of theta, so the
   parameters are identified up to an additive constant. This module centres them
   (mean zero) and says so. An implementation that returns uncentred values will
   appear to disagree with every reference implementation for no real reason.

2. UNBOUNDED MLE. If a model never loses -- or the comparison graph is
   disconnected, so two groups never meet -- the MLE is at infinity and no finite
   ranking exists. The honest return is COULD_NOT_RUN. This is the interesting case
   in practice because it happens constantly with small arenas and new entrants,
   and an implementation that silently returns whatever the optimiser had after N
   iterations reports a confident ranking of a quantity that is not identified.
   `regularise=` allows an explicitly-requested ridge penalty, which makes the
   estimate finite and BIASED, and the bias is recorded in the result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from evalplat.core import Verdict


class PreferenceError(ValueError):
    pass


Game = tuple[str, str]
"""(winner, loser). Ties are not accepted: see `from_games`."""


@dataclass(frozen=True, slots=True)
class BTResult:
    items: tuple[str, ...]
    theta: tuple[float, ...]
    theta_ci: tuple[tuple[float, float], ...]
    verdict: Verdict
    reason: str
    n_games: int
    iterations: int
    regularised: float

    @property
    def ranking(self) -> tuple[str, ...]:
        """Strongest first. Ties broken by item name for determinism -- an unstable
        ranking makes a gate flaky for reasons unrelated to model quality."""
        return tuple(
            i for _, i in sorted(zip(self.theta, self.items), key=lambda t: (-t[0], t[1]))
        )

    def strength(self, item: str) -> float:
        return self.theta[self.items.index(item)]

    def render(self) -> str:
        head = (
            f"[{self.verdict.value:>13}] bradley-terry over {len(self.items)} items, "
            f"{self.n_games} games, {self.iterations} iters"
            + (f", ridge={self.regularised}" if self.regularised else "")
            + f" -- {self.reason}"
        )
        rows = [
            f"    {r+1}. {i:<12} theta={self.strength(i):+.4f} "
            f"[{self.theta_ci[self.items.index(i)][0]:+.4f}, "
            f"{self.theta_ci[self.items.index(i)][1]:+.4f}]"
            for r, i in enumerate(self.ranking)
        ]
        return "\n".join([head, *rows])


def _tally(games: Sequence[Game], items: Sequence[str]) -> np.ndarray:
    idx = {it: i for i, it in enumerate(items)}
    w = np.zeros((len(items), len(items)), dtype=float)
    for winner, loser in games:
        if winner not in idx or loser not in idx:
            raise PreferenceError(f"game ({winner}, {loser}) names an unknown item")
        if winner == loser:
            raise PreferenceError(f"self-play game ({winner}, {loser}) carries no information")
        w[idx[winner], idx[loser]] += 1
    return w


def _connected(w: np.ndarray) -> bool:
    """The comparison graph must be connected, else relative strength between the
    components is genuinely undefined -- not merely hard to estimate."""
    n = w.shape[0]
    adj = (w + w.T) > 0
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def _has_unbounded_mle(w: np.ndarray) -> tuple[bool, str]:
    """An item that never loses (or never wins) drives its theta to +/-infinity."""
    wins = w.sum(axis=1)
    losses = w.sum(axis=0)
    played = wins + losses
    for i in range(w.shape[0]):
        if played[i] == 0:
            return True, f"item {i} played no games"
        if losses[i] == 0:
            return True, f"item {i} never lost ({int(wins[i])}-0), so its theta is unbounded above"
        if wins[i] == 0:
            return True, f"item {i} never won (0-{int(losses[i])}), so its theta is unbounded below"
    return False, ""


def bradley_terry(
    games: Sequence[Game],
    items: Sequence[str] | None = None,
    *,
    regularise: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    n_boot: int = 400,
    seed: int = 20260920,
) -> BTResult:
    """Fit Bradley-Terry by the MM algorithm (Hunter 2004), centred, with bootstrap
    intervals on each theta.

    Returns COULD_NOT_RUN -- not a ranking -- when the MLE is unbounded or the
    comparison graph is disconnected, unless `regularise > 0` is explicitly passed.
    """
    if not games:
        raise PreferenceError(
            "no games supplied. An empty arena is a COULD_NOT_RUN for the caller to "
            "surface, not an empty ranking."
        )
    its = tuple(sorted(items)) if items is not None else tuple(
        sorted({x for g in games for x in g})
    )
    if len(its) < 2:
        raise PreferenceError(f"need at least 2 items, got {its}")
    w = _tally(games, its)

    if not _connected(w):
        return BTResult(
            its, tuple([float("nan")] * len(its)), tuple([(float("nan"),) * 2] * len(its)),
            Verdict.COULD_NOT_RUN,
            "the comparison graph is disconnected -- relative strength across "
            "components is undefined, not merely uncertain; collect a bridging match",
            len(games), 0, regularise,
        )

    unbounded, why = _has_unbounded_mle(w)
    if unbounded and regularise <= 0:
        return BTResult(
            its, tuple([float("nan")] * len(its)), tuple([(float("nan"),) * 2] * len(its)),
            Verdict.COULD_NOT_RUN,
            f"the maximum-likelihood estimate is unbounded: {why}. Pass "
            "regularise>0 to obtain a finite but BIASED estimate, and record that "
            "you did.",
            len(games), 0, regularise,
        )

    theta, iters = _fit_mm(w, regularise, max_iter, tol)

    rng = np.random.default_rng(seed)
    boots = []
    g = np.array(games, dtype=object)
    for _ in range(n_boot):
        sample = g[rng.integers(0, len(games), len(games))]
        try:
            wb = _tally([tuple(x) for x in sample], its)  # type: ignore[misc]
            if not _connected(wb):
                continue
            ub, _ = _has_unbounded_mle(wb)
            if ub and regularise <= 0:
                continue
            tb, _ = _fit_mm(wb, max(regularise, 1e-6), max_iter, tol)
            boots.append(tb)
        except PreferenceError:
            continue

    if len(boots) < max(30, n_boot // 10):
        ci = tuple([(float("nan"), float("nan"))] * len(its))
        reason = (
            f"fitted, but only {len(boots)}/{n_boot} bootstrap resamples were "
            "estimable, so intervals are not reported -- a point ranking without "
            "intervals must not be used for promotion"
        )
        verdict = Verdict.COULD_NOT_RUN
    else:
        B = np.vstack(boots)
        lo = np.quantile(B, 0.025, axis=0)
        hi = np.quantile(B, 0.975, axis=0)
        ci = tuple((float(a), float(b)) for a, b in zip(lo, hi))
        reason = f"fitted; intervals from {len(boots)}/{n_boot} estimable resamples"
        verdict = Verdict.PASS

    return BTResult(
        its, tuple(float(t) for t in theta), ci, verdict, reason,
        len(games), iters, regularise,
    )


def _fit_mm(
    w: np.ndarray, regularise: float, max_iter: int, tol: float
) -> tuple[np.ndarray, int]:
    """Minorisation-maximisation iteration (Hunter 2004) on the exponentiated scale,
    centred in log space at the end.

    Regularisation is applied as PSEUDO-COUNTS: `regularise` phantom wins are added
    to every ordered pair. That is a principled smoothing -- equivalent to a Beta
    prior on each pairwise win probability -- and it is what makes an otherwise
    unbounded MLE finite. An earlier version added an ad-hoc penalty term directly to
    the MM denominator, which had no probabilistic interpretation and therefore no
    defensible answer to "how biased is this estimate?".

    Working on p=exp(theta) keeps every update positive by construction, which avoids
    the sign errors a naive gradient step on theta invites.
    """
    n = w.shape[0]
    if regularise > 0:
        w = w + regularise * (1.0 - np.eye(n))
    N = w + w.T
    wins = w.sum(axis=1)
    p = np.ones(n, dtype=float)
    it = 0
    for it in range(1, max_iter + 1):
        prev = p / p.sum()
        for i in range(n):
            denom = sum(
                N[i, j] / (p[i] + p[j]) for j in range(n) if j != i and N[i, j] > 0
            )
            if denom > 0 and wins[i] > 0:
                p[i] = wins[i] / denom
        p = np.clip(p, 1e-300, None)
        p /= p.sum()
        if np.max(np.abs(np.log(p) - np.log(prev))) < tol:
            break
    theta = np.log(p)
    return theta - theta.mean(), it


def win_rate_ranking(games: Sequence[Game], items: Sequence[str] | None = None) -> dict[str, float]:
    """THE PLANTED DEFECT, retained as a counter-example.

    Raw win rate, ignoring who each item played. Correct only under balanced
    pairings, and nothing about the output reveals whether pairings were balanced.
    """
    its = tuple(sorted(items)) if items is not None else tuple(
        sorted({x for g in games for x in g})
    )
    wins: dict[str, int] = defaultdict(int)
    played: dict[str, int] = defaultdict(int)
    for winner, loser in games:
        wins[winner] += 1
        played[winner] += 1
        played[loser] += 1
    return {i: (wins[i] / played[i] if played[i] else float("nan")) for i in its}


def rank_displacement(truth: Sequence[str], estimate: Sequence[str]) -> int:
    """Sum of absolute rank displacements -- the total number of positions every
    item is away from where it belongs. Reported alongside Kendall tau because a
    single badly-misplaced item and many slightly-misplaced ones have very
    different operational consequences, and tau alone cannot distinguish them."""
    pos = {it: r for r, it in enumerate(estimate)}
    missing = [t for t in truth if t not in pos]
    if missing:
        raise PreferenceError(f"estimate is missing items {missing}")
    return int(sum(abs(r - pos[it]) for r, it in enumerate(truth)))
