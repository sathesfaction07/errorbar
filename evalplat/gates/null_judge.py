"""Null-judge fail-closed (CAP-M7/S6). The signature test, shared with CAP-A1.

The defect, stated plainly: the judge endpoint is unreachable. Every item fails to
score. The eval suite catches the exceptions, scores nothing, reports "0 failures",
and the pipeline goes green. Nobody is lying; the report is just structurally unable
to say "I could not look".

This module exists to make that outcome unreachable. Three rules:

  1. A judge error is COULD_NOT_RUN for that item -- never a score, never a zero.
     Scoring an unreachable judge as 0 would also block the pipeline, but for the
     wrong reason and with a number that would then pollute the leaderboard.
  2. The gate's verdict is COULD_NOT_RUN if coverage falls below a declared floor,
     and COULD_NOT_RUN BLOCKS. `Verdict.blocks_promotion` is True for it.
  3. The runner's EXIT CODE distinguishes it (2, not 0 and not 1), because CI reads
     exit codes, not prose. The acceptance criterion for this scenario is explicitly
     "assert the exit code, not just the report text".

The coverage floor is the subtle part. A suite where 3 of 500 items failed to score
is usually fine; one where 200 failed is not, even though both "have results". The
floor makes that judgement explicit and reviewable instead of implicit in whether
anyone happened to look at the skipped count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from evalplat.audit.provenance import EvalSet, JudgeBundle
from evalplat.core import Result, Verdict


class JudgeUnavailable(RuntimeError):
    """Raised by a judge client that cannot reach its backend. Distinct from a
    scoring error: this one says nothing about the item."""


class JudgeClient(Protocol):
    """Minimal judge interface. `score` either returns a float or raises."""

    bundle: JudgeBundle

    def score(self, item: dict) -> float: ...


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    scores: tuple[float, ...]
    scored_ids: tuple[str, ...]
    failed_ids: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def coverage(self) -> float:
        total = len(self.scored_ids) + len(self.failed_ids)
        return len(self.scored_ids) / total if total else 0.0


def score_eval_set(judge: JudgeClient, eval_set: EvalSet) -> ScoringOutcome:
    """Score every item, recording failures rather than dropping them.

    Note what is NOT done here: no retry, no default score, no silent skip. Retries
    belong to the client; a default score is a fabricated measurement; and a silent
    skip is the defect itself.
    """
    scores: list[float] = []
    ok_ids: list[str] = []
    bad_ids: list[str] = []
    errs: list[str] = []
    for item in eval_set.items:
        iid = str(item.get("id"))
        try:
            v = judge.score(item)
        except Exception as exc:  # noqa: BLE001 -- any judge failure is a non-result
            bad_ids.append(iid)
            errs.append(f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
            bad_ids.append(iid)
            errs.append(f"judge returned {v!r}, which is not a score")
            continue
        scores.append(float(v))
        ok_ids.append(iid)
    return ScoringOutcome(tuple(scores), tuple(ok_ids), tuple(bad_ids), tuple(errs))


def null_judge_gate(
    judge: JudgeClient,
    eval_set: EvalSet,
    *,
    coverage_floor: float = 0.98,
) -> Result:
    """The gate. Returns COULD_NOT_RUN -- which blocks -- when coverage is short.

    Deliberately returns a `Result` rather than raising: a raised exception can be
    caught by an over-broad `except` in a pipeline step and converted into a pass,
    whereas a Result carrying COULD_NOT_RUN has to be actively ignored.
    """
    if not (0.0 < coverage_floor <= 1.0):
        raise ValueError(f"coverage_floor must lie in (0,1], got {coverage_floor}")

    out = score_eval_set(judge, eval_set)
    n_total = len(eval_set)
    detail = {
        "judge_bundle_hash": judge.bundle.hash,
        "eval_set_hash": eval_set.hash,
        "n_items": n_total,
        "n_scored": len(out.scored_ids),
        "n_failed": len(out.failed_ids),
        "coverage": out.coverage,
        "coverage_floor": coverage_floor,
        "first_errors": list(out.errors[:3]),
    }

    if not out.scored_ids:
        return Result(
            "s6.null_judge",
            Verdict.COULD_NOT_RUN,
            (
                f"the judge scored 0 of {n_total} items -- this is NOT '0 failures'. "
                f"First error: {out.errors[0] if out.errors else 'none recorded'}"
            ),
            detail,
        )

    if out.coverage < coverage_floor:
        return Result(
            "s6.null_judge",
            Verdict.COULD_NOT_RUN,
            (
                f"coverage {out.coverage:.1%} is below the floor {coverage_floor:.1%} "
                f"({len(out.failed_ids)} of {n_total} items unscored). The measured "
                "subset may be biased toward items the judge finds easy, so the "
                "aggregate is not trustworthy even though it exists."
            ),
            detail,
        )

    return Result(
        "s6.null_judge",
        Verdict.PASS,
        f"judge scored {len(out.scored_ids)}/{n_total} items "
        f"({out.coverage:.1%} >= {coverage_floor:.1%})",
        detail,
    )


class UnreachableJudge:
    """A judge whose backend is down. THE FIXTURE for the signature test.

    It is not a mock that returns nothing -- it raises on every call, which is what a
    real unreachable endpoint does, and is precisely the input that a naive suite
    converts into a green build.
    """

    def __init__(self, bundle: JudgeBundle, message: str = "connection refused") -> None:
        self.bundle = bundle
        self._message = message
        self.calls = 0

    def score(self, item: dict) -> float:
        self.calls += 1
        raise JudgeUnavailable(f"{self._message} (item {item.get('id')})")


class FlakyJudge:
    """Fails a declared fraction of items. Used to test the coverage FLOOR rather
    than the total-outage case -- the floor is where the interesting judgement is,
    and a gate that only catches total outages misses partial ones entirely."""

    def __init__(self, bundle: JudgeBundle, fail_every: int, score: float = 0.8) -> None:
        if fail_every < 1:
            raise ValueError("fail_every must be >= 1")
        self.bundle = bundle
        self._fail_every = fail_every
        self._score = score
        self.calls = 0

    def score(self, item: dict) -> float:
        self.calls += 1
        if self.calls % self._fail_every == 0:
            raise JudgeUnavailable(f"transient failure (item {item.get('id')})")
        return self._score


class WorkingJudge:
    """A judge that works. Present because a gate that cannot PASS is not a gate --
    the reverse test matters as much as the forward one."""

    def __init__(self, bundle: JudgeBundle, scorer: Callable[[dict], float] | None = None) -> None:
        self.bundle = bundle
        self._scorer = scorer or (lambda item: float(item.get("gold", 0.8)))
        self.calls = 0

    def score(self, item: dict) -> float:
        self.calls += 1
        return float(self._scorer(item))


def naive_score_eval_set_DO_NOT_USE(judge: JudgeClient, eval_set: EvalSet) -> list[float]:
    """THE PLANTED DEFECT, retained as a counter-example.

    This is the code that produces "0 failures" from a dead judge. It is four lines,
    it looks defensive, and it is the exact shape found in real eval harnesses. The
    bug is not the `except` -- it is that the caller receives a list and has no way
    to distinguish "scored 500 items" from "scored none".
    """
    scores = []
    for item in eval_set.items:
        try:
            scores.append(judge.score(item))
        except Exception:  # noqa: BLE001, S110
            pass
    return scores
