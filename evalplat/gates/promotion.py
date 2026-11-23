"""The promotion decision (CAP-M7). The seam this capstone exists to own.

Everything else in `evalplat` measures one thing well. This module is the only place
those measurements become an ACTION, and it is where real systems fail -- not because
any component is wrong, but because nobody defined what has to be true simultaneously
for a promotion to be legitimate.

`PromotionDecision` is the teaching core here, and it mirrors `Estimate`: it cannot
be constructed in a self-contradictory state.

    PromotionDecision(promote=True, gates=[... a COULD_NOT_RUN gate ...])
        -> raises. There is no way to express "ship it anyway".

    PromotionDecision(promote=True, gates=[])
        -> raises. A decision backed by no gates is vacuous, and a vacuous pass is
           indistinguishable from a real one once it reaches a dashboard.

WHAT MUST BE TRUE SIMULTANEOUSLY, and why each is not optional:

    coverage        we actually measured enough of the eval set   (S6)
    instrument      the judge agrees with humans above a floor    (S3)
    comparability   both sides share judge bundle and eval set    (S2)
    primary         the headline metric is BETTER, not merely up  (S1)
    guardrails      nothing else regressed                        (S1/S10)
    slices          no subgroup regressed under the aggregate     (S9)

A STATISTICAL DECISION WORTH ARGUING WITH. The primary metric is a single
pre-registered hypothesis, so it is NOT multiplicity-adjusted. The guardrails are a
family of "did anything break" tests, so they ARE adjusted among themselves via BH.
Adjusting the primary alongside the guardrails would penalise the one hypothesis you
committed to in advance, which is backwards; not adjusting the guardrails would mean
a wide guardrail suite blocks good changes by chance. Slices are adjusted within
their own family inside `gates.slice`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evalplat.core import Result, Verdict
from evalplat.gates.slice import SliceReport, SliceSpec, slice_gate
from evalplat.judges.agreement import AgreementReport
from evalplat.numbers.estimate import (
    EXACT_JUDGE, ComparisonVerdict, Estimate, compare,
)
from evalplat.stats.multiplicity import benjamini_hochberg, paired_pvalue


class PromotionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """A promotion decision that cannot contradict its own evidence."""

    promote: bool
    verdict: Verdict
    gates: tuple[Result, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.gates:
            raise PromotionError(
                "a PromotionDecision with no gates is vacuous. An empty gate list is "
                "not 'nothing went wrong' -- it is 'nothing was checked', and once it "
                "reaches a dashboard the two are indistinguishable."
            )
        blocking = [g.claim for g in self.gates if g.verdict.blocks_promotion]
        if self.promote and blocking:
            raise PromotionError(
                f"refusing to construct promote=True with blocking gate(s): "
                f"{', '.join(blocking)}. This is the central invariant of the module -- "
                "there is deliberately no override parameter, because an override is "
                "how every fail-closed gate eventually becomes advisory."
            )
        if self.promote and self.verdict is not Verdict.PASS:
            raise PromotionError(
                f"promote=True requires verdict PASS, got {self.verdict.value}"
            )

    @property
    def blocking(self) -> tuple[str, ...]:
        return tuple(g.claim for g in self.gates if g.verdict.blocks_promotion)

    def render(self) -> str:
        head = (
            f"[{self.verdict.value:>13}] PROMOTE={self.promote}  "
            f"({len(self.gates)} gates, {len(self.blocking)} blocking) -- {self.reason}"
        )
        return "\n".join([head, *(f"    {g.render()}" for g in self.gates)])

    def as_result(self) -> Result:
        return Result(
            "promotion.decision", self.verdict, self.reason,
            {
                "promote": self.promote,
                "blocking": list(self.blocking),
                "gates": {g.claim: g.verdict.value for g in self.gates},
            },
        )


def promotion_gate(
    *,
    primary: tuple[Estimate, Estimate],
    guardrails: Sequence[tuple[Estimate, Estimate]] = (),
    slices: Sequence[SliceSpec] = (),
    judge_coverage: Result | None = None,
    judge_agreement: AgreementReport | None = None,
    agreement_floor: float = 0.667,
    alpha: float = 0.05,
    require_slice_breakdown: bool = True,
) -> tuple[PromotionDecision, tuple[SliceReport, ...]]:
    """Compose every measurement into one four-state decision.

    `primary` is `(baseline, candidate)` on the headline metric. `guardrails` are
    further metric pairs that must not regress -- latency and cost belong here, and
    their direction is carried by their own `Metric`, so no sign handling is needed.

    Returns the decision plus the slice reports, so a caller can render the detail
    without re-running the comparisons.
    """
    baseline, candidate = primary
    gates: list[Result] = []

    # --- coverage (S6) -----------------------------------------------------------
    if judge_coverage is None:
        gates.append(Result(
            "gate.coverage", Verdict.COULD_NOT_RUN,
            "no coverage result supplied. Promotion without knowing how much of the "
            "eval set was actually scored is the null-judge defect wearing a hat.",
        ))
    else:
        gates.append(Result(
            "gate.coverage", judge_coverage.verdict,
            judge_coverage.reason, judge_coverage.detail,
        ))

    # --- instrument validity (S3) ------------------------------------------------
    judged = candidate.provenance.judge_bundle_hash != EXACT_JUDGE
    if not judged:
        gates.append(Result(
            "gate.instrument", Verdict.SKIPPED,
            "metric is measured exactly (no judge), so judge-human agreement does "
            "not apply; SKIPPED is recorded rather than omitted so the gate list is "
            "the same shape every run",
        ))
    elif judge_agreement is None:
        gates.append(Result(
            "gate.instrument", Verdict.COULD_NOT_RUN,
            "the metric is judge-scored but no agreement report was supplied. An "
            "uncalibrated judge produces numbers, not measurements.",
        ))
    else:
        ok = (
            judge_agreement.verdict is Verdict.PASS
            and judge_agreement.value == judge_agreement.value
            and judge_agreement.value >= agreement_floor
        )
        gates.append(Result(
            "gate.instrument",
            Verdict.PASS if ok else judge_agreement.verdict if not judge_agreement.verdict.is_conclusive else Verdict.FAIL,
            f"{judge_agreement.statistic}={judge_agreement.value:.4f} vs floor "
            f"{agreement_floor:.3f} -- {judge_agreement.reason}",
            {"value": judge_agreement.value, "floor": agreement_floor},
        ))

    # --- comparability (S2) ------------------------------------------------------
    comparable, why = baseline.provenance.comparable_with(candidate.provenance)
    gates.append(Result(
        "gate.comparability", Verdict.PASS if comparable else Verdict.FAIL, why,
        {"baseline": baseline.provenance.system_version,
         "candidate": candidate.provenance.system_version},
    ))

    # --- primary metric (S1) -- NOT multiplicity-adjusted, see module docstring ---
    pc = compare(baseline, candidate)
    gates.append(Result(
        "gate.primary", Verdict.PASS if pc.verdict.permits_promotion else Verdict.FAIL,
        pc.render(),
        {"verdict": pc.verdict.value, "difference": pc.difference,
         "diff_ci": None if pc.diff_interval is None
         else [pc.diff_interval.lo, pc.diff_interval.hi]},
    ))

    # --- guardrails -- adjusted as a family --------------------------------------
    if guardrails:
        pending = []
        for gb, gc in guardrails:
            gcmp = compare(gb, gc)
            try:
                p = paired_pvalue(gb, gc)
            except Exception:  # noqa: BLE001 -- no per-item scores; fall back to the CI
                p = 0.0 if gcmp.verdict.is_decisive else 1.0
            pending.append((gc.metric.name, gcmp, p))
        adj = benjamini_hochberg([p for *_, p in pending], alpha,
                                 [n for n, *_ in pending])
        for (name, gcmp, _), q, rejected in zip(pending, adj.qvalues, adj.rejected):
            regressed = gcmp.verdict is ComparisonVerdict.WORSE and rejected
            unknown = gcmp.verdict is ComparisonVerdict.INCOMPARABLE
            gates.append(Result(
                f"gate.guardrail[{name}]",
                Verdict.COULD_NOT_RUN if unknown else Verdict.FAIL if regressed else Verdict.PASS,
                f"{gcmp.verdict.value} (q={q:.4g}) -- {gcmp.reason}",
                {"verdict": gcmp.verdict.value, "qvalue": float(q)},
            ))

    # --- slices (S9) -------------------------------------------------------------
    reports: list[SliceReport] = []
    if not slices:
        if require_slice_breakdown:
            gates.append(Result(
                "gate.slices", Verdict.COULD_NOT_RUN,
                "no slice breakdown was supplied. The spec requires every headline "
                "number to carry one, because an aggregate can absorb a severe "
                "minority regression without moving. Pass require_slice_breakdown="
                "False only with a written reason.",
            ))
        else:
            gates.append(Result(
                "gate.slices", Verdict.SKIPPED,
                "slice breakdown explicitly waived by the caller",
            ))
    else:
        for spec in slices:
            rep = slice_gate(baseline, candidate, spec, alpha=alpha)
            reports.append(rep)
            gates.append(rep.as_result())

    # --- compose -----------------------------------------------------------------
    blocking = [g for g in gates if g.verdict.blocks_promotion]
    fails = [g for g in blocking if g.verdict is Verdict.FAIL]
    unknown = [g for g in blocking if g.verdict is Verdict.COULD_NOT_RUN]

    if fails:
        verdict, promote = Verdict.FAIL, False
        reason = (
            f"{len(fails)} gate(s) failed: {', '.join(g.claim for g in fails)}"
            + (f"; {len(unknown)} could not be evaluated" if unknown else "")
        )
    elif unknown:
        verdict, promote = Verdict.COULD_NOT_RUN, False
        reason = (
            f"{len(unknown)} gate(s) could not be evaluated "
            f"({', '.join(g.claim for g in unknown)}); refusing to promote on an "
            "incomplete measurement -- this is not a failure, it is an absence, and "
            "the two require different fixes"
        )
    else:
        verdict, promote = Verdict.PASS, True
        reason = (
            f"all {len(gates)} gate(s) satisfied: coverage, instrument, "
            "comparability, primary metric, guardrails and slices"
        )

    return PromotionDecision(promote, verdict, tuple(gates), reason), tuple(reports)


def naive_promotion_DO_NOT_USE(baseline: Estimate, candidate: Estimate) -> bool:
    """THE PLANTED DEFECT, retained as a counter-example.

    Compares the two point estimates and returns a bool. One line, obviously wrong
    once stated -- and yet functionally what a leaderboard sort does. It cannot
    express INDISTINGUISHABLE, cannot notice a judge change, cannot see a slice, and
    cannot say "I could not check". Every defect this capstone measures fits in here.
    """
    return candidate.point > baseline.point
