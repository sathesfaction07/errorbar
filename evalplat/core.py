"""Four-state verdicts for the whole platform.

The distinction that matters, and the reason this is not a bool:

    FAIL           we measured, and it is bad
    COULD_NOT_RUN  we did not measure

A pipeline that renders the second as "0 failures" is the defect this capstone
exists to prevent (SD-4, and CAP-M7/S6). Every gate here returns one of four
states, and the runner's exit code distinguishes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    COULD_NOT_RUN = "COULD_NOT_RUN"
    SKIPPED = "SKIPPED"

    @property
    def is_conclusive(self) -> bool:
        """Only PASS and FAIL are measurements. The other two are absences, and an
        absence must never be aggregated into a pass rate."""
        return self in (Verdict.PASS, Verdict.FAIL)

    @property
    def blocks_promotion(self) -> bool:
        """COULD_NOT_RUN blocks. A gate that could not evaluate has not approved
        anything, and treating it as approval is how unverified changes ship."""
        return self in (Verdict.FAIL, Verdict.COULD_NOT_RUN)


@dataclass(frozen=True, slots=True)
class Result:
    claim: str
    verdict: Verdict
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"[{self.verdict.value:>13}] {self.claim} -- {self.reason}"


@dataclass
class Report:
    """Aggregates results and computes an exit code that cannot launder an absence.

    `required` names claims that MUST be conclusive. If any required claim is
    COULD_NOT_RUN the report exits 2 -- never 0 -- because "we could not check the
    thing that matters" is not success.
    """

    results: list[Result] = field(default_factory=list)
    required: set[str] = field(default_factory=set)

    def add(self, result: Result) -> Result:
        self.results.append(result)
        return result

    def require(self, *claims: str) -> None:
        self.required.update(claims)

    @property
    def conclusive(self) -> list[Result]:
        return [r for r in self.results if r.verdict.is_conclusive]

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if r.verdict is Verdict.FAIL]

    @property
    def unverified_required(self) -> list[Result]:
        seen = {r.claim: r for r in self.results}
        out = [
            seen[c]
            for c in sorted(self.required)
            if c in seen and not seen[c].verdict.is_conclusive
        ]
        out += [
            Result(c, Verdict.COULD_NOT_RUN, "required claim was never attempted")
            for c in sorted(self.required)
            if c not in seen
        ]
        return out

    def pass_rate(self) -> tuple[float, int]:
        """Denominator is CONCLUSIVE results only, and the count is returned so a
        reader can see how thin the evidence is. A 100% pass rate over 1 conclusive
        result out of 40 claims is not a 100% pass rate."""
        conc = self.conclusive
        if not conc:
            return (float("nan"), 0)
        return (sum(r.verdict is Verdict.PASS for r in conc) / len(conc), len(conc))

    def exit_code(self) -> int:
        if self.unverified_required:
            return 2
        if self.failures:
            return 1
        if not self.conclusive:
            return 2
        return 0

    def render(self) -> str:
        lines = [r.render() for r in self.results]
        rate, n = self.pass_rate()
        lines.append(
            f"pass_rate: {'n/a' if n == 0 else f'{rate:.0%}'} of {n} conclusive "
            f"({len(self.results)} claims total)"
        )
        for r in self.unverified_required:
            lines.append(f"UNVERIFIED REQUIRED: {r.claim} -- {r.reason}")
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(
                {
                    "results": [
                        {
                            "claim": r.claim,
                            "verdict": r.verdict.value,
                            "reason": r.reason,
                            "detail": r.detail,
                        }
                        for r in self.results
                    ],
                    "exit_code": self.exit_code(),
                    "pass_rate": self.pass_rate()[0],
                    "conclusive": self.pass_rate()[1],
                    "unverified_required": [r.claim for r in self.unverified_required],
                },
                fh,
                indent=2,
                default=str,
            )
