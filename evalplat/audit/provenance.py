"""Canonical hashing for provenance. The mechanism behind "judge upgrade
invalidates history" (CAP-M7/S2).

The whole defect class this prevents: someone edits a judge prompt, redeploys, and
every historical score silently becomes incomparable to every new score. Nothing
errors. The leaderboard keeps ranking. The comparison is just meaningless.

The only defence is a hash that covers EVERYTHING affecting the output, so an edit
necessarily changes the identity of the instrument. The hard part is not the hash,
it is the completeness of what goes into it -- a hash over the prompt text alone
misses the model id, the temperature, the rubric, the few-shot examples and the
parser, any one of which shifts scores.

So `canonical_hash` is strict by construction:

  - Unknown/unhashable types are REFUSED, not coerced via `str()`. `str()` on an
    object includes its memory address, which makes the hash unstable across runs;
    on a dict it is order-dependent in older Pythons. Either way the hash silently
    stops meaning anything.
  - Floats are serialised with `repr`, which round-trips exactly in Python 3.
  - Dict keys are sorted, so insertion order cannot change the hash.
  - NaN is refused: `nan != nan`, so a config containing one could never compare
    equal to itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


class ProvenanceError(ValueError):
    """A refusal to hash. Always a bug in the caller's config, not a transient."""


_ALLOWED = (str, int, float, bool, type(None), list, tuple, dict)


def _canonicalise(obj: Any, path: str = "$") -> Any:
    """Convert to a JSON-serialisable form with a stable ordering, refusing
    anything whose textual form is not guaranteed stable across processes."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ProvenanceError(
                f"{path}: NaN cannot appear in provenance -- NaN != NaN, so the "
                "config could never compare equal to itself"
            )
        if math.isinf(obj):
            raise ProvenanceError(f"{path}: infinity cannot appear in provenance")
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj, key=str):
            if not isinstance(k, str):
                raise ProvenanceError(
                    f"{path}: dict key {k!r} is {type(k).__name__}, not str. Non-string "
                    "keys serialise inconsistently and would make the hash unstable."
                )
            out[k] = _canonicalise(obj[k], f"{path}.{k}")
        return out
    raise ProvenanceError(
        f"{path}: {type(obj).__name__} is not hashable as provenance. Refusing to "
        f"fall back to str() -- for most objects that embeds a memory address, which "
        f"would change the hash every run and silently destroy comparability. "
        f"Convert it to a dict/list/scalar explicitly, so the choice is visible."
    )


def canonical_hash(obj: Any, *, prefix: str = "") -> str:
    """Stable SHA-256 over a canonicalised structure. The `prefix` is a domain tag
    (`judge`, `evalset`) so two different kinds of artifact with identical contents
    do not collide into one identity."""
    payload = json.dumps(
        {"prefix": prefix, "body": _canonicalise(obj)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A pinned evaluation set. `items` is the full content, not a filename: two
    files with the same name and different contents are the exact failure this
    guards against."""

    name: str
    items: tuple[dict[str, Any], ...]
    version: str

    def __post_init__(self) -> None:
        if not self.items:
            raise ProvenanceError(
                f"EvalSet {self.name!r} is empty. An eval over zero items produces no "
                "measurement, and a gate reading it would pass vacuously."
            )
        ids = [it.get("id") for it in self.items]
        if any(i is None for i in ids):
            raise ProvenanceError(f"EvalSet {self.name!r}: every item needs an 'id'")
        if len(set(map(str, ids))) != len(ids):
            dupes = len(ids) - len(set(map(str, ids)))
            raise ProvenanceError(
                f"EvalSet {self.name!r} has {dupes} duplicate item id(s). Duplicates "
                "silently reweight the metric toward the repeated items."
            )

    @property
    def hash(self) -> str:
        return canonical_hash(
            {"name": self.name, "version": self.version, "items": list(self.items)},
            prefix="evalset",
        )

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class JudgeBundle:
    """Everything that determines a judge's output, hashed as one identity.

    Each field is here because omitting it is a real incident:
      `model_id`      swapping the backing model changes scores
      `prompt`        the obvious one, and the only one most systems version
      `rubric`        scoring criteria drift independently of the prompt
      `temperature`   non-zero sampling makes scores irreproducible
      `few_shot`      examples steer the judge as strongly as instructions
      `parser_version`how text becomes a number is part of the measurement
    """

    model_id: str
    prompt: str
    rubric: str
    temperature: float
    parser_version: str
    few_shot: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ProvenanceError("JudgeBundle.temperature must be a real number")
        if self.temperature < 0:
            raise ProvenanceError(f"negative temperature {self.temperature}")
        for f in ("model_id", "prompt", "rubric", "parser_version"):
            if not str(getattr(self, f)).strip():
                raise ProvenanceError(
                    f"JudgeBundle.{f} is empty. Every field here is part of the "
                    "instrument's identity; a blank one means the identity is a lie."
                )

    @property
    def hash(self) -> str:
        return canonical_hash(
            {
                "model_id": self.model_id,
                "prompt": self.prompt,
                "rubric": self.rubric,
                "temperature": float(self.temperature),
                "parser_version": self.parser_version,
                "few_shot": list(self.few_shot),
            },
            prefix="judge",
        )

    def with_changes(self, **kw: Any) -> "JudgeBundle":
        """Produce a modified bundle. Used by the planted-defect fixtures to show
        that ANY change -- including a single character of the prompt -- yields a
        different hash and therefore breaks comparability."""
        cur = {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "rubric": self.rubric,
            "temperature": self.temperature,
            "parser_version": self.parser_version,
            "few_shot": self.few_shot,
        }
        unknown = set(kw) - set(cur)
        if unknown:
            raise ProvenanceError(
                f"unknown JudgeBundle field(s): {sorted(unknown)}. Silently ignoring "
                "them would let a caller believe they had changed the judge when they "
                "had not."
            )
        cur.update(kw)
        return JudgeBundle(**cur)  # type: ignore[arg-type]
