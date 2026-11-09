"""LLM-as-judge for explanation quality, and the calibration that licenses it.

Binary accuracy is not enough. A detector can answer DEFECT for completely the
wrong reason and still score a true positive, so we also ask whether the stated
mechanism matches the real one.

But a judge is a model too, and an unvalidated judge just moves the trust
problem one layer down. So before any judge score is reported, the judge is
tested on cases whose answer is known by construction:

    CEILING — the ground truth is handed back as the prediction. A judge that
              cannot score this MATCH is broken.
    FLOOR   — another item's ground truth is handed in. A judge that scores
              this MATCH is rubber-stamping.

If either probe fails, judged numbers are suppressed rather than printed with a
caveat. A number nobody should trust is worse than no number.
"""
from __future__ import annotations

import sys

import re
from dataclasses import dataclass

from .dataset import Item
from .detect import Prediction
from .providers import Provider

PROMPT = """A reviewer was asked to find the methodology defect in some code.

THE ACTUAL DEFECT:
{truth}

THE REVIEWER SAID:
{claim}

Did the reviewer identify the same underlying mechanism? Wording may differ \
completely; what matters is whether they found the same problem. Naming a \
different real problem, or being vague enough to fit anything, is NOT a match.

Answer with one word, MATCH or MISS, then a brief reason:
GRADE: MATCH or MISS
REASON: ..."""

_G = re.compile(r"GRADE:\s*(MATCH|MISS)", re.I)


@dataclass
class Calibration:
    ceiling: float          # fraction of identical-truth probes graded MATCH
    floor: float            # fraction of mismatched probes graded MATCH
    n: int

    @property
    def usable(self) -> bool:
        """Demands real discrimination, not just a gap."""
        return self.ceiling >= 0.80 and self.floor <= 0.20

    def __str__(self) -> str:
        verdict = "USABLE" if self.usable else "NOT USABLE — judged scores suppressed"
        return (f"judge calibration on {self.n} probes: "
                f"ceiling {self.ceiling:.0%} (want >=80%), "
                f"floor {self.floor:.0%} (want <=20%) -> {verdict}")


def _grade(provider: Provider, truth: str, claim: str) -> bool:
    try:
        raw = provider.complete(PROMPT.format(truth=truth, claim=claim))
    except Exception:
        return False
    m = _G.search(raw)
    return bool(m) and m.group(1).upper() == "MATCH"


def calibrate(items: list[Item], provider: Provider,
              n: int = 8, progress: bool | None = None) -> Calibration:
    progress = sys.stdout.isatty() if progress is None else progress
    defective = [i for i in items if i.has_defect][:n]
    if len(defective) < 3:
        return Calibration(0.0, 1.0, len(defective))
    ceil_hits = floor_hits = 0
    for k, it in enumerate(defective, 1):
        other = defective[(k) % len(defective)]
        ceil_hits += _grade(provider, it.ground_truth, it.ground_truth)
        floor_hits += _grade(provider, it.ground_truth, other.ground_truth)
        if progress:
            print(f"\r  calibrating judge: {k}/{len(defective)}", end="", flush=True)
    if progress:
        print()
    m = len(defective)
    return Calibration(ceil_hits / m, floor_hits / m, m)


def judge_all(items: list[Item], preds: list[Prediction], provider: Provider,
              progress: bool | None = None) -> dict[str, bool]:
    """Only defective items are judged — there is no mechanism to match on a
    clean one, and asking would invent credit."""
    progress = sys.stdout.isatty() if progress is None else progress
    by_id = {i.id: i for i in items}
    out: dict[str, bool] = {}
    todo = [p for p in preds if by_id[p.item_id].has_defect and p.has_defect]
    for k, p in enumerate(todo, 1):
        out[p.item_id] = _grade(provider, by_id[p.item_id].ground_truth, p.why)
        if progress:
            print(f"\r  judging explanations: {k}/{len(todo)}", end="", flush=True)
    if progress and todo:
        print()
    return out
