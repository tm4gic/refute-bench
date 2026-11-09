"""The system under test: given code + context, is there a methodology defect?"""
from __future__ import annotations

import sys

import re
from dataclasses import dataclass

from .dataset import Item
from .providers import Provider

PROMPT = """You are auditing research code for methodology defects — the kind \
that make a result look real when it is not. Examples of what counts: using \
information that did not exist at decision time, uncontrolled multiple \
comparisons, treating correlated observations as independent, comparing \
quantities with different units, self-selected comparison groups, optimistic \
fill assumptions, stale or corrupt input data, valuations at the wrong price.

Not every snippet has a defect. Some are correct. Saying "DEFECT" when the code \
is fine is just as wrong as missing a real one.

{body}

Answer in exactly this format and nothing else:
VERDICT: DEFECT or CLEAN
CLASS: a short snake_case label, or none
WHY: one sentence naming the specific mechanism"""


@dataclass
class Prediction:
    item_id: str
    has_defect: bool
    defect_class: str
    why: str
    raw: str


_V = re.compile(r"VERDICT:\s*(DEFECT|CLEAN)", re.I)
_C = re.compile(r"CLASS:\s*([a-z_\- ]+)", re.I)
_W = re.compile(r"WHY:\s*(.+)", re.I | re.S)


def parse(item_id: str, raw: str) -> Prediction:
    """Parse the model's answer.

    An unparseable answer counts as CLEAN, not as a skip. Dropping malformed
    responses would quietly grade the model only on the questions it answered
    tidily, which inflates every metric.
    """
    v = _V.search(raw)
    c = _C.search(raw)
    w = _W.search(raw)
    return Prediction(
        item_id=item_id,
        has_defect=bool(v) and v.group(1).upper() == "DEFECT",
        defect_class=(c.group(1).strip().replace(" ", "_").lower() if c else "none"),
        why=(w.group(1).strip().split("\n")[0][:400] if w else ""),
        raw=raw,
    )


def run(items: list[Item], provider: Provider,
        progress: bool | None = None) -> list[Prediction]:
    # a redirected run should not emit 32 progress lines into the log
    progress = sys.stdout.isatty() if progress is None else progress
    preds = []
    for n, it in enumerate(items, 1):
        try:
            raw = provider.complete(PROMPT.format(body=it.prompt_body))
        except Exception as e:                      # a dead backend is a result
            raw = f"ERROR: {e}"
        preds.append(parse(it.id, raw))
        if progress:
            print(f"\r  {provider.name}: {n}/{len(items)}", end="", flush=True)
    if progress:
        print()
    return preds
