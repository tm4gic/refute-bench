"""Load and validate the benchmark.

The eval set is the asset. Everything else in this repo is replaceable; the
labels are not. So they get validated on every load rather than trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "defects.jsonl"

REQUIRED = {"id", "defect_class", "has_defect", "difficulty",
            "context", "snippet", "ground_truth"}
DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class Item:
    id: str
    defect_class: str
    has_defect: bool
    difficulty: str
    context: str
    snippet: str
    ground_truth: str
    hint_surface: str = ""

    @property
    def prompt_body(self) -> str:
        return f"CONTEXT\n{self.context}\n\nCODE\n{self.snippet}"


def load(path: Path | str = DATA) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n} is not valid JSON — {e}") from e

        missing = REQUIRED - raw.keys()
        if missing:
            raise ValueError(f"{path}:{n} missing fields {sorted(missing)}")
        if raw["id"] in seen:
            raise ValueError(f"{path}:{n} duplicate id {raw['id']!r}")
        if raw["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"{path}:{n} bad difficulty {raw['difficulty']!r}")
        # A clean item must be labelled defect_class 'none', and vice versa.
        # Getting this wrong silently corrupts every per-class metric.
        if raw["has_defect"] != (raw["defect_class"] != "none"):
            raise ValueError(
                f"{path}:{n} id={raw['id']} has_defect={raw['has_defect']} "
                f"contradicts defect_class={raw['defect_class']!r}")
        seen.add(raw["id"])
        items.append(Item(**{k: raw.get(k, "") for k in
                             list(REQUIRED) + ["hint_surface"]}))
    if not items:
        raise ValueError(f"{path} contained no items")
    return items


def summary(items: list[Item]) -> str:
    pos = sum(i.has_defect for i in items)
    classes = sorted({i.defect_class for i in items if i.has_defect})
    by_diff = {d: sum(i.difficulty == d for i in items) for d in
               ("easy", "medium", "hard")}
    return (f"{len(items)} items · {pos} with a defect, {len(items)-pos} clean "
            f"· {len(classes)} defect classes · "
            + " ".join(f"{d}={n}" for d, n in by_diff.items()))
