#!/usr/bin/env python3
"""Run the benchmark.

    python3 scripts/run_eval.py --systems baseline:always-yes baseline:keyword
    python3 scripts/run_eval.py --systems ollama:llama3.1:8b --judge ollama:llama3.1:8b
    python3 scripts/run_eval.py --systems claude --judge claude

Baselines always run, because a model score is uninterpretable without them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refutebench import dataset, detect, evaluate, judge, providers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=[],
                    help="providers to test, e.g. ollama:llama3.1:8b claude")
    ap.add_argument("--judge", default=None,
                    help="provider for explanation grading (optional)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args()

    items = dataset.load()
    print(f"\nREFUTE-BENCH\n{dataset.summary(items)}\n")

    specs = list(a.systems)
    if not a.no_baselines:
        specs += [s for s in providers.ALL_BASELINES if s not in specs]
    if not specs:
        print("nothing to run"); return 1

    # ---- judge calibration BEFORE any judged number is produced -----------
    jp, cal = None, None
    if a.judge:
        jp = providers.get(a.judge)
        print(f"calibrating judge ({jp.name}) before trusting it")
        cal = judge.calibrate(items, jp)
        print(f"  {cal}\n")
        if not cal.usable:
            jp = None                      # suppress rather than caveat

    results, scored = {}, []
    for spec in specs:
        prov = providers.get(spec)
        print(f"running {prov.name}")
        preds = detect.run(items, prov)
        judged = judge.judge_all(items, preds, jp) if jp else None
        s = evaluate.score(items, preds, prov.name, judged)
        scored.append(s)
        results[prov.name] = dict(
            accuracy=s.accuracy, precision=s.precision, recall=s.recall,
            f1=s.f1, ci=list(s.ci), errors=s.errors,
            explanation_accuracy=s.explanation_accuracy,
            predictions=[dict(id=p.item_id, has_defect=p.has_defect,
                              defect_class=p.defect_class, why=p.why)
                         for p in preds])

    print("\n" + "=" * 72)
    print("RESULTS".center(72))
    print("=" * 72)
    print(evaluate.table(scored, judge_usable=bool(jp)))

    for s in scored:
        if not s.provider.startswith("baseline:"):
            print(evaluate.breakdown(s))

    # ---- the comparison that decides whether anything was learned ---------
    # Compare on BALANCED accuracy, not raw accuracy.
    #
    # The first version of this block picked the best baseline by F1 and then
    # tested raw accuracy against it — which is exactly the gameable metric the
    # rest of the benchmark exists to warn about. always-yes posts acc 0.78 on a
    # 78%-positive set purely from class balance, so a genuinely good detector
    # was reported as "not distinguishable" from a constant answer. Fixed, and
    # recorded here because the benchmark should not contain the defect class it
    # tests for.
    base = max((s for s in scored if s.provider.startswith("baseline:")),
               key=lambda s: (s.balanced_accuracy
                              if s.balanced_accuracy == s.balanced_accuracy else -1),
               default=None)
    real = [s for s in scored if not s.provider.startswith("baseline:")]
    print("\n" + "-" * 72)
    if base:
        print(f"best baseline: {base.provider}  "
              f"balanced-acc {base.balanced_accuracy:.2f}  (raw acc {base.accuracy:.2f})")
    for s in real:
        lo, hi = evaluate.balanced_ci(items, s)
        inside = lo <= base.balanced_accuracy <= hi if base else False
        print(f"{s.provider}: balanced-acc {s.balanced_accuracy:.2f} "
              f"[{lo:.2f}, {hi:.2f}]"
              + (" — best baseline INSIDE the interval. Not distinguishable."
                 if inside else
                 f" — baseline {base.balanced_accuracy:.2f} is outside it."
                 if base else ""))
    if not real:
        print("baselines only. Add --systems to test a model against them.")
    print(f"\nn = {len(items)}. Intervals are wide by construction; treat "
          f"small gaps as noise.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(summary=dataset.summary(items),
                 judge_calibration=(dict(ceiling=cal.ceiling, floor=cal.floor,
                                         usable=cal.usable) if cal else None),
                 results=results), indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
