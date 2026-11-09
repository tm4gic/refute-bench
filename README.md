# refute-bench

**Can a language model find the bug that makes a research result wrong?**

A small evaluation benchmark for detecting *methodology* defects in research
code — look-ahead bias, uncontrolled multiple comparisons, units errors,
self-selected comparison groups, stale data. Not syntax bugs. The kind of
defect that leaves the code running fine and the conclusion false.

Built as a chapter 3–4 exercise from Chip Huyen's *AI Engineering*, where the
subject is evaluation methodology rather than model building.

---

## Why the eval set is the interesting part

Every defect in `data/defects.jsonl` is real. Each one was found in a live
quantitative research codebase, by an audit, after the wrong answer had already
been reported. The ground truth is what the audit concluded, not what someone
imagined a bug might look like.

```
32 items · 25 with a defect, 7 clean · 20 defect classes · easy=3 medium=15 hard=14
```

A sample of what is in there:

| class | the defect |
|---|---|
| `look_ahead` | Signal rows stamped with the bar date, written by a cron job that runs the next morning — so entries use a close that was already known. |
| `clustering_noop` | A "ticker-clustered" t-statistic where each ticker appears once, making the clustering arithmetically identical to not clustering. |
| `vacuous_test` | A shuffle-label self-test run against a null of 1.0 when the pooled mean is 0.67 — shuffling cannot change a pooled mean, so it flagged 35 of 35 cells on pure noise and validated nothing. |
| `units_benchmark` | A high–low range divided by an expected *absolute move*. For any random walk the range is exactly 2× the mean absolute move, manufacturing a 1.6× "edge" from a unit mismatch. |
| `tie_allocation` | A median split using `>=` where 12 of 32 observations sit exactly on the median — reversing the tie direction reverses the sign of the result. |
| `grouping_collapse` | `groupby(['date','symbol'])` merging 69 round trips into 39, scoring twelve sequential scalps as one averaged-down position. |

Seven items are **clean controls** — correct code that looks suspicious. Without
negatives, precision is undefined and "always say DEFECT" wins.

---

## The headline result: the baselines are hard to beat

```
system                 bal-acc    acc          95% CI   prec    rec     F1
--------------------------------------------------------------------------
baseline:keyword          0.60   0.53    [0.38, 0.72]   0.86   0.48   0.62
baseline:always-yes       0.50   0.78    [0.62, 0.91]   0.78   1.00   0.88
baseline:always-no        0.50   0.22    [0.09, 0.38]    nan   0.00    nan
baseline:random           0.47   0.41    [0.25, 0.56]   0.75   0.36   0.49
```

**`always-yes` scores an F1 of 0.88 while knowing nothing**, because the set is
78% positive. Any paper, blog post or eval harness reporting F1 on an imbalanced
set without this comparison is reporting the class balance.

That is why the headline metric here is **balanced accuracy**, which pins both
trivial strategies at exactly 0.50. The number to beat is **0.60** — twenty
lines of regex.

---

## Design decisions worth arguing with

**Baselines always run.** You cannot opt out. A model score with no baseline is
not a result.

**The judge is calibrated before it is trusted.** Binary accuracy lets a
detector be right for the wrong reason, so an LLM judge grades whether the
stated *mechanism* matches the real one. But an unvalidated judge just moves the
trust problem down a layer, so before any judged number is produced the judge
faces two probes with answers known by construction:

- **ceiling** — hand it the ground truth as the prediction. It must say MATCH.
- **floor** — hand it a *different* item's ground truth. It must say MISS.

Fail either and judged scores are **suppressed, not caveated**. A number nobody
should trust is worse than no number.

**Unparseable answers count as CLEAN, not as skips.** Dropping malformed
responses silently grades a model only on the questions it answered tidily.

**Bootstrap CIs on everything.** n=32. A bare `0.81` implies a precision the
sample cannot support; `[0.62, 0.91]` tells the truth.

**Per-class and per-difficulty breakdowns.** An aggregate hides a detector that
catches every easy look-ahead and no hard stale-data case.

---

## Run it

```bash
python3 -m pytest tests/ -q          # validates the eval set itself
python3 scripts/run_eval.py          # baselines only, no model needed
```

Against a local model (no API key):

```bash
ollama serve & ollama pull llama3.1:8b
python3 scripts/run_eval.py --systems ollama:llama3.1:8b --judge ollama:llama3.1:8b
```

Against the Claude CLI, if installed:

```bash
python3 scripts/run_eval.py --systems claude --judge claude --out results.json
```

Only `numpy` and `pytest` are required. Model backends are optional.

---

## Layout

```
data/defects.jsonl      the eval set — the asset; everything else is replaceable
refutebench/
  dataset.py            load + validate (it caught 3 mislabelled rows on day one)
  providers.py          ollama / claude CLI / four baselines
  detect.py             the system under test
  judge.py              explanation grading + the calibration that licenses it
  evaluate.py           metrics, bootstrap CIs, breakdowns
scripts/run_eval.py
tests/
```

`dataset.load()` enforces that `has_defect` agrees with `defect_class`, that ids
are unique, and that enough negatives exist to make precision meaningful. It
rejected three of my own rows the first time it ran.

---

## Known limits

- **n=32.** Every interval is wide. This distinguishes "much better than regex"
  from "not"; it will not resolve a three-point gap between two models.
- **Single annotator.** Ground truth comes from one audit process. No
  inter-annotator agreement is reported because there is no second annotator.
- **Contamination risk.** The snippets are paraphrased from a private codebase,
  but the defect *classes* are textbook and a model may recognise the pattern
  rather than reason about the code.
- **Detection is not repair.** Nothing here measures whether a model can fix
  what it finds.

Contributions of new defect items are welcome, with one rule: it has to be a
defect that actually happened, with the real number it changed.

MIT.
