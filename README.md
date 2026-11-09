# refute-bench

**Can a language model catch the bug that makes a research result wrong?**

A 32-item benchmark built from real methodology defects — look-ahead bias,
uncontrolled multiple comparisons, units errors, self-selected comparison
groups, stale data. Not syntax bugs. The kind that leaves the code running
perfectly and the conclusion false.

Built working through chapters 3–4 of Chip Huyen's *AI Engineering*, where the
subject is evaluation methodology rather than model building.

---

## Why I built it

Over about a month of quantitative research I reported a series of findings
that turned out to be wrong. Not wrong because the code crashed — the code ran
fine every time. Wrong because of how the measurement was set up.

A representative one. I told a trader his stop-loss rule was worth **+$845**,
"the largest verified effect in the program." An audit then found the script
that produced it had **crashed**, and the results file I quoted was timestamped
**49 seconds before** the code that supposedly wrote it. Underneath that were
fourteen more defects, including a grouping key that silently merged 69 round
trips into 39.

Every one of those was caught by *re-reading the code with fresh suspicion.*
Which raises an obvious question: can that step be automated? And a less
obvious one that turned out to matter more — **how would I know if it worked?**

---

## Building the test

I took the defects I had actually found and wrote each one up as a small
exercise: a few lines of code, the context it ran in, and a known answer.

```
32 items · 25 with a defect, 7 clean · 20 defect classes · easy=3 medium=15 hard=14
```

Here's one, verbatim from the set:

> **Context:** A signal is stamped with the bar's date. The scan that produces
> it runs from cron at 08:00 the *next* morning. Entries are simulated at the
> signal-date close.

```python
asof = hist.index[-1].date()        # last completed bar
log_signal(date=asof, price=float(hist['Close'].iloc[-1]))
# cron: 0 8 * * 1-5  python3 ath_knn_screener.py
```

**Answer: flawed.** The backtest enters at a closing price nobody could have
traded, because the signal did not exist until the next morning. That was 29 of
62 rows in a live signal log.

The expensive part was the **seven clean items** — correct code that looks
suspicious. Without them, a reviewer who shouts "flaw!" at everything scores
100%, and I would have learned nothing.

---

## What I learned

### 1. The laziest possible strategy won the most popular metric

Before testing any model I ran four deliberately stupid baselines. One of them
just answers "there's a defect" to all 32 questions without reading them.

It scored an **F1 of 0.88** — higher than the keyword baseline, and within 0.03
of a real model.

Not because it is clever. Because 78% of my test set contains a defect, so
blind pessimism looks like insight. **If I had reported only F1, I would have
shipped a benchmark whose winning strategy is to not read the question.**

That is roughly how a lot of impressive-sounding eval numbers are produced. The
fix is boring: report **balanced accuracy**, which averages performance on each
class separately and pins both constant strategies at exactly 0.50.

### 2. A real model does clear the bar, and precision is where it wins

```
system                 bal-acc    acc          95% CI   prec    rec     F1   expl
----------------------------------------------------------------------------------
claude-cli:default        0.92   0.88    [0.75, 0.97]   1.00   0.84   0.91   0.95
baseline:keyword          0.60   0.53    [0.38, 0.72]   0.86   0.48   0.62   0.00
baseline:always-yes       0.50   0.78    [0.62, 0.91]   0.78   1.00   0.88   0.00
baseline:always-no        0.50   0.22    [0.09, 0.38]    nan   0.00    nan     —
baseline:random           0.47   0.41    [0.25, 0.56]   0.75   0.36   0.49   0.00

balanced accuracy, 95% CI:  claude 0.92 [0.84, 0.98] · keyword 0.60 [0.39, 0.78]
judge calibration: ceiling 100%, floor 12% -> USABLE
```

**Precision 1.00 is the result that matters.** Zero false alarms on all seven
clean controls. A reviewer that cries wolf on correct code is worse than no
reviewer, and this one never did.

Recall is the softer side at **0.84** — it missed four:

| missed | difficulty | why it is hard |
|---|---|---|
| `clustering_noop` | hard | requires noticing a `groupby` that collapses nothing |
| `truncation_bias` | hard | the dropped rows are invisible in the code shown |
| `stale_data` | hard | needs knowing one bar does not make a price level |
| `units_benchmark` | medium | a 252-vs-365 annualization convention |

Accuracy fell cleanly with difficulty — **100% easy, 93% medium, 79% hard** —
which is weak evidence my difficulty labels track something real rather than
being my own guesses.

### 3. Grading the explanation needs its own guard

Binary accuracy lets a model be right for the wrong reason, so a second model
grades whether the *mechanism* it described matches the real one.

But an unvalidated judge just moves the trust problem down a layer. So before
any judged number is produced, the judge faces two probes whose answers are
known by construction: hand it the ground truth (must say MATCH), then hand it
a *different* item's ground truth (must say MISS). It scored 100% and 12%.

Had it failed, those scores would be **deleted, not footnoted**. A number nobody
should trust is worse than no number.

### 4. My benchmark contained the exact defect it tests for

The harness ends by comparing the model against the best baseline. On the first
real run it printed:

> *"the best baseline sits INSIDE its CI. Not distinguishable."*

about a detector scoring 0.92.

It was comparing **raw accuracy**. `always-yes` posts 0.78 on raw accuracy from
class balance alone, which falls inside the model's interval. So the verdict
logic was gamed by precisely the metric the rest of the project exists to warn
about — and it was gamed in my own code, two hours after I wrote the warning.

Fixed to compare balanced accuracy, bootstrapping the positive and negative
pools separately because there are only seven negatives.

This is the part I would actually put on a slide. Knowing a failure mode and
writing it down does not stop you walking into it. The only thing that caught
it was having the baselines there to produce an obviously absurd verdict.

---

## Run it

```bash
python3 -m pytest tests/ -q          # validates the eval set itself
python3 scripts/run_eval.py          # baselines only — no model, no API key
```

Against a local model:

```bash
ollama serve & ollama pull llama3.1:8b
python3 scripts/run_eval.py --systems ollama:llama3.1:8b --judge ollama:llama3.1:8b
```

Against the Claude CLI:

```bash
python3 scripts/run_eval.py --systems claude --judge claude --out results.json
```

Only `numpy` and `pytest` are required. Baselines always run — you cannot opt
out, because a model score with no baseline is not a result.

---

## Layout

```
data/defects.jsonl      the eval set — the asset; everything else is replaceable
refutebench/
  dataset.py            load + validate — it rejected 3 of my own mislabelled rows
  providers.py          ollama / claude CLI / four baselines
  detect.py             the system under test
  judge.py              explanation grading + the calibration that licenses it
  evaluate.py           metrics, bootstrap CIs, breakdowns
scripts/run_eval.py
results_claude.json     the full run, every prediction and explanation
```

Two smaller decisions that changed numbers:

- **Unparseable answers count as CLEAN, not as skips.** Dropping malformed
  responses silently grades a model only on the questions it answered tidily.
- **`dataset.load()` validates the labels on every load.** It caught three rows
  where I had marked an item clean while leaving its defect class set — my own
  labelling error, found by my own validator, on day one.

---

## What this does not show

- **n=32.** Every interval is wide. This separates "much better than regex"
  from "not". It will not resolve a three-point gap between two models.
- **One annotator.** Ground truth is my own audit process. No inter-annotator
  agreement, because there is no second annotator.
- **Contamination.** Snippets are paraphrased from a private codebase, but the
  defect *classes* are textbook. A model may be recognising a pattern rather
  than reasoning about the code.
- **Detection is not repair.** Nothing here measures whether a model can fix
  what it finds.
- **One run.** No temperature sweep, no repeated sampling, so run-to-run
  variance is unmeasured.

New defect items are welcome, with one rule: it has to be a defect that
actually happened, with the real number it changed.

MIT.
