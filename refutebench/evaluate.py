"""Metrics.

Three deliberate choices:

  * Bootstrap confidence intervals on everything. n=32 is a small benchmark and
    a bare 0.81 invites a precision the sample cannot support.
  * Per-class and per-difficulty breakdowns. An aggregate hides that a detector
    catches every easy look-ahead and no hard stale-data case.
  * Explanation accuracy reported separately from binary accuracy, because
    being right for the wrong reason is a different thing from being right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import Item
from .detect import Prediction


@dataclass
class Scores:
    provider: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    judged_match: int = 0
    judged_total: int = 0
    by_class: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    ci: tuple[float, float] = (float("nan"), float("nan"))
    errors: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    @property
    def balanced_accuracy(self) -> float:
        """Mean of recall on each class.

        The set is deliberately ~78% positive, which lets "always say DEFECT"
        post an F1 of 0.88 while knowing nothing. Balanced accuracy pins that
        strategy at exactly 0.50, so it is the honest headline here.
        """
        pos = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
        neg = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else float("nan")
        return (pos + neg) / 2

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r and (p + r) else float("nan")

    @property
    def explanation_accuracy(self) -> float:
        """Of the real defects it flagged, how many did it explain correctly."""
        return (self.judged_match / self.judged_total
                if self.judged_total else float("nan"))


def _bootstrap_acc(correct: np.ndarray, n_boot: int = 10_000,
                   seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    draws = rng.choice(correct, size=(n_boot, n), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def score(items: list[Item], preds: list[Prediction], provider: str,
          judged: dict[str, bool] | None = None) -> Scores:
    by_id = {i.id: i for i in items}
    s = Scores(provider=provider)
    correct = []
    for p in preds:
        it = by_id[p.item_id]
        s.errors += p.raw.startswith("ERROR:")
        ok = p.has_defect == it.has_defect
        correct.append(ok)

        if it.has_defect and p.has_defect:
            s.tp += 1
        elif it.has_defect and not p.has_defect:
            s.fn += 1
        elif (not it.has_defect) and p.has_defect:
            s.fp += 1
        else:
            s.tn += 1

        cls = it.defect_class
        c = s.by_class.setdefault(cls, {"n": 0, "hit": 0})
        c["n"] += 1
        c["hit"] += ok

        d = s.by_difficulty.setdefault(it.difficulty, {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += ok

    s.ci = _bootstrap_acc(np.array(correct, dtype=float))
    if judged:
        s.judged_total = len(judged)
        s.judged_match = sum(judged.values())
    return s


def table(all_scores: list[Scores], judge_usable: bool = True) -> str:
    w = max(len(s.provider) for s in all_scores) + 2
    head = (f"{'system':<{w}}{'bal-acc':>9}{'acc':>7}{'95% CI':>16}"
            f"{'prec':>7}{'rec':>7}{'F1':>7}{'expl':>7}{'err':>5}")
    lines = [head, "-" * len(head)]
    for s in sorted(all_scores, key=lambda x: -(x.balanced_accuracy
                                                 if x.balanced_accuracy == x.balanced_accuracy else -1)):
        expl = (f"{s.explanation_accuracy:.2f}"
                if judge_usable and s.judged_total else "  — ")
        lines.append(
            f"{s.provider:<{w}}{s.balanced_accuracy:>9.2f}{s.accuracy:>7.2f}"
            f"{f'[{s.ci[0]:.2f}, {s.ci[1]:.2f}]':>16}"
            f"{s.precision:>7.2f}{s.recall:>7.2f}{s.f1:>7.2f}{expl:>7}"
            f"{s.errors:>5}")
    return "\n".join(lines)


def breakdown(s: Scores) -> str:
    out = [f"\n{s.provider} — by defect class"]
    for cls, d in sorted(s.by_class.items(), key=lambda kv: -kv[1]["n"]):
        bar = "#" * d["hit"] + "." * (d["n"] - d["hit"])
        out.append(f"  {cls:<26}{d['hit']}/{d['n']}  {bar}")
    out.append(f"{s.provider} — by difficulty")
    for diff in ("easy", "medium", "hard"):
        d = s.by_difficulty.get(diff)
        if d:
            out.append(f"  {diff:<26}{d['hit']}/{d['n']}  "
                       f"{d['hit']/d['n']:.0%}")
    return "\n".join(out)


def balanced_ci(items: list[Item], s: Scores, n_boot: int = 10_000,
                seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for balanced accuracy.

    Resamples the positive and negative pools SEPARATELY, because balanced
    accuracy is the mean of two class-conditional rates. Resampling the pooled
    set would let a draw contain almost no negatives and make the interval
    meaningless on a set with only 7 of them.
    """
    pos_correct = np.array([1.0] * s.tp + [0.0] * s.fn)
    neg_correct = np.array([1.0] * s.tn + [0.0] * s.fp)
    if not len(pos_correct) or not len(neg_correct):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = rng.choice(pos_correct, size=(n_boot, len(pos_correct)), replace=True).mean(axis=1)
    n = rng.choice(neg_correct, size=(n_boot, len(neg_correct)), replace=True).mean(axis=1)
    draws = (p + n) / 2
    return tuple(np.percentile(draws, [2.5, 97.5]))
