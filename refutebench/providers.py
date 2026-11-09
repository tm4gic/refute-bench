"""Model backends, plus the trivial baselines every benchmark needs.

The baselines are not filler. A detector that scores 0.75 F1 means nothing
until you know that "always say yes" scores 0.86 on a set that is 75% positive.
Most reported LLM eval numbers are missing exactly this comparison.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import urllib.error
import urllib.request
from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(self, prompt: str) -> str: ...


class Ollama:
    """Local model. Free, no key, and reproducible enough for a benchmark."""

    def __init__(self, model: str = "llama3.1:8b",
                 host: str = "http://127.0.0.1:11434", temperature: float = 0.0):
        self.model, self.host, self.temperature = model, host, temperature
        self.name = f"ollama:{model}"

    def complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["response"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach ollama at {self.host} — start it with "
                f"`ollama serve` and `ollama pull {self.model}`") from e


class ClaudeCLI:
    """Uses the locally installed `claude` binary, so no API key is handled."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.name = f"claude-cli:{model or 'default'}"

    def complete(self, prompt: str) -> str:
        cmd = ["claude", "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {out.stderr[:300]}")
        return out.stdout


# ---------------------------------------------------------------- baselines
class AlwaysDefect:
    """The score to beat. On a mostly-positive set this looks strong."""
    name = "baseline:always-yes"

    def complete(self, prompt: str) -> str:
        return 'VERDICT: DEFECT\nCLASS: unknown\nWHY: asserted without looking.'


class NeverDefect:
    name = "baseline:always-no"

    def complete(self, prompt: str) -> str:
        return 'VERDICT: CLEAN\nCLASS: none\nWHY: asserted without looking.'


class RandomGuess:
    name = "baseline:random"

    def __init__(self, seed: int = 0, p: float = 0.5):
        self.rng, self.p = random.Random(seed), p

    def complete(self, prompt: str) -> str:
        hit = self.rng.random() < self.p
        return (f"VERDICT: {'DEFECT' if hit else 'CLEAN'}\n"
                f"CLASS: {'unknown' if hit else 'none'}\nWHY: coin flip.")


class Keyword:
    """A dumb-but-real detector: does the snippet contain a suspicious token?

    Included because a surprising number of 'LLM beats baseline' claims do not
    survive comparison against twenty lines of regex.
    """
    name = "baseline:keyword"
    SUSPICIOUS = re.compile(
        r"\.max\(\)|\.min\(\)|shift\(-|>=\s*med|groupby\(\[|except\s+Exception|"
        r"\.High|\.Low|for\s+\w+\s+in\s+\[.*\]:\s*$|sqrt\(365\)|/\s*390",
        re.M)

    def complete(self, prompt: str) -> str:
        code = prompt.split("CODE\n", 1)[-1]
        hit = bool(self.SUSPICIOUS.search(code))
        return (f"VERDICT: {'DEFECT' if hit else 'CLEAN'}\n"
                f"CLASS: {'unknown' if hit else 'none'}\nWHY: keyword match.")


def get(spec: str) -> Provider:
    """'ollama:llama3.1:8b' | 'claude' | 'baseline:always-yes' | ..."""
    if spec.startswith("baseline:"):
        return {"baseline:always-yes": AlwaysDefect,
                "baseline:always-no": NeverDefect,
                "baseline:random": RandomGuess,
                "baseline:keyword": Keyword}[spec]()
    if spec.startswith("ollama"):
        _, _, model = spec.partition(":")
        return Ollama(model or "llama3.1:8b")
    if spec.startswith("claude"):
        _, _, model = spec.partition(":")
        return ClaudeCLI(model or None)
    raise ValueError(f"unknown provider {spec!r}")


ALL_BASELINES = ["baseline:always-yes", "baseline:always-no",
                 "baseline:random", "baseline:keyword"]
