"""Model providers. `ReplayProvider` serves recorded completions so suites run in CI with no API key and no spend."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

# USD per 1M tokens (input, output), first-party Claude API rates.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str = "end_turn"

    def cost(self, model: str) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000


class Provider(Protocol):
    def complete(self, model: str, system: str, prompt: str, max_tokens: int) -> Completion: ...


def cache_key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, system, prompt], ensure_ascii=False).encode()).hexdigest()[:24]


class ReplayProvider:
    """Reads a JSONL cassette: one {"key", "model", "completion": {...}} per line."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, Completion] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.items[row["key"]] = Completion(**row["completion"])

    def complete(self, model, system, prompt, max_tokens):
        key = cache_key(model, system, prompt)
        if key not in self.items:
            raise LookupError(f"no recording for this prompt (key {key}) in {self.path.name}; run once with --record")
        return self.items[key]


class RecordingProvider:
    """Wraps a live provider and appends every completion to a cassette."""

    def __init__(self, inner: Provider, path: Path):
        self.inner, self.path = inner, path
        self.replay = ReplayProvider(path)

    def complete(self, model, system, prompt, max_tokens):
        try:
            return self.replay.complete(model, system, prompt, max_tokens)
        except LookupError:
            pass
        c = self.inner.complete(model, system, prompt, max_tokens)
        key = cache_key(model, system, prompt)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "completion": asdict(c)}, ensure_ascii=False) + "\n")
        self.replay.items[key] = c
        return c


class AnthropicProvider:
    """Live Claude calls through the official SDK (`pip install evalgate[anthropic]`).

    No server-side model fallback on purpose: an eval must measure the model it names, so a refusal is
    recorded as an outcome (stop_reason "refusal") instead of being silently answered by another model.
    """

    def __init__(self) -> None:
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()

    def complete(self, model, system, prompt, max_tokens):
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Completion(text=text, input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1), stop_reason=msg.stop_reason or "")


def make_provider(kind: str, cassette: Path | None, record: bool) -> Provider:
    if kind == "replay":
        if cassette is None:
            raise ValueError("replay provider needs a cassette")
        return ReplayProvider(cassette)
    live = AnthropicProvider()
    return RecordingProvider(live, cassette) if record and cassette else live
