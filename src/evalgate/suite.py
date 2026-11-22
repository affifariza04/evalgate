"""Eval suite format (YAML), validated with Pydantic so typos fail loudly instead of silently passing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    critical: bool = False  # a regression on a critical check fails the gate on its own


class ContainsAny(_Check):
    type: Literal["contains_any"]
    values: list[str]


class NotContains(_Check):
    type: Literal["not_contains"]
    values: list[str]


class Regex(_Check):
    type: Literal["regex"]
    pattern: str


class MaxWords(_Check):
    type: Literal["max_words"]
    n: int


class Judge(_Check):
    type: Literal["judge"]
    rubric: str


Check = Annotated[ContainsAny | NotContains | Regex | MaxWords | Judge, Field(discriminator="type")]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: str
    checks: list[Check]
    tags: list[str] = []


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["anthropic", "replay"] = "replay"
    model: str = "claude-opus-5-5"
    system: str  # path to the system prompt under test, relative to the suite file
    max_tokens: int = 1024


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["anthropic", "replay"] = "replay"
    model: str = "claude-sonnet-5-5"


class Gate(BaseModel):
    """When `evalgate compare` should fail the build."""
    model_config = ConfigDict(extra="forbid")
    min_pass_rate: float = 0.8           # candidate must stay above this
    max_new_failures: int = 2            # pass -> fail flips allowed (non-critical)
    significance: float = 0.05           # fail if pass rate drops with McNemar p below this


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    target: Target
    judge: JudgeConfig = JudgeConfig()
    gate: Gate = Gate()
    cassette: str | None = None          # replay file, relative to the suite file
    cases: list[Case]

    @property
    def check_count(self) -> int:
        return sum(len(c.checks) for c in self.cases)


def load_suite(path: str | Path, system_override: str | None = None) -> tuple[Suite, str, Path]:
    """Returns (suite, system prompt text, suite directory)."""
    path = Path(path)
    suite = Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [c.id for c in suite.cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    system_path = path.parent / (system_override or suite.target.system)
    return suite, system_path.read_text(encoding="utf-8"), path.parent
