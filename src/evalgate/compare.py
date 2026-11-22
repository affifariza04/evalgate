"""Paired comparison of two runs on the same suite, and the gate decision."""

from __future__ import annotations

from dataclasses import dataclass, field

from .runner import CaseResult, RunResult
from .stats import mcnemar_exact
from .suite import Gate


@dataclass
class Flip:
    id: str
    kind: str                      # "regression" | "fix"
    critical: bool
    base: CaseResult
    cand: CaseResult
    failed_checks: list[str] = field(default_factory=list)


@dataclass
class Comparison:
    base: RunResult
    cand: RunResult
    flips: list[Flip]
    p_value: float
    verdict: str                   # "pass" | "fail"
    reasons: list[str]

    @property
    def regressions(self) -> list[Flip]:
        return [f for f in self.flips if f.kind == "regression"]

    @property
    def fixes(self) -> list[Flip]:
        return [f for f in self.flips if f.kind == "fix"]


def compare(base: RunResult, cand: RunResult, gate: Gate) -> Comparison:
    b_by_id = {c.id: c for c in base.cases}
    shared = [c for c in cand.cases if c.id in b_by_id]
    if not shared:
        raise ValueError("the two runs share no case ids")
    flips: list[Flip] = []
    for c in shared:
        b = b_by_id[c.id]
        if b.passed and not c.passed:
            failed = [k for k in c.checks if not k.passed]
            flips.append(Flip(c.id, "regression", any(k.critical for k in failed), b, c,
                              [f"{k.type}: {k.detail}" for k in failed] or [c.error or c.stop_reason]))
        elif c.passed and not b.passed:
            flips.append(Flip(c.id, "fix", False, b, c))

    reg = [f for f in flips if f.kind == "regression"]
    fixes = [f for f in flips if f.kind == "fix"]
    p = mcnemar_exact(len(reg), len(fixes))
    reasons: list[str] = []
    rate = cand.summary["pass_rate"]
    if rate < gate.min_pass_rate:
        reasons.append(f"pass rate {rate:.0%} is below the {gate.min_pass_rate:.0%} floor")
    crit = [f.id for f in reg if f.critical]
    if crit:
        reasons.append(f"critical check regressed: {', '.join(crit)}")
    if len(reg) - len(crit) > gate.max_new_failures:
        reasons.append(f"{len(reg) - len(crit)} new failures (max {gate.max_new_failures})")
    if len(reg) > len(fixes) and p < gate.significance:
        reasons.append(f"pass rate dropped significantly (McNemar p={p:.3f})")
    return Comparison(base, cand, flips, p, "fail" if reasons else "pass", reasons)
