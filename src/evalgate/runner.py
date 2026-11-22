"""Run a suite against the system under test and grade every case."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .graders import CheckResult, grade
from .providers import Provider
from .stats import percentile, wilson
from .suite import Case, Suite


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    input: str
    output: str
    passed: bool
    checks: list[CheckResult]
    latency_ms: float = 0.0
    cost: float = 0.0
    stop_reason: str = ""
    error: str | None = None


@dataclass
class RunResult:
    suite: str
    label: str
    model: str
    system_sha: str
    cases: list[CaseResult]
    summary: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RunResult:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["cases"] = [CaseResult(**{**c, "checks": [CheckResult(**k) for k in c["checks"]]}) for c in d["cases"]]
        return cls(**d)


def summarize(cases: list[CaseResult]) -> dict:
    n, passed = len(cases), sum(c.passed for c in cases)
    lo, hi = wilson(passed, n)
    tags: dict[str, list[int]] = {}
    for c in cases:
        for t in c.tags:
            tags.setdefault(t, [0, 0])
            tags[t][0] += c.passed
            tags[t][1] += 1
    lat = [c.latency_ms for c in cases if not c.error]
    return {
        "n": n, "passed": passed, "pass_rate": round(passed / n, 4) if n else 0.0, "ci95": [round(lo, 3), round(hi, 3)],
        "cost_usd": round(sum(c.cost for c in cases), 6),
        "p50_ms": round(percentile(lat, 0.5), 1), "p95_ms": round(percentile(lat, 0.95), 1),
        "refusals": sum(c.stop_reason == "refusal" for c in cases),
        "by_tag": {t: {"passed": p, "n": k} for t, (p, k) in sorted(tags.items())},
    }


def run_case(case: Case, suite: Suite, system: str, target: Provider, judge: Provider | None) -> CaseResult:
    try:
        comp = target.complete(suite.target.model, system, case.input, suite.target.max_tokens)
    except Exception as e:  # a provider failure is a failed case, not a crashed run
        return CaseResult(case.id, case.tags, case.input, "", False, [], error=f"{type(e).__name__}: {e}")
    checks = [grade(chk, case.input, comp.text, judge, suite.judge.model) for chk in case.checks]
    return CaseResult(
        id=case.id, tags=case.tags, input=case.input, output=comp.text,
        passed=comp.stop_reason != "refusal" and all(c.passed for c in checks), checks=checks,
        latency_ms=comp.latency_ms, cost=round(comp.cost(suite.target.model) + sum(c.cost for c in checks), 6),
        stop_reason=comp.stop_reason,
    )


def run_suite(suite: Suite, system: str, target: Provider, judge: Provider | None, label: str, workers: int = 8) -> RunResult:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(lambda c: run_case(c, suite, system, target, judge), suite.cases))
    sha = hashlib.sha256(system.encode()).hexdigest()[:12]
    return RunResult(suite.name, label, suite.target.model, sha, cases, summarize(cases))
