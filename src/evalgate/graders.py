"""Graders: cheap deterministic checks first, an LLM judge only where a rule can't express the requirement."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .providers import Provider
from .suite import Check, ContainsAny, Judge, MaxWords, NotContains, Regex

JUDGE_SYSTEM = (
    "You grade one reply from a customer-support assistant against a rubric. "
    "Judge only what the rubric asks. Reply with JSON only: "
    '{"pass": true|false, "reason": "<one sentence>"}'
)


def judge_prompt(question: str, answer: str, rubric: str) -> str:
    return f"<question>\n{question}\n</question>\n<reply>\n{answer}\n</reply>\n<rubric>\n{rubric}\n</rubric>"


@dataclass
class CheckResult:
    type: str
    passed: bool
    detail: str
    critical: bool = False
    cost: float = 0.0


def parse_verdict(text: str) -> tuple[bool, str]:
    """Tolerant parse of the judge's JSON (it may wrap it in prose or a code fence)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"judge returned no JSON: {text[:80]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data.get("pass"), bool):
        raise ValueError(f"judge JSON has no boolean 'pass': {data}")
    return data["pass"], str(data.get("reason", ""))


def grade(check: Check, question: str, answer: str, judge: Provider | None = None, judge_model: str = "") -> CheckResult:
    low = answer.lower()
    match check:
        case ContainsAny(values=vals):
            hit = [v for v in vals if v.lower() in low]
            return CheckResult("contains_any", bool(hit), f"found {hit[0]!r}" if hit else f"none of {vals}", check.critical)
        case NotContains(values=vals):
            hit = [v for v in vals if v.lower() in low]
            return CheckResult("not_contains", not hit, f"contains {hit[0]!r}" if hit else "clean", check.critical)
        case Regex(pattern=p):
            ok = re.search(p, answer, re.I | re.M) is not None
            return CheckResult("regex", ok, f"/{p}/ {'matched' if ok else 'no match'}", check.critical)
        case MaxWords(n=n):
            words = len(answer.split())
            return CheckResult("max_words", words <= n, f"{words} words (max {n})", check.critical)
        case Judge(rubric=rubric):
            if judge is None:
                raise ValueError("suite has judge checks but no judge provider")
            c = judge.complete(judge_model, JUDGE_SYSTEM, judge_prompt(question, answer, rubric), 300)
            ok, reason = parse_verdict(c.text)
            return CheckResult("judge", ok, reason, check.critical, c.cost(judge_model))
    raise TypeError(f"unknown check {check!r}")
