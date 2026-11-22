import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalgate.compare import compare
from evalgate.graders import grade, parse_verdict
from evalgate.providers import Completion, RecordingProvider, ReplayProvider, cache_key
from evalgate.runner import CaseResult, RunResult, summarize
from evalgate.stats import mcnemar_exact, percentile, wilson
from evalgate.suite import Gate, Suite

EXAMPLE = Path(__file__).parent.parent / "examples" / "support-bot"


# ---------------------------------------------------------------- stats

def test_mcnemar_matches_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert math.isclose(mcnemar_exact(0, 3), 0.25)            # 2 * 0.5^3
    assert math.isclose(mcnemar_exact(10, 1), 2 * 12 / 2048)   # 2 * P(X<=1), X~Bin(11, .5)
    assert mcnemar_exact(8, 2) == mcnemar_exact(2, 8)          # symmetric


def test_wilson_interval_is_sane():
    lo, hi = wilson(9, 12)
    assert 0.4 < lo < 0.75 < hi < 1.0
    assert wilson(12, 12)[1] == 1.0 and wilson(0, 12)[0] == 0.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.95) == 0.0


# ---------------------------------------------------------------- suite + graders

def test_suite_rejects_unknown_check_types_and_fields():
    base = {"name": "s", "target": {"system": "p.md"}, "cases": []}
    Suite.model_validate(base)
    with pytest.raises(ValidationError):
        Suite.model_validate({**base, "cases": [{"id": "a", "input": "x", "checks": [{"type": "contain", "values": []}]}]})
    with pytest.raises(ValidationError):
        Suite.model_validate({**base, "target": {"system": "p.md", "modle": "x"}})


@pytest.mark.parametrize("check, answer, ok", [
    ({"type": "contains_any", "values": ["store credit"]}, "We can offer Store Credit.", True),
    ({"type": "contains_any", "values": ["store credit"]}, "Full refund!", False),
    ({"type": "not_contains", "values": ["full refund"]}, "Full refund!", False),
    ({"type": "regex", "pattern": r"300[.,]?000"}, "free over Rp300.000", True),
    ({"type": "max_words", "n": 3}, "one two three four", False),
])
def test_rule_graders(check, answer, ok):
    from evalgate.suite import Case
    case = Case.model_validate({"id": "x", "input": "q", "checks": [check]})
    assert grade(case.checks[0], "q", answer).passed is ok


@pytest.mark.parametrize("text, ok", [
    ('{"pass": true, "reason": "fine"}', True),
    ('Sure.\n```json\n{"pass": false, "reason": "no"}\n```', False),
])
def test_judge_verdict_parsing(text, ok):
    assert parse_verdict(text)[0] is ok


def test_judge_verdict_must_be_boolean():
    with pytest.raises(ValueError):
        parse_verdict('{"pass": "yes"}')
    with pytest.raises(ValueError):
        parse_verdict("looks good to me")


# ---------------------------------------------------------------- providers

def test_replay_and_recording(tmp_path):
    cassette = tmp_path / "c.jsonl"

    class Fake:
        calls = 0

        def complete(self, model, system, prompt, max_tokens):
            Fake.calls += 1
            return Completion(f"echo {prompt}", 10, 5, 12.0)

    rec = RecordingProvider(Fake(), cassette)
    assert rec.complete("m", "s", "hi", 10).text == "echo hi"
    assert rec.complete("m", "s", "hi", 10).text == "echo hi"      # served from the cassette
    assert Fake.calls == 1
    assert ReplayProvider(cassette).complete("m", "s", "hi", 10).output_tokens == 5
    with pytest.raises(LookupError):
        ReplayProvider(cassette).complete("m", "s", "other", 10)
    assert cache_key("m", "s", "hi") != cache_key("m", "s2", "hi")


def test_cost_uses_model_prices():
    assert math.isclose(Completion("", 1_000_000, 1_000_000, 0).cost("claude-haiku-5-5"), 0.60)
    assert Completion("", 10, 10, 0).cost("unknown-model") == 0.0


# ---------------------------------------------------------------- compare / gate

def _run(label, passes, critical=()):
    cases = []
    for i, p in enumerate(passes):
        from evalgate.graders import CheckResult
        chk = CheckResult("judge", p, "ok" if p else "bad", critical=(i in critical))
        cases.append(CaseResult(f"c{i}", ["t"], "q", "a", p, [chk]))
    return RunResult("s", label, "m", "sha", cases, summarize(cases))


def test_gate_passes_on_pure_improvement():
    cmp = compare(_run("a", [1, 1, 0, 0, 1]), _run("b", [1, 1, 1, 1, 1]), Gate(min_pass_rate=0.8))
    assert cmp.verdict == "pass" and len(cmp.fixes) == 2 and not cmp.regressions


def test_gate_fails_on_a_critical_regression_even_if_rate_improves():
    cmp = compare(_run("a", [1, 0, 0, 1, 1]), _run("b", [0, 1, 1, 1, 1], critical={0}), Gate(min_pass_rate=0.5))
    assert cmp.verdict == "fail" and any("critical" in r for r in cmp.reasons)


def test_gate_fails_on_significant_drop():
    base, cand = _run("a", [1] * 12), _run("b", [0] * 9 + [1] * 3)
    cmp = compare(base, cand, Gate(min_pass_rate=0.0, max_new_failures=100))
    assert cmp.verdict == "fail" and any("McNemar" in r for r in cmp.reasons)


def test_run_result_roundtrip(tmp_path):
    r = _run("a", [1, 0])
    r.save(tmp_path / "r.json")
    assert RunResult.load(tmp_path / "r.json") == r


# ---------------------------------------------------------------- the shipped example, end to end

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "evalgate.cli", *args], cwd=EXAMPLE, capture_output=True, text=True,
                          encoding="utf-8")


def test_example_end_to_end(tmp_path):
    for v in ("v1", "v2", "v3"):
        res = _cli("run", "suite.yaml", "-o", str(tmp_path / f"{v}.json"), "--label", v, "--system", f"prompts/{v}.md")
        assert res.returncode == 0, res.stderr
    rates = {v: json.loads((tmp_path / f"{v}.json").read_text(encoding="utf-8"))["summary"]["pass_rate"] for v in ("v1", "v2", "v3")}
    assert rates == {"v1": 0.75, "v2": 0.75, "v3": 1.0}

    bad = _cli("compare", "suite.yaml", str(tmp_path / "v1.json"), str(tmp_path / "v2.json"), "--html", str(tmp_path / "r.html"))
    assert bad.returncode == 1 and "refund-opened-bag" in bad.stdout and "critical" in bad.stdout
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "GATE FAIL" in html and "SORRY20" in html          # the offending reply is shown, HTML-escaped

    good = _cli("compare", "suite.yaml", str(tmp_path / "v2.json"), str(tmp_path / "v3.json"))
    assert good.returncode == 0 and "PASS" in good.stdout
