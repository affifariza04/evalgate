"""Reports: a Markdown summary for PR / job summaries and a self-contained HTML report."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, select_autoescape

from .compare import Comparison


def _pct(x: float) -> str:
    return f"{x:.0%}"


def render_markdown(cmp: Comparison) -> str:
    b, c = cmp.base.summary, cmp.cand.summary
    icon = "✅" if cmp.verdict == "pass" else "❌"
    lines = [
        f"### {icon} evalgate · {cmp.cand.suite}: **{cmp.verdict.upper()}**",
        "",
        "| | baseline | candidate |",
        "|---|---|---|",
        f"| pass rate | {_pct(b['pass_rate'])} ({b['passed']}/{b['n']}) | {_pct(c['pass_rate'])} ({c['passed']}/{c['n']}) |",
        f"| cost | ${b['cost_usd']:.4f} | ${c['cost_usd']:.4f} |",
        f"| p95 latency | {b['p95_ms']:.0f} ms | {c['p95_ms']:.0f} ms |",
        "",
        f"**{len(cmp.regressions)} regressed · {len(cmp.fixes)} fixed** (McNemar p = {cmp.p_value:.3f})",
    ]
    if cmp.reasons:
        lines += ["", "Gate failed because:"] + [f"- {r}" for r in cmp.reasons]
    if cmp.regressions:
        lines += ["", "| regressed case | why |", "|---|---|"]
        lines += [f"| `{f.id}`{' (critical)' if f.critical else ''} | {'; '.join(f.failed_checks)} |" for f in cmp.regressions]
    return "\n".join(lines)


def render_html(cmp: Comparison) -> str:
    env = Environment(loader=PackageLoader("evalgate", "templates"), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = _pct
    base_by_id = {c.id: c for c in cmp.base.cases}
    rows = [(base_by_id.get(c.id), c) for c in cmp.cand.cases]
    tags = sorted(set(cmp.base.summary["by_tag"]) | set(cmp.cand.summary["by_tag"]))
    return env.get_template("report.html.j2").render(cmp=cmp, b=cmp.base.summary, c=cmp.cand.summary, rows=rows, tags=tags)
