"""evalgate CLI: `run` a suite, `compare` two runs (non-zero exit when the gate fails)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from .compare import compare as do_compare
from .providers import make_provider
from .report import render_html, render_markdown
from .runner import RunResult, run_suite
from .suite import load_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Regression-test LLM prompts and gate CI on the result.")


@app.callback()
def _utf8_console() -> None:
    # Windows consoles default to a legacy code page; reports contain non-ASCII (✅ → …)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def run(
    suite_path: Path = typer.Argument(..., help="Suite YAML"),
    out: Path = typer.Option(..., "-o", help="Where to write the run JSON"),
    label: str = typer.Option("candidate", help="Name for this run in reports"),
    system: str | None = typer.Option(None, help="Override the system prompt file (relative to the suite)"),
    record: bool = typer.Option(False, help="Call the live model and append new completions to the cassette"),
) -> None:
    """Run every case and grade it."""
    suite, system_text, root = load_suite(suite_path, system)
    cassette = root / suite.cassette if suite.cassette else None
    target = make_provider("anthropic" if record else suite.target.provider, cassette, record)
    judge = make_provider("anthropic" if record else suite.judge.provider, cassette, record)
    result = run_suite(suite, system_text, target, judge, label)
    result.save(out)
    s = result.summary
    typer.echo(f"{suite.name} [{label}]  {s['passed']}/{s['n']} passed ({s['pass_rate']:.0%}, 95% CI "
               f"{s['ci95'][0]:.0%}-{s['ci95'][1]:.0%})  cost ${s['cost_usd']:.4f}  p95 {s['p95_ms']:.0f} ms")
    for c in result.cases:
        if not c.passed:
            why = "; ".join(f"{k.type}: {k.detail}" for k in c.checks if not k.passed) or c.error or c.stop_reason
            typer.echo(f"  FAIL {c.id}: {why}")


@app.command()
def compare(
    suite_path: Path = typer.Argument(..., help="Suite YAML (for the gate settings)"),
    base: Path = typer.Argument(..., help="Baseline run JSON"),
    cand: Path = typer.Argument(..., help="Candidate run JSON"),
    html: Path | None = typer.Option(None, help="Write an HTML report here"),
) -> None:
    """Compare a candidate run with a baseline; exit 1 if the gate fails."""
    suite, _, _ = load_suite(suite_path)
    cmp = do_compare(RunResult.load(base), RunResult.load(cand), suite.gate)
    md = render_markdown(cmp)
    typer.echo(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if html:
        html.write_text(render_html(cmp), encoding="utf-8")
        typer.echo(f"report: {html}")
    raise typer.Exit(1 if cmp.verdict == "fail" else 0)


if __name__ == "__main__":
    app()
