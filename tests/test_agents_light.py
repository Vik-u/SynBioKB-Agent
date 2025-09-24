from __future__ import annotations

from pathlib import Path

from biosyn_kb.agents.crew import build_crew, run_crew_on_local


def test_agents_metadata():
    crew = build_crew()
    assert crew.name and crew.purpose
    assert len(crew.agents) >= 3
    for a in crew.agents:
        assert a.role and a.task and a.background and a.backstory
        assert a.expertise and isinstance(a.expertise, list)
        assert a.description and a.expected_outputs


def test_agents_run_light(tmp_path: Path):
    # Create a tiny HTML file
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "a.html").write_text(
        """
        <html><title>Demo</title><body>
        <h1>Isobutanol production</h1>
        <p>We observed a yield of 90% and a titer of 10 g/L at pH 6.5.</p>
        <p>EC 4.1.1.86 was identified as a key enzyme.</p>
        </body></html>
        """,
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    db = tmp_path / "biosyn.db"
    out_jsonl = tmp_path / "summaries.jsonl"

    res = run_crew_on_local(html_dir, work_dir, db, out_jsonl, limit=1, skip_llm=True)
    assert res["status"] == "ok"
    # Summaries file should exist and contain at least one line
    assert out_jsonl.exists()
    assert out_jsonl.read_text(encoding="utf-8").strip() != ""

