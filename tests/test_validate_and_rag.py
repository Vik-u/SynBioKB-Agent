from __future__ import annotations

from pathlib import Path
import json

from biosyn_kb.validate import normalize_summary_record, validate_summary_record
from biosyn_kb.store.db import init_db, import_pages_from_html_dir, retrieve_paragraphs


def test_normalize_and_validate():
    rec = {
        "url": "https://example.org/x",
        "title": "T",
        "metrics": [{"kind": "yield", "value": 105, "unit": "%"}, {"kind": "titer", "value": 1.5, "unit": "g L-1"}],
        "enzymes": [{"name": "ADH", "ec_number": "EC 1.1.1.1"}],
    }
    rec2 = normalize_summary_record(rec)
    warns = validate_summary_record(rec2)
    assert any("Yield out of range" in w for w in warns)
    assert rec2["metrics"][1]["unit"] == "g/L"
    assert rec2["enzymes"][0]["ec_number"] == "1.1.1.1"


def test_retrieve_paragraphs(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "a.html").write_text(
        """
        <html><title>Demo</title><body>
        <p>Isobutanol production achieved a titer of 10 g/L.</p>
        <p>Another paragraph without numbers.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    import_pages_from_html_dir(db, html_dir)
    res = retrieve_paragraphs(db, "titer g/L isobutanol", k=2)
    assert res and any("titer" in r["para"].lower() for r in res)

