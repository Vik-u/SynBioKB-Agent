from __future__ import annotations

import json
from pathlib import Path

from biosyn_kb.summarize.schema import PageSummary
from biosyn_kb.store.db import init_db, import_summaries_jsonl, query_metrics


def test_schema_and_db_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    sample = {
        "url": "https://example.org/x",
        "title": "Example",
        "chemical": "isobutanol",
        "metrics": [
            {"kind": "titer", "value": 10.5, "unit": "g/L"},
            {"kind": "yield", "value": 80.0, "unit": "%"},
        ],
        "organisms": [{"name": "E. coli", "role": "host"}],
        "enzymes": [
            {"name": "ketoisovalerate decarboxylase", "ec_number": "4.1.1.86"}
        ],
        "starting_substrates": ["glucose"],
        "reaction_steps": [
            {
                "substrate": "2-ketoisovalerate",
                "product": "isobutyraldehyde",
                "enzyme": {"name": "kivd", "ec_number": "4.1.1.86"},
            }
        ],
        "strain_design": [{"gene": "kivd", "action": "overexpression"}],
        "summary_long": "Long summary",
    }
    # schema validate
    ps = PageSummary.model_validate(sample)
    # write jsonl
    jl = tmp_path / "s.jsonl"
    jl.write_text(ps.model_dump_json() + "\n", encoding="utf-8")
    # import
    n = import_summaries_jsonl(db, jl)
    assert n == 1
    rows = query_metrics(db, chemical="isobutanol", limit=10)
    assert any(r["kind"] == "titer" for r in rows)
    assert any(r["kind"] == "yield" for r in rows)

