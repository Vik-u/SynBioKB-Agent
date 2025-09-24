from __future__ import annotations

from biosyn_kb.extract.clean import extract_clean_text
from biosyn_kb.filter.relevance import filter_text
from biosyn_kb.entities import extract_entities
from biosyn_kb.validate import validate_summary_record


def test_3hp_extraction_and_validation():
    html = """
    <html><head><title>3-HP biosynthesis</title></head>
    <body>
      <h1>Biosynthesis of 3-hydroxypropionic acid (3-HP)</h1>
      <p>We engineered E. coli for 3-HP production using the malonyl-CoA pathway.</p>
      <p>Titer reached 80 g/L with a yield of 60% under fed-batch conditions.</p>
      <p>Key enzymes included malonyl-CoA reductase; other enzymes with EC 4.2.1.9 are also involved.</p>
    </body></html>
    """

    # Clean extract
    clean = extract_clean_text(html)
    assert "3-hydroxypropionic" in clean.text or "3-HP" in clean.text

    # Relevance filter keeps metric paragraph
    filt, ratio = filter_text(clean.text)
    assert "80 g/L" in filt or "yield of 60%" in filt

    # Entity extraction (regex-based) should pick yield and EC number
    ents = extract_entities(clean.text, seed_chemical="3-hydroxypropionic acid")
    assert 60.0 in ents.yields_percent
    assert any(ec == "4.2.1.9" for ec in ents.enzymes_ec)

    # Build a tiny summary-like record and validate
    rec = {
        "url": "https://example.org/3hp",
        "title": "3-HP biosynthesis",
        "chemical": "3-hydroxypropionic acid",
        "metrics": [
            {"kind": "titer", "value": 80.0, "unit": "g/L"},
            {"kind": "yield", "value": 60.0, "unit": "%"},
        ],
        "enzymes": [{"name": "enzymeX", "ec_number": "4.2.1.9"}],
        "organisms": [{"name": "Escherichia coli"}],
    }
    warns = validate_summary_record(rec)
    assert warns == []

