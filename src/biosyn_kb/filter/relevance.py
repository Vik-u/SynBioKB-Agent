from __future__ import annotations

import re
from typing import Tuple


PATTERNS = [
    r"\bEC\s*\d+\.\d+\.\d+\.\d+\b",
    r"\b(yield|titer|titre|productivity|conversion|selectivity)\b",
    r"\b(g/L|g L-1|g\s*L\-1|mg/L|%|percent|ppm)\b",
    r"\b(knockout|deletion|knock-down|overexpression|upregulation|downregulation)\b",
    r"\b(fermentation|bioreactor|batch|fed\-batch|pH|temperature|\d+\s*C)\b",
]

COMPILED = [re.compile(pat, re.IGNORECASE) for pat in PATTERNS]


def filter_text(text: str, *, min_len: int = 40) -> Tuple[str, float]:
    """Return high-signal paragraphs plus coverage ratio.

    Splits on blank lines, scores paragraphs by matching any of the patterns, and returns
    the concatenation of matched paragraphs along with the ratio of kept chars to total.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
    kept = []
    total_chars = sum(len(p) for p in paras) or 1
    for p in paras:
        if len(p) < min_len:
            continue
        if any(rx.search(p) for rx in COMPILED):
            kept.append(p)
    out = "\n\n".join(kept)
    ratio = (len(out) / total_chars) if total_chars else 0.0
    return out, ratio

