from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


EC_RE = re.compile(r"\bEC\s*(\d+\.\d+\.\d+\.\d+)\b")
SPECIES_RE = re.compile(r"\b([A-Z][a-z]+\s[a-z]+(?:\s[a-z]+)?)\b")
YIELD_RE = re.compile(r"\b(yield|titer)\b[^%\n\r]{0,40}?(\d+(?:\.\d+)?)\s?%", re.IGNORECASE)


@dataclass
class Extraction:
    chemical: Optional[str]
    enzymes_ec: List[str] = field(default_factory=list)
    organisms: List[str] = field(default_factory=list)
    yields_percent: List[float] = field(default_factory=list)
    evidence: Optional[str] = None
    confidence: float = 0.3


def extract_entities(text: str, *, seed_chemical: Optional[str] = None) -> Extraction:
    ecs = list(dict.fromkeys(m.group(1) for m in EC_RE.finditer(text)))
    species = list(dict.fromkeys(m.group(1) for m in SPECIES_RE.finditer(text)))
    yields = [float(m.group(2)) for m in YIELD_RE.finditer(text)]

    # crude evidence: first 240 chars around first biosynthesis mention
    idx = text.lower().find("biosynth")
    evidence = None
    if idx != -1:
        start = max(0, idx - 120)
        end = min(len(text), idx + 240)
        evidence = text[start:end]

    conf = 0.3
    if seed_chemical and idx != -1:
        conf = 0.5
    if ecs:
        conf += 0.1
    if species:
        conf += 0.1
    if yields:
        conf += 0.1
    conf = min(conf, 0.9)

    return Extraction(
        chemical=seed_chemical,
        enzymes_ec=ecs,
        organisms=species[:10],
        yields_percent=yields[:10],
        evidence=evidence,
        confidence=conf,
    )

