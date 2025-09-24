from __future__ import annotations

from typing import Any, Dict, List, Optional
import re


_EC_RX = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def deterministic_checks(summary: Dict[str, Any]) -> List[str]:
    """Return deterministic warnings for a PageSummary-like object.

    Checks:
    - year plausible
    - chemical present
    - metrics sane (yield 0..100, titer/productivity > 0)
    - EC numbers format
    - evidence present if metrics claimed
    """
    warns: List[str] = []
    yr = summary.get("year")
    if yr is not None:
        try:
            y = int(yr)
            if y < 1900 or y > 2100:
                warns.append(f"implausible year: {yr}")
        except Exception:
            warns.append(f"invalid year: {yr}")
    chem = summary.get("chemical")
    if not chem:
        warns.append("missing chemical")
    # metrics
    for m in summary.get("metrics", []) or []:
        kind = (m.get("kind") or "").lower()
        val = m.get("value")
        if isinstance(val, (int, float)):
            if kind == "yield" and not (0 <= val <= 100):
                warns.append(f"yield out of range: {val}")
            if kind in ("titer", "productivity") and val < 0:
                warns.append(f"negative {kind}: {val}")
    # EC numbers
    for e in summary.get("enzymes", []) or []:
        ec = (e.get("ec_number") or "").strip()
        if ec and not _EC_RX.match(ec):
            warns.append(f"invalid ec_number: {ec}")
    # evidence presence if metrics exist
    if (summary.get("metrics") or []) and not (summary.get("evidence") or []):
        warns.append("metrics without evidence quotes")
    return warns

