from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from ..llm.factory import get_llm_client


PROMPT = (
    "You are an expert information extractor for biosynthesis papers.\n"
    "Given the text of a page, extract ONLY a compact JSON object with these fields: \n"
    "- chemical: string or null\n"
    "- enzymes_ec: array of EC numbers as strings (e.g., ['4.2.1.9'])\n"
    "- organisms: array of organism names (binomials or strains)\n"
    "- yields_percent: array of numeric yields in percent (e.g., [60.0])\n"
    "- evidence: short quote (<=240 chars) supporting biosynthesis context\n"
    "Guidelines: prefer precise EC numbers; keep only organism names (no boilerplate).\n"
    "Return ONLY valid JSON. No markdown, no commentary.\n"
)


async def llm_extract_entities_text(
    text: str,
    *,
    model: str,
    ollama_url: str = "http://localhost:11434",
    seed_chemical: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = 42,
) -> dict:
    t = text[:12000]  # hard cap per record
    head = "Seed chemical: " + (seed_chemical or "") + "\n\n" if seed_chemical else ""
    prompt = PROMPT + "\n" + head + t
    opts = {"temperature": temperature, "top_p": top_p}
    if seed is not None:
        opts["seed"] = seed
    async with get_llm_client("ollama", ollama_url, None) as oc:
        out = await oc.generate(model=model, prompt=prompt, options=opts)
        s = out.strip()
        # try direct parse, else JSON substring
        try:
            return json.loads(s)
        except Exception:
            try:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start : end + 1])
            except Exception:
                pass
    return {"chemical": seed_chemical, "enzymes_ec": [], "organisms": [], "yields_percent": [], "evidence": None}
