from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..llm.factory import get_llm_client


REPORT_PROMPT = (
    "You are a scientific synthesis assistant. Given multiple structured summaries of studies,\n"
    "produce a single consolidated report with these requirements:\n"
    "- Organize chronologically by year when available.\n"
    "- Summarize approaches, pathways, organisms, enzymes (with EC numbers), strain designs.\n"
    "- Extract and compare key numeric metrics (yields/titers/productivities), include a compact table.\n"
    "- Provide inline citations as [n] mapping to a reference list at the end with titles and URLs.\n"
    "- Be concise but specific; avoid generic text.\n"
    "Return Markdown.\n"
)


async def compose_report_from_jsonl(
    summaries_jsonl: Path,
    out_path: Path,
    *,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    model: str = "gpt-oss:20b",
) -> None:
    lines = [json.loads(ln) for ln in summaries_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Build compact input with IDs for citation mapping
    srcs: List[str] = []
    for i, obj in enumerate(lines, 1):
        title = obj.get("title") or obj.get("url")
        url = obj.get("url")
        year = obj.get("year")
        chem = obj.get("chemical")
        key = obj.get("key_findings") or []
        metrics = obj.get("metrics") or []
        enzymes = obj.get("enzymes") or []
        organisms = obj.get("organisms") or []
        srcs.append(json.dumps({
            "id": i,
            "title": title,
            "url": url,
            "year": year,
            "chemical": chem,
            "key_findings": key,
            "metrics": metrics,
            "enzymes": enzymes,
            "organisms": organisms,
        }, ensure_ascii=False))
    prompt = REPORT_PROMPT + "\nSources (JSON, one per line):\n" + "\n".join(srcs) + "\n\nReport:"
    async with get_llm_client(provider, base_url, api_key) as oc:
        md = await oc.generate(model, prompt, options={"temperature": 0.2, "top_p": 1.0})
    out_path.write_text(md.strip(), encoding="utf-8")

