from __future__ import annotations

import re
from typing import Any, Dict, List

from ..summarize.schema import PageSummary, Metric, Enzyme, Organism
from ..store.db import retrieve_paragraphs
from ..normalize.registry import pubchem_inchikey_for_name, uniprot_for_ec, taxonomy_id_for_name
import asyncio
from ..llm.factory import get_llm_client


def _norm_unit(unit: str | None) -> str | None:
    if not unit:
        return unit
    u = unit.strip()
    u = u.replace("g L-1 h-1", "g/L/h").replace("g L-1", "g/L").replace("L⁻¹", "/L").replace("h⁻¹", "/h")
    u = u.replace("mg g-1", "mg/g")
    return u


def _norm_ec(ec: str | None) -> str | None:
    if not ec:
        return ec
    ec = ec.strip()
    if ec.lower().startswith("ec "):
        ec = ec[3:].strip()
    return ec


def normalize_summary_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Units
    for m in obj.get("metrics", []) or []:
        m["unit"] = _norm_unit(m.get("unit"))
    # EC numbers
    for e in obj.get("enzymes", []) or []:
        e["ec_number"] = _norm_ec(e.get("ec_number"))
    for rs in obj.get("reaction_steps", []) or []:
        enz = rs.get("enzyme") or {}
        enz["ec_number"] = _norm_ec(enz.get("ec_number"))
        rs["enzyme"] = enz
    return obj


def validate_summary_record(obj: Dict[str, Any]) -> List[str]:
    """Return a list of warnings (empty if OK)."""
    warns: List[str] = []
    try:
        PageSummary.model_validate(obj)
    except Exception as e:
        warns.append(f"Schema error: {e}")
        return warns
    # Metrics sanity
    for m in obj.get("metrics", []) or []:
        kind = (m.get("kind") or "").lower()
        val = m.get("value")
        if isinstance(val, (int, float)):
            if kind == "yield" and not (0 <= val <= 100):
                warns.append(f"Yield out of range: {val}")
            if kind == "titer" and val < 0:
                warns.append(f"Negative titer: {val}")
    return warns


def rag_validate_summary(db_path: str | None, obj: Dict[str, Any], *, k: int = 5) -> List[str]:
    """Validate that key numeric metrics appear in retrieved contexts.

    Best-effort: for each metric, search paragraphs and check if value/unit are present in text.
    """
    if not db_path:
        return []
    warns: List[str] = []
    try:
        title = obj.get("title") or ""
        chemical = obj.get("chemical") or ""
        query = f"{chemical} {title} "
        for m in obj.get("metrics", []) or []:
            val = m.get("value")
            unit = _norm_unit(m.get("unit")) or ""
            kind = m.get("kind") or ""
            q = f"{query} {kind} {val} {unit}"
            ctxs = retrieve_paragraphs(db_path, q, k=k)
            hit = False
            sv = f"{val}"
            for c in ctxs:
                text = c.get("para", "")
                if sv in text and (unit == "" or unit in text):
                    hit = True
                    break
            if not hit:
                warns.append(f"RAG mismatch for {kind}: {val} {unit} not found in top contexts")
    except Exception:
        pass
    return warns


async def _llm_check_metric(provider: str, base_url: str, api_key: str | None, model: str, metric: Dict[str, Any], ctxs: List[Dict[str, Any]]) -> str | None:
    q = metric.get("kind")
    val = metric.get("value")
    unit = metric.get("unit") or ""
    prompt = (
        "You are verifying scientific claims. Given context paragraphs from sources, check whether the metric appears supported.\n"
        "Return either 'OK' or 'WARN: <reason>'.\n\n"
        f"Metric: {q} = {val} {unit}\n\nContexts:\n"
    )
    for i, c in enumerate(ctxs, 1):
        prompt += f"[{i}] {c['para']}\nSource: {c['url']}\n\n"
    prompt += "Answer:"
    async with get_llm_client(provider, base_url, api_key) as oc:
        out = await oc.generate(model, prompt, options={"temperature": 0.0, "top_p": 1.0})
    s = out.strip()
    if s.upper().startswith("OK"):
        return None
    if s:
        return s
    return "WARN: no confirmation from contexts"


async def async_llm_rag_validate_summary(
    db_path: str | None,
    obj: Dict[str, Any],
    *,
    k: int = 5,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    model: str = "gpt-oss:20b",
) -> List[str]:
    """Async LLM-backed validation that metrics appear in retrieved contexts."""
    if not db_path:
        return []
    try:
        ctxs = retrieve_paragraphs(db_path, f"{obj.get('chemical','')} {obj.get('title','')}", k=k)
        tasks = [
            _llm_check_metric(provider, base_url, api_key, model, m, ctxs)
            for m in (obj.get("metrics", []) or [])
        ]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        warns: List[str] = []
        for o in outs:
            if isinstance(o, Exception):
                continue
            if o:
                warns.append(o)
        return warns
    except Exception:
        return []


def llm_rag_validate_summary(
    db_path: str | None,
    obj: Dict[str, Any],
    *,
    k: int = 5,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    model: str = "gpt-oss:20b",
) -> List[str]:
    """Sync wrapper for environments without an event loop.

    If already inside a running event loop, return [] to avoid nested loop errors;
    async contexts should call async_llm_rag_validate_summary directly.
    """
    if not db_path:
        return []
    try:
        loop = asyncio.get_running_loop()
        # In a running loop; skip (call async version from async context instead)
        return []
    except RuntimeError:
        # No running loop; safe to run
        return asyncio.run(
            async_llm_rag_validate_summary(
                db_path,
                obj,
                k=k,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        )


async def _enrich_async(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Chemical → PubChem
    chem = obj.get("chemical")
    if chem and not obj.get("chemical_xrefs"):
        ikey, cid = await pubchem_inchikey_for_name(chem)
        if ikey or cid:
            obj["chemical_xrefs"] = {"inchikey": ikey, "pubchem_cid": cid}
    # Enzymes → UniProt
    for e in obj.get("enzymes", []) or []:
        if e.get("ec_number") and not e.get("uniprot_id"):
            up = await uniprot_for_ec(e["ec_number"])
            if up:
                e["uniprot_id"] = up
    # Organisms → Taxonomy
    for o in obj.get("organisms", []) or []:
        if o.get("name") and not o.get("taxonomy_id"):
            tid = await taxonomy_id_for_name(o["name"])
            if tid:
                o["taxonomy_id"] = tid
    return obj


def enrich_summary_xrefs(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return asyncio.get_event_loop().run_until_complete(_enrich_async(obj))
    except RuntimeError:
        # No event loop; create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_enrich_async(obj))
        finally:
            loop.close()
