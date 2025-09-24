from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def crawl_report(pages_dir: Path) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    total = 0
    for p in pages_dir.glob("*.html"):
        # status is not embedded; we approximate by presence of content length
        # For richer stats, upstream prints crawl JSON lines; here we just count files.
        try:
            s = p.stat().st_size
            total += 1
            if s < 2000:
                statuses["tiny"] = statuses.get("tiny", 0) + 1
            else:
                statuses[">=2k"] = statuses.get(">=2k", 0) + 1
        except Exception:
            pass
    return {"html_files": total, "size_buckets": statuses}


def extract_report(extracted_jsonl: Path) -> Dict[str, Any]:
    lines = [json.loads(ln) for ln in extracted_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(lines)
    lengths = [len((x.get("text") or "")) for x in lines]
    if lengths:
        avg = sum(lengths) / len(lengths)
        gt2k = sum(1 for L in lengths if L >= 2000)
        gt8k = sum(1 for L in lengths if L >= 8000)
    else:
        avg = 0
        gt2k = 0
        gt8k = 0
    return {"records": n, "avg_chars": int(avg), ">=2k": gt2k, ">=8k": gt8k}


def summaries_quality(summaries_jsonl: Path) -> Dict[str, Any]:
    lines = [json.loads(ln) for ln in summaries_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(lines)
    with_metrics = sum(1 for x in lines if (x.get("metrics") or []))
    with_evidence = sum(1 for x in lines if (x.get("evidence") or []))
    with_enz = sum(1 for x in lines if (x.get("enzymes") or []))
    return {"records": n, "with_metrics": with_metrics, "with_evidence": with_evidence, "with_enzymes": with_enz}


def write_checkpoint(path: Path, obj: Any) -> None:
    _write_json(path, obj)

