from __future__ import annotations

from pathlib import Path
from typing import Optional

from .orchestrator import PipelineConfig, run_local_pipeline
from .crew import build_crew, run_crew_on_local
from .crewai_impl import run_crewai_pipeline


async def run_crewai_or_fallback(
    html_dir: Path,
    work_dir: Path,
    db_path: Path,
    out_jsonl: Path,
    *,
    model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434",
    limit: Optional[int] = None,
    max_chars: int = 6000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> None:
    # CrewAI is required as the primary agent runtime; no non-LLM fallback
    try:
        import crewai  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "CrewAI is required. Install it via 'make crewai' (or pip install crewai langchain-community langchain-ollama)."
        ) from e

    from .crewai_impl import run_crewai_pipeline_async
    await run_crewai_pipeline_async(
        html_dir=html_dir,
        work_dir=work_dir,
        db_path=db_path,
        out_jsonl=out_jsonl,
        model=model,
        ollama_url=ollama_url,
        limit=limit,
        max_chars=max_chars,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
