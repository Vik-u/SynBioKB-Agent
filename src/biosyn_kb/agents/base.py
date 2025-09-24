from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Agent:
    name: str
    role: str
    task: str
    background: str
    backstory: str
    expertise: List[str]
    description: str
    expected_outputs: List[str]
    run_fn: Callable[["Context"], Any]

    def run(self, ctx: "Context") -> Any:
        return self.run_fn(ctx)


@dataclass
class Context:
    # IO and config
    html_dir: Path
    work_dir: Path
    db_path: Path
    summaries_jsonl: Path
    model: str = "gpt-oss:20b"
    ollama_url: str = "http://localhost:11434"
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    limit: Optional[int] = None
    max_chars: int = 6000
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    use_filter: bool = False
    chunked: bool = False
    chunk_chars: int = 3500
    skip_llm: bool = False

    # runtime state
    urls_file: Optional[Path] = None
    pages_jsonl: Optional[Path] = None
    metrics_preview: List[Dict[str, Any]] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    report_path: Optional[Path] = None


@dataclass
class Crew:
    name: str
    purpose: str
    agents: List[Agent]

    def run(self, ctx: Context) -> None:
        for a in self.agents:
            a.run(ctx)
