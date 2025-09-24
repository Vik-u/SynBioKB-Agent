from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import yaml


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 42
    api_key: Optional[str] = None


def load_llm_config(path: Optional[str | Path]) -> Optional[LLMConfig]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    llm = (data or {}).get("llm") or {}
    cfg = LLMConfig()
    cfg.provider = llm.get("provider", cfg.provider)
    cfg.base_url = llm.get("base_url", cfg.base_url)
    cfg.model = llm.get("model", cfg.model)
    cfg.temperature = float(llm.get("temperature", cfg.temperature))
    cfg.top_p = float(llm.get("top_p", cfg.top_p))
    cfg.seed = llm.get("seed", cfg.seed)
    cfg.api_key = llm.get("api_key")
    return cfg
