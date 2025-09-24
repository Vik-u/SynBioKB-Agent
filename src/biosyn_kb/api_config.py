from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os
import yaml


@dataclass
class SerpApiConfig:
    api_key: Optional[str] = None
    engine: str = "google"


@dataclass
class BraveApiConfig:
    api_key: Optional[str] = None


@dataclass
class UnpaywallConfig:
    email: Optional[str] = None


@dataclass
class ApiConfig:
    serpapi: SerpApiConfig = field(default_factory=SerpApiConfig)
    brave: BraveApiConfig = field(default_factory=BraveApiConfig)
    unpaywall: UnpaywallConfig = field(default_factory=UnpaywallConfig)


def load_api_config(path: Optional[str | Path]) -> ApiConfig:
    """Load API config from YAML, merging with environment fallbacks.

    Precedence for keys:
      1) YAML file values (if provided)
      2) Environment variables (SERPAPI_API_KEY, BRAVE_API_KEY)
    """

    cfg = ApiConfig()
    data = {}
    p: Optional[Path] = None
    if path:
        p = Path(path)
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
    else:
        # Try common defaults
        for candidate in ("apis.yaml", "apis.yml", "config/apis.yaml"):
            pc = Path(candidate)
            if pc.is_file():
                with pc.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                p = pc
                break

    serp = (data or {}).get("serpapi") or {}
    brv = (data or {}).get("brave") or {}
    upw = (data or {}).get("unpaywall") or {}

    # YAML values first
    cfg.serpapi.api_key = serp.get("api_key") or cfg.serpapi.api_key
    cfg.serpapi.engine = serp.get("engine") or cfg.serpapi.engine
    cfg.brave.api_key = brv.get("api_key") or cfg.brave.api_key
    cfg.unpaywall.email = upw.get("email") or cfg.unpaywall.email

    # Env fallbacks
    cfg.serpapi.api_key = cfg.serpapi.api_key or os.environ.get("SERPAPI_API_KEY")
    cfg.brave.api_key = cfg.brave.api_key or os.environ.get("BRAVE_API_KEY")
    cfg.unpaywall.email = cfg.unpaywall.email or os.environ.get("UNPAYWALL_EMAIL")

    return cfg
