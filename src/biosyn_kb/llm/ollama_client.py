from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.aclose()
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        # GET /api/tags returns installed models
        url = f"{self.base_url}/api/tags"
        r = await self._client.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("models", [])

    async def generate(self, model: str, prompt: str, *, options: Optional[Dict[str, Any]] = None, format: Optional[str] = None) -> str:
        # POST /api/generate
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        # The response includes the full completion in 'response'
        return data.get("response", "")


async def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    async with OllamaClient(base_url) as oc:
        models = await oc.list_models()
        return [m.get("name", "") for m in models if m.get("name")]
