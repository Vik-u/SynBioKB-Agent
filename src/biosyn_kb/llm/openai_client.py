from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class OpenAIClient:
    def __init__(self, base_url: str = "https://api.openai.com/v1", api_key: Optional[str] = None, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAIClient":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.aclose()
        return None

    async def generate(self, model: str, prompt: str, *, options: Optional[Dict[str, Any]] = None, format: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": (options or {}).get("temperature", 0.0),
            "top_p": (options or {}).get("top_p", 1.0),
            "stream": False,
        }
        # If JSON is explicitly requested, try response_format (not all models support it)
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = (((data.get("choices") or [{}])[0] or {}).get("message") or {}).get("content")
        return content or ""

