from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx


class RetrieverClient:
    def __init__(self, base_url: str, timeout_s: int = 10):
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def search(self, query: str, k: int) -> Dict[str, Any]:
        params = {"q": query, "k": k}
        async with httpx.AsyncClient(timeout=self.timeout_s) as s:
            r = await s.get(f"{self.base_url}/search", params=params)
            r.raise_for_status()
            return r.json()


class VLLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._headers = {"Authorization": f"Bearer {self.api_key}"}

    async def chat(
        self, model: str, messages: List[dict], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers) as s:
            r = await s.post(f"{self.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

    async def stream_chat(
        self, model: str, messages: List[dict], temperature: float, max_tokens: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers) as s:
            async with s.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            # Best effort; skip malformed chunk
                            continue
