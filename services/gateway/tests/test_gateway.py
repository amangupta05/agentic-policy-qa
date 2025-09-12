from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.gateway.app import app, deps
from services.gateway.config import Settings
from services.gateway.logger import get_logger


# --------- Fakes for network ----------

class _FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class _FakeStream:
    def __init__(self, lines, status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    def __init__(self, timeout=None, headers=None):
        self.timeout = timeout
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, params=None, headers=None):
        if url.endswith("/search"):
            return _FakeResponse(
                {
                    "query": params.get("q"),
                    "k": params.get("k"),
                    "latency_ms": 1.2,
                    "chunks": [
                        {"source": "doc1.txt", "score": 0.9, "text": "policy A. answer 42."},
                        {"source": "doc2.txt", "score": 0.8, "text": "policy B. limits apply."},
                    ],
                },
                200,
            )
        return _FakeResponse({}, 404)

    async def post(self, url: str, json=None):
        if url.endswith("/v1/chat/completions") and not json.get("stream", False):
            return _FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "Final answer with citations [1]."}}
                    ]
                },
                200,
            )
        return _FakeResponse({}, 404)

    def stream(self, method: str, url: str, json=None):
        if url.endswith("/v1/chat/completions") and json and json.get("stream"):
            lines = [
                "data: " + '{"choices":[{"delta":{"content":"Final"}}]}',
                "data: " + '{"choices":[{"delta":{"content":" answer"}}]}',
                "data: [DONE]",
            ]
            return _FakeStream(lines, status_code=200)
        return _FakeStream([], status_code=404)


# --------- Dummy limiter ----------

class DummyLimiter:
    async def consume(self, key: str, tokens: int = 1) -> bool:
        return True


# --------- Fixtures ----------

@pytest.fixture(autouse=True)
def patch_httpx_and_deps(monkeypatch):
    import types
    import services.gateway.app as gw_app

    # stub out httpx with fake AsyncClient + Exception for HTTPError
    monkeypatch.setattr(
        gw_app,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient, HTTPError=Exception),
    )

    async def deps_override():
        s = Settings()
        return (
            s,
            get_logger("test"),
            "http://retriever:9000",
            5,
            "http://vllm:8000",
            "",
            10,
            "http://reranker:9100",
            DummyLimiter(),
            "rid-test",
        )

    app.dependency_overrides[deps] = deps_override
    yield
    app.dependency_overrides.clear()


# --------- Tests ----------

def test_chat_non_stream():
    client = TestClient(app)
    r = client.post("/chat", json={"query": "What is policy A?", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["output"].startswith("Final answer")
    assert body["context_count"] >= 1
    assert isinstance(body["citations"], list)


def test_chat_stream_post():
    client = TestClient(app)
    with client.stream("POST", "/chat", json={"query": "What is policy A?", "stream": True}) as s:
        assert s.status_code == 200
        text = "".join([chunk.decode("utf-8") for chunk in s.iter_raw()])
        assert "data:" in text
        assert "[DONE]" in text


def test_chat_stream_get_alias():
    client = TestClient(app)
    with client.stream("GET", "/chat/stream", params={"q": "What is policy A?", "top_k": 2}) as s:
        assert s.status_code == 200
        text = "".join([chunk.decode("utf-8") for chunk in s.iter_raw()])
        assert "data:" in text
