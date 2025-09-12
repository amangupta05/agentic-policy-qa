# services/gateway/app.py
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import Settings, get_settings
from logger import configure_logging, get_logger
from metrics import PROM_APP, RATE_LIMIT_HITS, REQ_COUNT, REQ_LATENCY_MS
from models import ChatRequest, ChatResponse, Health, Ready
from rag import build_messages_with_context
from rate_limiter import RateLimiter

import re
# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------
configure_logging()
app = FastAPI(title="gateway", version="0.3.0")
app.mount("/metrics", PROM_APP)

# CORS (open for dev; restrict origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _new_rid(h: Optional[str]) -> str:
    return h or str(uuid.uuid4())

def _apply_cutoff(chunks: List[Dict], top_k: int, min_score: float) -> List[Dict]:
    xs = [c for c in chunks if float(c.get("score", 0.0)) >= min_score]
    xs.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return xs[:top_k] if top_k > 0 else xs

def _format_citations(chunks: List[Dict]) -> List[Dict]:
    cites: List[Dict] = []
    for c in chunks:
        cites.append(
            {
                "doc_id": c.get("source"),
                "page": c.get("page"),
                "span": (c.get("text") or "")[:200],
                "score": float(c.get("score", 0.0)),
            }
        )
    return cites


_WHITESPACE = re.compile(r"\s+")
_PREAMBLE = re.compile(
    r"^(sure|okay|ok|here(?:'s| is)|based on (?:the )?context|single-?sentence answer.*?:)\s*",
    re.IGNORECASE,
)

def _first_sentence(text: str) -> str:
    s = _WHITESPACE.sub(" ", (text or "").strip())
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)
    return parts[0].strip()

def _clean_sentence(s: str) -> str:
    s = _PREAMBLE.sub("", s or "").strip()
    # strip any leading/trailing quotes (straight or smart), even if only on one side
    s = re.sub(r'^[\"“”‘’\']+', "", s)
    s = re.sub(r'[\"“”‘’\']+$', "", s)
    # if the model returned meta like “X: …” keep only the part after colon
    if ":" in s and not s.lower().startswith("refund"):
        s = s.split(":", 1)[-1].strip()
    return s

def _enforce_format(text: str, citations: list[dict], enforce: bool) -> str:
    if not enforce:
        return text
    if not citations:
        return "Not found in docs."

    cand = _clean_sentence(_first_sentence(text))
    if not cand:
        cand = _clean_sentence(_first_sentence(citations[0].get("span") or ""))

    if not cand:
        return "Not found in docs."

    if cand[-1] not in ".!?":
        cand += "."

    # append a single bracketed citation if none present
    if "[" not in cand and "]" not in cand:
        cand += " [1]"

    return cand or "Not found in docs."


async def _call_reranker(base_url: str, query: str, candidates: List[Dict], timeout_s: float = 10.0) -> List[Dict]:
    payload = {
        "candidates": [
            {
                "chunk_id": str(c.get("chunk_id") or c.get("id") or ""),
                "source": str(c.get("source", "")),
                "text": str(c.get("text", "")),
                "score": float(c.get("score", 0.0)),
            }
            for c in candidates
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as s:
            r = await s.post(f"{base_url.rstrip('/')}/rerank", params={"query": query}, json=payload)
            r.raise_for_status()
            data = r.json()
            reranked = data.get("reranked", [])
            key = lambda x: (x.get("source"), x.get("text"))
            lut = {key(c): c for c in candidates}
            enriched: List[Dict] = []
            for item in reranked:
                k = key(item)
                base = dict(lut.get(k, {}))
                base.update(item)
                enriched.append(base)
            return enriched or candidates
    except Exception:
        return candidates

# ------------------------------------------------------------------------------
# Dependency bundle
# ------------------------------------------------------------------------------
async def deps(
    settings: Settings = Depends(get_settings),
    x_request_id: Optional[str] = Header(default=None, convert_underscores=False),
):
    rid = _new_rid(x_request_id)
    logger = get_logger(rid=rid)

    retriever_url = settings.RETRIEVER_URL
    vllm_url = settings.VLLM_URL
    vllm_api_key = settings.VLLM_API_KEY
    reranker_url = getattr(settings, "RERANKER_URL", "http://reranker:9100")

    retriever_timeout = 10
    vllm_timeout = 60
    limiter = RateLimiter(
        redis_url=settings.REDIS_URL,
        bucket_capacity=settings.RATE_BUCKET_CAPACITY,
        fill_rate_per_s=settings.RATE_BUCKET_FILL_RATE,
    )

    return (
        settings,
        logger,
        retriever_url,
        retriever_timeout,
        vllm_url,
        vllm_api_key,
        vllm_timeout,
        reranker_url,
        limiter,
        rid,
    )

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(status="ok")

@app.get("/ready", response_model=Ready)
async def ready(settings: Settings = Depends(get_settings)) -> Ready:
    try:
        async with httpx.AsyncClient(timeout=3) as s:
            r1 = await s.get(f"{settings.RETRIEVER_URL.rstrip('/')}/health")
            r2 = await s.get(f"{settings.VLLM_URL.rstrip('/')}/v1/models")
        return Ready(status="ready", details={"retriever": r1.json(), "vllm_models": r2.status_code})
    except Exception as e:
        return Ready(status="not_ready", details={"error": str(e)})

# ------------------------------------------------------------------------------
# Chat JSON (POST /chat) and SSE (GET /chat/stream)
# ------------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    settings_logger_clients=Depends(deps),
):
    return await _handle_chat(req, request, settings_logger_clients, stream=False)

@app.get("/chat/stream")
async def chat_stream(
    q: str,
    top_k: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    request: Request = None,
    settings_logger_clients=Depends(deps),
):
    # Provide a GET SSE alias for tooling compatibility
    req = ChatRequest(query=q, messages=None, stream=True, temperature=temperature, max_tokens=max_tokens, top_k=top_k)
    return await _handle_chat(req, request, settings_logger_clients, stream=True)

# Core handler used by both endpoints
async def _handle_chat(
    req: ChatRequest,
    request: Request,
    settings_logger_clients,
    stream: bool,
):
    t0 = time.perf_counter()
    (
        settings,
        logger,
        retriever_url,
        retriever_timeout,
        vllm_url,
        vllm_api_key,
        vllm_timeout,
        reranker_url,
        limiter,
        rid,
    ) = settings_logger_clients

    # Rate limit
    key = req.api_key or (request.client.host if request and request.client else "anon")
    if not await limiter.consume(key, tokens=1):
        RATE_LIMIT_HITS.inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    REQ_COUNT.labels(endpoint="chat").inc()

    # Retrieval
    try:
        params = {"q": req.query, "k": req.top_k or settings.RETRIEVER_TOP_K}
        async with httpx.AsyncClient(timeout=retriever_timeout, headers={"x-request-id": rid}) as s:
            r = await s.get(f"{retriever_url}/search", params=params)
            r.raise_for_status()
            retrieval = r.json()
            chunks = retrieval.get("chunks", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"retrieval error: {e}")

    # Rerank + cutoff
    reranked_chunks: List[Dict] = await _call_reranker(
        base_url=reranker_url, query=req.query, candidates=chunks, timeout_s=10.0
    )
    reranked_chunks = _apply_cutoff(
        reranked_chunks,
        top_k=getattr(settings, "RERANK_TOP_K_POST", 3),
        min_score=getattr(settings, "RERANK_MIN_SCORE", 0.0),
    )

    # Messages and citations
    messages = build_messages_with_context(
        user_query=req.query,
        history=req.messages or [],
        chunks=reranked_chunks,
        system_preamble=settings.SYSTEM_PROMPT,
    )
    citations = _format_citations(reranked_chunks)

    if not reranked_chunks:
        return JSONResponse(
            {"request_id": rid, "query": req.query, "context_count": 0,
            "output": "Not found in docs.", "latency_ms": 0.0, "citations": []},
            headers={"x-request-id": rid},
        )

    # vLLM call
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "temperature": min(req.temperature, 0.2),
        "max_tokens": min(req.max_tokens, 128),
        "stop": ["\n\n\n"],  # mild stop, optional
    }

    if stream or req.stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            nonlocal t0
            try:
                async with httpx.AsyncClient(timeout=vllm_timeout, headers=headers) as s:
                    stream_payload = dict(payload, stream=True)
                    async with s.stream(
                        "POST",
                        f"{vllm_url.rstrip('/')}/v1/chat/completions",
                        json=stream_payload,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if not data:
                                    continue
                                if data == "[DONE]":
                                    break
                                yield f"data: {data}\n\n".encode()
            except httpx.HTTPError as e:
                err = {"error": f"generation error: {str(e)}"}
                yield f"data: {json.dumps(err)}\n\n".encode()

            dt_ms = (time.perf_counter() - t0) * 1000.0
            footer = {"request_id": rid, "citations": citations, "latency_ms": round(dt_ms, 2)}
            yield f"data: {json.dumps(footer)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-request-id": rid})

    # Non-stream JSON
    try:
        async with httpx.AsyncClient(timeout=vllm_timeout, headers=headers) as s:
            nonstream_payload = dict(payload, stream=False)
            r = await s.post(f"{vllm_url.rstrip('/')}/v1/chat/completions", json=nonstream_payload)
            r.raise_for_status()
            resp_json = r.json()
            text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = _enforce_format(text, citations, enforce=True)
            
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"generation error: {e}")

    dt_ms = (time.perf_counter() - t0) * 1000.0
    REQ_LATENCY_MS.labels(endpoint="chat").observe(dt_ms)

    return JSONResponse(
        ChatResponse(
            request_id=rid,
            query=req.query,
            context_count=len(reranked_chunks),
            output=text,
            latency_ms=round(dt_ms, 2),
            citations=citations,
        ).model_dump(),   # <- replace .dict() with .model_dump()
        headers={"x-request-id": rid},
    )
