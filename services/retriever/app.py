# services/retriever/app.py
"""
Retriever API (FastAPI) with CrossEncoder reranking

Endpoints:
- GET /health
- GET /ready
- GET /metrics
- GET /search?q=...&k=5
"""
import os
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- Config ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://vectordb:6333")
COLLECTION = os.getenv("COLLECTION", "policies")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_OVERFETCH = int(os.getenv("RERANK_OVERFETCH", "20"))  # ANN top-N before rerank
EF_SEARCH = int(os.getenv("EF_SEARCH", "512"))  # ANN recall knob

# --- App ---
app = FastAPI(title="retriever", version="0.2.0")

# --- Metrics ---
SEARCH_REQS = Counter("retrieval_requests_total", "Total retrieval requests")
SEARCH_LAT_MS = Histogram(
    "retrieval_latency_ms",
    "Search latency in milliseconds",
    buckets=[10, 25, 50, 75, 100, 150, 250, 400, 600, 1000, 2000],
)


# --- Models ---
class Chunk(BaseModel):
    chunk_id: str
    source: str
    section: Optional[str] = None
    page: Optional[int] = None
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    k: int
    latency_ms: float
    chunks: List[Chunk]


# --- Startup ---
@app.on_event("startup")
def _startup():
    global qc, embedder, reranker
    qc = QdrantClient(url=QDRANT_URL)
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)


# --- Health/Ready ---
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, str]:
    try:
        _ = qc.get_collections()
        _ = embedder.get_sentence_embedding_dimension()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


# --- Metrics endpoint ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Search ---
@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), k: int = TOP_K_DEFAULT):
    t0 = time.perf_counter()
    try:
        vec = embedder.encode([q])[0].tolist()

        # Overfetch for reranking
        prelim = qc.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=max(k, RERANK_OVERFETCH),
            with_payload=True,
            search_params=qm.SearchParams(hnsw_ef=EF_SEARCH),
        )

        # Build (query, passage) pairs for CE
        pairs = [(q, (pt.payload or {}).get("text", "")) for pt in prelim]
        scores = reranker.predict(pairs).tolist() if pairs else []

        # Sort by CE score desc and take top-k
        ranked = sorted(
            [
                (scores[i] if i < len(scores) else float(pt.score), pt)
                for i, pt in enumerate(prelim)
            ],
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        chunks: List[Chunk] = []
        for sc, pt in ranked:
            pl = pt.payload or {}
            chunks.append(
                Chunk(
                    chunk_id=str(pt.id),
                    source=str(pl.get("source")),
                    section=pl.get("section"),
                    page=pl.get("page"),
                    score=float(sc),  # CE score
                    text=str(pl.get("text", ""))[:2000],
                )
            )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        SEARCH_REQS.inc()
        SEARCH_LAT_MS.observe(dt_ms)
        return SearchResponse(query=q, k=k, latency_ms=round(dt_ms, 2), chunks=chunks)

    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        SEARCH_REQS.inc()
        SEARCH_LAT_MS.observe(dt_ms)
        raise HTTPException(status_code=500, detail=str(e))
