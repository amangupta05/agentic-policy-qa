from __future__ import annotations
import os, time
from typing import List, Dict
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI(title="reranker", version="0.1.0")
MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ce = CrossEncoder(MODEL_NAME)

class Candidate(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float

class RerankResponse(BaseModel):
    query: str
    latency_ms: float
    reranked: List[Candidate]

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/rerank", response_model=RerankResponse)
def rerank(query: str = Query(..., min_length=1), candidates: List[Candidate] = []):
    t0 = time.perf_counter()
    pairs = [(query, c.text) for c in candidates]
    scores = ce.predict(pairs).tolist()
    reranked = sorted(
        [Candidate(**c.dict(), score=float(s)) for c, s in zip(candidates, scores)],
        key=lambda x: x.score,
        reverse=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return RerankResponse(query=query, latency_ms=round(dt_ms, 2), reranked=reranked)
