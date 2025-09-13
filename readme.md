# Agentic Policy Q&A (High-Trust RAG)

## Overview
FastAPI Gateway orchestrates Retrieval → Rerank → vLLM generation with citations. Qdrant stores vectors. Cross-encoder reranks. Metrics via Prometheus.

## Endpoints
- `GET /health` and `/ready` on gateway, retriever, reranker
- `POST /chat` on gateway

## Quickstart (Docker)
```bash
cp .env.example .env
# edit .env locally. keep tokens empty or local only
docker compose up -d --build
# wait for services, then:
curl -s "http://localhost:8080/ready"
python scripts/ingest.py  # one-time sample ingest
curl -s -X POST http://localhost:8080/chat -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy?","stream":false}'
