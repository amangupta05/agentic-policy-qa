# Agentic Policy Q\&A (High-Trust RAG)

[![Build](https://img.shields.io/github/actions/workflow/status/your-org/agentic-policy-qa/ci.yml?branch=main)](https://github.com/your-org/agentic-policy-qa/actions)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/your-org/agentic-policy-qa/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/your-org/agentic-policy-qa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Answer policy and compliance questions with verifiable citations, automated self-checks, and strict guarantees on hallucination, latency, and safety.

---

## 🎯 Goals & SLOs

* **Hallucination rate** < 2%
* **Latency** p95 ≤ 1200 ms
* **Availability** ≥ 99.5%
* **Safety**: every answer has valid citations or returns `Insufficient context`

---

## 🏗️ System Overview

Core components:

* **Gateway** (FastAPI, SSE/WebSocket) – request entry, rate limiting, citation formatting
* **Retriever** (Qdrant, SentenceTransformers) – ANN search + cross-encoder reranking
* **Reranker** (CrossEncoder) – improves passage ranking
* **Agents** (LangGraph + Temporal) – Retrieve → Draft → Cite → Verify → Escalate
* **Verifier** (NLI models) – entailment + coverage checks
* **Vector DB** (Qdrant) – HNSW ANN index
* **LLM Serving** (vLLM, TinyLlama baseline) – grounded answer generation
* **Feature Store** (Feast + Redis) – user/org features
* **Observability** (Prometheus, Grafana, OpenTelemetry, Evidently) – metrics, drift, traces
* **UI** (Next.js, WIP) – streaming tokens, clickable citations

---

## 📂 Repository Layout

```
/infra             # docker-compose, helm, terraform
/services
  /gateway         # FastAPI entrypoint, rate limiting, citations
  /retriever       # embed + ANN search + rerank
  /reranker        # CE reranking API
  /agents          # LangGraph + Temporal agents
  /verifier        # claim extraction + NLI checks
  /vllm            # LLM server config
  /feature_svc     # Feast online features
/scripts           # ingest.py, tune_qdrant.py
/monitoring        # prometheus, grafana, evidently
/tests             # unit, e2e, load tests
/docs              # SLOs, architecture, rollback
/ui                # Next.js streaming UI (future)
```

---

## ⚙️ Setup & Installation

### 1. Clone and prepare

```bash
git clone https://github.com/your-org/agentic-policy-qa.git
cd agentic-policy-qa
cp .env.example .env
```

Fill `.env` with:

```ini
RETRIEVER_URL=http://retriever:9000
VLLM_URL=http://vllm:8000
VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VLLM_API_KEY=not-needed
REDIS_URL=redis://redis:6379/0
```

### 2. Build & run services

```bash
docker compose up -d
```

Services exposed:

* Gateway → [http://localhost:8080](http://localhost:8080)
* Retriever → [http://localhost:9000](http://localhost:9000)
* Reranker → [http://localhost:9100](http://localhost:9100)
* vLLM → [http://localhost:8000](http://localhost:8000)
* Qdrant → [http://localhost:6333](http://localhost:6333)
* Grafana → [http://localhost:3000](http://localhost:3000) (admin/admin)

### 3. Ingest sample policies

```bash
printf "Refunds: Customers may request a refund within 30 days with receipt." > data/policies/sample_refund.txt
python scripts/ingest.py
```

---

## 🚀 Usage

### Health & readiness

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

### Non-streaming query

```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "stream": false}' | jq .
```

### Streaming query (SSE)

```bash
curl -N "http://localhost:8080/chat/stream?q=What+is+the+refund+policy"
```

---



---

## 🧪 Testing

Run tests:

```bash
pytest -q
```

Covers:

* Unit tests for retriever and gateway
* Streaming vs non-streaming API
* Fake retriever/LLM responses for safe integration

---

## 🔒 Security

* PII scrubbing in ingestion/logs
* Row-level access and allowlist enforcement
* TLS + auth token ready (for demo/cloud)
* Prompt contract: **only answer from context**

---

## 📈 Roadmap

* Triton-served ONNX encoder + CE
* Ingestion agents & multi-tenant isolation
* HIL (human-in-loop) review UI
* Canary rollout with MLflow + Helm rollback
* Evaluation with RAGAS + drift alerts

---

## 📸 Demo Screenshots

### Query Example

![Demo Query](docs/images/demo-query.png)

### Grafana Metrics

![Grafana Dashboard](docs/images/grafana-dashboard.png)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

