Agentic Policy Q&A (High-Trust RAG)GoalsAnswer policy/compliance questions with citations and self-verification.Demonstrate end-to-end skills: retrieval, generation, agents, observability.SLOsHallucination < 2% (RAGAS faithfulness ≥ 0.98 over 200 Qs)p95 latency ≤ 1200 msUptime 99.5% (gateway)Architectureflowchart LR
  U[User/UI] <--> |SSE| GW[FastAPI Gateway]
  subgraph RET [Retrieval]
    ENC[Encoder]
    VEC[Vector DB]
    RER[Cross-Encoder Reranker]
  end
  subgraph GEN [Generation]
    VLLM[vLLM Server]
  end
  subgraph AG [Agents]
    RETA[Retrieve]
    DRAFT[Draft]
    CITE[Citation]
    VERIFY[Verify]
  end
  subgraph OBS [Observability]
    PROM[Prometheus]
    OTL[OpenTelemetry]
    GRA[Grafana]
  end
  GW --> RETA --> ENC --> VEC --> RER --> DRAFT --> VLLM --> CITE --> VERIFY --> GW
  GW --> PROM
  GW --> OTL
