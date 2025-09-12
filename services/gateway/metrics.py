from __future__ import annotations

from prometheus_client import Counter, Histogram, make_asgi_app

REQ_COUNT = Counter("gateway_requests_total", "Total requests", ["endpoint"])
REQ_LATENCY_MS = Histogram(
    "gateway_latency_ms", "Latency in ms", ["endpoint"], buckets=[10, 25, 50, 100, 200, 400, 800, 1600]
)
RATE_LIMIT_HITS = Counter("gateway_rate_limit_hits_total", "Total rate limit rejections")

PROM_APP = make_asgi_app()
