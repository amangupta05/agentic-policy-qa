#!/usr/bin/env python
from __future__ import annotations
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    CollectionParamsDiff,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("COLLECTION", "policies")
DIM = 384  # sentence-transformers/all-MiniLM-L6-v2

qc = QdrantClient(url=QDRANT_URL)

# Drop if exists
existing = {c.name for c in qc.get_collections().collections}
if COLL in existing:
    qc.delete_collection(COLL)

# Create with tuned HNSW + optimizers
qc.create_collection(
    collection_name=COLL,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=32, ef_construct=256, full_scan_threshold=10_000),
    optimizers_config=OptimizersConfigDiff(
        default_segment_number=2,
        indexing_threshold=2_000,
        memmap_threshold=200_000,
        max_optimization_threads=0,
    ),
)

print("Collection recreated:", COLL)
