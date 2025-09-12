"""
Ingestion pipeline skeleton.

Goal:
- Load files from data/policies
- Normalize and chunk (~512 tokens with overlap)
- Embed chunks (e.g., all-MiniLM-L6-v2 ONNX)
- Upsert to Qdrant collection 'policies' with cosine metric

"""

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer


# ...
def _point_id(source: str, idx: int, text: str) -> uuid.UUID:
    # stable UUID5 from deterministic string
    base = f"{source}:{idx}:{text[:64]}"
    return uuid.uuid5(uuid.NAMESPACE_URL, base)


DATA_DIR = Path("data/policies")
COLLECTION = "policies"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def load_documents(data_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() in (".txt",):
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append(
                {"id": _hash(str(p)), "source": p.name, "text": text, "meta": {}}
            )
    return docs


def normalize(text: str) -> str:
    return "\n".join(
        line.strip() for line in text.replace("\r", "").splitlines() if line.strip()
    )


def _split_by_tokens(text: str, tokenizer, max_len: int, overlap: int) -> List[str]:
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + max_len]
        if not window:
            break
        chunk = tokenizer.convert_tokens_to_string(window)
        chunks.append(chunk.strip())
        if i + max_len >= len(tokens):
            break
        i += max_len - overlap
    return chunks


def chunk(doc: Dict[str, Any], tokenizer) -> List[Dict[str, Any]]:
    text = normalize(doc["text"])
    parts = _split_by_tokens(text, tokenizer, CHUNK_TOKENS, OVERLAP_TOKENS)
    out = []
    for idx, part in enumerate(parts):
        cid = _point_id(doc["source"], idx, part)
        out.append(
            {
                "chunk_id": cid,
                "text": part,
                "source": doc["source"],
                "section": None,
                "page": None,
            }
        )
    return out


def ensure_collection(qc: QdrantClient, dim: int):
    names = [c.name for c in qc.get_collections().collections]
    if COLLECTION not in names:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(m=32, ef_construct=128),
        )


def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).tolist()


def upsert(qc, chunks, vectors):
    points = []
    for c, v in zip(chunks, vectors):
        points.append(
            qm.PointStruct(
                id=str(c["chunk_id"]),  # <-- make it a string
                vector=v,
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "section": c["section"],
                    "page": c["page"],
                    "ingested_at": int(time.time()),
                },
            )
        )
    qc.upsert(collection_name=COLLECTION, points=points)


def main():
    print(f"[ingest] loading embedder: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    tokenizer = model.tokenizer  # uses model’s tokenizer

    print(f"[ingest] ensuring collection at {QDRANT_URL}")
    qc = QdrantClient(url=QDRANT_URL)
    ensure_collection(qc, dim=model.get_sentence_embedding_dimension())

    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit(f"No documents in {DATA_DIR}")

    total_chunks = 0
    for d in docs:
        ch = chunk(d, tokenizer)
        vecs = embed_texts(model, [c["text"] for c in ch])
        upsert(qc, ch, vecs)
        total_chunks += len(ch)
        print(f"[ingest] {d['source']}: {len(ch)} chunks")

    print(f"[ingest] done. total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
