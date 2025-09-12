# Retrieval MVP

## Goal
Search policy chunks and return top-k passages with metadata for grounding.

## Data schema
- `chunk_id`: stable unique id
- `text`: chunk content (normalized, ~512 tokens)
- `source`: canonical file name (e.g., hipaa.pdf)
- `section`: logical section or header if available
- `page`: int page index if from PDF
- `tags`: optional list of topical labels
- `ingested_at`: timestamp (later)
- Vector: embedding for ANN search (cosine)

## Qdrant
- Collection: `policies`
- Distance: cosine
- HNSW params: M=32, ef_construct=128; query ef_search ~256

## API
- `GET /search?q=...&k=5` → `{query, k, chunks:[{chunk_id, source, section, page, score, text}]}`

## Metrics (planned)
- `retrieval_requests_total`
- `retrieval_latency_ms` (histogram)
- `retrieval_topk_size`

## Readiness
- `GET /ready` checks: Qdrant reachable, encoder loaded.

## Next steps
- Implement chunker, encoder, upsert.
- Add Prometheus client instrumentation.
- Add simple synonym expansion for acronyms.
