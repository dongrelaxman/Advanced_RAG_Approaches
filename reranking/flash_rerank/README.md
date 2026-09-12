# Flash Rerank

## What it is
Flash Rerank is a two-stage retrieval pipeline. Stage 1 uses hybrid search (BM25 + vector + RRF) for high recall. Stage 2 uses a lightweight cross-encoder model (FlashRank) to re-score and reorder the candidates for high precision.

## The problem it solves
Initial retrieval (BM25 or vector search) optimizes for **recall** — it casts a wide net to make sure relevant documents are in the pool. But the ranking within that pool is imprecise. FlashRank adds a smarter ranking step on top.

## How it works

```
Query
  ↓
Hybrid Search (BM25 + Vector + RRF) → top-20 candidates   [recall stage]
  ↓
FlashRank cross-encoder reranker    → top-5 re-scored      [precision stage]
  ↓
Final results
```

## Why cross-encoders are more accurate
- **Bi-encoders** (vector search): embed query and document *separately*, then compare. Fast but less accurate.
- **Cross-encoders** (FlashRank): look at query and document *together* in a single pass. Slower but much more accurate.

FlashRank uses small, quantized cross-encoder models that run fast on CPU — no GPU required.

## RRF vs FlashRank — are they alternatives?

No, they serve different purposes and work together:

| | RRF | FlashRank |
|---|---|---|
| **Purpose** | Fuse results from multiple retrievers | Re-score a pool of candidates |
| **Stage** | During retrieval fusion | After retrieval |
| **How it works** | Math formula, no ML model | Neural cross-encoder model |

RRF fuses BM25 + vector results. FlashRank then reranks the fused pool.

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | Cross-encoder model used for reranking |
| `fetch_k` | 20 | Candidates passed to FlashRank from hybrid search |
| `top_k` | 5 | Final results returned after reranking |

## How to run

```bash
pip install -r flash_rerank/requirements.txt
python flash_rerank/reranker.py
```

## No API key needed — runs fully local.
