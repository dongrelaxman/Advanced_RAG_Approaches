# Rerank Fusion (RRF)

## What it is
Reciprocal Rank Fusion (RRF) is a score fusion algorithm that combines ranked result lists from multiple retrievers into a single unified ranking — without needing to normalize scores across retrievers.

## The problem it solves
Different retrievers (BM25, vector search) produce scores on completely different scales. You can't just average them. RRF sidesteps this by working with **ranks** instead of raw scores.

## How it works

```
BM25 results:    [chunk_A (rank 1), chunk_C (rank 2), chunk_B (rank 3)]
Vector results:  [chunk_B (rank 1), chunk_A (rank 2), chunk_D (rank 3)]

RRF score = sum of 1 / (k + rank) across all retrievers

chunk_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
chunk_B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
chunk_C: 1/(60+2) + 0         = 0.0161
chunk_D: 0        + 1/(60+3) = 0.0159

Final ranking: chunk_A > chunk_B > chunk_C > chunk_D
```

## The RRF formula

```
RRF_score(doc) = Σ  1 / (k + rank_i)
                 i
```

- `k = 60` — smoothing constant from the original paper (Cormack et al., 2009)
- `rank_i` — the rank of the document in retriever i
- Works with any number of retrievers

## Why k=60
The value 60 was empirically determined in the original paper. It prevents top-ranked documents from dominating too heavily and gives lower-ranked documents a fair contribution.

## Shared module
`rerank_fusion.py` exports `reciprocal_rank_fusion()` which is imported and used by:
- `hybrid_search/hybrid_search.py`
- `flash_rerank/reranker.py`

## How to run (standalone)

```bash
pip install -r rerank_fusion/requirements.txt
python rerank_fusion/rerank_fusion.py
```

## No API key needed — runs fully local.
