# Hybrid Search

## What it is
Hybrid Search combines two independent retrieval methods — keyword search and vector search — and fuses their results into a single ranked list using Reciprocal Rank Fusion (RRF).

## The three approaches inside

### Keyword Search (TF-IDF)
- Matches exact terms using TF-IDF scoring via scikit-learn
- Ranks chunks by cosine similarity of their TF-IDF vectors to the query
- Fast, no embeddings needed
- Misses synonyms and semantic variations

### BM25 Search
- Industry-standard sparse retrieval algorithm
- Improves on TF-IDF with term frequency saturation and document length normalization
- Better than raw TF-IDF for most keyword retrieval tasks

### Hybrid Search (BM25 + Vector + RRF)
- Runs BM25 and vector search independently
- Fuses their ranked results using RRF (see `rerank_fusion/`)
- Gets the best of both: keyword precision + semantic understanding

## How it works

```
Query
  ├── BM25 search  → ranked list A
  └── Vector search → ranked list B
            ↓
        RRF fusion
            ↓
     final ranked list
```

## Why combine them

| Method | Strength | Weakness |
|---|---|---|
| Keyword (BM25) | Exact term matching | Misses synonyms, paraphrases |
| Vector search | Semantic understanding | Misses exact keyword matches |
| Hybrid | Both | Slightly more complex |

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Size of each text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |

## How to run

```bash
pip install -r hybrid_search/requirements.txt

# Run keyword search (TF-IDF)
python hybrid_search/keyword_search.py

# Run BM25 search
python hybrid_search/bm25_search.py

# Run vector search
python hybrid_search/vector_search.py

# Run hybrid search (BM25 + vector + RRF)
python hybrid_search/hybrid_search.py
```

## No API key needed — runs fully local.
