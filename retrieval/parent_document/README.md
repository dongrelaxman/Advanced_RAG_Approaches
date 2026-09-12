# Parent Document Retrieval

## What it is
Parent Document Retrieval uses two chunk sizes simultaneously. Small child chunks are indexed for precise retrieval. When a child chunk matches, its full large parent chunk is returned to the LLM for rich context.

## The problem it solves
There is a fundamental tension in chunking:
- **Small chunks** → precise retrieval, but lose surrounding context
- **Large chunks** → rich context, but imprecise retrieval

Parent Document Retrieval solves both at once.

## How it works

```
Index time:
  PDF → split into large parent chunks (1500 tokens)
          → split each parent into small child chunks (300 tokens)
          → embed child chunks → store in Weaviate
          → store parent chunks → InMemoryStore (keyed by parent_id)

Query time:
  Query → match child chunks in Weaviate
        → look up their parent_id
        → fetch full parent from InMemoryStore
        → return parent to LLM
```

## Two-level storage

| Level | Size | Where stored | Purpose |
|---|---|---|---|
| Child chunks | 300 tokens | Weaviate (vector index) | Precise retrieval |
| Parent chunks | 1500 tokens | InMemoryStore | Rich context for LLM |

## How it differs from Sentence Window

| | Parent Document | Sentence Window |
|---|---|---|
| Index unit | Child chunks (300 tokens) | Individual sentences |
| Return unit | Pre-defined parent chunk | Dynamic window around matched sentence |
| Context size | Fixed (parent chunk size) | Flexible (window size) |
| Granularity | Chunk-level | Sentence-level |

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `PARENT_CHUNK_SIZE` | 1500 | How much context the LLM receives |
| `CHILD_CHUNK_SIZE` | 300 | Retrieval precision |
| `PARENT_CHUNK_OVERLAP` | 150 | Overlap between parent chunks |
| `CHILD_CHUNK_OVERLAP` | 30 | Overlap between child chunks |

## How to run

```bash
pip install -r parent_document/requirements.txt
python parent_document/retriever.py
```

## No API key needed — runs fully local.
