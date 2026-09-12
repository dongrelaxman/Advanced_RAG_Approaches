# RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

## What it is
RAPTOR builds a tree of LLM-generated summaries over the document, bottom-up. All nodes — raw chunks and summaries at every level — are indexed together, enabling retrieval at any level of abstraction simultaneously.

**Paper:** Sarthi et al., 2024 — https://arxiv.org/abs/2401.18059

## The problem it solves
All other retrieval methods retrieve at the **chunk level** — they find specific passages but miss the big picture. High-level questions like *"What is the overall contribution of this paper?"* have no single chunk answer.

## How the tree is built

```
Level 0 — Raw chunks (leaf nodes, indexed at start)
    ↓  embed → UMAP dimensionality reduction → K-Means clustering → LLM summarizes each cluster
Level 1 — Cluster summaries (broader topics)
    ↓  repeat clustering + summarization on the summaries
Level 2 — Summaries of summaries (even broader)
    ↓  ... recurse until too few texts to cluster further
Root   — Single document-level summary
```

All nodes at all levels are stored in the **same vector store** and searched together at query time.

## Query time

```
Query → similarity search across ALL levels simultaneously
      → specific chunks (level 0) for specific questions
      → summaries (level 1, 2) for broad questions
```

## Example

| Query | What gets retrieved |
|---|---|
| `"What is scaled dot product?"` | Level 0 leaf chunk (specific) |
| `"What are the main components?"` | Level 1 summary (section-level) |
| `"What is the paper about?"` | Level 2 summary (document-level) |

## How it differs from other methods

| Method | What is indexed | Handles high-level questions? |
|---|---|---|
| Hybrid / Vector | Raw chunks only | No |
| Parent Document | Raw chunks + parents | Partially |
| RAPTOR | Raw chunks + summaries at multiple levels | Yes |

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `MAX_LEVELS` | 3 | Maximum depth of the summary tree |
| `MIN_CLUSTER_SIZE` | 5 | Stops recursion when fewer texts remain |
| `CHUNK_SIZE` | 500 | Leaf chunk size |

## Requirements
- `OPENAI_API_KEY` in `.env` file (used for LLM summarization at index time)

## How to run

```bash
pip install -r raptor/requirements.txt
python raptor/retriever.py
```

Note: building the tree takes longer than other approaches since the LLM summarizes clusters at each level. This is a one-time cost at index time.
