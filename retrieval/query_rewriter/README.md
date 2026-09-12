# Query Rewriter

## What it is
Query Rewriter uses an LLM to rewrite the user's query into a clearer, more specific, retrieval-friendly version before running vector search. One rewrite, one retrieval.

## The problem it solves
User queries are often short, vague, or ambiguous. Poor phrasing leads to poor embedding alignment and weak retrieval results.

Examples of bad queries:
- `"attention?"` — too short
- `"how multi head works"` — grammatically poor
- `"position stuff"` — too vague

## How it works

```
Original query (vague/short)
       ↓
      LLM rewrites it
       ↓
Rewritten query (clear, specific, retrieval-friendly)
       ↓
Vector search → results
```

## Example rewrites

| Original | Rewritten |
|---|---|
| `"attention?"` | `"What is the attention mechanism in transformer models?"` |
| `"position stuff"` | `"How does positional encoding work in the Transformer architecture?"` |
| `"how multi head works"` | `"How does multi-head attention work in the Transformer model?"` |

## Related techniques

| Technique | LLM output | Queries run |
|---|---|---|
| **Query Rewriter** | One better-phrased query | 1 |
| **Multi-Query** | N different rephrasings | N |
| **Query Expansion** | Original + added synonyms | 1 |
| **HyDE** | Fake answer (not a query) | 1 |
| **Step-Back** | Broader, more abstract query | 1 |

## Requirements
- `OPENAI_API_KEY` in `.env` file

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Model used to rewrite the query |

## How to run

```bash
pip install -r query_rewriter/requirements.txt
python query_rewriter/retriever.py
```
