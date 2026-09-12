# Self-Querying Retrieval

## What it is
Self-Querying Retrieval uses an LLM to parse a natural language query into two parts: a semantic search string and structured metadata filters. Both are applied to the vector store together.

## The problem it solves
Standard vector search ignores structural intent in the query. A question like *"What does the paper say about attention on page 3?"* treats "page 3" as part of the semantic meaning — which produces poor results.

## How it works

```
User query: "What is discussed about attention on page 3?"
       ↓
      LLM parses it into:
       ├── Semantic query : "attention mechanism"
       └── Metadata filter: page == 3
       ↓
  Vector store applies both simultaneously:
       ├── similarity search on "attention mechanism"
       └── hard filter: only return chunks where page == 3
       ↓
  Precise, filtered results
```

## The three query types

| Query | What the LLM generates |
|---|---|
| `"What is attention?"` | Semantic only, no filter |
| `"What is on page 1?"` | page == 0 filter, minimal semantic |
| `"Positional encoding from first 3 pages"` | Semantic + page <= 2 filter |

## Metadata fields available

| Field | Type | Description |
|---|---|---|
| `page` | integer | Page number in the PDF (0-indexed) |
| `source` | string | Source PDF file path |

## How it differs from standard retrieval
Standard retrieval only does semantic search. Self-querying adds SQL-like filtering on metadata fields on top of semantic search — combining flexibility with precision.

## Requirements
- `OPENAI_API_KEY` in `.env` file (used to parse the query, not for retrieval itself)

## How to run

```bash
pip install -r self_query/requirements.txt
python self_query/retriever.py
```

Set `verbose=True` on the retriever (already enabled) to see the structured query the LLM generates for each question.
