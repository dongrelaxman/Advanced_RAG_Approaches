# HyDE — Hypothetical Document Embedding

## What it is
HyDE uses an LLM to generate a hypothetical (fake) answer to the query, then embeds that answer instead of the query itself. The fake answer lives closer to real document passages in embedding space, improving retrieval alignment.

## The problem it solves
A short query embeds very differently from a full document passage. They exist in different regions of the embedding space, so even relevant chunks may not rank highly.

```
Query embedding    ──────────────────────────●  (short, question-like)
Real chunk         ──────────────────●          (long, passage-like)
                                     ↑
                              large gap = poor match
```

## How it works

```
Query: "attention mechanism"
  ↓
LLM generates a hypothetical answer:
  "The attention mechanism computes a weighted sum of value vectors,
   where weights are determined by the compatibility of query and key
   vectors using scaled dot products..."
  ↓
Embed the hypothetical answer (not the original query)
  ↓
Vector search using hypothetical answer embedding
  ↓
Real matching chunks returned
```

## Why hallucination is acceptable
The hypothetical answer does not need to be factually correct. Even a wrong answer captures the **vocabulary, style, and structure** of real document passages — which is all that's needed for better embedding alignment.

```
Query embedding        ──────────────────────────●
Real chunk             ──────────────────●
Hypothetical doc       ─────────────────●           ← much closer to real chunk
```

## How it differs from Query Rewriter

| | Query Rewriter | HyDE |
|---|---|---|
| **LLM generates** | A better query | A fake answer/passage |
| **What is embedded** | Rewritten query | Fake answer |
| **Approach** | Improve the question | Simulate the answer |
| **Best for** | Vague/short queries | Short queries with embedding gap |

## Requirements
- `OPENAI_API_KEY` in `.env` file

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Model used to generate the hypothesis |
| `CHUNK_SIZE` | 500 | Size of indexed document chunks |

## How to run

```bash
pip install -r hyde/requirements.txt
python hyde/retriever.py
```

The output shows both the generated hypothesis and the retrieved chunks so you can see exactly what was used as the search vector.
