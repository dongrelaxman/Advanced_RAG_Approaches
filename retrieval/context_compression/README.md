# Context Compression Retrieval

## What it is
Context Compression retrieves full chunks as usual, then compresses each chunk down to only the sentences that are relevant to the query — removing noise before sending context to the LLM.

## The problem it solves
Retrieved chunks often contain irrelevant sentences mixed in with the relevant ones. Sending the full chunk to the LLM adds noise, wastes tokens, and can confuse the answer.

## How it works

```
Query
  ↓
Base retriever (Weaviate vector search) → top-10 full chunks
  ↓
EmbeddingsFilter compressor
  → drops passages whose similarity to query < threshold
  → keeps only relevant sentences
  ↓
Compressed, relevant passages → LLM
```

## Two compressor strategies

| Compressor | How it works | Needs LLM? |
|---|---|---|
| **EmbeddingsFilter** | Keeps sentences semantically similar to the query above a threshold | No |
| **LLMChainExtractor** | Uses an LLM to extract relevant sentences | Yes |

This implementation uses `EmbeddingsFilter` — no API key required.

## Benefits
- Less noise in LLM context → better answers
- Fewer tokens sent → cheaper and faster
- Works on top of any base retriever

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.75 | Passages below this similarity score are dropped |
| `k` in base retriever | 10 | How many chunks are fetched before compression |

### Tuning the threshold
- Lower (e.g. `0.6`) → more passages pass through, more context, more noise
- Higher (e.g. `0.85`) → stricter filtering, less context, less noise

## How to run

```bash
pip install -r context_compression/requirements.txt
python context_compression/retriever.py
```

## No API key needed — runs fully local.
