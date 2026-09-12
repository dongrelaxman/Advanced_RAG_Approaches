# Sentence Window Retrieval

## What it is
Sentence Window Retrieval indexes documents at the individual sentence level for maximum retrieval precision. When a sentence matches, it returns that sentence plus a configurable window of surrounding sentences for context.

## The problem it solves
The same tension as Parent Document Retrieval — small units are precise, large units have context. Sentence Window solves it at a finer granularity: the sentence level.

## How it works

```
Index time:
  PDF → split into individual sentences
      → embed each sentence
      → store in Weaviate with sentence_id metadata

Query time:
  Query → match most similar sentence (sentence_id = 42)
        → fetch window: sentences [40, 41, 42, 43, 44]  (window_size=2)
        → return joined window as context
```

## Example

```
All sentences in document:
  [40] The Transformer follows an encoder-decoder structure.
  [41] The encoder maps input to continuous representations.
  [42] Each layer has multi-head attention and a feed-forward network.  ← matched
  [43] Residual connections are applied around each sub-layer.
  [44] Layer normalization follows each sub-layer.

Window size = 2 → returns sentences 40 through 44
```

## How it differs from Parent Document Retrieval

| | Parent Document | Sentence Window |
|---|---|---|
| Index unit | Child chunks (300 tokens) | Individual sentences |
| Return unit | Pre-defined parent chunk | Dynamic window around match |
| Context size | Fixed (parent chunk size) | Flexible (window size × sentence length) |
| Granularity | Chunk-level | Sentence-level (finer) |

Sentence Window is more granular — it retrieves at the exact sentence that matched, then expands outward. Parent Document uses fixed boundaries set at index time.

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `WINDOW_SIZE` | 2 | Sentences to include before and after the matched sentence |

### Tuning window size

| Value | Context returned |
|---|---|
| `1` | 1 before + matched + 1 after (3 sentences) |
| `2` | 2 before + matched + 2 after (5 sentences) |
| `3` | 3 before + matched + 3 after (7 sentences) |

## How to run

```bash
pip install -r sentence_window/requirements.txt
python sentence_window/retriever.py
```

## No API key needed — runs fully local.
