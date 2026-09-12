# Auto-Merging (Hierarchical) Retrieval

## What it is
Auto-Merging Retrieval builds a parent-child chunk hierarchy at index time. At query time, if enough child chunks from the same parent are retrieved, it automatically replaces them with the full parent chunk instead.

## The problem it solves
When multiple small chunks from the same parent section are retrieved, you end up sending fragmented, overlapping context to the LLM. Auto-merging detects this and returns the coherent parent section instead.

## How it works

```
Index time:
  PDF → split into parent chunks (1000 tokens)
          → split each parent into child chunks (256 tokens)
          → embed child chunks → store in Weaviate with parent_id metadata
          → store parent chunks in a dict keyed by parent_id

Query time:
  Query → retrieve top-20 child chunks from Weaviate
        → group matched children by parent_id
        → for each parent:
            if matched_children / total_children >= threshold → return parent
            else                                              → return child chunks
        → return top-k results
```

## The merge decision

```
Parent A has 4 child chunks total.
Query matched child 1, child 2, child 3 → 3/4 = 75% ≥ threshold (50%) → MERGE → return parent A

Parent B has 6 child chunks total.
Query matched child 1 only → 1/6 = 17% < threshold (50%) → return child 1 only
```

## How it differs from Parent Document Retrieval

| | Parent Document | Auto-Merging |
|---|---|---|
| **Always returns** | Parent (always) | Child OR parent (depends on matches) |
| **Decision** | Fixed — always return parent | Dynamic — based on how many children matched |
| **Specific query** | Returns parent (may have noise) | Returns just the child chunk |
| **Broad query** | Returns parent | Returns parent (merged) |

Auto-merging is smarter — it adapts based on the query. Specific questions get child chunks. Broad questions that touch many children get the full parent.

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `PARENT_CHUNK_SIZE` | 1000 | Size of parent chunks returned to LLM |
| `CHILD_CHUNK_SIZE` | 256 | Size of child chunks used for retrieval |
| `MERGE_THRESHOLD` | 0.5 | Fraction of children that must match to trigger merge |

### Tuning the merge threshold

| Value | Behaviour |
|---|---|
| `0.3` | Merge aggressively (30% children enough) |
| `0.5` | Balanced — default |
| `0.8` | Merge rarely, prefer precise child chunks |

## How to run

```bash
pip install -r auto_merging/requirements.txt
python auto_merging/retriever.py
```

## No API key needed — runs fully local.
