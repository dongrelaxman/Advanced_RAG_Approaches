# Advanced RAG Approaches

A hands-on reference project implementing 14 advanced Retrieval-Augmented Generation (RAG) retrieval and reranking methods, served as a REST API with Swagger UI.

All approaches run against the **"Attention Is All You Need"** paper (Vaswani et al., 2017) as the source document.

---

## Project Structure

```
Advanced_RAG_Approaches/
├── main.py                      # FastAPI app entry point
├── models.py                    # Shared Pydantic request/response models
├── requirements.txt             # All dependencies
├── data/
│   └── attention_is_all_you_need.pdf
├── retrieval/                   # Retrieval methods
│   ├── router.py                # Aggregates all retrieval module routers
│   ├── hybrid_search/           # BM25 + Vector + RRF
│   ├── context_compression/     # Retrieval + EmbeddingsFilter
│   ├── self_query/              # LLM-parsed metadata filtering
│   ├── parent_document/         # Child retrieval → parent context
│   ├── sentence_window/         # Sentence-level retrieval with window
│   ├── raptor/                  # Recursive summary tree retrieval
│   ├── auto_merging/            # Dynamic child→parent merging
│   ├── hyde/                    # Hypothetical document embedding
│   └── query_rewriter/          # LLM query rewriting before retrieval
└── reranking/                   # Reranking methods
    ├── router.py                # Aggregates all reranking module routers
    ├── rerank_fusion/           # Reciprocal Rank Fusion (RRF)
    └── flash_rerank/            # FlashRank cross-encoder reranking
```

Each module contains:
- `retriever.py` / main implementation file
- `router.py` — FastAPI router with endpoint + lazy-initialized pipeline cache
- `README.md` — explanation of the approach

---

## Retrieval Methods

| Method | Description | Needs API Key |
|---|---|---|
| **Hybrid Search** | BM25 + vector search fused with RRF | No |
| **Vector Search** | Semantic similarity via HuggingFace embeddings | No |
| **BM25 Search** | Sparse keyword retrieval (Okapi BM25) | No |
| **Keyword Search** | TF-IDF cosine similarity | No |
| **Context Compression** | Retrieves chunks then filters to relevant sentences only | No |
| **Self Query** | LLM splits query into semantic search + metadata filter | Yes |
| **Parent Document** | Indexes small child chunks, returns large parent chunks | No |
| **Sentence Window** | Indexes individual sentences, returns surrounding window | No |
| **RAPTOR** | Builds a recursive summary tree; searches all levels | Yes |
| **Auto Merging** | Returns parent chunk when enough of its children are matched | No |
| **HyDE** | Generates a hypothetical answer and embeds that instead of the query | Yes |
| **Query Rewriter** | LLM rewrites vague queries before retrieval | Yes |

## Reranking Methods

| Method | Description |
|---|---|
| **Rerank Fusion** | Reciprocal Rank Fusion over BM25 + vector ranked lists |
| **Flash Rerank** | Hybrid retrieval → FlashRank cross-encoder reranker |

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| Vector database | Weaviate (Docker) |
| Embeddings | `BAAI/bge-base-en-v1.5` via HuggingFace |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranking | FlashRank (`ms-marco-MiniLM-L-12-v2`) |
| LLM (API key methods) | OpenAI `gpt-4o-mini` |
| PDF loading | LangChain + PyPDF |

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```
> Only needed for: `self_query`, `query_rewriter`, `hyde`, `raptor`

### 3. Start Weaviate
```bash
docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
```

### 4. Start the API server
```bash
uvicorn main:app --reload
```

### 5. Open Swagger UI
```
http://localhost:8000/docs
```

---

## API Endpoints

All endpoints accept a POST request with:
```json
{ "query": "What is the attention mechanism?" }
```

And return:
```json
{
  "method": "hybrid_search",
  "query": "What is the attention mechanism?",
  "results": [
    { "rank": 1, "text": "...", "label": "RRF=0.03200" }
  ],
  "metadata": {}
}
```

| Endpoint | Method |
|---|---|
| `POST /retrieval/hybrid-search` | Hybrid Search |
| `POST /retrieval/vector-search` | Vector Search |
| `POST /retrieval/bm25-search` | BM25 Search |
| `POST /retrieval/keyword-search` | Keyword Search |
| `POST /retrieval/context-compression` | Context Compression |
| `POST /retrieval/self-query` | Self Query |
| `POST /retrieval/parent-document` | Parent Document |
| `POST /retrieval/sentence-window` | Sentence Window |
| `POST /retrieval/raptor` | RAPTOR |
| `POST /retrieval/auto-merging` | Auto Merging |
| `POST /retrieval/hyde` | HyDE |
| `POST /retrieval/query-rewriter` | Query Rewriter |
| `POST /reranking/rerank-fusion` | Rerank Fusion |
| `POST /reranking/flash-rerank` | Flash Rerank |

> **Note:** The first request to each endpoint is slow as it initializes the pipeline (loads PDF, embeds chunks). Subsequent requests are fast due to module-level caching.

---

## Router Architecture

Each module owns its routing logic — `main.py` stays thin regardless of how many methods are added.

```
main.py
  └── retrieval/router.py        (group router)
        ├── hybrid_search/router.py
        ├── context_compression/router.py
        └── ...
  └── reranking/router.py        (group router)
        ├── rerank_fusion/router.py
        └── flash_rerank/router.py
```

**Adding a new method:**
1. Create `retrieval/new_method/router.py` with a FastAPI `APIRouter`
2. Add one `include_router` line in `retrieval/router.py`
