from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Hybrid Search"])

_cache: dict = {}


@router.post("/hybrid-search", response_model=SearchResponse)
def hybrid_search(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "hybrid" not in _cache:
        from retrieval.hybrid_search.hybrid_search import load_chunks, HybridSearcher
        _cache["hybrid"] = HybridSearcher(load_chunks(), client)
    raw = _cache["hybrid"].search(body.query)
    return SearchResponse(
        method="hybrid_search",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"RRF={s:.5f}") for i, (t, s) in enumerate(raw)],
    )


@router.post("/vector-search", response_model=SearchResponse)
def vector_search(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vector_vs" not in _cache:
        from retrieval.hybrid_search.vector_search import load_chunks, build_vectorstore
        _cache["vector_vs"] = build_vectorstore(load_chunks(), client)
    from retrieval.hybrid_search.vector_search import search
    raw = search(body.query, _cache["vector_vs"])
    return SearchResponse(
        method="vector_search",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"score={s:.4f}") for i, (t, s) in enumerate(raw)],
    )


@router.post("/bm25-search", response_model=SearchResponse)
def bm25_search(body: QueryRequest):
    if "bm25" not in _cache:
        from retrieval.hybrid_search.bm25_search import load_chunks, build_index
        _cache["bm25"], _cache["bm25_texts"] = build_index(load_chunks())
    from retrieval.hybrid_search.bm25_search import search
    raw = search(body.query, _cache["bm25"], _cache["bm25_texts"])
    return SearchResponse(
        method="bm25_search",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"score={s:.4f}") for i, (t, s) in enumerate(raw)],
    )


@router.post("/keyword-search", response_model=SearchResponse)
def keyword_search(body: QueryRequest):
    if "kw" not in _cache:
        from retrieval.hybrid_search.keyword_search import load_chunks, build_index
        _cache["kw_vec"], _cache["kw_mat"], _cache["kw_texts"] = build_index(load_chunks())
    from retrieval.hybrid_search.keyword_search import search
    raw = search(body.query, _cache["kw_vec"], _cache["kw_mat"], _cache["kw_texts"])
    return SearchResponse(
        method="keyword_search",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"score={s:.4f}") for i, (t, s) in enumerate(raw)],
    )
