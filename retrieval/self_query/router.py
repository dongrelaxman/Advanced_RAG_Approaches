from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Self Query"])

_cache: dict = {}


@router.post("/self-query", response_model=SearchResponse)
def self_query(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "retriever" not in _cache:
        from retrieval.self_query.retriever import load_chunks, build_pipeline
        _cache["retriever"] = build_pipeline(load_chunks(), client)
    from retrieval.self_query.retriever import search
    raw = search(body.query, _cache["retriever"])
    return SearchResponse(
        method="self_query",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"page={meta.get('page', '?')}") for i, (t, _, meta) in enumerate(raw)],
    )
