from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Context Compression"])

_cache: dict = {}


@router.post("/context-compression", response_model=SearchResponse)
def context_compression(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "retriever" not in _cache:
        from retrieval.context_compression.retriever import load_chunks, build_pipeline
        _cache["retriever"] = build_pipeline(load_chunks(), client)
    from retrieval.context_compression.retriever import search
    raw = search(body.query, _cache["retriever"])
    results = (
        [Result(rank=i + 1, text=t, label="compressed") for i, (t, _) in enumerate(raw)]
        if raw else
        [Result(rank=1, text="No passages passed the similarity threshold.", label="—")]
    )
    return SearchResponse(method="context_compression", query=body.query, results=results)
