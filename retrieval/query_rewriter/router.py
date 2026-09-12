from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Query Rewriter"])

_cache: dict = {}


@router.post("/query-rewriter", response_model=SearchResponse)
def query_rewriter(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vs" not in _cache:
        from retrieval.query_rewriter.retriever import load_chunks, build_pipeline
        _cache["vs"], _cache["rewriter"] = build_pipeline(load_chunks(), client)
    from retrieval.query_rewriter.retriever import search
    rewritten, raw = search(body.query, _cache["vs"], _cache["rewriter"])
    return SearchResponse(
        method="query_rewriter",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"rank={r}") for i, (t, r) in enumerate(raw)],
        metadata={"rewritten_query": rewritten},
    )
