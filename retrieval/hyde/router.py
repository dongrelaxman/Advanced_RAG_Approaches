from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["HyDE"])

_cache: dict = {}


@router.post("/hyde", response_model=SearchResponse)
def hyde(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vs" not in _cache:
        from retrieval.hyde.retriever import load_chunks, build_pipeline
        _cache["vs"], _cache["chain"] = build_pipeline(load_chunks(), client)
    from retrieval.hyde.retriever import search
    hypothesis, raw = search(body.query, _cache["vs"], _cache["chain"])
    return SearchResponse(
        method="hyde",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"rank={r}") for i, (t, r) in enumerate(raw)],
        metadata={"hypothesis": hypothesis},
    )
