from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Auto Merging"])

_cache: dict = {}


@router.post("/auto-merging", response_model=SearchResponse)
def auto_merging(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vs" not in _cache:
        from retrieval.auto_merging.retriever import load_documents, build_pipeline
        _cache["vs"], _cache["parent_store"], _cache["children_count"] = build_pipeline(load_documents(), client)
    from retrieval.auto_merging.retriever import search
    raw = search(body.query, _cache["vs"], _cache["parent_store"], _cache["children_count"])
    return SearchResponse(
        method="auto_merging",
        query=body.query,
        results=[
            Result(rank=i + 1, text=t, label="MERGED→parent" if src == "parent" else "child")
            for i, (t, src) in enumerate(raw)
        ],
    )
