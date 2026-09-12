from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["RAPTOR"])

_cache: dict = {}


@router.post("/raptor", response_model=SearchResponse)
def raptor(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vs" not in _cache:
        from retrieval.raptor.retriever import load_chunks, build_pipeline
        _cache["vs"] = build_pipeline(load_chunks(), client)
    from retrieval.raptor.retriever import search
    raw = search(body.query, _cache["vs"])
    return SearchResponse(
        method="raptor",
        query=body.query,
        results=[
            Result(rank=i + 1, text=t, label=f"level={meta.get('level', 0)} ({meta.get('type', 'chunk')})")
            for i, (t, meta) in enumerate(raw)
        ],
    )
