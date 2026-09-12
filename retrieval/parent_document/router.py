from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Parent Document"])

_cache: dict = {}


@router.post("/parent-document", response_model=SearchResponse)
def parent_document(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "retriever" not in _cache:
        from retrieval.parent_document.retriever import load_documents, build_pipeline
        _cache["retriever"] = build_pipeline(load_documents(), client)
    from retrieval.parent_document.retriever import search
    raw = search(body.query, _cache["retriever"])
    return SearchResponse(
        method="parent_document",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"{len(t)} chars") for i, (t, _) in enumerate(raw)],
    )
