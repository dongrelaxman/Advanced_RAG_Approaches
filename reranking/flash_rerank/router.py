from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Flash Rerank"])

_cache: dict = {}


@router.post("/flash-rerank", response_model=SearchResponse)
def flash_rerank(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "pipeline" not in _cache:
        from reranking.flash_rerank.reranker import load_chunks, FlashRerankPipeline
        _cache["pipeline"] = FlashRerankPipeline(load_chunks(), client)
    raw = _cache["pipeline"].search(body.query)
    return SearchResponse(
        method="flash_rerank",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"score={s:.4f}") for i, (t, s) in enumerate(raw)],
    )
