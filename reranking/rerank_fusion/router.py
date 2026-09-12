from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Rerank Fusion"])

_cache: dict = {}


@router.post("/rerank-fusion", response_model=SearchResponse)
def rerank_fusion(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "searcher" not in _cache:
        from reranking.rerank_fusion.rerank_fusion import load_chunks, RRFSearcher
        _cache["searcher"] = RRFSearcher(load_chunks(), client)
    raw = _cache["searcher"].search(body.query)
    return SearchResponse(
        method="rerank_fusion",
        query=body.query,
        results=[Result(rank=i + 1, text=t, label=f"RRF={s:.5f}") for i, (t, s) in enumerate(raw)],
    )
