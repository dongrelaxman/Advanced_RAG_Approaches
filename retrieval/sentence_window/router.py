from fastapi import APIRouter, Request
from models import QueryRequest, SearchResponse, Result

router = APIRouter(tags=["Sentence Window"])

_cache: dict = {}


@router.post("/sentence-window", response_model=SearchResponse)
def sentence_window(body: QueryRequest, request: Request):
    client = request.app.state.weaviate_client
    if "vs" not in _cache:
        from retrieval.sentence_window.retriever import load_sentences, build_index
        _cache["sentences"] = load_sentences()
        _cache["vs"] = build_index(_cache["sentences"], client)
    from retrieval.sentence_window.retriever import search
    raw = search(body.query, _cache["vs"], _cache["sentences"])
    return SearchResponse(
        method="sentence_window",
        query=body.query,
        results=[Result(rank=i + 1, text=window, label=f"sentence_id={sid}") for i, (_, window, sid) in enumerate(raw)],
    )
