from fastapi import APIRouter

from retrieval.hybrid_search.router    import router as hybrid_router
from retrieval.context_compression.router import router as context_compression_router
from retrieval.self_query.router       import router as self_query_router
from retrieval.parent_document.router  import router as parent_document_router
from retrieval.sentence_window.router  import router as sentence_window_router
from retrieval.raptor.router           import router as raptor_router
from retrieval.auto_merging.router     import router as auto_merging_router
from retrieval.hyde.router             import router as hyde_router
from retrieval.query_rewriter.router   import router as query_rewriter_router

router = APIRouter()

router.include_router(hybrid_router)
router.include_router(context_compression_router)
router.include_router(self_query_router)
router.include_router(parent_document_router)
router.include_router(sentence_window_router)
router.include_router(raptor_router)
router.include_router(auto_merging_router)
router.include_router(hyde_router)
router.include_router(query_rewriter_router)
