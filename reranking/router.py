from fastapi import APIRouter

from reranking.rerank_fusion.router import router as rerank_fusion_router
from reranking.flash_rerank.router  import router as flash_rerank_router

router = APIRouter()

router.include_router(rerank_fusion_router)
router.include_router(flash_rerank_router)
