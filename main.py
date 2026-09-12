import os
import sys
from contextlib import asynccontextmanager

import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval.router import router as retrieval_router
from reranking.router import router as reranking_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.weaviate_client = weaviate.connect_to_local()
    yield
    app.state.weaviate_client.close()


app = FastAPI(
    title="Advanced RAG Approaches",
    description="A collection of RAG retrieval and reranking methods over the Attention Is All You Need paper.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(retrieval_router, prefix="/retrieval", tags=["Retrieval"])
app.include_router(reranking_router, prefix="/reranking", tags=["Reranking"])
