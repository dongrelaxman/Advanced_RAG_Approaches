"""
Flash Rerank pipeline over the Attention Is All You Need paper.
Stage 1: Hybrid retrieval (BM25 + Weaviate vector search with RRF fusion) to get top-N candidates.
Stage 2: FlashRank cross-encoder reranks candidates for precision.

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install flashrank rank-bm25 langchain-community langchain-huggingface
         pip install langchain-weaviate pypdf langchain-text-splitters
         pip install sentence-transformers weaviate-client
"""

import re
import sys
import os
import weaviate
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from flashrank import Ranker, RerankRequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rerank_fusion.rerank_fusion import reciprocal_rank_fusion

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperFlashRerank"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


class FlashRerankPipeline:
    def __init__(self, chunks, client: weaviate.WeaviateClient):
        self.texts = [chunk.page_content for chunk in chunks]

        # BM25 index
        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Weaviate vector store
        print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=INDEX_NAME,
        )

        # FlashRank cross-encoder
        print(f"Loading FlashRank reranker ({RERANK_MODEL})...")
        self.ranker = Ranker(model_name=RERANK_MODEL)
        print("Pipeline ready.\n")

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.texts[i], rank + 1) for rank, (i, _) in enumerate(ranked)]

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [(doc.page_content, rank + 1) for rank, doc in enumerate(results)]

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20) -> list[tuple[str, float]]:
        # Stage 1: hybrid retrieval
        bm25_results = self._bm25_search(query, fetch_k)
        vector_results = self._vector_search(query, fetch_k)
        candidates = [text for text, _ in reciprocal_rank_fusion([bm25_results, vector_results])[:fetch_k]]

        # Stage 2: FlashRank reranking
        passages = [{"id": i, "text": text} for i, text in enumerate(candidates)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(rerank_request)

        return [(r["text"], r["score"]) for r in reranked[:top_k]]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        pipeline = FlashRerankPipeline(chunks, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = pipeline.search(query, top_k=5)
            for rank, (text, score) in enumerate(results, 1):
                print(f"  Rank {rank} (score={score:.4f}): {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
