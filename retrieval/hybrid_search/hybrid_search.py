"""
Hybrid search over the Attention Is All You Need paper.
Combines BM25 (sparse/keyword) + Weaviate vector search (dense/semantic)
using Reciprocal Rank Fusion (RRF) to merge ranked results.

RRF score = sum(1 / (k + rank_i)) for each retriever i
This is retriever-agnostic and does not require score normalization.

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install rank-bm25 langchain-community langchain-huggingface
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from reranking.rerank_fusion.rerank_fusion import reciprocal_rank_fusion

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperHybrid"


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


class HybridSearcher:
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
        print("Index ready.\n")

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.texts[i], rank + 1) for rank, (i, _) in enumerate(ranked)]

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [(doc.page_content, rank + 1) for rank, doc in enumerate(results)]

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        fetch_k = top_k * 3

        bm25_results = self._bm25_search(query, fetch_k)
        vector_results = self._vector_search(query, fetch_k)

        fused = reciprocal_rank_fusion([bm25_results, vector_results])
        return fused[:top_k]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        searcher = HybridSearcher(chunks, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = searcher.search(query, top_k=5)
            for rank, (text, score) in enumerate(results, 1):
                print(f"  Rank {rank} (RRF={score:.5f}): {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
