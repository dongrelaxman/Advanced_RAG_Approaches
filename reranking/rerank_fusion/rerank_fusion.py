"""
Reciprocal Rank Fusion (RRF) — Cormack et al., 2009.

RRF score = sum(1 / (k + rank_i)) across all retrievers.
Retriever-agnostic: works with any number of ranked lists without score normalization.

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install rank-bm25 langchain-community langchain-huggingface
         pip install langchain-weaviate pypdf langchain-text-splitters
         pip install sentence-transformers weaviate-client
"""

import re
import weaviate
from collections import defaultdict
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperRRF"
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, int]]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists into one using RRF.

    Args:
        ranked_lists: each list contains (text, rank) tuples from one retriever.
        k: smoothing constant (default 60 per the original paper).

    Returns:
        List of (text, rrf_score) sorted by descending score.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for results in ranked_lists:
        for text, rank in results:
            rrf_scores[text] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


class RRFSearcher:
    def __init__(self, chunks, client: weaviate.WeaviateClient):
        self.texts = [chunk.page_content for chunk in chunks]

        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

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
        searcher = RRFSearcher(chunks, client)

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
