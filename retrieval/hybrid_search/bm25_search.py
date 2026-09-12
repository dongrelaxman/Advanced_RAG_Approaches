"""
BM25 keyword search over the Attention Is All You Need paper.
BM25 (Okapi BM25) is the industry-standard sparse retrieval algorithm —
it improves on TF-IDF with term frequency saturation and document length normalization.

Install: pip install rank-bm25 langchain-community pypdf langchain-text-splitters
"""

import re
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    texts = [chunk.page_content for chunk in chunks]
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts


def search(query: str, bm25: BM25Okapi, texts: list[str], top_k: int = 5):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(texts[i], score) for i, score in ranked if score > 0]


def main():
    print("Loading and indexing PDF with BM25...")
    chunks = load_chunks()
    bm25, texts = build_index(chunks)
    print(f"Indexed {len(texts)} chunks.\n")

    queries = [
        "What is the attention mechanism?",
        "multi-head attention",
        "positional encoding",
        "encoder decoder architecture",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = search(query, bm25, texts)
        if not results:
            print("  No results found.")
        for rank, (text, score) in enumerate(results, 1):
            print(f"  Rank {rank} (score={score:.4f}): {text[:120].strip()}...")
        print()


if __name__ == "__main__":
    main()
