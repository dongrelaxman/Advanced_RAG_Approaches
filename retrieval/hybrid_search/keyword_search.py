"""
TF-IDF keyword search over the Attention Is All You Need paper.
Uses LangChain for PDF loading/splitting and sklearn for TF-IDF ranking.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    texts = [chunk.page_content for chunk in chunks]
    preprocessed = [preprocess(t) for t in texts]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(preprocessed)
    return vectorizer, matrix, texts


def search(query: str, vectorizer, matrix, texts, top_k: int = 5):
    preprocessed_query = preprocess(query)
    query_vec = vectorizer.transform([preprocessed_query])
    scores = cosine_similarity(matrix, query_vec).flatten()
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(texts[i], scores[i]) for i in ranked if scores[i] > 0]


def main():
    print("Loading and indexing PDF...")
    chunks = load_chunks()
    vectorizer, matrix, texts = build_index(chunks)
    print(f"Indexed {len(texts)} chunks.\n")

    queries = [
        "What is the attention mechanism?",
        "multi-head attention",
        "positional encoding",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = search(query, vectorizer, matrix, texts)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  Rank {rank} (score={score:.4f}): {text[:120].strip()}...")
        print()


if __name__ == "__main__":
    main()
