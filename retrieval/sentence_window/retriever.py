"""
Sentence Window Retrieval over the Attention Is All You Need paper.

Problem: small chunks are precise but lose context; large chunks have context
but are imprecise. Parent Document uses fixed chunk boundaries set at index time.

Solution: index at the individual sentence level for maximum retrieval precision.
When a sentence matches, expand outward by a configurable window of surrounding
sentences to give the LLM enough context.

Pipeline:
  Index time : split into sentences → embed each → store with sentence_id metadata
  Query time : match sentence → fetch sentence_id → return window of N sentences
               around the match

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain-community langchain-huggingface langchain-weaviate
         pip install pypdf langchain-text-splitters sentence-transformers weaviate-client
"""

import re
import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore

PDF_PATH = "data/attention_is_all_you_need.pdf"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperSentenceWindow"
WINDOW_SIZE = 2


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def load_sentences(pdf_path: str = PDF_PATH) -> list[str]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = " ".join(page.page_content for page in pages)
    return split_into_sentences(full_text)


def build_index(sentences: list[str], client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    print(f"Embedding {len(sentences)} sentences with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs = [
        Document(page_content=sentence, metadata={"sentence_id": i})
        for i, sentence in enumerate(sentences)
    ]

    vectorstore = WeaviateVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        index_name=INDEX_NAME,
    )
    print("Index ready.\n")
    return vectorstore


def search(
    query: str,
    vectorstore: WeaviateVectorStore,
    sentences: list[str],
    top_k: int = 3,
    window_size: int = WINDOW_SIZE,
) -> list[tuple[str, str, int]]:
    """
    Returns list of (matched_sentence, windowed_context, sentence_id).
    """
    results = vectorstore.similarity_search(query, k=top_k)
    output = []
    for doc in results:
        sid = doc.metadata["sentence_id"]
        start = max(0, sid - window_size)
        end = min(len(sentences) - 1, sid + window_size)
        window = " ".join(sentences[start : end + 1])
        output.append((doc.page_content, window, sid))
    return output


def main():
    print("Loading PDF and splitting into sentences...")
    sentences = load_sentences()
    print(f"Found {len(sentences)} sentences.\n")

    client = weaviate.connect_to_local()
    try:
        vectorstore = build_index(sentences, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, vectorstore, sentences)
            for rank, (matched, window, sid) in enumerate(results, 1):
                print(f"  Rank {rank} (sentence_id={sid}):")
                print(f"    Matched  : {matched[:100].strip()}...")
                print(f"    Window   : {window[:200].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
