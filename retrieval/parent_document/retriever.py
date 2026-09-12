"""
Parent Document Retrieval over the Attention Is All You Need paper.

Problem: small chunks give precise retrieval but lose surrounding context;
large chunks give rich context but imprecise retrieval.

Solution: index small child chunks for retrieval, but return their large
parent chunks to the LLM — precise matching with rich context.

Two-level chunking:
  Parent chunks (1500 tokens) — stored in InMemoryStore, returned to LLM
  Child chunks  (300 tokens)  — embedded in Weaviate, used for retrieval

At query time:
  Query → match child chunks → look up their parent → return parent to LLM

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain langchain-community langchain-huggingface
         pip install langchain-weaviate pypdf langchain-text-splitters
         pip install sentence-transformers weaviate-client
"""

import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

PDF_PATH = "data/attention_is_all_you_need.pdf"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperParentDoc"

PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30


def load_documents(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def build_pipeline(docs, client: weaviate.WeaviateClient):
    print(f"Embedding with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=INDEX_NAME,
        text_key="text",
        embedding=embeddings,
    )

    docstore = InMemoryStore()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Indexing documents (embedding child chunks, storing parent chunks)...")
    retriever.add_documents(docs)
    print(f"Indexed {len(list(docstore.yield_keys()))} parent chunks.\n")

    return retriever


def search(query: str, retriever: ParentDocumentRetriever, top_k: int = 3) -> list[tuple[str, int]]:
    results = retriever.invoke(query)
    return [(doc.page_content, i + 1) for i, doc in enumerate(results[:top_k])]


def main():
    print("Loading PDF...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages.\n")

    client = weaviate.connect_to_local()
    try:
        retriever = build_pipeline(docs, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, retriever)
            for rank, (text, _) in enumerate(results, 1):
                print(f"  Rank {rank} ({len(text)} chars): {text[:200].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
