"""
Semantic vector search over the Attention Is All You Need paper.
Uses LangChain + HuggingFace sentence-transformers for embeddings and Weaviate as the vector store.

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain-community langchain-huggingface langchain-weaviate
         pip install pypdf langchain-text-splitters sentence-transformers weaviate-client
"""

import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaper"


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=INDEX_NAME,
    )
    return vectorstore


def load_vectorstore(client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return WeaviateVectorStore(
        client=client,
        index_name=INDEX_NAME,
        text_key="text",
        embedding=embeddings,
    )


def search(query: str, vectorstore: WeaviateVectorStore, top_k: int = 5):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, score) for doc, score in results]


def main():
    print("Loading PDF and building vector store...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding with {EMBEDDING_MODEL}...")

    client = weaviate.connect_to_local()
    try:
        vectorstore = build_vectorstore(chunks, client)
        print("Vector store ready.\n")

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, vectorstore)
            for rank, (text, score) in enumerate(results, 1):
                print(f"  Rank {rank} (score={score:.4f}): {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
