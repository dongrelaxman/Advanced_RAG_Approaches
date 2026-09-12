"""
Contextual Compression Retrieval over the Attention Is All You Need paper.

Problem: retrieved chunks often contain irrelevant sentences mixed with relevant ones,
adding noise to the LLM context.

Solution: after retrieval, a compressor filters each chunk down to only the
sentences semantically relevant to the query.

Pipeline:
  Query → Weaviate vector retriever → top-10 full chunks
        → EmbeddingsFilter compressor → trimmed, relevant passages

EmbeddingsFilter keeps only passages whose embedding similarity to the query
exceeds a threshold — no LLM or API key required.

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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperCompression"
SIMILARITY_THRESHOLD = 0.75


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_pipeline(chunks, client: weaviate.WeaviateClient):
    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=INDEX_NAME,
    )

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    compressor = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=SIMILARITY_THRESHOLD
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    print("Pipeline ready.\n")
    return compression_retriever


def search(query: str, retriever: ContextualCompressionRetriever) -> list[tuple[str, int]]:
    results = retriever.invoke(query)
    return [(doc.page_content, i + 1) for i, doc in enumerate(results)]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        retriever = build_pipeline(chunks, client)

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
            if not results:
                print("  No passages passed the similarity threshold.")
            for rank, (text, _) in enumerate(results, 1):
                print(f"  Rank {rank}: {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
