"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.
(Sarthi et al., 2024 — https://arxiv.org/abs/2401.18059)

Problem: chunk-level retrieval misses the big picture. High-level questions
like "What is this paper's main contribution?" have no single chunk answer.

Solution: build a tree of summaries bottom-up using clustering + LLM summarization.
All nodes (raw chunks + summaries at every level) are indexed together, so
retrieval works at any level of abstraction simultaneously.

Requires: OPENAI_API_KEY in .env
          Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain langchain-community langchain-huggingface
         pip install langchain-weaviate langchain-openai pypdf
         pip install langchain-text-splitters sentence-transformers
         pip install weaviate-client umap-learn scikit-learn numpy python-dotenv
"""

import os
import numpy as np
import weaviate
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import umap

load_dotenv()

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperRaptor"
LLM_MODEL = "gpt-4o-mini"

MAX_LEVELS = 3
MIN_CLUSTER_SIZE = 5

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert summarizer. Given several text passages, write a concise "
        "summary that captures the key ideas. Be thorough but avoid repetition.",
    ),
    ("human", "{texts}"),
])


def load_chunks(pdf_path: str = PDF_PATH) -> list[str]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [chunk.page_content for chunk in splitter.split_documents(docs)]


def embed_texts(texts: list[str], embeddings_model: HuggingFaceEmbeddings) -> np.ndarray:
    return np.array(embeddings_model.embed_documents(texts))


def cluster_texts(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    n_neighbors = min(10, len(embeddings) - 1)
    n_components = min(10, len(embeddings) - 2)

    reduced = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(reduced)
    return labels


def summarize_cluster(texts: list[str], llm_chain) -> str:
    combined = "\n\n---\n\n".join(texts)
    return llm_chain.invoke({"texts": combined})


def build_raptor_tree(
    texts: list[str],
    embeddings_model: HuggingFaceEmbeddings,
    llm_chain,
    vectorstore: WeaviateVectorStore,
    level: int = 0,
):
    if len(texts) < MIN_CLUSTER_SIZE or level >= MAX_LEVELS:
        return

    n_clusters = max(2, len(texts) // 5)
    n_clusters = min(n_clusters, len(texts) - 1)

    print(f"  Level {level + 1}: clustering {len(texts)} texts into {n_clusters} clusters...")

    embeddings = embed_texts(texts, embeddings_model)
    labels = cluster_texts(embeddings, n_clusters)

    summaries = []
    for cluster_id in range(n_clusters):
        cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
        if not cluster_texts:
            continue
        summary = summarize_cluster(cluster_texts, llm_chain)
        summaries.append(summary)

        vectorstore.add_documents([
            Document(
                page_content=summary,
                metadata={"level": level + 1, "type": "summary"},
            )
        ])

    print(f"  Level {level + 1}: generated {len(summaries)} summaries.")
    build_raptor_tree(summaries, embeddings_model, llm_chain, vectorstore, level + 1)


def build_pipeline(chunks: list[str], client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    print(f"Embedding {len(chunks)} leaf chunks with {EMBEDDING_MODEL}...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    leaf_docs = [
        Document(page_content=text, metadata={"level": 0, "type": "chunk"})
        for text in chunks
    ]
    vectorstore = WeaviateVectorStore.from_documents(
        documents=leaf_docs,
        embedding=embeddings_model,
        client=client,
        index_name=INDEX_NAME,
    )
    print("Leaf chunks indexed.\n")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
    )
    llm_chain = SUMMARIZE_PROMPT | llm | StrOutputParser()

    print("Building RAPTOR tree (clustering + summarizing)...")
    build_raptor_tree(chunks, embeddings_model, llm_chain, vectorstore)
    print("\nRAPTOR tree complete.\n")

    return vectorstore


def search(query: str, vectorstore: WeaviateVectorStore, top_k: int = 5) -> list[tuple[str, dict]]:
    results = vectorstore.similarity_search(query, k=top_k)
    return [(doc.page_content, doc.metadata) for doc in results]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        vectorstore = build_pipeline(chunks, client)

        queries = [
            "What is the scaled dot product attention formula?",
            "How does positional encoding work?",
            "What is the overall contribution of this paper?",
            "What are the main components of the Transformer?",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, vectorstore)
            for rank, (text, metadata) in enumerate(results, 1):
                level = metadata.get("level", 0)
                kind = metadata.get("type", "chunk")
                print(f"  Rank {rank} [level={level} ({kind})]: {text[:150].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
