"""
Auto-Merging (Hierarchical) Retrieval over the Attention Is All You Need paper.

Problem: retrieving multiple small child chunks from the same parent section
sends fragmented, overlapping context to the LLM instead of the full section.

Solution: build a two-level hierarchy at index time. At query time, if enough
child chunks from the same parent are matched, automatically replace them with
the full parent chunk.

Hierarchy:
  Parent chunks (1000 tokens) — stored in a dict, returned to LLM
  Child chunks  (256 tokens)  — embedded in Weaviate with parent_id metadata

Merge rule:
  matched_children / total_children >= MERGE_THRESHOLD → return parent
  otherwise                                            → return child chunks

Requires: Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain langchain-community langchain-huggingface
         pip install langchain-weaviate pypdf langchain-text-splitters
         pip install sentence-transformers weaviate-client
"""

import weaviate
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_core.documents import Document

PDF_PATH = "data/attention_is_all_you_need.pdf"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperAutoMerging"

PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 25
MERGE_THRESHOLD = 0.5


def load_documents(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def build_hierarchy(docs) -> tuple[dict[str, str], list[Document], dict[str, int]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    parent_chunks = parent_splitter.split_documents(docs)

    parent_store: dict[str, str] = {}
    child_docs: list[Document] = []
    children_count: dict[str, int] = {}

    for p_idx, parent in enumerate(parent_chunks):
        parent_id = f"parent_{p_idx}"
        parent_store[parent_id] = parent.page_content

        children = child_splitter.split_documents([parent])
        children_count[parent_id] = len(children)

        for c_idx, child in enumerate(children):
            child_docs.append(Document(
                page_content=child.page_content,
                metadata={"parent_id": parent_id, "child_index": c_idx},
            ))

    return parent_store, child_docs, children_count


def build_pipeline(docs, client: weaviate.WeaviateClient):
    print("Building parent-child hierarchy...")
    parent_store, child_docs, children_count = build_hierarchy(docs)
    print(f"  {len(parent_store)} parent chunks, {len(child_docs)} child chunks.\n")

    print(f"Embedding child chunks with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = WeaviateVectorStore.from_documents(
        documents=child_docs,
        embedding=embeddings,
        client=client,
        index_name=INDEX_NAME,
    )
    print("Pipeline ready.\n")
    return vectorstore, parent_store, children_count


def search(
    query: str,
    vectorstore: WeaviateVectorStore,
    parent_store: dict[str, str],
    children_count: dict[str, int],
    top_k: int = 5,
    fetch_k: int = 20,
    merge_threshold: float = MERGE_THRESHOLD,
) -> list[tuple[str, str]]:
    raw_results = vectorstore.similarity_search(query, k=fetch_k)

    matched_children: dict[str, list[str]] = defaultdict(list)
    for doc in raw_results:
        parent_id = doc.metadata["parent_id"]
        matched_children[parent_id].append(doc.page_content)

    final_results: list[tuple[str, str]] = []
    seen_parents: set[str] = set()

    for doc in raw_results:
        parent_id = doc.metadata["parent_id"]

        if parent_id in seen_parents:
            continue

        n_matched = len(matched_children[parent_id])
        n_total = children_count[parent_id]
        ratio = n_matched / n_total

        if ratio >= merge_threshold:
            final_results.append((parent_store[parent_id], "parent"))
            seen_parents.add(parent_id)
        else:
            final_results.append((doc.page_content, "child"))

        if len(final_results) >= top_k:
            break

    return final_results


def main():
    print("Loading PDF...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages.\n")

    client = weaviate.connect_to_local()
    try:
        vectorstore, parent_store, children_count = build_pipeline(docs, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, vectorstore, parent_store, children_count)
            for rank, (text, source) in enumerate(results, 1):
                label = "MERGED → parent" if source == "parent" else "child"
                print(f"  Rank {rank} [{label}] ({len(text)} chars): {text[:150].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
