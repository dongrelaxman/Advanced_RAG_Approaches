"""
Self-Querying Retrieval over the Attention Is All You Need paper.

Problem: standard vector search ignores structural intent in the query.
"What is attention on page 3?" treats "page 3" as semantic — which is wrong.

Solution: an LLM parses the natural language query into two parts:
  1. Semantic query  → used for vector similarity search
  2. Metadata filter → applied as a hard filter (e.g. page == 3)

This combines the flexibility of semantic search with the precision of
structured filtering, similar to SQL WHERE clauses on top of vector search.

Requires: OPENAI_API_KEY in .env
          Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain langchain-community langchain-huggingface
         pip install langchain-weaviate langchain-openai
         pip install pypdf langchain-text-splitters sentence-transformers weaviate-client
         pip install python-dotenv
"""

import os
import weaviate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.weaviate import WeaviateTranslator
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

load_dotenv()

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperSelfQuery"
LLM_MODEL = "gpt-4o-mini"

DOCUMENT_DESCRIPTION = (
    "Chunks from the 'Attention Is All You Need' paper by Vaswani et al. "
    "covering the Transformer architecture, attention mechanisms, and experiments."
)

METADATA_FIELDS = [
    AttributeInfo(
        name="page",
        description="The page number in the PDF (0-indexed, so page 0 is the first page)",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="The source PDF file path",
        type="string",
    ),
]


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

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
    )

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELDS,
        structured_query_translator=WeaviateTranslator(),
        verbose=True,
    )

    print("Pipeline ready.\n")
    return retriever


def search(query: str, retriever: SelfQueryRetriever) -> list[tuple[str, int, dict]]:
    results = retriever.invoke(query)
    return [(doc.page_content, i + 1, doc.metadata) for i, doc in enumerate(results)]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        retriever = build_pipeline(chunks, client)

        queries = [
            "What is the attention mechanism?",
            "How does multi-head attention work?",
            "What is discussed on page 1?",
            "What topics appear on pages 4 and 5?",
            "Explain positional encoding from the first 3 pages",
        ]

        for query in queries:
            print(f"Query: {query}")
            results = search(query, retriever)
            if not results:
                print("  No results found.")
            for rank, (text, _, metadata) in enumerate(results, 1):
                page = metadata.get("page", "?")
                print(f"  Rank {rank} (page={page}): {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
