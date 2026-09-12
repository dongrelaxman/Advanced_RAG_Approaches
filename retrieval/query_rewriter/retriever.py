"""
Query Rewriter Retrieval over the Attention Is All You Need paper.

Problem: user queries are often short, ambiguous, or poorly phrased,
leading to poor vector search matches.

Solution: before retrieval, an LLM rewrites the query into a clearer,
more retrieval-friendly version that better captures the user's intent.

Pipeline:
  Original query → LLM rewrites it → rewritten query → vector retrieval → results

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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = "data/attention_is_all_you_need.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_NAME = "AttentionPaperQueryRewriter"
LLM_MODEL = "gpt-4o-mini"

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at rewriting search queries to improve document retrieval. "
        "Rewrite the user's query to be more specific, clear, and retrieval-friendly. "
        "Return only the rewritten query — no explanation, no extra text.",
    ),
    ("human", "{query}"),
])


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

    rewriter = REWRITE_PROMPT | llm | StrOutputParser()

    print("Pipeline ready.\n")
    return vectorstore, rewriter


def search(
    query: str,
    vectorstore: WeaviateVectorStore,
    rewriter,
    top_k: int = 5,
) -> tuple[str, list[tuple[str, int]]]:
    rewritten = rewriter.invoke({"query": query})
    results = vectorstore.similarity_search(rewritten, k=top_k)
    return rewritten, [(doc.page_content, i + 1) for i, doc in enumerate(results)]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        vectorstore, rewriter = build_pipeline(chunks, client)

        queries = [
            "attention?",
            "how multi head works",
            "position stuff",
            "encoder and decoder",
            "dot product formula",
        ]

        for query in queries:
            rewritten, results = search(query, vectorstore, rewriter)
            print(f"Original : {query}")
            print(f"Rewritten: {rewritten}")
            for rank, (text, _) in enumerate(results, 1):
                print(f"  Rank {rank}: {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
