"""
HyDE (Hypothetical Document Embedding) Retrieval over the Attention Is All You Need paper.

Problem: a short query embeds differently from a full document passage — they
live in different regions of the embedding space, causing poor retrieval alignment.

Solution: use an LLM to generate a hypothetical (fake) answer to the query,
then embed that instead of the query. The fake answer mimics the style and
vocabulary of real passages, landing closer to relevant chunks in embedding space.

Pipeline:
  Query → LLM generates hypothetical answer → embed answer → vector search → real chunks

Note: hallucination is acceptable — even a factually wrong answer captures
the right vocabulary and passage structure for better embedding alignment.

Requires: OPENAI_API_KEY in .env
          Weaviate running locally via Docker
  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

Install: pip install langchain langchain-community langchain-huggingface
         pip install langchain-weaviate langchain-openai pypdf
         pip install langchain-text-splitters sentence-transformers weaviate-client
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
INDEX_NAME = "AttentionPaperHyde"
LLM_MODEL = "gpt-4o-mini"

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Given a question, write a short hypothetical "
        "passage (2-4 sentences) that would appear in an academic paper and directly "
        "answer the question. Write only the passage — no preamble, no explanation.",
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
    hypothesis_chain = HYDE_PROMPT | llm | StrOutputParser()

    print("Pipeline ready.\n")
    return vectorstore, hypothesis_chain


def search(
    query: str,
    vectorstore: WeaviateVectorStore,
    hypothesis_chain,
    top_k: int = 5,
) -> tuple[str, list[tuple[str, int]]]:
    hypothetical_doc = hypothesis_chain.invoke({"query": query})
    results = vectorstore.similarity_search(hypothetical_doc, k=top_k)
    return hypothetical_doc, [(doc.page_content, i + 1) for i, doc in enumerate(results)]


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    client = weaviate.connect_to_local()
    try:
        vectorstore, hypothesis_chain = build_pipeline(chunks, client)

        queries = [
            "What is the attention mechanism?",
            "multi-head attention",
            "positional encoding",
            "encoder decoder architecture",
            "scaled dot product attention formula",
        ]

        for query in queries:
            hypothetical_doc, results = search(query, vectorstore, hypothesis_chain)
            print(f"Query      : {query}")
            print(f"Hypothesis : {hypothetical_doc[:150].strip()}...")
            for rank, (text, _) in enumerate(results, 1):
                print(f"  Rank {rank}: {text[:120].strip()}...")
            print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
