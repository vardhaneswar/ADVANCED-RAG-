"""
RAG API Service
---------------
This service exposes an HTTP API endpoint that the dashboard
(or any client) can call with a natural-language question.

Flow:
    query -> retrieval (vector + graph) -> fusion -> LLM stub -> insights
"""

from fastapi import FastAPI
from pydantic import BaseModel

from config import get_settings
from retrieval.vector_retriever import retrieve_from_vector_index
from retrieval.graph_retriever import retrieve_from_graph
from retrieval.fusion import fuse_results
from llm.llm_client import generate_answer
from insights.formatter import format_insight


# ---------- FastAPI setup ----------

app = FastAPI(title="RAG API Service (Finance Demo)")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


# ---------- Core RAG pipeline ----------

def handle_query(query: str) -> str:
    """
    Main RAG pipeline for a single query.

    Steps:
    1. Vector retrieval
    2. (Optional) Graph retrieval
    3. Fusion of contexts
    4. LLM call (stub)
    5. Insight formatting
    """
    settings = get_settings()
    print(f"[rag-api] Received query: {query!r}")
    print(f"[rag-api] Environment: {settings.env}")
    print(f"[rag-api] Default model: {settings.default_model_name}")
    print(f"[rag-api] Graph retrieval enabled: {settings.enable_graph_retrieval}")

    # 1) Vector retrieval (pretend to query Pinecone)
    vector_docs = retrieve_from_vector_index(query, top_k=3)

    # 2) Graph retrieval (stub)
    graph_results = []
    if settings.enable_graph_retrieval:
        graph_results = retrieve_from_graph(query)
    else:
        print("[rag-api] Graph retrieval disabled.")

    # 3) Fusion
    fused_context = fuse_results(vector_docs, graph_results)

    # 4) LLM stub
    raw_answer = generate_answer(query, fused_context)

    # 5) Formatting
    final_answer = format_insight(raw_answer)
    return final_answer


# ---------- HTTP endpoint ----------

@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    """
    HTTP endpoint for RAG queries.

    Request JSON:
        { "query": "Compare NVIDIA vs Apple financial performance" }

    Response JSON:
        { "answer": "..." }
    """
    answer = handle_query(payload.query)
    return QueryResponse(answer=answer)


def main():
    """
    CLI entrypoint (optional).

    You can run:
        python main.py
    to test a single query without starting the HTTP server.
    """
    demo_query = "Compare NVIDIA vs Apple based on recent market behavior."
    answer = handle_query(demo_query)
    print("\n=== DEMO ANSWER ===")
    print(answer)
    print("===================")


if __name__ == "__main__":
    main()
