
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from main import (
    generate_grounded_answer,
    get_rag_components,
    LLM_MODEL,
    EMBEDDING_MODEL,
)

# Serialize local model inference requests
inference_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model initialization is deferred until first query.
    yield


app = FastAPI(
    title="Advanced Wikipedia RAG API",
    description=(
        "Local RAG using FAISS, BM25, "
        "cross-encoder reranking and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Request schema
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question about AI or machine learning",
    )


# Response schemas
class SourceResponse(BaseModel):
    id: int
    title: str
    url: str
    excerpt: str
    score: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    citation_warning: str | None
    latency_ms: float


def run_rag(question: str):
    start = time.perf_counter()

    with inference_lock:
        answer, evidence, warning = (
            generate_grounded_answer(question)
        )

    sources = [
        SourceResponse(
            id=item["id"],
            title=item["title"],
            url=item["url"],
            excerpt=item["text"][:1200],
            score=item["score"],
        )
        for item in evidence
    ]

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        citation_warning=warning,
        latency_ms=round(elapsed_ms, 2),
    )


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "note": (
            "Process health only; model readiness "
            "is checked separately."
        ),
    }


# Readiness endpoint
@app.get("/ready")
def ready():
    try:
        get_rag_components()
        return {"status": "ready"}
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"RAG initialization failed: {error}",
        )


# Main inference endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        return await run_in_threadpool(
            run_rag,
            question,
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG inference failed. Check Ollama, "
                "model availability and FAISS storage."
            ),
        )


@app.get("/")
def root():
    return {
        "application": "Advanced Wikipedia RAG API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }
