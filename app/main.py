"""FastAPI service: REST + Server-Sent-Events API and the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import app_graph, initial_state, serialize_sources
from app.config import get_settings
from app.loader import SUPPORTED_SUFFIXES
from app.retriever import get_hybrid_retriever, reindex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load models and build the index before accepting traffic.
    await asyncio.to_thread(get_hybrid_retriever)
    yield


app = FastAPI(
    title="Agentic RAG Compliance Engine",
    version="1.0.0",
    description="Self-RAG agent (LangGraph) over hybrid retrieval for compliance & financial documents.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["高风险客户多久复评一次？"])


class Source(BaseModel):
    index: int
    source: str | None
    title: str | None
    section: str | None
    content: str
    scores: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    candidates: list[Source]
    trace: list[dict]
    mode: str
    rewritten_query: str | None
    grounded: bool | None
    groundedness: float | None
    latency_ms: float


def _to_response(state: dict, started: float) -> QueryResponse:
    query = state.get("query")
    return QueryResponse(
        answer=state.get("generation", ""),
        sources=serialize_sources(state.get("relevant", [])),
        candidates=serialize_sources(state.get("documents", [])),
        trace=state.get("trace", []),
        mode=state.get("mode", "offline"),
        rewritten_query=query if query != state.get("question") else None,
        grounded=state.get("grounded"),
        groundedness=state.get("groundedness"),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        result = await app_graph.ainvoke(initial_state(request.question.strip()))
    except Exception as e:  # pragma: no cover - surfaced to the client
        logging.exception("query failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_response(result, started)


@app.post("/api/v1/query/stream")
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """Stream each graph step as it finishes (SSE), then the final result."""

    async def events():
        started = time.perf_counter()
        seen = 0
        state: dict = {}
        try:
            async for state in app_graph.astream(
                initial_state(request.question.strip()), stream_mode="values"
            ):
                trace = state.get("trace", [])
                for step in trace[seen:]:
                    yield f"data: {json.dumps({'type': 'step', 'step': step}, ensure_ascii=False)}\n\n"
                seen = len(trace)
            payload = _to_response(state, started).model_dump()
            yield f"data: {json.dumps({'type': 'result', 'result': payload}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logging.exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/v1/health")
async def health() -> dict:
    s = get_settings()
    r = get_hybrid_retriever()
    return {
        "status": "ok",
        "llm_enabled": s.llm_enabled,
        "llm_model": s.llm_model if s.llm_enabled else None,
        "embedding_model": s.embedding_model if s.embedding_backend != "hash" else "hash",
        "reranker_model": s.reranker_model if r.reranker is not None else None,
        "chunks": len(r.documents),
    }


@app.get("/api/v1/documents")
async def list_documents() -> list[dict]:
    r = get_hybrid_retriever()
    counts = Counter(d.metadata["source"] for d in r.documents)
    titles = {d.metadata["source"]: d.metadata["title"] for d in r.documents}
    return [{"source": s, "title": titles[s], "chunks": n} for s, n in sorted(counts.items())]


@app.post("/api/v1/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Only {', '.join(sorted(SUPPORTED_SUFFIXES))} files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")
    target = get_settings().docs_dir / name
    target.write_bytes(data)
    retriever = await asyncio.to_thread(reindex)
    chunks = sum(1 for d in retriever.documents if d.metadata["source"] == name)
    return {"source": name, "chunks": chunks, "total_chunks": len(retriever.documents)}


@app.get("/api/v1/graph")
async def graph_mermaid() -> dict:
    return {"mermaid": app_graph.get_graph().draw_mermaid()}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
