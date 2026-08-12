# FastAPI app - this is what ties everything together and exposes it over HTTP.
#
# Routes:
#   GET  /health   - just checks if the server is up
#   GET  /stats     - shows how many vectors are indexed, current settings
#   POST /ingest     - upload files, they get chunked + embedded + indexed
#   POST /query       - ask a question, get back an answer + citations
#   DELETE /index      - wipes the vector index (be careful with this one)

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.ingestion import ingest_paths
from app.rag_chain import RAGPipeline
from app.vectorstore import get_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(
    title="RAG-Based Enterprise Knowledge Assistant",
    description=(
        "A simple RAG API for searching and asking questions "
        "about a set of documents."
    ),
    version="1.0.0",
)

# letting all origins through for now since this is just a portfolio project
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    # only build the pipeline once and reuse it, don't want to recreate
    # the Groq client on every single request
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_manager())
    return _pipeline


# request/response models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class CitationOut(BaseModel):
    source: str
    chunk_id: str
    page: int
    snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationOut]


class IngestResponse(BaseModel):
    ingested_files: List[str]
    num_documents: int
    num_chunks: int
    total_vectors: int


class StatsResponse(BaseModel):
    embedding_model: str
    llm_model: str
    total_vectors: int
    chunk_size: int
    chunk_overlap: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    manager = get_manager()
    return StatsResponse(
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.LLM_MODEL,
        total_vectors=manager.document_count(),
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    manager = get_manager()
    saved_paths: List[Path] = []

    # save uploaded files to disk first, reject anything we can't handle
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}' for {upload.filename}",
            )
        dest = config.UPLOAD_DIR / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(dest)

    try:
        chunks, result = ingest_paths(saved_paths)
        manager.build(chunks)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # reset the cached pipeline so the next query sees the new documents
    global _pipeline
    _pipeline = None

    return IngestResponse(
        ingested_files=result.source_files,
        num_documents=result.num_documents,
        num_chunks=result.num_chunks,
        total_vectors=manager.document_count(),
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    pipeline = get_pipeline()
    try:
        result = pipeline.answer(payload.question, k=payload.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
    )


@app.delete("/index")
def wipe_index():
    # deletes everything in the vector index, mostly for testing/resetting
    if config.VECTOR_STORE_DIR.exists():
        shutil.rmtree(config.VECTOR_STORE_DIR)
        config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    global _pipeline
    _pipeline = None
    import app.vectorstore as vs_module

    vs_module._manager = None
    return {"status": "index cleared"}
