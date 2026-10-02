"""
FastAPI Application Entrypoint for the Defense-in-Depth Framework.

Provides REST API endpoints for:
- Full pipeline query processing
- Individual layer testing
- Health check with model status
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse
from app.router import DefenseInDepthRouter

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

# Remove default loguru handler and add custom format
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level=settings.log_level,
    colorize=True,
)

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

# Global router instance
_router: DefenseInDepthRouter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle — model loading and cleanup.

    Models are loaded on startup and released on shutdown.
    """
    global _router

    logger.info("=" * 60)
    logger.info("Defense-in-Depth Framework — Starting Up")
    logger.info("=" * 60)

    _router = DefenseInDepthRouter()

    try:
        await _router.initialize()
        logger.info("All systems initialized successfully")
    except Exception as exc:
        logger.error("Initialization failed: {}", exc)
        logger.warning("Framework will operate in degraded mode")

    yield

    # Shutdown
    logger.info("Shutting down framework...")
    if _router:
        await _router.shutdown()
    logger.info("Framework shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Defense-in-Depth Hallucination Mitigation Framework",
    description=(
        "A pure-Python, asynchronous, multi-layer framework for detecting "
        "and mitigating LLM hallucinations. Integrates internal Semantic Entropy "
        "Probes with external NLI verification and structured reasoning."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health_check() -> dict:
    """
    Health check endpoint with model and component status.
    """
    return {
        "status": "healthy",
        "framework": "Defense-in-Depth Hallucination Mitigation",
        "version": "0.1.0",
        "config": {
            "t_safe": settings.t_safe,
            "nli_confidence_threshold": settings.nli_confidence_threshold,
            "ollama_model": settings.ollama_model,
            "probe_model": settings.probe_model_name,
            "nli_model": settings.nli_model_name,
            "search_backend": settings.search_backend,
        },
        "initialized": _router is not None and _router._initialized,
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main pipeline entry point.

    Processes a user query through the full defense-in-depth pipeline:
    Layer 1 (SEP Triage) → Layer 2 (NLI-RAG) or Layer 3 (HalluClean).
    """
    if _router is None:
        raise HTTPException(
            status_code=503,
            detail="Framework not initialized. Please wait for startup to complete.",
        )

    try:
        return await _router.process_query(request)
    except Exception as exc:
        logger.error("Pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/query/layer1", response_model=QueryResponse)
async def process_layer1(request: QueryRequest) -> QueryResponse:
    """
    Layer 1 only — Semantic Entropy Probe triage.

    Forces the query through Layer 1 only (for benchmarking).
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Framework not initialized.")

    request.force_layer = 1
    try:
        return await _router.process_query(request)
    except Exception as exc:
        logger.error("Layer 1 error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/query/layer2", response_model=QueryResponse)
async def process_layer2(request: QueryRequest) -> QueryResponse:
    """
    Layer 2 only — NLI-RAG verification.

    Forces the query through Layer 2 (for benchmarking).
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Framework not initialized.")

    request.force_layer = 2
    try:
        return await _router.process_query(request)
    except Exception as exc:
        logger.error("Layer 2 error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/query/layer3", response_model=QueryResponse)
async def process_layer3(request: QueryRequest) -> QueryResponse:
    """
    Layer 3 only — HalluClean structured reasoning.

    Forces the query through Layer 3 (for benchmarking).
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Framework not initialized.")

    request.force_layer = 3
    try:
        return await _router.process_query(request)
    except Exception as exc:
        logger.error("Layer 3 error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Server Runner
# ---------------------------------------------------------------------------


def run_server() -> None:
    """
    Run the FastAPI server via Uvicorn.

    Entry point for the `did-serve` CLI command.
    """
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
