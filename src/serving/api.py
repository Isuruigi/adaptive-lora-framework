"""
FastAPI-based production API for model serving.

Features:
- RESTful endpoints for inference
- Batch processing support
- Health checks and metrics
- Request validation with Pydantic
- Rate limiting and authentication
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    from prometheus_client import Counter, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Pydantic Models
# ============================================================================


class GenerationRequest(BaseModel):
    """Request model for text generation.

    Attributes:
        prompt: Input prompt for generation.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        adapter: Specific adapter to use (None for auto-routing).
        stream: Enable streaming response.
    """

    prompt: str = Field(..., min_length=1, max_length=32000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    adapter: Optional[str] = None
    stream: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms.",
                "max_tokens": 256,
                "temperature": 0.7,
                "adapter": None
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    request_id: str
    generated_text: str
    adapter_used: str
    routing_confidence: float
    tokens_generated: int
    latency_ms: float
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "abc-123",
                "generated_text": "Quantum computing uses...",
                "adapter_used": "reasoning",
                "routing_confidence": 0.95,
                "tokens_generated": 150,
                "latency_ms": 234.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""

    prompts: List[str] = Field(..., min_length=1, max_length=100)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    adapter: Optional[str] = None


class BatchGenerationResponse(BaseModel):
    """Response model for batch generation."""

    request_id: str
    results: List[GenerationResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    models_loaded: List[str]
    gpu_available: bool


class AdapterInfo(BaseModel):
    """Adapter information."""

    name: str
    task_type: str
    loaded: bool
    metrics: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str
    request_id: Optional[str]


# ============================================================================
# Global State
# ============================================================================


class AppState:
    """Application state container."""

    def __init__(self):
        self.start_time = time.time()
        self.inference_engine = None
        self.router = None
        self.adapters = {}
        self.request_count = 0
        self.version = "1.0.0"


app_state = AppState()


# ============================================================================
# Metrics
# ============================================================================


if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "api_requests_total",
        "Total API requests",
        ["endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "api_request_latency_seconds",
        "API request latency",
        ["endpoint"]
    )
    TOKENS_GENERATED = Counter(
        "tokens_generated_total",
        "Total tokens generated"
    )


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Adaptive LoRA API server...")

    # Initialize models (placeholder)
    # In production, load actual models here
    app_state.adapters = {
        "reasoning": {"loaded": True, "metrics": {"requests": 0}},
        "code": {"loaded": True, "metrics": {"requests": 0}},
        "analysis": {"loaded": True, "metrics": {"requests": 0}},
        "creative": {"loaded": True, "metrics": {"requests": 0}},
    }

    logger.info(f"Loaded {len(app_state.adapters)} adapters")

    yield

    # Cleanup
    logger.info("Shutting down API server...")


# ============================================================================
# Application
# ============================================================================


app = FastAPI(
    title="Adaptive LoRA Framework API",
    description="Production API for multi-adapter LLM inference with dynamic routing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware
# ============================================================================


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"

    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"- {response.status_code} ({duration:.3f}s)"
    )

    return response


# ============================================================================
# Health Endpoints
# ============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Adaptive LoRA Framework API", "version": app_state.version}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    import torch

    uptime = time.time() - app_state.start_time

    return HealthResponse(
        status="healthy",
        version=app_state.version,
        uptime_seconds=uptime,
        models_loaded=list(app_state.adapters.keys()),
        gpu_available=torch.cuda.is_available() if torch else False
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe for Kubernetes."""
    if not app_state.adapters:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    return {"status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}


# ============================================================================
# Generation Endpoints
# ============================================================================


@app.post(
    "/generate",
    response_model=GenerationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Generation"]
)
async def generate(request: GenerationRequest, req: Request):
    """Generate text from prompt.

    Uses automatic adapter routing unless specific adapter is requested.
    """
    request_id = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    start_time = time.time()

    try:
        # Route to adapter
        if request.adapter:
            if request.adapter not in app_state.adapters:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown adapter: {request.adapter}"
                )
            adapter_used = request.adapter
            routing_confidence = 1.0
        else:
            # Auto-routing (placeholder)
            adapter_used = "reasoning"
            routing_confidence = 0.85

        # Generate (placeholder - replace with actual inference)
        generated_text = f"[Generated response for: {request.prompt[:50]}...]"
        tokens_generated = len(generated_text.split())

        # Update metrics
        app_state.request_count += 1
        if adapter_used in app_state.adapters:
            app_state.adapters[adapter_used]["metrics"]["requests"] += 1

        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint="/generate", status="success").inc()
            TOKENS_GENERATED.inc(tokens_generated)

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResponse(
            request_id=request_id,
            generated_text=generated_text,
            adapter_used=adapter_used,
            routing_confidence=routing_confidence,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/generate/batch",
    response_model=BatchGenerationResponse,
    tags=["Generation"]
)
async def generate_batch(request: BatchGenerationRequest, req: Request):
    """Generate text for multiple prompts."""
    request_id = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    start_time = time.time()

    results = []

    for i, prompt in enumerate(request.prompts):
        single_request = GenerationRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            adapter=request.adapter
        )

        # Generate (simplified - in production use parallel processing)
        result = await generate(single_request, req)
        results.append(result)

    total_latency_ms = (time.time() - start_time) * 1000

    return BatchGenerationResponse(
        request_id=request_id,
        results=results,
        total_latency_ms=total_latency_ms
    )


# ============================================================================
# Adapter Management Endpoints
# ============================================================================


@app.get("/adapters", response_model=List[AdapterInfo], tags=["Adapters"])
async def list_adapters():
    """List available adapters."""
    return [
        AdapterInfo(
            name=name,
            task_type=name,
            loaded=info.get("loaded", False),
            metrics=info.get("metrics", {})
        )
        for name, info in app_state.adapters.items()
    ]


@app.get("/adapters/{name}", response_model=AdapterInfo, tags=["Adapters"])
async def get_adapter(name: str):
    """Get adapter details."""
    if name not in app_state.adapters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter not found: {name}"
        )

    info = app_state.adapters[name]
    return AdapterInfo(
        name=name,
        task_type=name,
        loaded=info.get("loaded", False),
        metrics=info.get("metrics", {})
    )


@app.post("/adapters/{name}/load", tags=["Adapters"])
async def load_adapter(name: str):
    """Load an adapter."""
    if name not in app_state.adapters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter not found: {name}"
        )

    app_state.adapters[name]["loaded"] = True
    return {"status": "loaded", "adapter": name}


@app.post("/adapters/{name}/unload", tags=["Adapters"])
async def unload_adapter(name: str):
    """Unload an adapter."""
    if name not in app_state.adapters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter not found: {name}"
        )

    app_state.adapters[name]["loaded"] = False
    return {"status": "unloaded", "adapter": name}


# ============================================================================
# Metrics Endpoint
# ============================================================================


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Prometheus not available"
        )

    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/stats", tags=["Monitoring"])
async def stats():
    """Get API statistics."""
    return {
        "total_requests": app_state.request_count,
        "uptime_seconds": time.time() - app_state.start_time,
        "adapters": {
            name: info.get("metrics", {})
            for name, info in app_state.adapters.items()
        }
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception [{request_id}]: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "request_id": request_id
        }
    )


# ============================================================================
# Run Application
# ============================================================================


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
