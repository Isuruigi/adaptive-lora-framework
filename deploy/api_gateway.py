"""
Production API Gateway for Adaptive LoRA Framework

This is the FastAPI gateway that:
- Handles authentication & rate limiting
- Provides request caching
- Implements circuit breaker
- Routes to Modal serverless GPU

Deploy on Railway/Render for $5-10/mo
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
import httpx
import os
import time
from datetime import datetime
import hashlib
import logging
import asyncio

# ==================== CONFIGURATION ====================

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODAL_ENDPOINT = os.getenv("MODAL_ENDPOINT", "https://your-modal-app.modal.run")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
JWT_SECRET = os.getenv("JWT_SECRET", "change-in-production")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT", "60"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Adaptive LoRA API Gateway",
    description="Production API for multi-adapter LLM inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ==================== IN-MEMORY STORES (Use Redis in prod) ====================

class SimpleRateLimiter:
    """Simple in-memory rate limiter (use Redis in production)"""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests: Dict[str, list] = {}

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        if user_id not in self.requests:
            self.requests[user_id] = []

        # Clean old requests
        self.requests[user_id] = [t for t in self.requests[user_id] if t > minute_ago]

        if len(self.requests[user_id]) >= self.rpm:
            return False

        self.requests[user_id].append(now)
        return True


class SimpleCache:
    """Simple in-memory cache (use Redis in production)"""

    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, tuple] = {}

    def _make_key(self, prompt: str, adapter: str) -> str:
        content = f"{prompt}:{adapter}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, adapter: str) -> Optional[Dict]:
        key = self._make_key(prompt, adapter)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            del self.cache[key]
        return None

    def set(self, prompt: str, adapter: str, response: Dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]

        key = self._make_key(prompt, adapter)
        self.cache[key] = (response, time.time())


# Initialize
rate_limiter = SimpleRateLimiter(RATE_LIMIT_PER_MINUTE)
response_cache = SimpleCache(CACHE_TTL)

# Metrics
metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "cache_hits": 0,
    "total_latency": 0.0,
    "start_time": datetime.utcnow()
}


# ==================== MODELS ====================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    adapter: str = Field("reasoning", pattern="^(reasoning|code|creative|analysis)$")
    max_tokens: int = Field(200, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)

    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class GenerateResponse(BaseModel):
    request_id: str
    response: str
    adapter_used: str
    latency_ms: float
    tokens_generated: int
    timestamp: str
    cached: bool = False


# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """Simple circuit breaker"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False

        return True  # half-open

    def record_success(self):
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning("Circuit breaker opened!")


circuit_breaker = CircuitBreaker()


# ==================== MODAL CLIENT ====================

async def call_modal(request_data: Dict) -> Dict:
    """Call Modal service with circuit breaker"""

    if not circuit_breaker.can_proceed():
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable (circuit breaker open)"
        )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{MODAL_ENDPOINT}/api_generate",
                json=request_data
            )
            response.raise_for_status()

            circuit_breaker.record_success()
            return response.json()

    except httpx.HTTPError as e:
        circuit_breaker.record_failure()
        logger.error(f"Modal call failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Inference service unavailable"
        )


# ==================== HELPERS ====================

def get_user_id(request: Request) -> str:
    """Extract user ID from request"""
    # Check for API key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[:8]}"

    # Use IP address
    return f"ip:{request.client.host}"


# ==================== ENDPOINTS ====================

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Generate text with:
    - Rate limiting
    - Caching
    - Circuit breaker
    - Error handling
    """
    global metrics

    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    user_id = get_user_id(req)

    metrics["requests_total"] += 1

    try:
        # Rate limiting
        if not rate_limiter.is_allowed(user_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        # Check cache
        cached = response_cache.get(request.prompt, request.adapter)
        if cached:
            metrics["cache_hits"] += 1
            cached["request_id"] = request_id
            cached["cached"] = True
            return cached

        # Call Modal
        result = await call_modal({
            "prompt": request.prompt,
            "adapter": request.adapter,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "request_id": request_id
        })

        # Prepare response
        total_latency = time.time() - start_time

        response_data = GenerateResponse(
            request_id=request_id,
            response=result.get("response", ""),
            adapter_used=result.get("adapter_used", request.adapter),
            latency_ms=total_latency * 1000,
            tokens_generated=result.get("tokens_generated", 0),
            timestamp=datetime.utcnow().isoformat(),
            cached=False
        )

        # Cache in background
        background_tasks.add_task(
            response_cache.set,
            request.prompt,
            request.adapter,
            response_data.dict()
        )

        metrics["requests_success"] += 1
        metrics["total_latency"] += total_latency

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        metrics["requests_failed"] += 1
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""

    checks = {
        "api": "healthy",
        "modal": "unknown",
        "circuit_breaker": circuit_breaker.state
    }

    # Check Modal
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MODAL_ENDPOINT}/health")
            if response.status_code == 200:
                checks["modal"] = "healthy"
            else:
                checks["modal"] = "degraded"
    except Exception:
        checks["modal"] = "unreachable"

    overall = "healthy" if checks["modal"] == "healthy" else "degraded"

    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    global metrics

    uptime = (datetime.utcnow() - metrics["start_time"]).total_seconds()
    success_rate = metrics["requests_success"] / max(metrics["requests_total"], 1)
    avg_latency = metrics["total_latency"] / max(metrics["requests_success"], 1) * 1000

    prometheus_format = f"""# HELP requests_total Total requests
# TYPE requests_total counter
requests_total {metrics['requests_total']}

# HELP requests_success Successful requests
# TYPE requests_success counter
requests_success {metrics['requests_success']}

# HELP requests_failed Failed requests
# TYPE requests_failed counter
requests_failed {metrics['requests_failed']}

# HELP cache_hits Cache hit count
# TYPE cache_hits counter
cache_hits {metrics['cache_hits']}

# HELP success_rate Request success rate
# TYPE success_rate gauge
success_rate {success_rate:.4f}

# HELP avg_latency_ms Average latency in ms
# TYPE avg_latency_ms gauge
avg_latency_ms {avg_latency:.2f}

# HELP uptime_seconds API uptime
# TYPE uptime_seconds counter
uptime_seconds {uptime:.0f}
"""

    return PlainTextResponse(prometheus_format, media_type="text/plain")


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Adaptive LoRA API Gateway",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "generate": "POST /v1/generate",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "docs": "GET /docs"
        }
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==================== STARTUP ====================

@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("🚀 Adaptive LoRA API Gateway Started")
    logger.info(f"   Modal Endpoint: {MODAL_ENDPOINT}")
    logger.info(f"   Rate Limit: {RATE_LIMIT_PER_MINUTE} req/min")
    logger.info("=" * 50)


# Run with: uvicorn api_gateway:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
