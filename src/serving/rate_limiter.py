"""
Rate limiting middleware for API.

Features:
- Token bucket algorithm
- Per-user/API key limits
- Sliding window counter
- Redis-backed distributed limiting
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    remaining: int
    reset_after: float  # seconds until reset
    limit: int
    retry_after: Optional[float] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(time.time() + self.reset_after)),
        }
        
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
            
        return headers


class RateLimiter(ABC):
    """Abstract rate limiter."""
    
    @abstractmethod
    async def check(self, key: str) -> RateLimitResult:
        """Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., user ID, API key).
            
        Returns:
            RateLimitResult.
        """
        pass
    
    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        rate: float = 10.0,
        capacity: int = 100,
        per: float = 60.0
    ):
        """Initialize limiter.
        
        Args:
            rate: Tokens per second.
            capacity: Maximum bucket capacity.
            per: Time period in seconds.
        """
        self.rate = rate / per  # tokens per second
        self.capacity = capacity
        self.per = per
        
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()
        
    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit using token bucket."""
        async with self._lock:
            now = time.time()
            
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": self.capacity,
                    "last_update": now
                }
                
            bucket = self._buckets[key]
            
            # Refill tokens
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.capacity,
                bucket["tokens"] + elapsed * self.rate
            )
            bucket["last_update"] = now
            
            # Check if request allowed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_after=self.per,
                    limit=self.capacity
                )
            else:
                # Calculate retry time
                tokens_needed = 1 - bucket["tokens"]
                retry_after = tokens_needed / self.rate
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after=self.per,
                    limit=self.capacity,
                    retry_after=retry_after
                )
                
    async def reset(self, key: str) -> None:
        """Reset bucket for key."""
        async with self._lock:
            if key in self._buckets:
                del self._buckets[key]


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    def __init__(
        self,
        limit: int = 100,
        window_seconds: float = 60.0,
        precision: int = 10
    ):
        """Initialize limiter.
        
        Args:
            limit: Maximum requests per window.
            window_seconds: Window size in seconds.
            precision: Number of sub-windows for smoothing.
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.precision = precision
        self.bucket_size = window_seconds / precision
        
        self._windows: Dict[str, Dict[int, int]] = {}
        self._lock = asyncio.Lock()
        
    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit using sliding window."""
        async with self._lock:
            now = time.time()
            current_bucket = int(now / self.bucket_size)
            
            if key not in self._windows:
                self._windows[key] = {}
                
            window = self._windows[key]
            
            # Clean old buckets
            min_bucket = current_bucket - self.precision
            window = {k: v for k, v in window.items() if k > min_bucket}
            self._windows[key] = window
            
            # Count requests in window
            total = sum(window.values())
            
            if total < self.limit:
                # Allow request
                window[current_bucket] = window.get(current_bucket, 0) + 1
                
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - total - 1,
                    reset_after=self.window_seconds,
                    limit=self.limit
                )
            else:
                # Calculate when oldest bucket expires
                oldest_bucket = min(window.keys()) if window else current_bucket
                reset_after = (oldest_bucket + self.precision + 1) * self.bucket_size - now
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after=self.window_seconds,
                    limit=self.limit,
                    retry_after=max(0, reset_after)
                )
                
    async def reset(self, key: str) -> None:
        """Reset window for key."""
        async with self._lock:
            if key in self._windows:
                del self._windows[key]


class RedisRateLimiter(RateLimiter):
    """Redis-backed rate limiter for distributed systems."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        limit: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:"
    ):
        """Initialize Redis limiter.
        
        Args:
            redis_url: Redis connection URL.
            limit: Maximum requests per window.
            window_seconds: Window size.
            key_prefix: Redis key prefix.
        """
        self.redis_url = redis_url
        self.limit = limit
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._client = None
        
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url)
            await self._client.ping()
        except ImportError:
            logger.warning("redis package not installed")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            
    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit using Redis."""
        if self._client is None:
            # Fallback: allow all
            return RateLimitResult(
                allowed=True,
                remaining=self.limit,
                reset_after=self.window_seconds,
                limit=self.limit
            )
            
        full_key = f"{self.key_prefix}{key}"
        now = time.time()
        
        try:
            pipe = self._client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(full_key, 0, now - self.window_seconds)
            # Count current entries
            pipe.zcard(full_key)
            # Add current request
            pipe.zadd(full_key, {str(now): now})
            # Set expiry
            pipe.expire(full_key, self.window_seconds + 1)
            
            results = await pipe.execute()
            count = results[1]
            
            if count < self.limit:
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - count - 1,
                    reset_after=self.window_seconds,
                    limit=self.limit
                )
            else:
                # Remove the added entry since we're rejecting
                await self._client.zrem(full_key, str(now))
                
                # Get oldest entry to calculate retry
                oldest = await self._client.zrange(full_key, 0, 0, withscores=True)
                if oldest:
                    retry_after = oldest[0][1] + self.window_seconds - now
                else:
                    retry_after = self.window_seconds
                    
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after=self.window_seconds,
                    limit=self.limit,
                    retry_after=max(0, retry_after)
                )
                
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open
            return RateLimitResult(
                allowed=True,
                remaining=self.limit,
                reset_after=self.window_seconds,
                limit=self.limit
            )
            
    async def reset(self, key: str) -> None:
        """Reset limit for key."""
        if self._client:
            await self._client.delete(f"{self.key_prefix}{key}")


class TieredRateLimiter(RateLimiter):
    """Rate limiter with different tiers/plans."""
    
    def __init__(
        self,
        tiers: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize tiered limiter.
        
        Args:
            tiers: Tier configurations.
        """
        self.tiers = tiers or {
            "free": {"limit": 10, "window": 60},
            "basic": {"limit": 100, "window": 60},
            "pro": {"limit": 1000, "window": 60},
            "enterprise": {"limit": 10000, "window": 60},
        }
        
        self._limiters: Dict[str, SlidingWindowLimiter] = {}
        self._user_tiers: Dict[str, str] = {}
        
        # Create limiter for each tier
        for tier_name, config in self.tiers.items():
            self._limiters[tier_name] = SlidingWindowLimiter(
                limit=config["limit"],
                window_seconds=config["window"]
            )
            
    def set_user_tier(self, user_key: str, tier: str) -> None:
        """Set tier for user.
        
        Args:
            user_key: User identifier.
            tier: Tier name.
        """
        if tier in self.tiers:
            self._user_tiers[user_key] = tier
        else:
            logger.warning(f"Unknown tier: {tier}")
            
    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit based on user tier."""
        tier = self._user_tiers.get(key, "free")
        limiter = self._limiters.get(tier)
        
        if limiter is None:
            limiter = self._limiters["free"]
            
        return await limiter.check(key)
        
    async def reset(self, key: str) -> None:
        """Reset limits for key across all tiers."""
        tier = self._user_tiers.get(key, "free")
        limiter = self._limiters.get(tier)
        
        if limiter:
            await limiter.reset(key)


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        limiter: Optional[RateLimiter] = None,
        key_func: Optional[callable] = None
    ):
        """Initialize middleware.
        
        Args:
            limiter: Rate limiter instance.
            key_func: Function to extract key from request.
        """
        self.limiter = limiter or TokenBucketLimiter()
        self.key_func = key_func or self._default_key_func
        
    def _default_key_func(self, request) -> str:
        """Extract key from request (IP address)."""
        # Try to get real IP from headers (for reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
        
    async def __call__(self, request, call_next):
        """Middleware handler."""
        key = self.key_func(request)
        result = await self.limiter.check(key)
        
        if not result.allowed:
            from fastapi.responses import JSONResponse
            
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers=result.to_headers()
            )
            
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header, value in result.to_headers().items():
            response.headers[header] = value
            
        return response
