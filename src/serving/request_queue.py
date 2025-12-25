"""
Request queue system with priority handling.

Features:
- Priority-based queue
- Redis backend support
- Request batching
- Rate limiting integration
"""

from __future__ import annotations

import asyncio
import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Priority(IntEnum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(order=True)
class QueuedRequest:
    """Request in the queue."""
    
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)
    timeout: float = field(default=30.0, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        # Negate priority for max-heap behavior (higher priority first)
        self.priority = -self.priority


class InMemoryQueue:
    """In-memory priority queue implementation."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize queue.
        
        Args:
            max_size: Maximum queue size.
        """
        self._queue: List[QueuedRequest] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
        
    async def push(self, request: QueuedRequest) -> bool:
        """Add request to queue.
        
        Args:
            request: Request to add.
            
        Returns:
            True if added successfully.
        """
        async with self._lock:
            if len(self._queue) >= self._max_size:
                logger.warning("Queue is full, rejecting request")
                return False
                
            heapq.heappush(self._queue, request)
            return True
            
    async def pop(self) -> Optional[QueuedRequest]:
        """Get next request from queue.
        
        Returns:
            Next request or None if empty.
        """
        async with self._lock:
            if self._queue:
                return heapq.heappop(self._queue)
            return None
            
    async def peek(self) -> Optional[QueuedRequest]:
        """Peek at next request without removing.
        
        Returns:
            Next request or None.
        """
        async with self._lock:
            if self._queue:
                return self._queue[0]
            return None
            
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0
    
    async def clear(self) -> int:
        """Clear all requests from queue.
        
        Returns:
            Number of cleared requests.
        """
        async with self._lock:
            count = len(self._queue)
            self._queue = []
            return count


class RedisQueue:
    """Redis-backed priority queue."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "inference_queue",
        max_size: int = 10000
    ):
        """Initialize Redis queue.
        
        Args:
            redis_url: Redis connection URL.
            queue_name: Queue key name.
            max_size: Maximum queue size.
        """
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.max_size = max_size
        self._client = None
        
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url)
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using mock")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
            
    async def push(self, request: QueuedRequest) -> bool:
        """Add request to Redis sorted set."""
        if self._client is None:
            return False
            
        try:
            import json
            
            # Check size
            size = await self._client.zcard(self.queue_name)
            if size >= self.max_size:
                return False
                
            # Score is negative priority + timestamp fraction for ordering
            score = -request.priority + (request.timestamp / 1e12)
            
            payload = json.dumps({
                "request_id": request.request_id,
                "payload": request.payload,
                "timeout": request.timeout,
                "retries": request.retries,
                "max_retries": request.max_retries,
                "metadata": request.metadata,
                "timestamp": request.timestamp,
            })
            
            await self._client.zadd(self.queue_name, {payload: score})
            return True
            
        except Exception as e:
            logger.error(f"Failed to push to Redis: {e}")
            return False
            
    async def pop(self) -> Optional[QueuedRequest]:
        """Get and remove next request."""
        if self._client is None:
            return None
            
        try:
            import json
            
            # Pop lowest score (highest priority)
            result = await self._client.zpopmin(self.queue_name)
            
            if result:
                payload_str, score = result[0]
                data = json.loads(payload_str)
                
                return QueuedRequest(
                    priority=-int(score),
                    timestamp=data["timestamp"],
                    request_id=data["request_id"],
                    payload=data["payload"],
                    timeout=data["timeout"],
                    retries=data["retries"],
                    max_retries=data["max_retries"],
                    metadata=data["metadata"]
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to pop from Redis: {e}")
            return None
            
    def size(self) -> int:
        """Get queue size (sync version)."""
        if self._client is None:
            return 0
        # Note: For async context, use async version
        return 0
        
    async def size_async(self) -> int:
        """Get queue size."""
        if self._client is None:
            return 0
        return await self._client.zcard(self.queue_name)


class RequestQueue:
    """High-level request queue with processing."""
    
    def __init__(
        self,
        backend: str = "memory",
        max_size: int = 10000,
        redis_url: Optional[str] = None,
        batch_size: int = 8,
        batch_timeout: float = 0.1
    ):
        """Initialize request queue.
        
        Args:
            backend: Queue backend ('memory' or 'redis').
            max_size: Maximum queue size.
            redis_url: Redis URL if using Redis backend.
            batch_size: Batch size for processing.
            batch_timeout: Timeout for batch collection.
        """
        if backend == "redis" and redis_url:
            self._queue = RedisQueue(redis_url=redis_url, max_size=max_size)
        else:
            self._queue = InMemoryQueue(max_size=max_size)
            
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._processing = False
        self._processor: Optional[Callable] = None
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "failed": 0,
            "retried": 0,
        }
        
    async def initialize(self) -> None:
        """Initialize queue (connect to Redis if needed)."""
        if isinstance(self._queue, RedisQueue):
            await self._queue.connect()
            
    async def enqueue(
        self,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add request to queue.
        
        Args:
            payload: Request payload.
            priority: Request priority.
            timeout: Request timeout.
            callback: Optional callback for completion.
            metadata: Additional metadata.
            
        Returns:
            Request ID.
        """
        request_id = str(uuid.uuid4())
        
        request = QueuedRequest(
            priority=priority,
            timestamp=time.time(),
            request_id=request_id,
            payload=payload,
            callback=callback,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        success = await self._queue.push(request)
        
        if success:
            self._stats["enqueued"] += 1
            logger.debug(f"Enqueued request {request_id} with priority {priority.name}")
        else:
            raise RuntimeError("Queue is full")
            
        return request_id
        
    async def process_one(self, processor: Callable) -> Optional[Any]:
        """Process single request from queue.
        
        Args:
            processor: Async function to process request.
            
        Returns:
            Processing result or None.
        """
        request = await self._queue.pop()
        
        if request is None:
            return None
            
        try:
            result = await asyncio.wait_for(
                processor(request.payload),
                timeout=request.timeout
            )
            
            self._stats["processed"] += 1
            
            if request.callback:
                await request.callback(result)
                
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {request.request_id} timed out")
            await self._handle_failure(request, "timeout")
            
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            await self._handle_failure(request, str(e))
            
        return None
        
    async def _handle_failure(
        self,
        request: QueuedRequest,
        error: str
    ) -> None:
        """Handle failed request with retry logic."""
        if request.retries < request.max_retries:
            # Re-queue with incremented retry count
            request.retries += 1
            request.metadata["last_error"] = error
            request.timestamp = time.time()  # Update timestamp
            
            await self._queue.push(request)
            self._stats["retried"] += 1
            logger.info(f"Retrying request {request.request_id} (attempt {request.retries})")
        else:
            self._stats["failed"] += 1
            logger.error(f"Request {request.request_id} failed after {request.max_retries} retries")
            
    async def process_batch(
        self,
        processor: Callable
    ) -> List[Any]:
        """Process batch of requests.
        
        Args:
            processor: Async function that processes list of payloads.
            
        Returns:
            List of results.
        """
        batch = []
        start_time = time.time()
        
        while len(batch) < self.batch_size:
            request = await self._queue.pop()
            
            if request is None:
                break
                
            batch.append(request)
            
            # Check timeout
            if time.time() - start_time > self.batch_timeout:
                break
                
        if not batch:
            return []
            
        try:
            payloads = [r.payload for r in batch]
            results = await processor(payloads)
            
            self._stats["processed"] += len(batch)
            
            # Call callbacks
            for request, result in zip(batch, results):
                if request.callback:
                    await request.callback(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Re-queue all requests
            for request in batch:
                await self._handle_failure(request, str(e))
                
            return []
            
    async def start_processing(
        self,
        processor: Callable,
        batch: bool = True
    ) -> None:
        """Start continuous processing loop.
        
        Args:
            processor: Processing function.
            batch: Whether to use batch processing.
        """
        self._processing = True
        self._processor = processor
        
        logger.info("Starting queue processing")
        
        while self._processing:
            if batch:
                await self.process_batch(processor)
            else:
                await self.process_one(processor)
                
            # Small delay to prevent busy-waiting on empty queue
            if self._queue.is_empty() if hasattr(self._queue, 'is_empty') else True:
                await asyncio.sleep(0.01)
                
    def stop_processing(self) -> None:
        """Stop processing loop."""
        self._processing = False
        logger.info("Stopping queue processing")
        
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.size() if hasattr(self._queue, 'size') else 0
        }
