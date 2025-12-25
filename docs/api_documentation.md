# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Set `Authorization: Bearer <your-token>` header for protected endpoints.

---

## Health Endpoints

### GET /

Root endpoint.

**Response:**
```json
{"message": "Adaptive LoRA Framework API", "version": "1.0.0"}
```

### GET /health

Health check with system status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "models_loaded": ["reasoning", "code", "analysis", "creative"],
  "gpu_available": true
}
```

### GET /ready

Kubernetes readiness probe.

### GET /live

Kubernetes liveness probe.

---

## Generation Endpoints

### POST /generate

Generate text from a prompt with automatic adapter routing.

**Request:**
```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "adapter": null,
  "stream": false
}
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | string | required | Input prompt (1-32000 chars) |
| max_tokens | int | 512 | Maximum tokens to generate (1-4096) |
| temperature | float | 0.7 | Sampling temperature (0.0-2.0) |
| top_p | float | 0.9 | Top-p sampling (0.0-1.0) |
| top_k | int | 50 | Top-k sampling (1-1000) |
| adapter | string | null | Specific adapter (null for auto-routing) |
| stream | bool | false | Enable streaming response |

**Response:**
```json
{
  "request_id": "abc-123",
  "generated_text": "Quantum computing uses quantum mechanics...",
  "adapter_used": "reasoning",
  "routing_confidence": 0.95,
  "tokens_generated": 150,
  "latency_ms": 234.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### POST /generate/batch

Generate text for multiple prompts.

**Request:**
```json
{
  "prompts": [
    "What is machine learning?",
    "Write a Python function to sort a list."
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "adapter": null
}
```

**Response:**
```json
{
  "request_id": "batch-456",
  "results": [
    {"request_id": "...", "generated_text": "...", ...},
    {"request_id": "...", "generated_text": "...", ...}
  ],
  "total_latency_ms": 589.2
}
```

---

## Adapter Management

### GET /adapters

List all available adapters.

**Response:**
```json
[
  {
    "name": "reasoning",
    "task_type": "reasoning",
    "loaded": true,
    "metrics": {"requests": 1542}
  },
  ...
]
```

### GET /adapters/{name}

Get specific adapter details.

### POST /adapters/{name}/load

Load an adapter into memory.

### POST /adapters/{name}/unload

Unload an adapter from memory.

---

## Monitoring

### GET /metrics

Prometheus metrics endpoint.

### GET /stats

Get API statistics.

**Response:**
```json
{
  "total_requests": 15420,
  "uptime_seconds": 86400.5,
  "adapters": {
    "reasoning": {"requests": 8000},
    "code": {"requests": 5000},
    "analysis": {"requests": 2420}
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "request_id": "abc-123"
}
```

**Status Codes:**
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (unknown adapter)
- `500` - Internal Server Error
- `503` - Service Unavailable (models not loaded)

---

## Rate Limits

- **Default**: 100 requests/minute
- **Batch**: 10 requests/minute
- Headers: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

---

## Examples

### cURL

```bash
# Single generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?", "max_tokens": 100}'

# Batch generation
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Query 1", "Query 2"]}'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Explain machine learning",
        "max_tokens": 256,
        "temperature": 0.7
    }
)

result = response.json()
print(result["generated_text"])
```
