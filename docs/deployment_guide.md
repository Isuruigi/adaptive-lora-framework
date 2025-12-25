# Production Deployment Guide

This guide covers deploying the Adaptive Multi-Agent LoRA Framework in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Deployment Options](#deployment-options)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Scaling Strategies](#scaling-strategies)
7. [Security Best Practices](#security-best-practices)
8. [Cost Optimization](#cost-optimization)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x A10G (24GB) | 1x A100 (40/80GB) |
| CPU | 8 cores | 16+ cores |
| RAM | 32GB | 64GB+ |
| Storage | 100GB SSD | 500GB NVMe |
| Network | 1 Gbps | 10 Gbps |

### Software Requirements

- Python 3.10+
- CUDA 11.8+ / CUDA 12.x
- Docker 24.0+
- Kubernetes 1.28+ (for K8s deployment)
- Redis 7.0+ (for caching/rate limiting)

### Model Requirements

- Base model: Meta-Llama-3-8B (or similar 7-13B model)
- ~16GB VRAM for 8B model with 4-bit quantization
- ~32GB VRAM for full precision or larger models

---

## Infrastructure Requirements

### Cloud Provider Options

#### AWS (Recommended)
```
Instance Types:
- Development: g5.xlarge (1x A10G, 24GB VRAM)
- Production: g5.2xlarge or p4d.24xlarge
- High Scale: p4de.24xlarge (8x A100)

Services:
- EKS for Kubernetes
- ElastiCache for Redis
- S3 for model storage
- CloudWatch for monitoring
- ALB for load balancing
```

#### GCP
```
Instance Types:
- Development: g2-standard-4 (1x L4)
- Production: a2-highgpu-1g (1x A100)

Services:
- GKE for Kubernetes
- Memorystore for Redis
- GCS for model storage
```

#### Azure
```
Instance Types:
- Development: NC6s_v3 (1x V100)
- Production: NC24ads_A100_v4 (1x A100)

Services:
- AKS for Kubernetes
- Azure Cache for Redis
- Blob Storage for models
```

---

## Deployment Options

### Option 1: Docker Compose (Simplest)

Best for: Single-server deployments, testing, small-scale production.

```bash
# Clone repository
git clone https://github.com/your-org/adaptive-lora-framework.git
cd adaptive-lora-framework

# Configure environment
cp .env.example .env
# Edit .env with your settings:
# - HF_TOKEN=your_huggingface_token
# - OPENAI_API_KEY=your_openai_key
# - BASE_MODEL=meta-llama/Meta-Llama-3-8B

# Build and run
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose ps
docker-compose logs -f api
```

### Option 2: Kubernetes (Recommended for Scale)

Best for: Multi-GPU deployments, auto-scaling, high availability.

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (copy template and fill values)
kubectl apply -f k8s/secrets.yaml

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/networkpolicy.yaml

# Check status
kubectl get pods -n ml-inference
kubectl logs -f deployment/adaptive-lora-api -n ml-inference
```

### Option 3: Managed ML Platforms

- **AWS SageMaker**: Use custom inference containers
- **GCP Vertex AI**: Deploy as custom prediction endpoint
- **Azure ML**: Deploy as managed online endpoint

---

## Step-by-Step Deployment

### Step 1: Prepare Models

```bash
# Download and prepare models
python scripts/prepare_data.py --download-model

# Train adapters (or use pre-trained)
python experiments/train_router.py --config configs/router/training_config.yaml

# Upload to S3/GCS (for cloud deployments)
aws s3 sync ./models s3://your-bucket/models/
```

### Step 2: Build Docker Image

```dockerfile
# Dockerfile.prod
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
# Build
docker build -f Dockerfile.prod -t adaptive-lora:latest .

# Push to registry
docker tag adaptive-lora:latest your-registry/adaptive-lora:v1.0.0
docker push your-registry/adaptive-lora:v1.0.0
```

### Step 3: Configure Environment

```yaml
# configs/serving/production.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1  # 1 worker per GPU
  timeout: 300

model:
  base_model: "meta-llama/Meta-Llama-3-8B"
  max_model_len: 4096
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 1  # Increase for multi-GPU

inference:
  max_batch_size: 32
  enable_streaming: true
  cache_enabled: true

rate_limiting:
  enabled: true
  requests_per_minute: 60
  burst_size: 10

monitoring:
  prometheus_enabled: true
  log_level: "INFO"
```

### Step 4: Deploy to Kubernetes

```bash
# 1. Create namespace and secrets
kubectl create namespace ml-inference
kubectl create secret generic adaptive-lora-secrets \
  --from-literal=hf-token=$HF_TOKEN \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  -n ml-inference

# 2. Apply configurations
kubectl apply -f k8s/

# 3. Wait for rollout
kubectl rollout status deployment/adaptive-lora-api -n ml-inference

# 4. Get external endpoint
kubectl get ingress -n ml-inference
```

### Step 5: Verify Deployment

```bash
# Health check
curl https://api.your-domain.com/health

# Test generation
curl -X POST https://api.your-domain.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain machine learning", "max_tokens": 256}'

# Check metrics
curl https://api.your-domain.com/metrics
```

---

## Monitoring & Observability

### Prometheus + Grafana Stack

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

# Apply ServiceMonitor
kubectl apply -f k8s/servicemonitor.yaml
```

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `api_request_latency_seconds_p95` | > 2s | P95 latency too high |
| `api_requests_total{status="error"}` | > 5% error rate | High error rate |
| `lora_gpu_memory_usage` | > 90% | GPU memory pressure |
| `lora_active_adapters` | < 1 | No adapters loaded |
| `lora_queue_size` | > 100 | Request queue backlog |

### Logging

```python
# Configure structured logging
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "request_id": getattr(record, 'request_id', None)
        }
        return json.dumps(log_obj)

# Send to CloudWatch/Stackdriver/ELK
```

---

## Scaling Strategies

### Horizontal Scaling (Multiple Pods)

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: adaptive-lora-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: adaptive-lora-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### Vertical Scaling (Larger GPUs)

```yaml
# For high-memory models
resources:
  requests:
    nvidia.com/gpu: "1"
    memory: "64Gi"
  limits:
    nvidia.com/gpu: "1"
    memory: "128Gi"
```

### Multi-GPU with Tensor Parallelism

```python
# For 70B+ models across 4+ GPUs
engine = VLLMInferenceEngine(
    model_name="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=4,  # Distribute across 4 GPUs
    gpu_memory_utilization=0.9
)
```

---

## Security Best Practices

### 1. API Authentication

```python
# Enable JWT authentication
from src.serving.auth import JWTAuth

auth = JWTAuth(
    secret_key=os.environ["JWT_SECRET"],
    algorithm="HS256"
)

# Protected endpoint
@app.post("/generate")
async def generate(
    request: GenerationRequest,
    user: User = Depends(auth.get_current_user)
):
    ...
```

### 2. Network Security

```yaml
# NetworkPolicy - Only allow specific ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
spec:
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

### 3. Secret Management

```bash
# Use external secret stores
# AWS Secrets Manager
kubectl apply -f external-secrets.yaml

# HashiCorp Vault
helm install vault hashicorp/vault
```

### 4. Input Validation

```python
# Validate and sanitize all inputs
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    
    @validator('prompt')
    def sanitize_prompt(cls, v):
        # Remove potential injection attempts
        return sanitize_input(v)
```

---

## Cost Optimization

### 1. Spot/Preemptible Instances (Dev/Test)

```yaml
# AWS Spot for non-production
nodeSelector:
  node.kubernetes.io/lifecycle: spot
tolerations:
- key: "spot"
  operator: "Equal"
  value: "true"
```

### 2. Request Caching

```python
# Cache frequent responses
from src.serving.inference_engine import InferenceCache

cache = InferenceCache(redis_client, ttl=3600)

# Check cache before generating
cached = cache.get(query, params)
if cached:
    return cached
```

### 3. Batch Processing

```python
# Batch requests for efficiency
async def batch_generate(requests: List[GenerationRequest]):
    # Group by similar parameters
    # Process in batches of 8-32
    results = await engine.generate_batch(prompts, config)
```

### 4. Model Quantization

```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# Reduces VRAM by ~75%
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Model too large | Use quantization or larger GPU |
| Slow startup | Model loading | Use model caching, warmup |
| High latency | Queue backlog | Scale horizontally |
| 503 errors | Pod not ready | Check liveness/readiness probes |
| Auth failures | Token expired | Refresh tokens, check secrets |

### Debug Commands

```bash
# Check pod status
kubectl describe pod <pod-name> -n ml-inference

# View logs
kubectl logs -f <pod-name> -n ml-inference

# Check GPU utilization
kubectl exec -it <pod-name> -- nvidia-smi

# Test connectivity
kubectl port-forward svc/adaptive-lora-service 8000:8000 -n ml-inference
curl localhost:8000/health
```

### Performance Profiling

```python
# Enable profiling endpoint
@app.get("/debug/profile")
async def profile():
    import torch
    return {
        "gpu_memory_allocated": torch.cuda.memory_allocated(),
        "gpu_memory_cached": torch.cuda.memory_reserved(),
        "adapters_loaded": len(engine.adapters),
        "queue_size": request_queue.current_size
    }
```

---

## Quick Start Checklist

- [ ] Configure `HF_TOKEN` for model downloads
- [ ] Set up Redis for caching/rate limiting
- [ ] Configure SSL/TLS certificates
- [ ] Set up Prometheus/Grafana monitoring
- [ ] Configure alerting (PagerDuty/Slack)
- [ ] Load test before go-live
- [ ] Set up log aggregation
- [ ] Configure backup strategy
- [ ] Document runbooks for common issues
- [ ] Set up CI/CD pipeline

---

## Support

For issues, please open a GitHub issue with:
1. Environment details (GPU, CUDA version, etc.)
2. Logs from the failing component
3. Steps to reproduce
4. Expected vs actual behavior
