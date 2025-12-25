# Architecture Overview

This document describes the architecture of the Adaptive Multi-Adapter LLM System.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API Gateway                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Rate Limit│ │ Auth JWT │ │ Cache    │ │ Metrics  │ │ Logging  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Router Network       │
                    │  ┌─────────────────────┐  │
                    │  │ BERT Encoder        │  │
                    │  │ + Gumbel-Softmax    │  │
                    │  │ + Complexity Head   │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        ▼             ▼           ▼           ▼             ▼
   ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌─────────┐
   │Reasoning│  │  Code   │ │Creative │ │Analysis │  │ Fallback│
   │ Adapter │  │ Adapter │ │ Adapter │ │ Adapter │  │  (Base) │
   │ (r=32)  │  │ (r=64)  │ │ (r=16)  │ │ (r=32)  │  │         │
   └────┬────┘  └────┬────┘ └────┬────┘ └────┬────┘  └────┬────┘
        │            │           │           │            │
        └────────────┴───────────┼───────────┴────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Base Model         │
                    │  (Llama 3 / Phi-3)      │
                    │  + QLoRA (4-bit NF4)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Self-Evaluator       │
                    │  ┌──────────────────┐   │
                    │  │ Multi-Metric     │   │
                    │  │ Uncertainty      │   │
                    │  │ LLM-as-Judge     │   │
                    │  └──────────────────┘   │
                    └─────────────────────────┘
```

## Component Details

### 1. Router Network (`src/router/`)

The router analyzes incoming queries and selects the optimal adapter.

**Components:**
- `AdapterRouter`: Main router with Gumbel-Softmax for differentiable training
- `HierarchicalRouter`: Two-stage routing (domain → adapter)
- `DynamicRouter`: Adaptive behavior with fallback mechanisms
- `RouterEnsemble`: Multiple routers for robust predictions

**Training:**
- Multi-objective loss: classification + complexity + ranking
- Curriculum learning for difficult examples
- Optional RL refinement using REINFORCE

### 2. LoRA Adapters (`src/adapters/`)

Specialized adapters for different task types.

| Adapter | Rank | Alpha | Target Modules | Use Case |
|---------|------|-------|----------------|----------|
| Reasoning | 32 | 64 | All attention + MLP | Math, logic |
| Code | 64 | 128 | All | Programming |
| Creative | 16 | 32 | Attention only | Writing |
| Analysis | 32 | 64 | All | Research |

**Features:**
- QLoRA for 4-bit quantization (75% memory reduction)
- Hot-swapping between adapters
- Adapter merging and fusion

### 3. Self-Evaluation (`src/evaluation/`)

Multi-faceted quality assessment.

**Metrics:**
- Relevance: Query-response alignment
- Coherence: Logical flow and consistency
- Completeness: Coverage of the query
- Safety: Content safety checks

**Uncertainty Quantification:**
- MC Dropout for prediction uncertainty
- Ensemble disagreement
- Temperature scaling for calibration
- Conformal prediction for coverage guarantees

### 4. Serving Infrastructure (`src/serving/`)

Production-ready API layer.

**Components:**
- FastAPI with async request handling
- Request queue with priority scheduling
- Response caching (Redis)
- Rate limiting per user/API key
- JWT authentication
- Prometheus metrics

### 5. Data Pipeline (`src/data/`)

Data preparation and augmentation.

**Components:**
- Synthetic data generation with quality scoring
- Data deduplication and cleaning
- Difficulty-based curriculum generation
- Adversarial example generation

## Deployment Architecture

### Local Development
```
Gradio UI → Local Model → Response
```

### Production (Kubernetes)
```
           ┌─────────────────────────────────────┐
           │            Ingress (nginx)          │
           └─────────────────┬───────────────────┘
                             │
           ┌─────────────────▼───────────────────┐
           │         API Service (FastAPI)       │
           │         Replicas: 2-10 (HPA)        │
           └─────────────────┬───────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
┌────▼────┐           ┌──────▼──────┐         ┌──────▼──────┐
│  Redis  │           │ Prometheus  │         │  Grafana    │
│ (Cache) │           │  (Metrics)  │         │ (Dashboard) │
└─────────┘           └─────────────┘         └─────────────┘
```

### Serverless (Modal)
```
API Gateway → Modal Function (GPU) → Response
   (Railway)    (Auto-scaling)
```

## Data Flow

1. **Request Ingestion**: API receives query with optional parameters
2. **Routing**: Router analyzes query, predicts adapter probabilities
3. **Adapter Selection**: Highest probability adapter is loaded (if not cached)
4. **Generation**: Base model + selected adapter generates response
5. **Evaluation**: Self-evaluator scores response quality
6. **Caching**: Response cached if quality above threshold
7. **Response**: Result returned with metadata (adapter used, latency, confidence)

## Monitoring & Observability

### Metrics Collected
- Request latency (p50, p95, p99)
- Adapter usage distribution
- Router accuracy
- Quality scores
- GPU utilization
- Cache hit rate

### Alerting Rules
- High latency (>5s)
- Low quality scores (<0.5)
- Error rate spike (>5%)
- GPU memory exhaustion
