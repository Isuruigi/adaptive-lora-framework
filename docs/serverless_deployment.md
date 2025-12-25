# Cost-Effective Serverless Production Deployment

This guide covers deploying the Adaptive LoRA Framework using **Modal** for serverless GPU and **Railway/Render** for the API gateway.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│         PRODUCTION SYSTEM - SERVERLESS ARCHITECTURE            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Users (1000+ req/day)                                          │
│     │                                                            │
│     ▼                                                            │
│  Cloudflare (CDN + DDoS + Rate Limiting) - FREE                │
│     │                                                            │
│     ▼                                                            │
│  Railway/Render API Gateway ($5-10/mo)                         │
│  ├─ Authentication (API keys/JWT)                               │
│  ├─ Rate limiting (in-memory or Redis)                          │
│  ├─ Response caching                                            │
│  └─ Circuit breaker                                             │
│     │                                                            │
│     ▼                                                            │
│  Modal Serverless GPU ($20-50/mo)                              │
│  ├─ Auto-scaling (0 → 100 instances)                           │
│  ├─ Health checks                                               │
│  └─ T4/A10G GPUs on demand                                      │
│     │                                                            │
│     ▼                                                            │
│  Storage & Monitoring                                           │
│  ├─ Redis (Upstash) - FREE→$10/mo                             │
│  ├─ Sentry (Error tracking) - FREE                             │
│  └─ Better Stack (Logs) - FREE                                  │
│                                                                  │
│  💰 Total: $30-70/month (handles 100K+ requests)                │
│  ⚡ SLA: 99.5% uptime, <500ms P99 latency                       │
└─────────────────────────────────────────────────────────────────┘
```

## Cost Breakdown

| Service | Free Tier | Paid Tier | Notes |
|---------|-----------|-----------|-------|
| Modal (GPU) | $30 credits/mo | ~$0.60/hr T4 | Pay per second |
| Railway | $5/mo | $5-10/mo | API Gateway |
| Upstash Redis | 10K commands/day | $10/mo | Caching + Rate Limit |
| Sentry | 5K errors/mo | FREE | Error tracking |
| Cloudflare | Unlimited | FREE | CDN + DDoS |

**Total: $30-70/month for 100K+ requests**

---

## Step 1: Deploy Modal Serverless GPU

### 1.1 Install Modal CLI

```bash
pip install modal
modal token new
```

### 1.2 Create HuggingFace Secret

```bash
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 1.3 Test Locally

```bash
cd deploy
modal run modal_inference.py
```

### 1.4 Deploy to Production

```bash
modal deploy modal_inference.py
```

You'll get endpoints like:
- `https://your-app--api-generate.modal.run`
- `https://your-app--health.modal.run`
- `https://your-app--metrics.modal.run`

---

## Step 2: Deploy API Gateway

### Option A: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Create project
railway init

# Set environment variables
railway variables set MODAL_ENDPOINT=https://your-app--api-generate.modal.run
railway variables set RATE_LIMIT=60
railway variables set JWT_SECRET=your-secret-key

# Deploy
railway up
```

### Option B: Render

1. Create new Web Service on render.com
2. Connect your GitHub repo
3. Set build command: `pip install -r deploy/requirements-gateway.txt`
4. Set start command: `uvicorn deploy.api_gateway:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in dashboard

### Option C: Docker (Any cloud)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY deploy/requirements-gateway.txt .
RUN pip install -r requirements-gateway.txt

COPY deploy/api_gateway.py .

CMD ["uvicorn", "api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 3: Configure Cloudflare (Optional but Recommended)

1. Add your domain to Cloudflare (free)
2. Point DNS to your Railway/Render URL
3. Enable:
   - **SSL/TLS**: Full (strict)
   - **Rate Limiting**: 100 req/10s per IP
   - **DDoS Protection**: Enabled
   - **Caching**: Cache API responses

---

## Step 4: Add Monitoring

### Sentry (Error Tracking)

```bash
# Set in both Modal and API Gateway
SENTRY_DSN=https://xxx@sentry.io/xxx
```

### Upstash Redis (Caching)

```bash
# Get free Redis from upstash.com
REDIS_URL=redis://default:xxx@xxx.upstash.io:6379
```

### Better Stack (Logs)

Connect via Railway/Render log drain.

---

## API Usage

### Generate Text

```bash
curl -X POST https://api.yourdomain.com/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Explain machine learning",
    "adapter": "reasoning",
    "max_tokens": 200
  }'
```

### Health Check

```bash
curl https://api.yourdomain.com/health
```

### Metrics

```bash
curl https://api.yourdomain.com/metrics
```

---

## Scaling

### Modal Auto-Scaling

Modal automatically scales from 0 to 100+ containers based on demand.

```python
@app.cls(
    keep_warm=2,              # Keep 2 containers warm
    allow_concurrent_inputs=50,  # 50 concurrent requests per container
)
```

### Cost Optimization

1. **Use T4 GPUs** (~$0.60/hr vs $1.10/hr for A10G)
2. **Enable caching** (reduces GPU calls by 30-50%)
3. **Batch similar requests** (higher GPU utilization)
4. **Set container_idle_timeout** (180s is good balance)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cold start slow | Increase `keep_warm` value |
| GPU OOM | Use smaller model or 4-bit quantization |
| Rate limited | Increase limits or add caching |
| Circuit breaker open | Check Modal health endpoint |

---

## Production Checklist

- [ ] HuggingFace token configured
- [ ] Modal deployed and tested
- [ ] API Gateway deployed
- [ ] Rate limiting enabled
- [ ] Caching configured
- [ ] Sentry error tracking
- [ ] Health checks passing
- [ ] SSL/TLS enabled
- [ ] API keys distributed
- [ ] Monitoring dashboard set up
