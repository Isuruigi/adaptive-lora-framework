# Colab Pro Production Deployment

Use Google Colab Pro/Pro+ as your GPU backend for the lowest cost.

## Cost Comparison

| Option | Monthly Cost | GPU | Best For |
|--------|--------------|-----|----------|
| Colab Pro | $10/mo | T4 (limited hrs) | Light usage |
| Colab Pro+ | $50/mo | A100/V100 | Production |
| Modal | $30-70/mo | T4/A10G | Serverless |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            COLAB PRO PRODUCTION SETUP                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Railway API Gateway ($5/mo)                            │
│  ├─ Rate limiting                                        │
│  ├─ Caching                                              │
│  └─ Circuit breaker                                      │
│          │                                               │
│          ▼                                               │
│  ngrok Tunnel (free or $8/mo for reserved domain)      │
│          │                                               │
│          ▼                                               │
│  Google Colab Pro ($10-50/mo)                           │
│  ├─ FastAPI server                                       │
│  ├─ Llama 3.2 3B (4-bit)                                │
│  └─ T4/A100 GPU                                          │
│                                                          │
│  💰 Total: $15-65/month                                 │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Open Colab Notebook

1. Upload `notebooks/colab_production_server.ipynb` to Google Drive
2. Open with Google Colab
3. Go to Runtime → Change runtime type → T4 GPU

### Step 2: Configure

Set these values in the notebook:
```python
HF_TOKEN = "hf_your_token"  # From huggingface.co
NGROK_TOKEN = "your_ngrok_token"  # From ngrok.com (free)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
```

### Step 3: Run All Cells

1. Click Runtime → Run all
2. Wait for model to load (~2-3 min)
3. Copy the ngrok URL shown:
   ```
   📡 Public URL: https://xxxx-xx-xx-xxx-xx.ngrok-free.app
   ```

### Step 4: Connect API Gateway

```bash
# Railway
railway variables set MODAL_ENDPOINT=https://xxxx.ngrok-free.app
railway up

# Or local testing
export MODAL_ENDPOINT=https://xxxx.ngrok-free.app
python deploy/api_gateway.py
```

### Step 5: Test

```bash
curl -X POST https://your-gateway.railway.app/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI", "max_tokens": 100}'
```

## Keeping Colab Running

⚠️ Colab disconnects after idle timeout. Solutions:

1. **Browser Extension**: Use "Colab Alive" extension
2. **Reserved ngrok Domain**: $8/mo for stable URL
3. **Cloudflare Tunnel**: Free alternative to ngrok
4. **Pro+**: Longer runtime limits

## For True Production

If you need 24/7 uptime, consider:
- **Modal**: Serverless, auto-scales, ~$30-70/mo
- **RunPod**: Dedicated GPU, ~$0.20/hr
- **Lambda Labs**: A100 instances, ~$1.10/hr
- **Self-hosted**: Buy RTX 4090, one-time $1600

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/generate` | POST | Generate text |
| `/metrics` | GET | Prometheus metrics |
