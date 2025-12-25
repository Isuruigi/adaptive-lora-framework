# Deploy to HuggingFace Spaces

Deploy your Adaptive LoRA demo to HuggingFace Spaces in minutes!

## Quick Deploy (5 minutes)

### Step 1: Create a HuggingFace Account
- Go to [huggingface.co](https://huggingface.co) and sign up (free)

### Step 2: Create a New Space
1. Click your profile → **New Space**
2. Fill in:
   - **Space name**: `adaptive-lora-demo` (or your choice)
   - **License**: MIT
   - **SDK**: Gradio
   - **Visibility**: Public
3. Click **Create Space**

### Step 3: Upload Files
Upload these files to your Space (via web UI or git):

```
app.py                    # Main Gradio app
requirements_space.txt    # Rename to requirements.txt in Space
SPACE_README.md          # Rename to README.md in Space
```

Or use git:
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/adaptive-lora-demo
cd adaptive-lora-demo

# Copy files
cp /path/to/adaptive-lora-framework/app.py .
cp /path/to/adaptive-lora-framework/requirements_space.txt requirements.txt
cp /path/to/adaptive-lora-framework/SPACE_README.md README.md

# Push
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Add Secrets
1. Go to Space Settings → **Repository secrets**
2. Add: `GROQ_API_KEY` = your Groq API key
   - Get free key at [console.groq.com](https://console.groq.com)

### Step 5: Done! 🎉
Your Space will build automatically. Access at:
```
https://huggingface.co/spaces/YOUR_USERNAME/adaptive-lora-demo
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check requirements.txt has valid packages |
| API errors | Verify GROQ_API_KEY secret is set |
| Slow response | Free tier has rate limits, wait and retry |

## Updating Your Space

To update after changes:
```bash
cd adaptive-lora-demo
git pull
# make changes
git add .
git commit -m "Update"
git push
```

## Features Shown in Demo

- ✅ Auto-routing between 4 adapters
- ✅ Real-time generation via Groq
- ✅ Temperature/max tokens controls
- ✅ Example prompts
