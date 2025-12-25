# Railway Setup Guide

## Step 1: Install Railway CLI

### Windows (PowerShell):
```powershell
iwr https://railway.app/install.ps1 | iex
```

### Alternative (using npm):
```bash
npm install -g @railway/cli
```

### Alternative (using Homebrew on WSL):
```bash
brew install railway
```

## Step 2: Login to Railway

```bash
railway login
```

This will open your browser for authentication.

## Step 3: Initialize Railway Project

```bash
cd "D:\Projects\Multi agent lora\adaptive-lora-framework"
railway init
```

Follow the prompts to:
- Create a new project OR link to existing project
- Select your service

## Step 4: Set Environment Variable

Once your Colab server is running and you have the ngrok URL, run:

```bash
railway variables set MODAL_ENDPOINT=https://xxxx.ngrok-free.app
```

Replace `https://xxxx.ngrok-free.app` with your actual ngrok URL from the notebook output.

## Step 5: Deploy (Optional)

If you want to deploy code to Railway:

```bash
railway up
```

## Summary

After Colab is running:
1. Copy the ngrok URL from notebook output
2. Run: `railway variables set MODAL_ENDPOINT=<your-ngrok-url>`
3. Your Railway app can now connect to your Colab server!

## Troubleshooting

**Railway command not found:**
- Make sure you restart your terminal after installation
- Try the npm installation method if PowerShell method fails

**Authentication issues:**
- Make sure you have a Railway account at https://railway.app
- Check that your browser allows popups for the login
