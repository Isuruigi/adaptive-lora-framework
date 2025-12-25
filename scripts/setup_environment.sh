#!/bin/bash
# Setup script for Adaptive LoRA Framework
# This script installs all dependencies and configures the environment

set -e

echo "=================================="
echo "Adaptive LoRA Framework Setup"
echo "=================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention (optional, requires CUDA)
echo "Installing Flash Attention (optional)..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention installation skipped (requires CUDA 11.6+)"

# Install main dependencies
echo "Installing main dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,synthetic}
mkdir -p outputs/{checkpoints,adapters,logs}
mkdir -p experiments
mkdir -p models

# Setup environment file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠ Please edit .env file with your API keys"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || true

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python scripts/download_models.py"
echo "3. Start training: python -m src.adapters.lora_trainer"
echo ""
