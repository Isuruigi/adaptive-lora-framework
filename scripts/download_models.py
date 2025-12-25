"""
Download and cache models for the framework.

Usage:
    python scripts/download_models.py --model meta-llama/Llama-3-8B
    python scripts/download_models.py --all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


# Load environment variables
load_dotenv()


# Default models to download
DEFAULT_MODELS = [
    "meta-llama/Llama-3-8B",
    "prajjwal1/bert-tiny",  # Router encoder
]

# Router encoder models (lightweight)
ROUTER_ENCODERS = [
    "prajjwal1/bert-tiny",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def get_cache_dir() -> Path:
    """Get HuggingFace cache directory."""
    cache_dir = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    return Path(cache_dir)


def download_model(
    model_name: str,
    cache_dir: Path,
    token: str | None = None
) -> None:
    """Download a model from HuggingFace Hub.

    Args:
        model_name: Model identifier on HuggingFace Hub.
        cache_dir: Local cache directory.
        token: HuggingFace API token.
    """
    print(f"\n📥 Downloading: {model_name}")

    try:
        # Download full model snapshot
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            token=token,
            resume_download=True,
        )

        print(f"✅ Downloaded to: {local_path}")

    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        raise


def download_tokenizer(
    model_name: str,
    cache_dir: Path,
    token: str | None = None
) -> None:
    """Download only the tokenizer files.

    Args:
        model_name: Model identifier.
        cache_dir: Cache directory.
        token: API token.
    """
    print(f"\n📥 Downloading tokenizer: {model_name}")

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for filename in tokenizer_files:
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                cache_dir=str(cache_dir),
                token=token,
            )
        except Exception:
            pass  # Some files may not exist


def verify_download(model_name: str, cache_dir: Path) -> bool:
    """Verify that a model was downloaded successfully.

    Args:
        model_name: Model identifier.
        cache_dir: Cache directory.

    Returns:
        True if model exists.
    """
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
        )
        return True
    except Exception:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download models for Adaptive LoRA Framework")

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all default models"
    )
    parser.add_argument(
        "--routers",
        action="store_true",
        help="Download router encoder models"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads"
    )

    args = parser.parse_args()

    # Get HuggingFace token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  HF_TOKEN not set. Some models may not be accessible.")
        print("   Set HF_TOKEN in your environment or .env file.")

    # Get cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache directory: {cache_dir}")

    # Determine which models to download
    models_to_download: List[str] = []

    if args.model:
        models_to_download.append(args.model)
    elif args.all:
        models_to_download.extend(DEFAULT_MODELS)
    elif args.routers:
        models_to_download.extend(ROUTER_ENCODERS)
    else:
        # Default: download router encoders (lightweight)
        models_to_download.extend(ROUTER_ENCODERS)

    # Verify mode
    if args.verify:
        print("\n🔍 Verifying downloads...")
        for model in models_to_download:
            if verify_download(model, cache_dir):
                print(f"  ✅ {model}")
            else:
                print(f"  ❌ {model}")
        return

    # Download models
    print(f"\n📦 Downloading {len(models_to_download)} model(s)...")

    for model in tqdm(models_to_download, desc="Models"):
        try:
            download_model(model, cache_dir, token)
        except Exception as e:
            print(f"Error downloading {model}: {e}")
            continue

    print("\n✨ Download complete!")

    # Final verification
    print("\n🔍 Verifying downloads...")
    for model in models_to_download:
        if verify_download(model, cache_dir):
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model} - May need re-download")


if __name__ == "__main__":
    main()
