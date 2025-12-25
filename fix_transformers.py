"""
Force upgrade transformers to fix Llama 3.2 compatibility
Run this script, then restart your Python kernel
"""

import subprocess
import sys

def upgrade_transformers():
    print("=" * 60)
    print("🔧 Fixing Transformers Version")
    print("=" * 60)

    # Uninstall old version
    print("\n1️⃣ Uninstalling old transformers...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"], check=True)

    # Install new version
    print("\n2️⃣ Installing transformers 4.45.0...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "transformers==4.45.0"
    ], check=True)

    print("\n" + "=" * 60)
    print("✅ Transformers upgraded to 4.45.0")
    print("⚠️  RESTART YOUR KERNEL/RUNTIME NOW!")
    print("=" * 60)

if __name__ == "__main__":
    upgrade_transformers()
