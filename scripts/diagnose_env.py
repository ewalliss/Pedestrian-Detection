"""Environment diagnostic script — run this FIRST on Windows if you get DLL errors.

Usage:
    python scripts/diagnose_env.py

Checks Python version, PyTorch installation, CUDA availability, and DLL loading.
"""

import sys
import platform

print("=" * 60)
print("  Environment Diagnostic")
print("=" * 60)

# ── Python ────────────────────────────────────────────────────
print(f"\nPython  : {sys.version}")
print(f"Platform: {platform.platform()}")

if sys.version_info < (3, 10):
    print("  [WARN] Python < 3.10 — recommend Python 3.13")
else:
    print("  [OK] Python version")

# ── PyTorch ───────────────────────────────────────────────────
print("\n── PyTorch ──────────────────────────────────────────────")
try:
    import torch
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version  : {torch.version.cuda}")
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute cap   : sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print("  [WARN] GPU older than Volta (sm_70) — FP16 may be slow")
        else:
            print("  [OK] GPU supports FP16")
    else:
        print("  [WARN] CUDA not available — will run on CPU (slow)")
        print("         Fix: reinstall PyTorch with CUDA support")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("  [OK] torch imports successfully")
except ImportError as e:
    print(f"  [FAIL] torch import failed: {e}")
    print("\n  Fix:")
    print("  1. Install Visual C++ Redistributable 2015-2022:")
    print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("  2. Reinstall PyTorch:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)
except OSError as e:
    print(f"  [FAIL] DLL load error: {e}")
    print("\n  Fix — try in order:")
    print("  1. Install Visual C++ Redistributable (MOST COMMON FIX):")
    print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("     Then REBOOT and try again.")
    print()
    print("  2. Reinstall PyTorch for your CUDA version:")
    print("     Check CUDA: run  nvidia-smi  in a new CMD window")
    print("     For CUDA 11.8 (V100 recommended):")
    print("       pip uninstall torch torchvision torchaudio -y")
    print("       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("     For CUDA 12.1:")
    print("       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("  3. Test CPU-only (no GPU) to isolate the issue:")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ── FP16 smoke test ───────────────────────────────────────────
print("\n── FP16 test ────────────────────────────────────────────")
try:
    if torch.cuda.is_available():
        x = torch.randn(4, 4, dtype=torch.float16, device="cuda")
        y = x @ x.T
        print(f"  [OK] FP16 matmul on GPU: {y.shape}")
    else:
        x = torch.randn(4, 4)
        print(f"  [OK] FP32 tensor on CPU: {x.shape}")
        print("  [SKIP] FP16 GPU test — no CUDA")
except Exception as e:
    print(f"  [FAIL] {e}")

# ── transformers ──────────────────────────────────────────────
print("\n── HuggingFace transformers ─────────────────────────────")
try:
    import transformers
    print(f"  [OK] transformers=={transformers.__version__}")
except ImportError:
    print("  [FAIL] transformers not installed")
    print("         pip install transformers")

# ── peft ──────────────────────────────────────────────────────
print("\n── PEFT (for LoRA) ──────────────────────────────────────")
try:
    import peft
    print(f"  [OK] peft=={peft.__version__}")
except ImportError:
    print("  [WARN] peft not installed (only needed for LoRA fine-tuning)")
    print("         pip install peft")

# ── Dataset check ─────────────────────────────────────────────
print("\n── Market-1501 dataset ──────────────────────────────────")
from pathlib import Path
market_path = Path(r"C:\Users\Administrator\Downloads\pedestrian\Market-1501-v15.09.15")
if not market_path.exists():
    # try common locations
    candidates = [
        Path(r"C:\Users\Administrator\Downloads\pedestrian\Market-1501-v15.09.15"),
        Path(r"D:\pedestrian\Market-1501-v15.09.15"),
        Path(r"C:\datasets\Market-1501-v15.09.15"),
    ]
    market_path = next((p for p in candidates if p.exists()), None)

if market_path and market_path.exists():
    train_dir = market_path / "bounding_box_train"
    n_imgs = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    print(f"  [OK] Found at {market_path}")
    print(f"       bounding_box_train: {n_imgs} images")
else:
    print("  [WARN] Market-1501 not found at default locations")
    print("         Pass the path with: --market1501-root D:\\path\\to\\Market-1501-v15.09.15")

print("\n" + "=" * 60)
print("  Diagnostic complete")
print("=" * 60)
