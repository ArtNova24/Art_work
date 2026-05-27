"""
Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Reconstruction Orchestrator.
Parses options, enforces deterministic random seeding, and triggers training.
All prints and comments are kept strictly in ASCII.
"""
import os
import sys
import argparse
import random
import numpy as np
import torch

# Add project root to system path for correct imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Central config paths
from phase3.config import RANDOM_SEED
from phase3.train_jepa import run_training

def main():
    parser = argparse.ArgumentParser(description="Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Reconstruction Suite")
    parser.add_argument("--dry-run", action="store_true", help="Execute 2 training epochs over a mini-dataset for fast shape & gradient flow verification.")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count.")
    args = parser.parse_args()

    # On Windows, force UTF-8 encoding via environment variable.
    # NOTE: Do NOT wrap sys.stdout with TextIOWrapper here — it re-buffers
    # output and prevents real-time log streaming with `python -u`.
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print("============================================================")
    print("  Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Model")
    print("============================================================")

    # 1. Enforce determinism and set random seeds
    print("  Setting seeds for reproducibility...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"    Set seeds to {RANDOM_SEED} on CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"    Set seeds to {RANDOM_SEED} on CPU.")

    # 2. Trigger orchestrator training
    run_training(dry_run=args.dry_run, num_epochs=args.epochs)

    print("============================================================")
    if args.dry_run:
        print("  [DRY RUN STATUS] Verification Dry-Run successfully completed!")
    else:
        print("  PHASE 3 STATUS: SUCCESS")
    print("============================================================")

if __name__ == "__main__":
    main()
