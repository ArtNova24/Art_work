"""
Historic Image Restoration Phase 4 -- Master Orchestrator
Supports two modes via CLI:
  --evaluate   Runs the full quantitative test-set evaluation suite.
  --demo       Launches the interactive Gradio web application.
  --dry-run    (For --evaluate only) Runs on a single mini-batch for fast verification.
All prints and comments are kept strictly in ASCII.
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
from pathlib import Path

# Make all project modules importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase3"))

from phase4.config import RANDOM_SEED

def set_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_evaluate(dry_run=False):
    from phase4.evaluator import Phase4Evaluator

    print("============================================================")
    print("  Historic Image Restoration Phase 4 -- Evaluation & Metrics Suite")
    print("============================================================")

    set_seeds()
    evaluator = Phase4Evaluator(dry_run=dry_run)
    results   = evaluator.evaluate_all()

    print("\n  ====== Final Metric Summary ======")
    for method, m in results.items():
        fid_str = f"{m['fid']:.2f}" if m['fid'] < 900 else "N/A"
        print(f"  [{method:22s}]  SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}dB  "
              f"FID={fid_str}  StyleFidelity={m['style_fidelity']*100:.1f}%")

    print("\n  ============================================================")
    if dry_run:
        print("  [DRY RUN] Phase 4 evaluation dry-run completed successfully.")
    else:
        print("  PHASE 4 STATUS: SUCCESS")
    print("  ============================================================")


def run_demo():
    print("============================================================")
    print("  Historic Image Restoration Phase 4 -- Interactive Gradio Demo")
    print("============================================================")
    print("  Launching on http://localhost:7860 ...")
    print("  Press Ctrl+C to stop the server.")
    print("============================================================")

    # Import here so model loading happens inside run_demo()
    try:
        from phase4.gradio_app import build_app
        app, theme, css = build_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            theme=theme,
            css=css
        )
    except ImportError as e:
        print(f"\n  [ERROR] Gradio is not installed: {e}")
        print("  Please install it with:")
        print("      venv\\Scripts\\pip install gradio")
        sys.exit(1)


def main():
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    parser = argparse.ArgumentParser(
        description="Historic Image Restoration Phase 4 -- Evaluation and Demo Orchestrator"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full quantitative evaluation on test partition."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch interactive Gradio web demo."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(For --evaluate) Process only one mini-batch for fast shape/import verification."
    )

    args = parser.parse_args()

    if not args.evaluate and not args.demo:
        parser.print_help()
        print("\n  Hint: Use --evaluate to run the full metrics suite, or --demo to launch the web app.")
        sys.exit(0)

    if args.evaluate:
        run_evaluate(dry_run=args.dry_run)

    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
