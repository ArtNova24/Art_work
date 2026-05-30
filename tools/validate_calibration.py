"""
Calibration & Reliability Diagram Utility

Usage examples:

# 1) From saved prediction arrays (fast):
python tools/validate_calibration.py --mode from_files \
    --probs_np features/pred_probs.npy \
    --labels_np features/true_labels.npy \
    --out_dir diagnostics/calibration

# 2) Run end-to-end (reconstruct + classify) - requires Phase 4 checkpoints:
python tools/validate_calibration.py --mode reconstruct \
    --recon_method conditioned \
    --out_dir diagnostics/calibration

Outputs:
- PNG reliability diagram at <out_dir>/reliability_diagram.png
- JSON summary at <out_dir>/calibration_summary.json

This script computes multiclass ECE using the "top-label" method
(i.e., probability assigned to the predicted class vs correctness), and
also computes per-class ECE and plots a reliability diagram for the
predicted-class probabilities.
"""

import argparse
import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def expected_calibration_error(probs, labels, n_bins=15):
    """Compute ECE for predicted-class probabilities.
    probs: (N,) predicted probabilities for the predicted class
    labels: (N,) binary correctness (1 if predicted class == true label)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs > lo) & (probs <= hi)
        count = mask.sum()
        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_counts.append(0)
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (count / len(labels)) * abs(conf - acc)
        bin_acc.append(float(acc))
        bin_conf.append(float(conf))
        bin_counts.append(int(count))
    return ece, bins, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def load_from_files(probs_path, labels_path):
    probs = np.load(probs_path)
    labels = np.load(labels_path)
    return probs, labels


def probs_and_labels_from_multiclass_preds(preds_probs, true_labels):
    # preds_probs: (N, C)
    pred_cls = preds_probs.argmax(axis=1)
    pred_conf = preds_probs[np.arange(len(preds_probs)), pred_cls]
    correct = (pred_cls == true_labels).astype(int)
    return pred_conf, correct, pred_cls


def plot_reliability(bin_conf, bin_acc, bins, out_path, title='Reliability Diagram'):
    centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], 'k:', label='Perfectly calibrated')
    plt.plot(centers, bin_acc, 's-', label='Accuracy')
    plt.bar(centers, (bin_conf - bin_acc), width=0.025, alpha=0.3, color='C1', label='Confidence - Accuracy')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_from_files(probs_np, labels_np, out_dir, n_bins=15):
    os.makedirs(out_dir, exist_ok=True)
    probs_array = np.load(probs_np)
    labels_array = np.load(labels_np)
    if probs_array.ndim == 1:
        # already predicted-class probs and binary labels
        pred_conf = probs_array
        correct = labels_array.astype(int)
    else:
        pred_conf, correct, _ = probs_and_labels_from_multiclass_preds(probs_array, labels_array)
    ece, bins, bin_acc, bin_conf, bin_counts = expected_calibration_error(pred_conf, correct, n_bins=n_bins)
    summary = {
        'mode': 'from_files',
        'ece': float(ece),
        'n_bins': n_bins,
        'bin_counts': bin_counts.tolist()
    }
    plot_reliability(bin_conf, bin_acc, bins, os.path.join(out_dir, 'reliability_diagram.png'))
    with open(os.path.join(out_dir, 'calibration_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote reliability diagram and summary to {out_dir}')
    return summary


def run_reconstruct_and_evaluate(out_dir, n_bins=15, batch_size=16):
    """Run Phase 4 reconstruction pipeline and evaluate calibration on reconstructions.
    This requires Phase 3 checkpoints and Phase 2 style predictor to exist in `features/`.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Lazy imports to avoid heavy dependencies unless this mode is used
    from phase4.evaluator import Phase4Evaluator
    import torch
    import torchvision.transforms as T

    evaluator = Phase4Evaluator(dry_run=False)
    # Prepare arrays to collect
    all_pred_probs = []
    all_true = []

    # Use evaluator to reconstruct images and compute hybrid features for reconstructions
    # We'll process the test dataset via evaluator.test_loader
    for batch in evaluator.test_loader:
        # Depending on dataset structure: StyleJEPAImageDataset yields (img, style_vec, label)
        imgs, style_vecs, labels = batch
        # For simplicity run evaluator._reconstruct on batch elements
        for i in range(imgs.shape[0]):
            img = imgs[i]
            sv = style_vecs[i].numpy()
            mask = evaluator.mask_gen.collate_masks(1)[0]
            recon = evaluator._reconstruct(img, sv, mask, sampling_mode='guided')
            # extract hybrid features from recon
            feat = evaluator.extract_hybrid(recon)
            # apply preprocessing for classifier
            if evaluator.block_scalers is not None and evaluator.shap_weights is not None:
                # block_scalers is list or dict per block
                X_blocks = []
                # Assuming block order: glcm(20), lbp(256), color(201), cnn(512)
                X_blocks.append(feat[:20])
                X_blocks.append(feat[20:276])
                X_blocks.append(feat[276:477])
                X_blocks.append(feat[477:])
                X_scaled = np.concatenate([sc.transform(b.reshape(1,-1)) for sc,b in zip(evaluator.block_scalers, X_blocks)], axis=1)
                X_scaled = X_scaled.flatten()
                X_scaled = X_scaled * evaluator.shap_weights
                X = X_scaled.reshape(1,-1)
            else:
                X = feat.reshape(1,-1)
            probs = evaluator.classifier.predict_proba(X)  # (1, C)
            all_pred_probs.append(probs[0])
            all_true.append(int(labels[i].item()))

    probs_arr = np.stack(all_pred_probs, axis=0)
    true_arr = np.array(all_true, dtype=int)
    # Save arrays
    np.save(os.path.join(out_dir, 'pred_probs.npy'), probs_arr)
    np.save(os.path.join(out_dir, 'true_labels.npy'), true_arr)
    print(f'Saved predictions to {out_dir}')
    # Delegate to run_from_files for plotting
    return run_from_files(os.path.join(out_dir, 'pred_probs.npy'), os.path.join(out_dir, 'true_labels.npy'), out_dir, n_bins=n_bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['from_files', 'reconstruct'], required=True)
    parser.add_argument('--probs_np', type=str, default=None)
    parser.add_argument('--labels_np', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_bins', type=int, default=15)
    args = parser.parse_args()

    if args.mode == 'from_files':
        if args.probs_np is None or args.labels_np is None:
            raise SystemExit('For from_files mode, provide --probs_np and --labels_np')
        summary = run_from_files(args.probs_np, args.labels_np, args.out_dir, n_bins=args.n_bins)
        print(summary)
    else:
        summary = run_reconstruct_and_evaluate(args.out_dir, n_bins=args.n_bins)
        print(summary)

if __name__ == '__main__':
    main()
