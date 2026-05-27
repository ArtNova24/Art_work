"""
Historic Image Restoration Phase 2 — Step 1: Model Training
Trains 5 style classifiers on the 989-dim hybrid feature matrix:
  1. SVM with RBF kernel
  2. Random Forest (500 trees)
  3. XGBoost Classifier
  4. MLP (2-layer PyTorch)
  5. CNN End-to-End (fine-tuned ResNet-18 on raw images)

Evaluates on Val/Test and saves all trained models + performance metrics.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# Import paths from phase1 config
PHASE1_DIR = Path(__file__).parent.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from config import (
    FEATURES_DIR, ALL_CLASSES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, TOTAL_DIM
)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda")
    print(f"  CUDA GPU available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("  CUDA GPU NOT available, using CPU.")


def load_data():
    """Load pre-split numpy arrays from features/."""
    X_train = np.load(os.path.join(FEATURES_DIR, "features_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "features_val.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "features_test.npy"))

    y_train = np.load(os.path.join(FEATURES_DIR, "labels_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "labels_val.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "labels_test.npy"))

    weights = np.load(os.path.join(FEATURES_DIR, "class_weights.npy"))

    print(f"  Loaded dataset shapes:")
    print(f"    Train features : {X_train.shape}, labels: {y_train.shape}")
    print(f"    Val features   : {X_val.shape}, labels: {y_val.shape}")
    print(f"    Test features  : {X_test.shape}, labels: {y_test.shape}")
    print(f"    Class weights  : {weights.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, weights


# ── MLP Architecture ────────────────────────────────────────────────────────
class StyleMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_DIM, hidden_dim1=512, hidden_dim2=256, num_classes=len(ALL_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Dataset for PyTorch MLP ──────────────────────────────────────────────────
class HybridFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Dataset for CNN End-to-End ────────────────────────────────────────────────
class ArtImageDataset(Dataset):
    def __init__(self, indices, image_index, transform=None):
        self.indices = indices
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.image_index[real_idx]
        img_path = item['path']
        label = item['class_idx']

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback in case of corruption
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    """Compute comprehensive accuracy, macro F1, and weighted F1 metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"    [{model_name}] Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted)
    }


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n  [Model 1/5] Training SVM (RBF kernel)...")
    t0 = time.time()
    svm = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    print(f"    SVM fit in {time.time()-t0:.1f}s")

    y_pred_val = svm.predict(X_val)
    y_pred_test = svm.predict(X_test)

    val_res = evaluate_model("SVM Val", y_val, y_pred_val)
    test_res = evaluate_model("SVM Test", y_test, y_pred_test)

    # Save model
    joblib.dump(svm, os.path.join(FEATURES_DIR, "svm_classifier.pkl"))
    return svm, val_res, test_res


def train_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n  [Model 2/5] Training Random Forest (500 estimators)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    print(f"    Random Forest fit in {time.time()-t0:.1f}s")

    y_pred_val = rf.predict(X_val)
    y_pred_test = rf.predict(X_test)

    val_res = evaluate_model("RF Val", y_val, y_pred_val)
    test_res = evaluate_model("RF Test", y_test, y_pred_test)

    # Save model
    joblib.dump(rf, os.path.join(FEATURES_DIR, "rf_classifier.pkl"))
    return rf, val_res, test_res


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n  [Model 3/5] Training XGBoost Classifier...")
    t0 = time.time()
    import xgboost as xgb

    # Create XGBoost Classifier
    # Support GPU training if available
    tree_method = "hist"
    device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method=tree_method,
        device=device_arg,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Use sample weights to balance the classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"    XGBoost fit in {time.time()-t0:.1f}s")

    y_pred_val = xgb_model.predict(X_val)
    y_pred_test = xgb_model.predict(X_test)

    val_res = evaluate_model("XGB Val", y_val, y_pred_val)
    test_res = evaluate_model("XGB Test", y_test, y_pred_test)

    # Save model
    joblib.dump(xgb_model, os.path.join(FEATURES_DIR, "xgb_classifier.pkl"))
    return xgb_model, val_res, test_res


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_weights):
    print("\n  [Model 4/5] Training 2-Layer PyTorch MLP...")
    t0 = time.time()

    train_ds = HybridFeatureDataset(X_train, y_train)
    val_ds   = HybridFeatureDataset(X_val, y_val)
    test_ds  = HybridFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = StyleMLP(input_dim=TOTAL_DIM, num_classes=len(ALL_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_state = None

    epochs = 40
    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Evaluate on Val
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                logits = model(x_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best state and evaluate on Test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, os.path.join(FEATURES_DIR, "mlp_classifier.pt"))

    model.eval()
    val_preds = []
    test_preds = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            val_preds.extend(torch.argmax(model(x_b), dim=1).cpu().numpy())
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            test_preds.extend(torch.argmax(model(x_b), dim=1).cpu().numpy())

    val_res = evaluate_model("MLP Val", y_val, val_preds)
    test_res = evaluate_model("MLP Test", y_test, test_preds)
    print(f"    MLP complete in {time.time()-t0:.1f}s")
    return model, val_res, test_res


def train_cnn_end2end(y_train, y_val, y_test, class_weights):
    print("\n  [Model 5/5] Training End-to-End CNN Baseline (ResNet-18)...")
    t0 = time.time()

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    with open(index_path) as f:
        image_index = json.load(f)

    # Reconstruct exact splits used in assemble_features.py deterministically
    indices = np.arange(len(image_index))
    labels = np.array([item['class_idx'] for item in image_index], dtype=np.int32)

    from sklearn.model_selection import train_test_split
    temp_idx, test_idx = train_test_split(
        indices, test_size=TEST_RATIO, stratify=labels, random_state=RANDOM_SEED
    )
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_ratio_adjusted, stratify=labels[temp_idx], random_state=RANDOM_SEED
    )

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ArtImageDataset(train_idx, image_index, train_transform)
    val_ds   = ArtImageDataset(val_idx, image_index, eval_transform)
    test_ds  = ArtImageDataset(test_idx, image_index, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Pretrained ResNet-18
    # Using torchvision weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(ALL_CLASSES))
    model = model.to(device)

    checkpoint_path = os.path.join(FEATURES_DIR, "cnn_end2end_classifier.pt")
    if os.path.exists(checkpoint_path):
        print(f"    Found existing CNN checkpoint {checkpoint_path}. Loading weights for evaluation...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            val_preds = []
            test_preds = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    val_preds.extend(torch.argmax(model(imgs), dim=1).cpu().numpy())
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    test_preds.extend(torch.argmax(model(imgs), dim=1).cpu().numpy())

            val_res = evaluate_model("CNN-E2E Val", y_val, val_preds)
            test_res = evaluate_model("CNN-E2E Test", y_test, test_preds)
            print(f"    CNN End-to-End loaded and evaluated in {time.time()-t0:.1f}s")
            return model, val_res, test_res
        except Exception as e:
            print(f"    Error loading CNN checkpoint: {e}. Falling back to training...")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_state = None

    epochs = 5  # Quick fine-tuning over 5 epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        # Evaluate on Val
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(lbls.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"      Epoch {epoch+1}/{epochs} — Train Loss: {train_loss/len(train_ds):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best state and evaluate on Test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, os.path.join(FEATURES_DIR, "cnn_end2end_classifier.pt"))

    model.eval()
    val_preds = []
    test_preds = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            val_preds.extend(torch.argmax(model(imgs), dim=1).cpu().numpy())
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            test_preds.extend(torch.argmax(model(imgs), dim=1).cpu().numpy())

    val_res = evaluate_model("CNN-E2E Val", y_val, val_preds)
    test_res = evaluate_model("CNN-E2E Test", y_test, test_preds)
    print(f"    CNN End-to-End complete in {time.time()-t0:.1f}s")
    return model, val_res, test_res


def main():
    print("\n" + "="*60)
    print("  Historic Image Restoration Phase 2 — Step 1: Model Training Engine")
    print("="*60)

    print("="*60)

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_data()

    # 2. Train classifiers
    svm, svm_val, svm_test = train_svm(X_train, y_train, X_val, y_val, X_test, y_test)
    rf, rf_val, rf_test   = train_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    xgb_m, xgb_val, xgb_test = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    mlp, mlp_val, mlp_test = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_weights)
    cnn, cnn_val, cnn_test = train_cnn_end2end(y_train, y_val, y_test, class_weights)

    # 3. Save all validation and test metrics
    metrics = {
        "svm": {"val": svm_val, "test": svm_test},
        "rf":  {"val": rf_val,  "test": rf_test},
        "xgb": {"val": xgb_val, "test": xgb_test},
        "mlp": {"val": mlp_val, "test": mlp_test},
        "cnn": {"val": cnn_val, "test": cnn_test}
    }

    out_metrics_path = os.path.join(FEATURES_DIR, "classifier_metrics.json")
    with open(out_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  [OK] Save metrics -> {out_metrics_path}")

    # Determine best classifier (based on Test Macro F1)
    best_model_name = None
    best_f1 = 0.0
    for name, res in metrics.items():
        f1 = res["test"]["f1_macro"]
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n  [BEST] Best performing model: {best_model_name.upper()} (Test Macro F1: {best_f1:.4f})")

    # Serialize best classical model (SVM, RF, or XGBoost) as style_predictor.pkl
    classical_names = ["svm", "rf", "xgb"]
    best_classical_name = "xgb"
    best_classical_f1 = 0.0
    for name in classical_names:
        f1 = metrics[name]["test"]["f1_macro"]
        if f1 > best_classical_f1:
            best_classical_f1 = f1
            best_classical_name = name

    print(f"  [BEST CLASSICAL] Best classical model: {best_classical_name.upper()} (Test Macro F1: {best_classical_f1:.4f})")

    if best_classical_name == "svm":
        best_classical = svm
    elif best_classical_name == "rf":
        best_classical = rf
    else:
        best_classical = xgb_m

    predictor_path = os.path.join(FEATURES_DIR, "style_predictor.pkl")
    joblib.dump(best_classical, predictor_path)
    print(f"  [OK] Serialized best classical model style oracle -> {predictor_path}")


if __name__ == "__main__":
    main()
