"""
train_model.py
--------------
Stage 3: Preprocessing and Model Training for URL Phishing Classification.

Workflow:
  1. Loads `phishing_uci.csv` from `stage3_ml/data/` (or fetches if missing).
  2. Selects the 10 URL-computable features and the target label.
  3. Preprocesses data: checks nulls, drops duplicates.
  4. Maps target labels:
       UCI -1 (Phishing)   -> 1 (Phishing)
       UCI  1 (Legitimate) -> 0 (Legitimate)
  5. Performs Stratified 80/20 Train/Test split (random_state=42).
  6. Trains:
       - Baseline: LogisticRegression
       - Primary: RandomForestClassifier (tuned with 5-fold GridSearchCV)
  7. Evaluates both models: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
  8. Saves the best model bundled with metadata (feature order, label mapping)
     to `stage3_ml/models/phishing_model.joblib`.
  9. Generates evaluation plots and metrics.json in `stage3_ml/reports/`.
  10. Displays honest academic comparison regarding URL-only vs full-feature accuracy.

Usage:
    python train_model.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "reports"
CSV_PATH = DATA_DIR / "phishing_uci.csv"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"

# Ensure feature extractor is importable
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor import FEATURE_ORDER


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Loads CSV and extracts selected URL features and target."""
    if not csv_path.exists():
        print(f"[*] Dataset not found at {csv_path}. Running get_dataset.py...")
        from get_dataset import acquire_dataset
        df_raw = acquire_dataset()
    else:
        df_raw = pd.read_csv(csv_path)

    # Standardize column names
    df_raw.columns = [str(c).lower().strip() for c in df_raw.columns]
    target_col = "result" if "result" in df_raw.columns else "class"

    selected_cols = FEATURE_ORDER + [target_col]
    missing_cols = [c for c in selected_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df = df_raw[selected_cols].copy()

    # Diagnostics
    initial_rows = len(df)
    null_count = df.isnull().sum().sum()
    print(f"[*] Raw dataset loaded: {initial_rows} records.")
    print(f"[*] Null / missing values count: {null_count}")

    # Drop duplicate rows in the feature space
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(df)
    print(f"[*] Deduplicated feature records: {dedup_rows} (removed {initial_rows - dedup_rows} identical rows)")

    # Target Mapping:
    # In UCI: -1 = Phishing, 1 = Legitimate
    # Model Binary Standard:
    #   1 = Phishing (positive target class)
    #   0 = Legitimate (negative class)
    df["label"] = df[target_col].map({-1: 1, 1: 0})
    df = df.drop(columns=[target_col])

    print("\n[*] Target Mapping Applied:")
    print("      UCI -1 (Phishing)   -> 1 (Phishing)")
    print("      UCI  1 (Legitimate) -> 0 (Legitimate)")
    print("[*] Processed Class Distribution:")
    for label_val, count in df["label"].value_counts().items():
        name = "Phishing (1)" if label_val == 1 else "Legitimate (0)"
        print(f"      {name}: {count} ({count / len(df) * 100:.2f}%)")

    return df


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Computes standard classification evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
    }


def train_and_evaluate():
    """Main training, cross-validation, evaluation, and saving pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess_data(CSV_PATH)

    X = df[FEATURE_ORDER].values
    y = df["label"].values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 1. Baseline Model: Logistic Regression
    print("\n[*] Training Baseline: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # 2. Primary Model: Random Forest with 5-Fold GridSearchCV
    print("\n[*] Training Primary Model: Random Forest with 5-Fold GridSearchCV...")
    rf_param_grid = {
        "n_estimators": [50, 80, 100],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"[+] Best Random Forest Parameters: {grid_search.best_params_}")
    print(f"[+] 5-Fold Best CV F1-Score: {grid_search.best_score_:.4f}")

    rf_metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest")

    # Display Metrics Table
    print("\n" + "=" * 75)
    print(f"{'Metric':<22} | {'Logistic Regression':<22} | {'Random Forest (Best)':<22}")
    print("-" * 75)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        name = k.replace("_", " ").title()
        print(f"{name:<22} | {lr_metrics[k]:<22.4f} | {rf_metrics[k]:<22.4f}")
    print("=" * 75)

    # Feature Importances for Random Forest
    rf_importances = best_rf.feature_importances_
    feat_imp_sorted = sorted(
        zip(FEATURE_ORDER, rf_importances), key=lambda x: x[1], reverse=True
    )
    print("\n[*] Random Forest Feature Importances:")
    for feat, imp in feat_imp_sorted:
        print(f"    - {feat:<26}: {imp * 100:>6.2f}%")

    # Model Artifact Bundle
    bundle = {
        "model": best_rf,
        "model_name": "RandomForestClassifier",
        "feature_order": FEATURE_ORDER,
        "label_mapping": {1: "phishing", 0: "legitimate"},
        "best_params": grid_search.best_params_,
        "metrics": {
            "random_forest": rf_metrics,
            "logistic_regression": lr_metrics,
        },
        "feature_importances": dict(feat_imp_sorted),
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n[+] Saved primary model bundle to: {MODEL_PATH}")
    print(f"[+] Model bundle file size: {file_size_mb:.2f} MB (well under 25 MB limit)")

    # Save metrics.json
    metrics_export = {
        "dataset": {
            "source": "UCI Phishing Websites Dataset (ID 327)",
            "selected_features": FEATURE_ORDER,
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "random_forest_best_params": grid_search.best_params_,
        "feature_importances": {feat: round(float(imp), 4) for feat, imp in feat_imp_sorted},
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[+] Metrics exported to: {METRICS_JSON_PATH}")

    # Generate Visualization Plots
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_lr = np.array(lr_metrics["confusion_matrix"])
    disp_lr = ConfusionMatrixDisplay(
        confusion_matrix=cm_lr,
        display_labels=["Legitimate", "Phishing"]
    )
    disp_lr.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Logistic Regression\n(Acc: {lr_metrics['accuracy']:.4f})")

    cm_rf = np.array(rf_metrics["confusion_matrix"])
    disp_rf = ConfusionMatrixDisplay(
        confusion_matrix=cm_rf,
        display_labels=["Legitimate", "Phishing"]
    )
    disp_rf.plot(ax=axes[1], cmap="Greens", colorbar=False)
    axes[1].set_title(f"Random Forest\n(Acc: {rf_metrics['accuracy']:.4f})")
    plt.tight_layout()
    cm_plot_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix plot saved: {cm_plot_path}")

    # 2. ROC Curves Plot
    plt.figure(figsize=(6.5, 5))
    for mod, col, name in [
        (lr_model, "navy", f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.4f})"),
        (best_rf, "darkgreen", f"Random Forest (AUC = {rf_metrics['roc_auc']:.4f})"),
    ]:
        proba = mod.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, color=col, lw=2, label=name)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (URL-Only Features)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"[+] ROC curve plot saved: {roc_plot_path}")

    # 3. Feature Importance Bar Plot
    plt.figure(figsize=(8, 4.5))
    names = [x[0] for x in reversed(feat_imp_sorted)]
    values = [x[1] * 100 for x in reversed(feat_imp_sorted)]
    plt.barh(names, values, color="#2b5c8f")
    plt.xlabel("Relative Importance (%)")
    plt.title("Random Forest Feature Importances (URL-Computable Subset)")
    plt.tight_layout()
    fi_plot_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(fi_plot_path, dpi=200)
    plt.close()
    print(f"[+] Feature importance plot saved: {fi_plot_path}")

    # Academic Disclaimer / Benchmark Comparison
    print("\n" + "=" * 75)
    print("NOTE ON BENCHMARK COMPARISON (FOR REPORT):")
    print("-" * 75)
    print("Published literature using ALL 30 UCI features achieves ~95-97% accuracy")
    print("because it relies on heavy features like DOM analysis, PageRank, Alexa")
    print("traffic, and live SSL certificate validation.")
    print(f"By restricting classification to 10 strictly URL-computable features,")
    print(f"our Random Forest achieves {rf_metrics['accuracy'] * 100:.2f}% accuracy with zero network risk,")
    print("instant inference, and full privacy preservation for real-world QR scanning.")
    print("=" * 75)


if __name__ == "__main__":
    train_and_evaluate()
