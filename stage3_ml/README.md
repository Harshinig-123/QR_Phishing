# Stage 3: ML Classification & Feature Explanation (Member 3)

Part of the **Quishing Shield** multi-stage phishing detection framework. This module provides a 100% static, machine-learning-based classification and local feature explanation layer for candidate URLs.

---

## 🛡️ Core Safety & Privacy Guarantees

1. **100% Static Execution (Zero Network I/O):**
   Neither feature extraction (`feature_extractor.py`) nor inference (`ml_predict.py`) ever contacts the network. There are **zero DNS lookups, zero WHOIS queries, zero socket connections, and zero HTTP requests**. The final destination site is **never contacted** by Stage 3.
2. **Drive-by Malware & Zero-Day Immunity:**
   By completely rejecting page-content inspection (DOM, scripts, images), Stage 3 eliminates exposure to malicious browser exploits and server-side cloaking kits.
3. **Domain Memorization Resistance:**
   TLD and domain identity strings are **deliberately excluded** from the feature vector. The classifier learns structural, lexical, and syntactic attack patterns rather than memorizing domain registries.

---

## 📁 Directory Structure

```
stage3_ml/
├── data/
│   ├── urls_labeled.csv          # Clean labeled URLs (235,370 records; gitignored)
│   ├── phiusiil_dataset.zip      # Cached raw UCI 967 dataset archive (gitignored)
│   ├── sanity_urls.csv           # ~30 curated hand-written synthetic test cases (tracked)
│   ├── qr_testset_manifest.csv   # Stage 4 testset manifest with rendering variations
│   └── testset_images/           # Generated QR images for multi-layer evaluation
├── legacy_uci327/                # Archived legacy artifacts from UCI ID 327
│   ├── feature_extractor.py
│   ├── feature_selection.md
│   ├── get_dataset.py
│   └── phishing_model.joblib
├── models/
│   └── phishing_model.joblib     # Serialized bundle: RF, LR + scaler, metadata (~0.90 MB)
├── reports/
│   ├── confusion_matrix.png      # Comparative confusion matrices (LR vs RF)
│   ├── roc_curve.png             # ROC curves for LR, RF, and No-HTTPS Ablation
│   ├── feature_importance.png    # Top 15 Random Forest Gini feature importances
│   ├── metrics.json              # Full quantitative metrics export
│   └── testset_results.csv       # Multi-layer testset evaluation results
├── benchmark_trad_chehab.py      # Optional benchmark for Trad & Chehab (2025) pixel arrays
├── demo_stage3.py                # Standalone demonstration runner on sanity URLs
├── evaluate_on_testset.py        # Stage 4 evaluation suite & layer coverage analysis
├── feature_extractor.py          # 23 static URL features, safe fallbacks, zero network I/O
├── feature_selection.md          # 23-feature specification, distribution analysis & rationale
├── get_dataset.py                # PhiUSIIL (UCI 967) acquisition & verified label mapping
├── ml_predict.py                 # Prediction API, local LR explanations, should_run_stage3()
├── requirements.txt              # Stage 3 Python dependencies
├── test_feature_extractor.py     # Unit tests & socket-blocking monkeypatch verification
├── test_render_consistency.py    # Visual rendering invariance verification (qrcode)
└── train_model.py                # Grouped domain split, 5-fold CV, ablation study
```

---

## 🚀 Setup & Execution Guide

### 1. Install Dependencies
```bash
pip install -r stage3_ml/requirements.txt
```

### 2. Acquire and Clean Dataset (PhiUSIIL - UCI ID 967)
Fetches the PhiUSIIL Phishing URL dataset, verifies metadata label encodings, strips all 54 precomputed content/traffic columns, and outputs clean `url,label` pairs:
```bash
python stage3_ml/get_dataset.py
```
* **Dataset Encoding:** Raw `0` = Phishing, Raw `1` = Legitimate.
* **Standard Project Mapping:** Mapped `1` = Phishing (positive target), Mapped `0` = Legitimate.

### 3. Run Unit Tests (Static Verification)
Runs unit tests across standard and edge-case URLs, verifying safe defaults and proving that socket connections raise an error if attempted:
```bash
python -m unittest stage3_ml/test_feature_extractor.py
```

### 4. Train Models & Run Ablation Study
Performs domain-grouped train/test splitting (`GroupShuffleSplit` ensuring 0 domain leakage), 5-fold CV hyperparameter tuning for Random Forest, trains the standardized Logistic Regression explanation baseline, executes the No-HTTPS ablation study, and generates plots in `reports/`:
```bash
python stage3_ml/train_model.py
```

### 5. Test Predictions & Local Explanations (CLI)
```bash
python stage3_ml/ml_predict.py "https://www.google.com"
python stage3_ml/ml_predict.py "http://192.168.1.50/login/update"
```

### 6. Run Sanity Set Demonstration
Evaluates the hand-written synthetic test cases in `data/sanity_urls.csv`:
```bash
python stage3_ml/demo_stage3.py
```

### 7. Run Stage 4 Testset Evaluation & Cross-Layer Coverage
Executes full Stage 1 -> Stage 2 -> Stage 3 evaluation across diverse QR rendering variations:
```bash
python stage3_ml/evaluate_on_testset.py
```

### 8. Verify QR Rendering Invariance
Proves that visual QR parameters (version, error correction L/M/Q/H, box size, colors) do not alter Stage 3 predictions:
```bash
python -m unittest stage3_ml/test_render_consistency.py
```

### 9. Optional Trad & Chehab Benchmark
Runs optional evaluation on local pixel arrays (skips gracefully if not provided):
```bash
python stage3_ml/benchmark_trad_chehab.py --data-dir /path/to/dataset/
```

---

## 📊 Evaluation & Ablation Results

Evaluated on a held-out test split with **zero domain leakage** (`GroupShuffleSplit`, 80/20 by domain):

| Metric | Logistic Regression (Baseline) | Random Forest (Primary) | RF (No-HTTPS Ablation) |
|---|:---:|:---:|:---:|
| **Accuracy** | 99.71% | **99.74%** | 95.76% |
| **Precision** | 100.00% | **100.00%** | 98.83% |
| **Recall** | 99.34% | **99.40%** | 91.42% |
| **F1-Score** | 99.67% | **99.70%** | 94.98% |
| **ROC-AUC** | 0.9983 | **0.9980** | 0.9948 |

### Honest Academic Benchmark Note
1. **Protocol Artifacts:** In the full feature set, `is_https` carries significant Gini importance (40.4%). Removing `is_https` drops accuracy from **99.74% to 95.76%** (recall drops to 91.42%), demonstrating the degree to which modern URL classifiers rely on encryption headers.
2. **Comparison with Structural / Pixel Papers:** These results are URL-string-based and are **not directly comparable** to pixel-based or structural QR-feature papers (such as Trad & Chehab, arXiv:2505.03451), which inspect physical QR module arrangements.

---

## 🔌 Integration Contract for Teammates

### What Stage 3 Consumes:
* **From Member 1 (Stage 1):** The decoded `result` from `process_qr_image()`. Stage 3 provides `should_run_stage3(stage1_result) -> bool` to filter out non-URL payloads (`WIFI:`, `tel:`, `SMSTO:`, `BEGIN:VCARD`).
* **From Member 2 (Stage 2):** Stage 3 expects `analysis["resolved_url"]` — the expanded destination URL string following safe shortener resolution.

### Prediction Output Format:
```json
{
  "url": "http://192.168.1.50/login/update",
  "prediction": "phishing",
  "probability": 1.0,
  "phishing_probability": 1.0,
  "top_factors": [
    {
      "feature": "count_slashes",
      "value": 4.0,
      "direction": "phishing",
      "contribution": 14.4492
    },
    {
      "feature": "is_https",
      "value": 0.0,
      "direction": "phishing",
      "contribution": 12.3981
    },
    {
      "feature": "path_depth",
      "value": 2.0,
      "direction": "phishing",
      "contribution": 7.5541
    }
  ],
  "features": { ... }
}
```
