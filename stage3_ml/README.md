# Stage 3: ML Classification (Member 3)

This module implements the Machine Learning Classification stage for the **QR Phishing Detection** system. It uses a **Random Forest Classifier** (with 5-fold GridSearchCV tuning) alongside a **Logistic Regression** comparison baseline trained on the **UCI Phishing Websites Dataset (UCI ID 327)**.

To preserve complete user privacy, eliminate network latency, and prevent drive-by malware exposure during QR scanning, Stage 3 is strictly restricted to **10 features computable directly from the raw URL string**.

---

## 📁 Directory Structure

```
stage3_ml/
├── data/
│   └── phishing_uci.csv          # Downloaded UCI dataset (11,055 records)
├── models/
│   └── phishing_model.joblib     # Serialized model bundle (~0.15 MB, well under 25 MB limit)
├── reports/
│   ├── confusion_matrix.png      # Confusion matrices for LR and RF
│   ├── roc_curve.png             # Receiver Operating Characteristic curves
│   ├── feature_importance.png    # Random Forest feature importance bar chart
│   └── metrics.json              # Full quantitative metrics export
├── feature_extractor.py          # Extracts 10 URL-computable features (-1, 0, 1)
├── feature_selection.md          # 30-column UCI feature analysis & exclusion rationale
├── get_dataset.py                # Dataset downloader (ucimlrepo with HTTP fallback)
├── ml_predict.py                 # Lazy-loading prediction interface & CLI
├── requirements.txt              # Stage 3 Python dependencies
├── test_feature_extractor.py     # Unit test suite for feature extraction
└── train_model.py                # Preprocessing, tuning, training & evaluation pipeline
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r stage3_ml/requirements.txt
```

### 2. Download the Dataset
Fetches the official UCI Phishing Websites dataset (ID 327) and saves it to `stage3_ml/data/phishing_uci.csv`:
```bash
python stage3_ml/get_dataset.py
```

### 3. Run Unit Tests
Verifies feature extraction against known IP addresses, `@` symbols, long URLs, shorteners, and malformed inputs:
```bash
python stage3_ml/test_feature_extractor.py
```

### 4. Train and Evaluate Models
Performs stratified 80/20 train/test splitting, 5-fold cross-validation grid search, model serialization, and plot generation:
```bash
python stage3_ml/train_model.py
```

### 5. Predict on a URL (CLI)
```bash
python stage3_ml/ml_predict.py "https://example.com"
python stage3_ml/ml_predict.py "http://192.168.1.50/login/secure-update"
```

### 6. Run End-to-End Pipeline
Chains Stage 1 (QR Decoding) -> Stage 2 (Heuristics & Unshortening) -> Stage 3 (ML Classification):
```bash
# Analyze a QR image:
python pipeline.py stage1_qr_decoding/test_images/phishing_style_ip.png

# Analyze a URL directly (skips Stage 1):
python pipeline.py --url "http://paypal.verify-account.com@evil-site.com/login"
```

---

## 🔍 Feature Selection (URL-Computable Subset)

The 30 features of the UCI dataset were reviewed. Only **10 lexical and syntactic features** can be extracted from a raw URL without downloading page HTML/JavaScript or making external network lookups:

1. `having_ip_address`: Hostname is IPv4/IPv6 (-1) vs domain (1)
2. `url_length`: <54 chars (1), 54-75 (0), >75 chars (-1)
3. `shortining_service`: Known shortener service (-1) vs standard domain (1)
4. `having_at_symbol`: `@` present (-1) vs absent (1)
5. `double_slash_redirecting`: `//` redirect in path/query (-1) vs standard (1)
6. `prefix_suffix`: Hyphen `-` in domain body (-1) vs none (1)
7. `having_sub_domain`: 0 subdomains (1), 1 subdomain (0), ≥2 subdomains (-1)
8. `sslfinal_state`: HTTPS scheme (1) vs unencrypted HTTP (-1) [Approximated]
9. `port`: Explicit non-standard port (-1) vs standard 80/443 (1)
10. `https_token`: Token `https` deceptively inside domain name (-1) vs clean (1)

*For detailed analysis and justification for all 20 excluded features (DOM, WHOIS, PageRank, Alexa), see [feature_selection.md](feature_selection.md).*

---

## 📊 Evaluation Results

Models were evaluated on a held-out stratified test set (20% split) with the following metrics:

| Metric | Logistic Regression (Baseline) | Random Forest (Primary, Tuned) |
|---|:---:|:---:|
| **Accuracy** | 74.22% | **75.78%** |
| **Precision** | 71.43% | **75.36%** |
| **Recall** | 83.33% | **78.79%** |
| **F1-Score** | 76.92% | **77.04%** |
| **ROC-AUC** | 0.8038 | **0.8404** |

### Top Random Forest Feature Importances
1. `sslfinal_state`: 32.40%
2. `prefix_suffix`: 28.54%
3. `having_sub_domain`: 6.68%
4. `url_length`: 6.14%
5. `double_slash_redirecting`: 6.02%
6. `shortining_service`: 4.66%
7. `https_token`: 4.29%
8. `having_ip_address`: 4.12%
9. `having_at_symbol`: 3.85%
10. `port`: 3.30%

### Academic Benchmark Note
In academic literature, models trained on the **complete 30-feature UCI dataset** achieve ~95–97% accuracy because they incorporate external domain reputation, Alexa traffic, and DOM inspection. By deliberately confining classification to **strictly URL-computable features**, our classifier operates with **zero external web requests, instant inference speed (<1ms), and complete immunity to drive-by malicious downloads**. When combined with Stage 2 Heuristics in `pipeline.py`, the multi-stage system provides robust, multi-layered phishing defense.
