"""
evaluate_on_testset.py
----------------------
Stage 4 Evaluation Support: End-to-End Testset Evaluation & Cross-Layer Analysis.

Evaluates QR code testsets through the full 3-layer architecture:
  Layer 1 (QR Decoding): Extracts and validates payload type.
  Layer 2 (Heuristics): Rule-based scoring and safe shortener expansion.
  Layer 3 (ML Classification): Static Random Forest inference on resolved URL.

Inputs:
  - Manifest CSV (`--manifest`) specifying:
    image_path,true_label,payload_type,url,qr_version,error_correction,color,size,source
  - If no manifest exists, generates a standardized synthetic evaluation suite
    in `stage3_ml/data/testset_images/` with rendering variations.

Outputs:
  - `stage3_ml/reports/testset_results.csv`: Row-by-row multi-stage execution log.
  - Console summary:
      * ML-only metrics
      * Heuristic-only metrics
      * Combined pipeline metrics (simple documented combination rule; note that
        Member 1's pipeline owns the final decision rule)
      * Per-layer coverage breakdown (caught by both, ML-only, heuristics-only, neither)
      * Separate tracking for shortened URLs and non-URL payloads
      * Rendering consistency check (% of URLs with invariant ML predictions across renderings)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1_DIR = REPO_ROOT / "stage1_qr_decoding"
STAGE2_DIR = REPO_ROOT / "stage2_heuristics"
STAGE3_DIR = REPO_ROOT / "stage3_ml"

for p in [str(STAGE1_DIR), str(STAGE2_DIR), str(STAGE3_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_H
from qr_decoder import process_qr_image
from heuristics import analyze_url
from ml_predict import predict_url, should_run_stage3

REPORTS_DIR = STAGE3_DIR / "reports"
DATA_DIR = STAGE3_DIR / "data"
DEFAULT_MANIFEST = DATA_DIR / "qr_testset_manifest.csv"
RESULTS_CSV = REPORTS_DIR / "testset_results.csv"


def generate_example_manifest(manifest_path: Path) -> Path:
    """
    Generates a small benchmark QR testset if Member 1's manifest is not yet present.
    Includes multiple visual variations per URL to test rendering consistency,
    along with shorteners and non-URL payloads.
    """
    print(f"[*] Generating example test-set manifest at: {manifest_path}...")
    img_dir = DATA_DIR / "testset_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    test_items = [
        # (url, true_label, payload_type, source)
        ("https://www.google.com", "legitimate", "url", "generated"),
        ("https://www.wikipedia.org", "legitimate", "url", "generated"),
        ("https://www.python.org", "legitimate", "url", "generated"),
        ("http://192.168.1.50/login/update", "phishing", "url", "generated"),
        ("http://paypal.com@evil-site.net/signin", "phishing", "url", "generated"),
        ("https://paypal-security-update-verify.com/signin", "phishing", "url", "generated"),
        ("http://xn--pple-43d.com/login", "phishing", "url", "generated"),
        ("https://bit.ly/3xPh1sh-sample", "phishing", "url", "generated"),
        ("WIFI:S:OfficeNetwork;T:WPA;P:SecretPass123;;", "non_url", "wifi", "generated"),
        ("tel:+18005550199", "non_url", "phone", "generated"),
        ("SMSTO:+18005550199:Verify your code", "non_url", "sms", "generated"),
    ]

    manifest_rows = []
    # Test rendering variations for URLs (2 variations per URL)
    variations = [
        {"version": 1, "ec": "M", "color": "black", "size": 6},
        {"version": 3, "ec": "H", "color": "#000080", "size": 8},
    ]

    item_idx = 1
    for text, label, ptype, source in test_items:
        if ptype == "url":
            for v_idx, var in enumerate(variations, 1):
                img_name = f"qr_{item_idx}_{v_idx}.png"
                img_path = img_dir / img_name
                ec_const = ERROR_CORRECT_H if var["ec"] == "H" else ERROR_CORRECT_M

                qr = qrcode.QRCode(
                    version=var["version"],
                    error_correction=ec_const,
                    box_size=var["size"],
                    border=4,
                )
                qr.add_data(text)
                qr.make(fit=True)
                img = qr.make_image(fill_color=var["color"], back_color="white")
                img.save(str(img_path))

                manifest_rows.append({
                    "image_path": f"stage3_ml/data/testset_images/{img_name}",
                    "true_label": label,
                    "payload_type": ptype,
                    "url": text,
                    "qr_version": var["version"],
                    "error_correction": var["ec"],
                    "color": var["color"],
                    "size": var["size"],
                    "source": source,
                })
        else:
            # Non-URL payload
            img_name = f"qr_non_url_{item_idx}.png"
            img_path = img_dir / img_name
            qr = qrcode.QRCode(version=1, box_size=6, border=4)
            qr.add_data(text)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(str(img_path))

            manifest_rows.append({
                "image_path": f"stage3_ml/data/testset_images/{img_name}",
                "true_label": label,
                "payload_type": ptype,
                "url": text,
                "qr_version": 1,
                "error_correction": "M",
                "color": "black",
                "size": 6,
                "source": source,
            })
        item_idx += 1

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"[+] Saved manifest with {len(df_manifest)} entries to {manifest_path}")
    return manifest_path


def combine_verdicts(
    stage1_res: Any,
    stage2_res: Optional[Dict[str, Any]],
    stage3_res: Optional[Dict[str, Any]]
) -> str:
    """
    Reference multi-layer combination rule for Stage 4 evaluation:
      - Non-URL payloads -> 'non_url'
      - QR decode failures -> 'unreadable'
      - Flagged as phishing if Stage 3 ML predicts 'phishing' OR Stage 2 evaluates as 'phishing' (score >= 55)
      - Flagged as suspicious if Stage 2 score in 25-54 or Stage 2 & Stage 3 disagree
      - Otherwise legitimate.

    NOTE: Member 1's main pipeline (pipeline.py) owns the production decision rule.
    """
    if not should_run_stage3(stage1_res):
        return "non_url"

    s2_verdict = stage2_res.get("verdict", "legitimate") if stage2_res else "legitimate"
    s2_score = stage2_res.get("score", 0) if stage2_res else 0
    s3_pred = stage3_res.get("prediction", "legitimate") if stage3_res else "legitimate"

    if s3_pred == "phishing" or s2_verdict == "phishing" or s2_score >= 55:
        return "phishing"
    if s2_verdict == "suspicious" or (s3_pred != s2_verdict):
        return "suspicious"
    return "legitimate"


def run_evaluation(manifest_path: Path, resolve_online: bool = False):
    """Executes the evaluation pipeline across all testset entries."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path = generate_example_manifest(manifest_path)

    df_manifest = pd.read_csv(manifest_path)
    print(f"\n[*] Loaded test-set manifest: {len(df_manifest)} records from {manifest_path}")
    print(f"[*] Online shortener resolution: {'ENABLED (--resolve)' if resolve_online else 'DISABLED (safe offline mode)'}")

    results = []
    for _, row in df_manifest.iterrows():
        raw_img_path = str(row["image_path"]).strip()
        img_p = Path(raw_img_path)
        if not img_p.is_file():
            if (REPO_ROOT / img_p).is_file():
                img_path = str(REPO_ROOT / img_p)
            elif (DATA_DIR / "testset_images" / img_p.name).is_file():
                img_path = str(DATA_DIR / "testset_images" / img_p.name)
            else:
                img_path = raw_img_path
        else:
            img_path = str(img_p)

        true_label = str(row["true_label"]).strip().lower()
        ptype = str(row.get("payload_type", "url")).strip().lower()
        expected_url = str(row.get("url", ""))

        # 1. Stage 1: QR Decoding
        s1_res = process_qr_image(img_path)
        s1_status = s1_res.status
        decoded_url = s1_res.url if s1_res.status == "ok" else None
        raw_payload = s1_res.raw_data

        is_url = should_run_stage3(s1_res)

        s2_score = None
        s2_verdict = None
        s2_resolved = None
        s2_is_shortener = False

        s3_pred = None
        s3_prob = None
        s3_phish_prob = None
        s3_top_factor = None

        if is_url and decoded_url:
            # 2. Stage 2: Heuristics & Shortener resolution
            s2_data = analyze_url(decoded_url, resolve=resolve_online)
            s2_score = s2_data["score"]
            s2_verdict = s2_data["verdict"]
            s2_resolved = s2_data["resolved_url"]
            s2_is_shortener = s2_data["is_shortener"]

            # 3. Stage 3: ML Classification on resolved URL
            s3_data = predict_url(s2_resolved)
            s3_pred = s3_data["prediction"]
            s3_prob = s3_data["probability"]
            s3_phish_prob = s3_data["phishing_probability"]
            s3_top_factor = s3_data["top_factors"][0]["feature"] if s3_data["top_factors"] else None
        else:
            s2_verdict = "skipped_non_url"
            s3_pred = "skipped_non_url"

        combined_verdict = combine_verdicts(s1_res, s2_data if is_url else None, s3_data if is_url else None)

        results.append({
            "image_path": img_path,
            "url": expected_url,
            "true_label": true_label,
            "payload_type": ptype,
            "qr_version": row.get("qr_version"),
            "error_correction": row.get("error_correction"),
            "color": row.get("color"),
            "stage1_status": s1_status,
            "stage1_decoded_url": decoded_url,
            "stage2_score": s2_score,
            "stage2_verdict": s2_verdict,
            "stage2_is_shortener": s2_is_shortener,
            "stage2_resolved_url": s2_resolved,
            "stage3_prediction": s3_pred,
            "stage3_confidence": s3_prob,
            "stage3_phishing_prob": s3_phish_prob,
            "stage3_top_factor": s3_top_factor,
            "combined_verdict": combined_verdict,
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_CSV, index=False)
    print(f"[+] Saved execution results to: {RESULTS_CSV}")

    # Compute Layer Metrics (for URL rows)
    url_mask = df_res["payload_type"] == "url"
    df_urls = df_res[url_mask].copy()

    y_true_binary = (df_urls["true_label"] == "phishing").astype(int)
    y_ml_binary = (df_urls["stage3_prediction"] == "phishing").astype(int)
    y_heur_binary = (df_urls["stage2_verdict"] == "phishing").astype(int)
    y_comb_binary = (df_urls["combined_verdict"] == "phishing").astype(int)

    def _calc_metrics(y_true, y_pred):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    m_ml = _calc_metrics(y_true_binary, y_ml_binary)
    m_heur = _calc_metrics(y_true_binary, y_heur_binary)
    m_comb = _calc_metrics(y_true_binary, y_comb_binary)

    # Coverage breakdown
    both_caught = sum((y_heur_binary == 1) & (y_ml_binary == 1) & (y_true_binary == 1))
    ml_only = sum((y_heur_binary == 0) & (y_ml_binary == 1) & (y_true_binary == 1))
    heur_only = sum((y_heur_binary == 1) & (y_ml_binary == 0) & (y_true_binary == 1))
    neither = sum((y_heur_binary == 0) & (y_ml_binary == 0) & (y_true_binary == 1))

    # Rendering consistency check
    grouped = df_urls.groupby("url")["stage3_prediction"].nunique()
    consistent_urls = (grouped == 1).sum()
    total_unique_urls = len(grouped)
    consistency_pct = (consistent_urls / total_unique_urls * 100) if total_unique_urls > 0 else 100.0

    print("\n" + "=" * 80)
    print("                     STAGE 4 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Items Evaluated    : {len(df_res)}")
    print(f"  - Valid URL Payloads        : {len(df_urls)}")
    print(f"  - Non-URL Payloads (Filtered): {len(df_res) - len(df_urls)}")
    print("-" * 80)
    print(f"{'Metric':<20} | {'ML-Only (Stage 3)':<20} | {'Heuristics-Only':<20} | {'Combined Pipeline'}")
    print("-" * 80)
    for k in ["accuracy", "precision", "recall", "f1_score"]:
        name = k.replace("_", " ").title()
        print(f"{name:<20} | {m_ml[k]:<20.4f} | {m_heur[k]:<20.4f} | {m_comb[k]:<20.4f}")
    print("=" * 80)

    print("\n[*] Per-Layer Phishing Detection Coverage:")
    print(f"    - Caught by Both (ML & Heuristics) : {both_caught}")
    print(f"    - Caught by ML Only                : {ml_only}")
    print(f"    - Caught by Heuristics Only        : {heur_only}")
    print(f"    - Missed by Both                   : {neither}")

    print("\n[*] QR Rendering Invariance Verification:")
    print(f"    - Unique URLs Evaluated            : {total_unique_urls}")
    print(f"    - URLs with Invariant ML Verdicts  : {consistent_urls} ({consistency_pct:.1f}%)")
    if consistency_pct == 100.0:
        print("    [+] Zero variance: Visual rendering (EC level, color, version) had 0 impact on Stage 3.")
    else:
        print("    [-] Noticeable variances detected:")
        for u, n in grouped[grouped > 1].items():
            print(f"        * URL: {u}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Testset Evaluation Suite")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest CSV")
    parser.add_argument("--resolve", action="store_true", help="Enable online shortener expansion")
    args = parser.parse_args()

    run_evaluation(Path(args.manifest), resolve_online=args.resolve)


if __name__ == "__main__":
    main()
