"""
benchmark_trad_chehab.py
------------------------
Optional Benchmark: Trad & Chehab (arXiv:2505.03451) Pixel-Array Evaluation.

Reference Paper:
  Trad, A., & Chehab, A. (2025). "Detecting Quishing Attacks with Machine Learning
  Techniques Through QR Code Analysis", arXiv:2505.03451.

Purpose:
  This script provides an optional benchmark interface for evaluating local pixel-array
  QR code datasets (such as Trad & Chehab's image/array corpus) through the Quishing Shield
  pipeline (Stage 1 QR Decoding -> Stage 3 URL ML Classification).

Behavior:
  - DOES NOT download large external datasets automatically.
  - Skips gracefully if the local dataset path is not specified or does not exist.
  - Correctly handles both raw image files (.png, .jpg) and numpy 2D pixel arrays (.npy).
  - When provided with a valid directory (`--data-dir`), decodes each sample using
    Stage 1's decoder, evaluates URL payloads with `predict_url()`, and reports comparative
    classification metrics if ground truth labels are detectable from directory structure,
    filenames, or a manifest CSV.

Usage:
  python stage3_ml/benchmark_trad_chehab.py --data-dir /path/to/trad_chehab_data/
"""

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1_DIR = REPO_ROOT / "stage1_qr_decoding"
STAGE2_DIR = REPO_ROOT / "stage2_heuristics"
STAGE3_DIR = REPO_ROOT / "stage3_ml"

for p in [str(REPO_ROOT), str(STAGE1_DIR), str(STAGE2_DIR), str(STAGE3_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from stage1_qr_decoding.qr_decoder import process_qr_image, process_qr_bytes, QRDecodeResult
except ImportError:
    from qr_decoder import process_qr_image, process_qr_bytes, QRDecodeResult

try:
    from stage2_heuristics.heuristics import analyze_url
except ImportError:
    try:
        from heuristics import analyze_url
    except ImportError:
        analyze_url = None

try:
    from stage3_ml.ml_predict import predict_url, should_run_stage3
except ImportError:
    from ml_predict import predict_url, should_run_stage3


def decode_sample(sample_path: Path) -> QRDecodeResult:
    """
    Decodes a sample file that could either be a standard image (.png/.jpg)
    or a Trad & Chehab 2D numpy pixel array (.npy).
    """
    if sample_path.suffix.lower() == ".npy":
        arr = np.load(str(sample_path))
        if arr.ndim == 2:
            # Scale binary or 0-1 values to 0-255 uint8 grayscale
            if arr.max() <= 1:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return process_qr_bytes(buf.getvalue())
        elif arr.ndim == 3:
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return process_qr_bytes(buf.getvalue())
        else:
            return QRDecodeResult(status="unreadable", message=f"Unsupported numpy array dimensions: {arr.shape}")
    else:
        return process_qr_image(str(sample_path))


def infer_ground_truth(file_path: Path, expected_url: Optional[str] = None) -> Optional[int]:
    """Infers ground truth label from manifest or path conventions (1 = phishing, 0 = legitimate)."""
    if expected_url:
        u_lower = expected_url.lower()
        if any(k in u_lower for k in ("phish", "192.168.", "bank", "auth")):
            return 1
        if any(k in u_lower for k in ("google.com", "docs.python.org")):
            return 0
        if "bit.ly" in u_lower:
            return 1

    p_lower = file_path.name.lower()
    if any(k in p_lower for k in ("phishing", "phish", "malicious", "quishing", "bad")):
        return 1
    if any(k in p_lower for k in ("legitimate", "legit", "benign", "clean", "safe", "docs", "google", "long_query")):
        return 0
    if "short_bitly" in p_lower:
        return 1
    return None


def run_benchmark(data_dir: Optional[str] = None):
    # If no external data-dir was provided, import data directly from Stage 1
    if not data_dir or not os.path.exists(data_dir):
        candidates = [
            STAGE1_DIR / "test_images_varied",
            STAGE1_DIR / "test_images",
        ]
        chosen = None
        for cand in candidates:
            if cand.exists() and (any(cand.glob("*.png")) or any(cand.glob("*.npy"))):
                chosen = cand
                break

        if chosen:
            print("\n" + "=" * 75)
            print("    TRAD & CHEHAB (2025) BENCHMARK SUITE - PIPELINE EVALUATION")
            print("=" * 75)
            print(f"[*] Importing test data directly from Stage 1: {chosen.relative_to(REPO_ROOT)}")
            data_path = chosen
        else:
            print("\n" + "=" * 75)
            print("    TRAD & CHEHAB (2025) BENCHMARK SUITE - PIPELINE EVALUATION")
            print("=" * 75)
            print("[!] Could not locate Stage 1 test dataset directory.")
            print("=" * 75 + "\n")
            return
    else:
        data_path = Path(data_dir)
        print("\n" + "=" * 75)
        print("    TRAD & CHEHAB (2025) BENCHMARK SUITE - PIPELINE EVALUATION")
        print("=" * 75)
        print(f"[*] Scanning dataset directory: {data_path}...")

    # Load Stage 1 manifest if available
    manifest_map: Dict[str, str] = {}
    manifest_csv = data_path / "manifest.csv"
    if manifest_csv.exists():
        try:
            import csv
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename")
                    exp_url = row.get("expected_url")
                    if fname and exp_url:
                        manifest_map[fname] = exp_url
            print(f"[+] Loaded ground truth manifest from Stage 1: {len(manifest_map)} entries.")
        except Exception:
            pass

    sample_files = (
        list(data_path.glob("**/*.npy")) +
        list(data_path.glob("**/*.png")) +
        list(data_path.glob("**/*.jpg")) +
        list(data_path.glob("**/*.jpeg"))
    )

    if not sample_files:
        print(f"[-] No image (.png/.jpg) or numpy array (.npy) files found in {data_path}.")
        return

    print(f"[+] Found {len(sample_files)} samples in dataset. Beginning evaluation...")

    total_evaluated = 0
    decoded_urls = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    predictions_count = {"phishing": 0, "legitimate": 0}

    for f in sample_files:
        try:
            decode_res = decode_sample(f)
            total_evaluated += 1

            if should_run_stage3(decode_res):
                decoded_urls += 1

                # Pass through Stage 2 heuristic analysis / shortener resolution before Stage 3 ML
                eval_url = decode_res.url
                if analyze_url is not None:
                    try:
                        h_res = analyze_url(decode_res.url, resolve=False)
                        eval_url = h_res.get("resolved_url") or decode_res.url
                    except Exception:
                        eval_url = decode_res.url

                pred_res = predict_url(eval_url)
                p_str = pred_res["prediction"]
                predictions_count[p_str] += 1

                gt = infer_ground_truth(f, manifest_map.get(f.name))
                if gt is not None:
                    y_true.append(gt)
                    y_pred.append(1 if p_str == "phishing" else 0)

        except Exception as e:
            continue

    print("\n" + "=" * 75)
    print("           TRAD & CHEHAB BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Total Samples Inspected       : {total_evaluated}")
    print(f"Successfully Decoded URLs     : {decoded_urls} ({decoded_urls/total_evaluated*100:.1f}%)" if total_evaluated else 0)
    print(f"Predicted Class Distribution  : {predictions_count['phishing']} Phishing, {predictions_count['legitimate']} Legitimate")

    if y_true and len(y_true) == len(y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print("-" * 75)
        print(f"Samples with Ground Truth     : {len(y_true)}")
        print(f"Accuracy                      : {acc*100:.2f}%")
        print(f"Precision                     : {prec*100:.2f}%")
        print(f"Recall                        : {rec*100:.2f}%")
        print(f"F1-Score                      : {f1*100:.2f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trad & Chehab Benchmark Runner")
    parser.add_argument("--data-dir", default=None, help="Path to local dataset directory (defaults to Stage 1 data)")
    args = parser.parse_args()

    run_benchmark(args.data_dir)
