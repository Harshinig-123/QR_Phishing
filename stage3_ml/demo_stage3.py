"""
demo_stage3.py
--------------
Demonstration script for Stage 3 (ML Classification & Feature Explanation).

Runs `predict_url()` against the curated hand-written sanity URLs in
`stage3_ml/data/sanity_urls.csv` (or an optional user-supplied URL),
printing the predictions, confidence levels, and top explanatory factors.

Usage:
    python stage3_ml/demo_stage3.py
    python stage3_ml/demo_stage3.py "http://192.168.1.50/login/update"
"""

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_predict import predict_url
SANITY_CSV = SCRIPT_DIR / "data" / "sanity_urls.csv"


def run_sanity_demo():
    if not SANITY_CSV.exists():
        print(f"Sanity set not found at {SANITY_CSV}")
        return

    print("\n" + "=" * 90)
    print("           STAGE 3 ML CLASSIFIER: SANITY DATASET DEMONSTRATION")
    print("=" * 90)
    print(f"{'Target URL':<48} | {'Expected':<11} | {'Predicted':<11} | {'Conf':<7} | {'Top Factor'}")
    print("-" * 90)

    total = 0
    correct = 0

    with open(SANITY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"].strip()
            exp = row["expected_type"].strip().lower()
            res = predict_url(url)
            pred = res["prediction"].lower()
            conf = res["probability"] * 100
            top_f = res["top_factors"][0] if res["top_factors"] else {"feature": "none", "value": 0}
            top_desc = f"{top_f['feature']}={top_f['value']}"

            total += 1
            is_match = (pred == exp)
            if is_match:
                correct += 1

            mark = "OK " if is_match else "DIFF"
            url_display = (url[:45] + "...") if len(url) > 48 else url
            print(f"{url_display:<48} | {exp:<11} | {pred.upper():<11} | {conf:>5.1f}% | [{mark}] {top_desc}")

    print("=" * 90)
    print(f"Sanity Test Results: {correct}/{total} matched expectations ({correct/total*100:.1f}%)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        u = sys.argv[1]
        r = predict_url(u)
        print(f"URL: {r['url']}")
        print(f"Prediction: {r['prediction'].upper()} ({r['probability']*100:.2f}%)")
        print("Top Factors:")
        for tf in r["top_factors"]:
            print(f"  - {tf['feature']} = {tf['value']} (favors {tf['direction']})")
    else:
        run_sanity_demo()
