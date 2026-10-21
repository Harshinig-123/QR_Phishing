"""
run_all_samples.py
------------------
Runs the FULL pipeline (Stage 1 -> 2 -> 3) over many QR images and prints a
results table. Put this file in the repo root, next to pipeline.py.

    python run_all_samples.py             # the 19 sample QR codes
    python run_all_samples.py --varied    # the 108 varied-rendering images

Writes pipeline_results.csv (samples) or pipeline_results_varied.csv (varied),
which you can use for the report's evaluation section.

Notes:
  * Shortener images (bit.ly) make Stage 2 contact the shortener service, so
    --varied needs an internet connection and takes longer.
  * Verdict-consistency check (--varied): the same URL rendered in 18 styles
    must get the SAME final verdict every time.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.join(ROOT, "stage1_qr_decoding")
if not os.path.isdir(S1):
    S1 = ROOT


def summarize_row(path):
    r = pipeline.run_pipeline(path)
    s1, s2, s3 = r["stage1"], r["stage2"] or {}, r["stage3"] or {}
    prob = s3.get("probability")
    return {
        "file": os.path.basename(path),
        "s1_status": s1["status"],
        "payload_type": s1["payload_type"],
        "s2_verdict": s2.get("verdict", ""),
        "s2_score": s2.get("score", ""),
        "s3_prediction": s3.get("prediction", ""),
        "s3_probability": round(prob, 3) if isinstance(prob, (int, float)) else "",
        "final_verdict": r["final_verdict"],
        "errors": "; ".join(
            f"{k}: {v['error']}" for k, v in (("stage2", s2), ("stage3", s3)) if "error" in v
        ),
    }


def print_table(rows):
    cols = ["file", "payload_type", "s2_verdict", "s2_score", "s3_prediction", "s3_probability", "final_verdict"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def run_samples():
    folder = os.path.join(S1, "test_images")
    if not os.path.isdir(folder):
        sys.exit(f"{folder} not found. Run generate_test_qr.py inside stage1_qr_decoding first.")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rows = [summarize_row(os.path.join(folder, f)) for f in files]
    print_table(rows)
    out = os.path.join(ROOT, "pipeline_results.csv")
    write_csv(out, rows)
    errs = [r for r in rows if r["errors"]]
    print(f"\n{len(rows)} images processed. Stage errors: {len(errs)}")
    for r in errs:
        print("  ", r["file"], "->", r["errors"])
    print(f"Saved {out}")


def run_varied():
    folder = os.path.join(S1, "test_images_varied")
    manifest = os.path.join(folder, "manifest.csv")
    if not os.path.exists(manifest):
        sys.exit("Run generate_varied_qr.py inside stage1_qr_decoding first.")
    rows, verdicts = [], defaultdict(set)
    with open(manifest, encoding="utf-8") as f:
        for m in csv.DictReader(f):
            row = summarize_row(os.path.join(folder, m["filename"]))
            row["expected_url"], row["variant"] = m["expected_url"], m["variant"]
            rows.append(row)
            verdicts[m["expected_url"]].add(row["final_verdict"])
    out = os.path.join(ROOT, "pipeline_results_varied.csv")
    write_csv(out, rows)

    print("Verdicts seen per URL across all renderings (should be exactly one each):\n")
    consistent = 0
    for url, seen in verdicts.items():
        ok = len(seen) == 1
        consistent += ok
        print(f"  {'OK  ' if ok else 'DIFF'} {sorted(seen)}  {url}")
    print(f"\nConsistent URLs: {consistent}/{len(verdicts)}   Images: {len(rows)}")
    errs = [r for r in rows if r["errors"]]
    print(f"Stage errors: {len(errs)}")
    print(f"Saved {out}")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--varied", action="store_true", help="run the 108 varied-rendering images")
    args = ap.parse_args()
    print(f"Stage 2 connected: {pipeline.STAGE2_READY}   Stage 3 connected: {pipeline.STAGE3_READY}\n")
    run_varied() if args.varied else run_samples()
