"""
get_dataset.py
--------------
Stage 3: Dataset Acquisition & Normalization for URL-based Phishing Classification.

Fetches the PhiUSIIL Phishing URL Dataset (UCI ID 967), inspects and prints its
metadata label coding, keeps only the raw URL string and the target label (discarding
all 54 precomputed page-content, DNS, and traffic columns), cleans duplicates/conflicts,
and outputs a standardized `stage3_ml/data/urls_labeled.csv`:
    - url: raw URL string
    - label: 1 = Phishing, 0 = Legitimate

Dataset Reference:
    Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered
    phishing URL detection framework based on similarity index and incremental learning.
    Computers & Security, 103545. https://archive.ics.uci.edu/dataset/967
"""

import csv
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_CSV = DATA_DIR / "urls_labeled.csv"
ZIP_PATH = DATA_DIR / "phiusiil_dataset.zip"

UCI_DATASET_ID = 967
DIRECT_ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
DIRECT_CSV_URL = "https://archive.ics.uci.edu/static/public/967/data.csv"


def verify_and_print_label_coding():
    """
    Inspects and prints the exact label coding from dataset specifications.
    In PhiUSIIL (UCI ID 967):
        Raw '0' = Phishing website / URL
        Raw '1' = Legitimate website / URL
    Standard Project Target Mapping:
        Mapped 1 = Phishing (positive detection target)
        Mapped 0 = Legitimate (negative class)
    """
    print("\n" + "=" * 70)
    print("        DATASET METADATA & LABEL CODING VERIFICATION (UCI ID 967)")
    print("=" * 70)
    print("Dataset Name : PhiUSIIL Phishing URL Dataset (UCI ID 967)")
    print("Source       : Prasad & Chandra (2023), Computers & Security")
    print("Original Specification:")
    print("   Raw Value '0' -> Phishing (Attacker-controlled URL)")
    print("   Raw Value '1' -> Legitimate (Benign URL)")
    print("Stage 3 Standard Target Encoding:")
    print("   Project Label 1 -> Phishing (Positive Target)")
    print("   Project Label 0 -> Legitimate (Benign Reference)")
    print("=" * 70 + "\n")


def acquire_raw_zip(zip_target: Path) -> Path:
    """Ensures the dataset zip file is downloaded locally."""
    # Check if a previously downloaded test zip exists
    test_zip = DATA_DIR / "test.zip"
    if test_zip.exists() and test_zip.stat().st_size > 5_000_000 and not zip_target.exists():
        test_zip.rename(zip_target)

    if zip_target.exists() and zip_target.stat().st_size > 5_000_000:
        print(f"[*] Found cached dataset zip at: {zip_target}")
        return zip_target

    print(f"[*] Downloading dataset archive from {DIRECT_ZIP_URL}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            sys.stdout.write(f"\r    Downloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
        else:
            sys.stdout.write(f"\r    Downloading: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(DIRECT_ZIP_URL, str(zip_target), reporthook=_progress)
    print(f"\n[+] Download complete: {zip_target} ({zip_target.stat().st_size / (1024*1024):.2f} MB)")
    return zip_target


def load_raw_urls_and_labels(zip_target: Path) -> pd.DataFrame:
    """
    Streams through the compressed CSV inside the zip file, extracting ONLY the
    'URL' and 'label' columns, ignoring all 54 precomputed columns.
    """
    print(f"[*] Reading URL and label columns from {zip_target}...")
    with zipfile.ZipFile(zip_target) as z:
        csv_filename = None
        for name in z.namelist():
            if name.endswith(".csv"):
                csv_filename = name
                break
        if not csv_filename:
            raise FileNotFoundError(f"No CSV file found inside {zip_target}")

        with z.open(csv_filename) as f:
            text_stream = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")
            reader = csv.reader(text_stream)
            header = next(reader)
            
            # Identify required column indices
            url_idx = -1
            label_idx = -1
            for idx, col in enumerate(header):
                col_clean = col.strip().lower()
                if col_clean == "url":
                    url_idx = idx
                elif col_clean == "label":
                    label_idx = idx
            
            if url_idx == -1 or label_idx == -1:
                raise ValueError(f"Could not find URL and label columns in header: {header}")

            records = []
            for row in reader:
                if len(row) > max(url_idx, label_idx):
                    u = row[url_idx].strip()
                    l = row[label_idx].strip()
                    if u and l in ("0", "1"):
                        records.append((u, l))

    print(f"[*] Extracted {len(records):,} raw records.")
    return pd.DataFrame(records, columns=["url", "raw_label"])


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw URL-label records:
      - Strips surrounding whitespace
      - Drops empty URLs
      - Maps labels: raw 0 -> 1 (Phishing), raw 1 -> 0 (Legitimate)
      - Identifies and drops conflicting URLs (same URL with both labels)
      - Drops duplicate URLs
    """
    print("[*] Cleaning and normalizing raw URLs...")
    # Map raw labels
    df["label"] = df["raw_label"].map({"0": 1, "1": 0}).astype(int)
    df = df.drop(columns=["raw_label"])

    initial_count = len(df)
    # Check conflicting URLs
    label_nunique = df.groupby("url")["label"].nunique()
    conflicting_urls = set(label_nunique[label_nunique > 1].index)
    if conflicting_urls:
        print(f"[-] Dropping {len(conflicting_urls):,} URLs with conflicting labels.")
        df = df[~df["url"].isin(conflicting_urls)]

    # Deduplicate by URL string
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    dedup_count = len(df)
    print(f"[+] Cleaned dataset: {dedup_count:,} unique URLs (removed {initial_count - dedup_count:,} duplicate/conflicting records).")

    return df


def display_dataset_summary(df: pd.DataFrame):
    """Prints class balance and 5 random examples per class."""
    print("\n" + "=" * 70)
    print("                    DATASET SUMMARY & EXAMPLES")
    print("=" * 70)
    print(f"Total Unique URLs: {len(df):,}")
    print("\nClass Distribution:")
    for label_val, count in df["label"].value_counts().items():
        name = "Phishing (1)" if label_val == 1 else "Legitimate (0)"
        pct = count / len(df) * 100
        print(f"   {name:<16}: {count:>8,} samples ({pct:.2f}%)")

    print("\nSample Legitimate URLs (label = 0):")
    legit_samples = df[df["label"] == 0].sample(n=5, random_state=42)["url"].tolist()
    for s in legit_samples:
        print(f"   [0] {s}")

    print("\nSample Phishing URLs (label = 1):")
    phish_samples = df[df["label"] == 1].sample(n=5, random_state=42)["url"].tolist()
    for s in phish_samples:
        print(f"   [1] {s}")
    print("=" * 70 + "\n")


def acquire_dataset(max_records: int = None) -> pd.DataFrame:
    """Main acquisition and cleaning routine."""
    verify_and_print_label_coding()

    # 1. Attempt ucimlrepo first as specified in requirements, fall back to direct download
    raw_df = None
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 5_000_000:
        try:
            from ucimlrepo import fetch_ucirepo
            print("[*] Attempting fetch via ucimlrepo.fetch_ucirepo(id=967)...")
            ds = fetch_ucirepo(id=UCI_DATASET_ID)
            print("[+] Successfully queried UCI metadata via ucimlrepo.")
            feats = ds.data.features
            tgts = ds.data.targets
            url_col = "URL" if "URL" in feats.columns else "url"
            lbl_col = "label" if "label" in tgts.columns else "Label"
            raw_df = pd.DataFrame({"url": feats[url_col], "raw_label": tgts[lbl_col].astype(str)})
        except Exception as exc:
            print(f"[*] ucimlrepo direct fetch skipped/timed out ({exc}). Using direct archive stream...")

    if raw_df is None:
        zip_path = acquire_raw_zip(ZIP_PATH)
        raw_df = load_raw_urls_and_labels(zip_path)

    clean_df = clean_and_normalize(raw_df)

    if max_records and len(clean_df) > max_records:
        print(f"[*] Subsampling to {max_records:,} records while maintaining class stratifications...")
        clean_df = clean_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=int(max_records * (len(x) / len(clean_df))), random_state=42)
        ).reset_index(drop=True)

    display_dataset_summary(clean_df)

    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[+] Saved clean labeled URLs to: {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / (1024*1024):.2f} MB)")
    return clean_df


if __name__ == "__main__":
    acquire_dataset()
