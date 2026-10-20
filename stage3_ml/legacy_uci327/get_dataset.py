"""
get_dataset.py
--------------
Stage 3: Dataset Acquisition for Phishing Websites Dataset (UCI ID 327).

Downloads the dataset via the official `ucimlrepo` library, with a fallback
to direct HTTP download of the dataset file from the UCI Machine Learning
Repository.

Saves the cleaned tabular dataset to `stage3_ml/data/phishing_uci.csv` and prints:
  - Dataset shape
  - Column names
  - Target class distribution
  - Missing values count

Usage:
    python get_dataset.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_CSV = DATA_DIR / "phishing_uci.csv"

# UCI Phishing Websites dataset direct ARFF link for fallback
FALLBACK_URL = (
    "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
)


def fetch_via_ucimlrepo() -> pd.DataFrame:
    """Fetch dataset using the ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo

    print("[*] Attempting to fetch dataset ID 327 via ucimlrepo...")
    dataset = fetch_ucirepo(id=327)
    x = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([x, y], axis=1)
    # Ensure column names are lowercase stripped
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def fetch_via_download() -> pd.DataFrame:
    """Fallback: download ARFF or zipped dataset directly from UCI."""
    import io
    import urllib.request
    import zipfile

    print(f"[*] Fallback: downloading dataset archive from {FALLBACK_URL}...")
    req = urllib.request.Request(
        FALLBACK_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Find .arff file in the archive
        arff_candidates = [n for n in zf.namelist() if n.endswith(".arff")]
        if not arff_candidates:
            raise FileNotFoundError("No .arff file found inside the downloaded archive")
        with zf.open(arff_candidates[0]) as arff_file:
            lines = [l.decode("utf-8", errors="replace") for l in arff_file.readlines()]

    # Parse ARFF data section into DataFrame
    header = []
    data_lines = []
    is_data = False
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("%"):
            continue
        if line_clean.lower().startswith("@attribute"):
            parts = line_clean.split()
            header.append(parts[1].replace("'", "").lower())
        elif line_clean.lower().startswith("@data"):
            is_data = True
        elif is_data:
            data_lines.append([int(v.strip()) for v in line_clean.split(",")])

    df = pd.DataFrame(data_lines, columns=header)
    return df


def acquire_dataset() -> pd.DataFrame:
    """Acquires the dataset using ucimlrepo or fallback, and saves to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = fetch_via_ucimlrepo()
        print("[+] Successfully fetched dataset via ucimlrepo.")
    except Exception as e:
        print(f"[-] ucimlrepo fetch failed ({e}). Attempting fallback download...")
        try:
            df = fetch_via_download()
            print("[+] Successfully fetched dataset via direct fallback download.")
        except Exception as fallback_err:
            print(f"[!] Fallback download also failed: {fallback_err}")
            raise

    # Standardize target column name to 'result'
    if "result" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "result"})

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Dataset saved to: {OUTPUT_CSV}")

    # Dataset diagnostics
    print("\n" + "=" * 60)
    print("UCI Phishing Websites Dataset Summary (ID 327)")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns:\n{list(df.columns)}")
    print("\nTarget class balance ('result'):")
    val_counts = df["result"].value_counts().to_dict()
    for val, count in val_counts.items():
        label_str = "Legitimate" if val == 1 else ("Phishing" if val == -1 else str(val))
        print(f"  Value {val:>2} ({label_str}): {count} ({count / len(df) * 100:.2f}%)")

    missing_cnt = df.isnull().sum().sum()
    print(f"\nTotal missing values (nulls): {missing_cnt}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    acquire_dataset()
