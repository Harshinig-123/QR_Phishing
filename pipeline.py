"""
pipeline.py
-----------
End-to-End QR Phishing Detection Pipeline.

Integrates all 3 project stages:
  1. Stage 1 (QR Decoding): Extracts and validates URL from an image.
  2. Stage 2 (Heuristics): Resolves shorteners and computes rule-based risk score.
  3. Stage 3 (ML Classification): Predicts probability of phishing on resolved URL.

Combination Logic:
------------------
The final pipeline verdict combines Stage 2 (Heuristics) and Stage 3 (ML):
  - "phishing":
      Triggered if Stage 3 ML predicts 'phishing' with probability >= 0.50
      OR Stage 2 Heuristics classifies the URL as 'phishing' (risk score >= 55).
  - "suspicious":
      Triggered if Stage 2 evaluates as 'suspicious' (risk score 25-54)
      OR if Stage 2 and Stage 3 disagree on legitimacy.
  - "legitimate":
      Triggered when both Stage 2 and Stage 3 concur that the URL is legitimate.

Usage:
    # 1. Full pipeline from QR image:
    python pipeline.py stage1_qr_decoding/test_images/phishing_style_ip.png

    # 2. Skip Stage 1 and analyze a raw URL:
    python pipeline.py --url "http://192.168.1.50/login/secure-update"

    # 3. Output raw JSON:
    python pipeline.py stage1_qr_decoding/test_images/legit_url.png --json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add all stage directories to sys.path
REPO_ROOT = Path(__file__).resolve().parent
STAGE1_DIR = REPO_ROOT / "stage1_qr_decoding"
STAGE2_DIR = REPO_ROOT / "stage2_heuristics"
STAGE3_DIR = REPO_ROOT / "stage3_ml"

for d in [str(STAGE1_DIR), str(STAGE2_DIR), str(STAGE3_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Stage Imports
try:
    from qr_decoder import process_qr_image, QRDecodeResult
except ImportError as e:
    raise ImportError(f"Could not import Stage 1 qr_decoder: {e}")

try:
    from heuristics import analyze_url
except ImportError as e:
    raise ImportError(f"Could not import Stage 2 heuristics: {e}")

try:
    from ml_predict import predict_url
except ImportError as e:
    raise ImportError(f"Could not import Stage 3 ml_predict: {e}")


def determine_final_verdict(
    stage2_verdict: str,
    stage2_score: int,
    stage3_prediction: str,
    stage3_probability: float
) -> str:
    """
    Combines rule-based heuristics (Stage 2) and machine learning (Stage 3)
    into a consolidated final verdict.

    Combination Rule:
      - 'phishing': If ML predicts 'phishing' with probability >= 0.5 OR
                    Stage 2 verdict is 'phishing'.
      - 'suspicious': If Stage 2 is 'suspicious' OR Stage 2 and Stage 3 disagree.
      - 'legitimate': If both systems concur the URL is legitimate.
    """
    ml_is_phishing = (stage3_prediction == "phishing" and stage3_probability >= 0.50)
    heur_is_phishing = (stage2_verdict == "phishing" or stage2_score >= 55)

    if ml_is_phishing or heur_is_phishing:
        return "phishing"

    if stage2_verdict == "suspicious" or (stage3_prediction != stage2_verdict):
        return "suspicious"

    return "legitimate"


def run_pipeline(
    qr_image_path: Optional[str] = None,
    url: Optional[str] = None,
    resolve_redirects: bool = True
) -> Dict[str, Any]:
    """
    Executes the 3-stage phishing detection pipeline.

    Args:
        qr_image_path: File path to a QR code image.
        url: Direct URL string (bypasses Stage 1 if provided).
        resolve_redirects: Whether Stage 2 should follow URL shorteners.

    Returns:
        Structured dictionary containing results from Stage 1, Stage 2,
        Stage 3, and the consolidated final verdict.
    """
    stage1_data: Optional[Dict[str, Any]] = None
    target_url: Optional[str] = None

    # Step 1: Stage 1 QR Decoding
    if url is not None:
        target_url = url.strip()
        stage1_data = {
            "status": "ok",
            "url": target_url,
            "raw_data": target_url,
            "message": "Direct URL provided; Stage 1 QR decoding skipped.",
            "all_detected": [target_url],
        }
    elif qr_image_path is not None:
        qr_res: QRDecodeResult = process_qr_image(qr_image_path)
        stage1_data = qr_res.to_dict()

        if qr_res.status != "ok":
            return {
                "stage1": stage1_data,
                "stage2": None,
                "stage3": None,
                "final_verdict": "unreadable_qr" if qr_res.status == "unreadable" else qr_res.status,
                "summary": f"Pipeline stopped at Stage 1: {qr_res.message}",
            }
        target_url = qr_res.url
    else:
        raise ValueError("Either 'qr_image_path' or 'url' must be provided.")

    # Step 2: Stage 2 Heuristics & Unshortening
    # Stage 2 safely unshortens the URL and inspects lexical markers
    stage2_raw = analyze_url(target_url, resolve_redirects=resolve_redirects)
    resolved_url = stage2_raw.get("resolved_url", target_url)

    stage2_data = {
        "original_url": stage2_raw.get("original_url"),
        "is_shortener": stage2_raw.get("is_shortener"),
        "resolved_url": resolved_url,
        "redirect_chain": stage2_raw.get("redirect_chain"),
        "score": stage2_raw.get("score"),
        "verdict": stage2_raw.get("verdict"),
        "triggered_rules": stage2_raw.get("triggered_rules"),
        "resolution_note": stage2_raw.get("resolution_note"),
    }

    # Step 3: Stage 3 ML Classification
    # Crucial Requirement: Pass the RESOLVED URL to Stage 3, not the shortened link
    stage3_raw = predict_url(resolved_url)
    stage3_data = {
        "url": stage3_raw.get("url"),
        "prediction": stage3_raw.get("prediction"),
        "probability": stage3_raw.get("probability"),
        "features": stage3_raw.get("features"),
    }

    # Step 4: Consolidated Final Verdict
    final_verdict = determine_final_verdict(
        stage2_verdict=stage2_data["verdict"],
        stage2_score=stage2_data["score"],
        stage3_prediction=stage3_data["prediction"],
        stage3_probability=stage3_data["probability"],
    )

    # Human-readable explanation summary
    summary_parts = [
        f"Final Verdict: {final_verdict.upper()}.",
        f"Stage 2 Heuristic Score: {stage2_data['score']}/100 ({stage2_data['verdict']}).",
        f"Stage 3 ML Prediction: {stage3_data['prediction']} ({stage3_data['probability'] * 100:.1f}% confidence).",
    ]
    if stage2_data["is_shortener"]:
        summary_parts.append(f"Shortener detected: '{stage2_data['original_url']}' -> '{resolved_url}'.")

    return {
        "stage1": stage1_data,
        "stage2": stage2_data,
        "stage3": stage3_data,
        "final_verdict": final_verdict,
        "summary": " ".join(summary_parts),
    }


def print_human_readable_report(result: Dict[str, Any]):
    """Pretty-prints a formatted report of the pipeline execution."""
    print("\n" + "=" * 70)
    print("           QR PHISHING DETECTION PIPELINE REPORT")
    print("=" * 70)

    # Stage 1 Details
    s1 = result.get("stage1") or {}
    print(f"[*] Stage 1 (QR Decoding):")
    print(f"    - Status  : {s1.get('status')}")
    print(f"    - Raw Data: {s1.get('raw_data')}")
    print(f"    - Message : {s1.get('message')}")

    if result.get("final_verdict") in ("no_qr", "unreadable", "not_a_url", "unreadable_qr"):
        print("\n" + "!" * 70)
        print(f"[-] PIPELINE TERMINATED EARLY: {result['summary']}")
        print("!" * 70 + "\n")
        return

    # Stage 2 Details
    s2 = result.get("stage2") or {}
    print(f"\n[*] Stage 2 (Heuristics & Shortener Resolution):")
    print(f"    - Original URL : {s2.get('original_url')}")
    if s2.get("is_shortener"):
        print(f"    - Shortener    : YES (Redirects followed: {len(s2.get('redirect_chain', [])) - 1})")
        print(f"    - Target URL   : {s2.get('resolved_url')}")
    else:
        print(f"    - Shortener    : NO")
    if s2.get("resolution_note"):
        print(f"    - Note         : {s2.get('resolution_note')}")
    print(f"    - Risk Score   : {s2.get('score')} / 100")
    print(f"    - Heur Verdict : {s2.get('verdict').upper()}")
    if s2.get("triggered_rules"):
        print("    - Flags Triggered:")
        for r in s2.get("triggered_rules", []):
            print(f"        * {r}")

    # Stage 3 Details
    s3 = result.get("stage3") or {}
    print(f"\n[*] Stage 3 (ML Classification - Random Forest):")
    print(f"    - Evaluated URL: {s3.get('url')}")
    print(f"    - ML Prediction: {s3.get('prediction').upper()}")
    print(f"    - Confidence   : {s3.get('probability') * 100:.2f}%")

    # Final Verdict Banner
    verdict = result.get("final_verdict", "unknown").upper()
    print("\n" + "-" * 70)
    if verdict == "PHISHING":
        badge = "[ALERT: PHISHING DETECTED]"
    elif verdict == "SUSPICIOUS":
        badge = "[CAUTION: SUSPICIOUS URL]"
    else:
        badge = "[SAFE: LEGITIMATE URL]"
    print(f"FINAL CONSOLIDATED VERDICT: {badge}")
    print(f"Summary: {result.get('summary')}")
    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End QR Phishing Detection Pipeline (Stage 1 -> Stage 2 -> Stage 3)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the QR code image file to analyze."
    )
    parser.add_argument(
        "--url",
        help="Analyze a raw URL directly (bypasses Stage 1 QR decoding)."
    )
    parser.add_argument(
        "--no-redirects",
        action="store_true",
        help="Disable online HTTP redirect following for shortened URLs."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of human-readable report."
    )

    args = parser.parse_args()

    if not args.image and not args.url:
        parser.print_help()
        sys.exit(1)

    try:
        result = run_pipeline(
            qr_image_path=args.image,
            url=args.url,
            resolve_redirects=(not args.no_redirects)
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_human_readable_report(result)

    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
