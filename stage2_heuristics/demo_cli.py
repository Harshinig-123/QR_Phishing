"""
demo_cli.py
-----------
Stage 2 CLI: Test URL heuristics directly, or chain from a Stage 1 QR code image.

Usage:
    # 1. Analyze a raw URL directly:
    python demo_cli.py "http://192.168.1.50/login/secure-update"

    # 2. Decode a QR code image with Stage 1 and run Stage 2 heuristics on it:
    python demo_cli.py --qr ../stage1_qr_decoding/test_images/phishing_style_ip.png
"""

import sys
import os
import json

# Ensure UTF-8 output when possible on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from heuristics import analyze_url

# Add Stage 1 to sys.path so we can optionally chain directly from a QR image
STAGE1_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage1_qr_decoding"))
if STAGE1_PATH not in sys.path:
    sys.path.insert(0, STAGE1_PATH)


def run_demo(target_url: str, resolve_redirects: bool = True):
    print(f"\nAnalyzing URL: {target_url}")
    print("-" * 60)

    result = analyze_url(target_url, resolve_redirects=resolve_redirects)

    # Pretty print full JSON output
    print(json.dumps(result, indent=2))
    print("-" * 60)

    # Summary Output
    verdict = result["verdict"]
    score = result["score"]

    if verdict == "phishing":
        badge = "[ALERT: PHISHING DETECTED]"
    elif verdict == "suspicious":
        badge = "[WARNING: SUSPICIOUS URL]"
    else:
        badge = "[OK: LEGITIMATE URL]"

    print(f"\n{badge} (Risk Score: {score}/100)")
    if result["is_shortener"]:
        print(f"[*] Shortened Link: Yes -> Resolved Target: {result['resolved_url']}")

    if result["triggered_rules"]:
        print("\nTriggered Heuristic Flags:")
        for rule in result["triggered_rules"]:
            print(f"  - {rule}")
    else:
        print("\nNo suspicious heuristic flags triggered.")

    print(f"\nReady for Stage 3 (ML): 'resolved_url' and {len(result['features'])} features extracted.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo_cli.py <url>")
        print("  python demo_cli.py --qr <path_to_qr_image>")
        sys.exit(1)

    if sys.argv[1] == "--qr":
        if len(sys.argv) != 3:
            print("Usage: python demo_cli.py --qr <path_to_qr_image>")
            sys.exit(1)

        qr_path = sys.argv[2]
        try:
            from qr_decoder import process_qr_image
        except ImportError:
            print("Error: Could not import qr_decoder from stage1_qr_decoding.")
            print(f"Make sure '{STAGE1_PATH}' exists.")
            sys.exit(1)

        print(f"Step 1: Decoding QR Image -> {qr_path}")
        qr_result = process_qr_image(qr_path)
        if qr_result.status != "ok":
            print(f"[-] Stage 1 failed: {qr_result.message}")
            sys.exit(1)

        print(f"[+] Stage 1 Decoded URL: {qr_result.url}")
        run_demo(qr_result.url)

    else:
        url_input = sys.argv[1]
        run_demo(url_input)


if __name__ == "__main__":
    main()
