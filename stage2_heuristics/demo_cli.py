#!/usr/bin/env python3
"""
Interactive CLI for Stage 2 (heuristics + shortener analysis).

Examples:
    python demo_cli.py "http://192.168.1.50/login/secure-update"
    python demo_cli.py --no-resolve "https://bit.ly/3example"
    python demo_cli.py --qr ../stage1_qr_decoding/test_images/phishing_style_ip.png
    python demo_cli.py --json "https://paypal-verify.com"

Exit codes: 0 legitimate, 1 suspicious, 2 phishing, 3 error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from heuristics import analyze_url

STAGE1_DIR = Path(__file__).resolve().parent.parent / "stage1_qr_decoding"
# Function names tried on Stage 1's qr_decoder module (first one found is used).
DECODER_CANDIDATES = (
    "decode_qr", "decode_qr_image", "decode_image", "decode",
    "scan_qr", "process_qr", "process_image", "read_qr",
)
EXIT_CODES = {"legitimate": 0, "suspicious": 1, "phishing": 2}


def _extract_url(result: Any) -> Optional[str]:
    """Pull a URL string out of whatever Stage 1 returns (object, dict, str, list)."""
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            url = _extract_url(item)
            if url:
                return url
        return None
    if isinstance(result, dict):
        for key in ("url", "data", "payload", "text"):
            if isinstance(result.get(key), str) and result[key]:
                return result[key]
        return None
    url = getattr(result, "url", None)
    return url if isinstance(url, str) and url else None


def decode_qr_image(image_path: str) -> str:
    """Stage 1 -> Stage 2 hand-off. Falls back to OpenCV if qr_decoder is unusable."""
    if not Path(image_path).is_file():
        raise SystemExit(f"error: image not found: {image_path}")

    sys.path.insert(0, str(STAGE1_DIR))
    try:
        import qr_decoder  # type: ignore
    except ImportError:
        qr_decoder = None

    if qr_decoder is not None:
        for name in DECODER_CANDIDATES:
            func = getattr(qr_decoder, name, None)
            if callable(func):
                try:
                    result = func(image_path)
                except Exception as exc:  # Stage 1 may reject invalid images
                    raise SystemExit(f"error: Stage 1 ({name}) failed: {exc}")
                url = _extract_url(result)
                if url:
                    return url
                raise SystemExit(f"error: Stage 1 ({name}) returned no URL: {result!r}")

    try:
        import cv2  # type: ignore
    except ImportError:
        raise SystemExit(
            "error: could not use Stage 1's qr_decoder (no known decode function "
            f"in {STAGE1_DIR}) and OpenCV is not installed.\n"
            "Add your decode function's name to DECODER_CANDIDATES in demo_cli.py "
            "or `pip install opencv-python`."
        )
    img = cv2.imread(image_path)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    if not data:
        raise SystemExit("error: no QR code could be decoded from the image")
    return data


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def print_report(analysis: dict) -> None:
    verdict = analysis["verdict"]
    colors = {"legitimate": "32", "suspicious": "33", "phishing": "31"}
    score = analysis["score"]
    bar = "#" * (score // 5) + "." * (20 - score // 5)

    print("=" * 64)
    print(f" Verdict : {_color(verdict.upper(), colors[verdict])}")
    print(f" Score   : {score}/100  [{bar}]")
    print("=" * 64)
    print(f" Original URL : {analysis['original_url']}")
    if analysis["is_shortener"]:
        print(f" Resolved URL : {analysis['resolved_url']}")
        print(" Redirect chain:")
        for i, hop in enumerate(analysis["redirect_chain"]):
            print(f"   {i}. {hop}")
    if analysis["resolution_note"]:
        print(f" Note         : {analysis['resolution_note']}")

    print("\n Triggered rules:")
    if analysis["triggered_rules"]:
        for rule in analysis["triggered_rules"]:
            print(f"   - {rule}")
    else:
        print("   (none)")

    print("\n Features (for Stage 3):")
    for key, value in analysis["features"].items():
        print(f"   {key:28s} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: heuristic + shortener analysis of a URL or QR image."
    )
    parser.add_argument("url", nargs="?", help="URL to analyse")
    parser.add_argument("--qr", metavar="IMAGE", help="QR image to decode with Stage 1 first")
    parser.add_argument("--no-resolve", action="store_true",
                        help="do not contact URL shorteners (fully offline analysis)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="per-hop timeout in seconds (default 5)")
    parser.add_argument("--json", action="store_true", help="print raw JSON output")
    args = parser.parse_args()

    if bool(args.url) == bool(args.qr):
        parser.error("provide exactly one of: a URL, or --qr IMAGE")

    try:
        url = decode_qr_image(args.qr) if args.qr else args.url
        if args.qr and not args.json:
            print(f"Stage 1 decoded: {url}\n")
        analysis = analyze_url(url, resolve=not args.no_resolve, timeout=args.timeout)
    except SystemExit:
        raise
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_report(analysis)
    return EXIT_CODES[analysis["verdict"]]


if __name__ == "__main__":
    sys.exit(main())