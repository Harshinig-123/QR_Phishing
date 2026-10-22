"""
live_demo.py
------------
Live demo helper: give it any link, it draws the QR code, then runs the FULL
pipeline (Stage 1 -> 2 -> 3) on that QR image and prints the verdict.

Put this file in the repo root, next to pipeline.py.

    python live_demo.py                       # interactive: type a link, press Enter
    python live_demo.py http://paypa1.example/login
    python live_demo.py --show <url>          # also open the QR image on screen (Windows)
    python live_demo.py --image some_qr.png   # run the pipeline on an existing QR image
    python live_demo.py --json <url>          # also print the full JSON

Use made-up domains for phishing demos (for example something.example).
Stage 1 never touches the network. Stage 2 resolves known URL shorteners
(bit.ly, tinyurl, ...) over the network; see stage2_heuristics/README.md for
exactly what it requests. Everything else works offline.
"""

import argparse
import os
import re
import sys
import json
import warnings

import qrcode

import pipeline

warnings.filterwarnings("ignore")  # hides library warnings so the demo output stays clean

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "live_demo_qr")
_counter = 0

NON_URL_PREFIXES = ("wifi:", "tel:", "sms:", "smsto:", "mms:", "mailto:", "upi:", "bitcoin:",
                    "ethereum:", "geo:", "begin:vcard", "javascript:", "market:", "intent:", "data:")


def prepare_payload(text):
    """Accept what a person would type. Returns (payload, note)."""
    text = text.strip()
    low = text.lower()
    if "://" in text or low.startswith(NON_URL_PREFIXES):
        return text, ""
    if " " not in text and "." in text:
        return "https://" + text, "No scheme typed, so https:// was added."
    return text, ""


def make_qr(payload):
    global _counter
    os.makedirs(OUT, exist_ok=True)
    _counter += 1
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", payload)[:40].strip("_") or "payload"
    path = os.path.join(OUT, f"{_counter:02d}_{safe}.png")
    qrcode.make(payload).save(path)
    return path


def run(image_path, show_image=False, as_json=False):
    if show_image and hasattr(os, "startfile"):
        try:
            os.startfile(image_path)
        except OSError:
            pass
    result = pipeline.run_pipeline(image_path)
    print()
    print(pipeline.summarize(result))
    s3 = result.get("stage3") or {}
    factors = s3.get("top_factors") or []
    if factors:
        text = ", ".join(f"{f.get('feature')} ({f.get('direction')})" for f in factors[:3])
        print(f"  ML top factors: {text}")
    if as_json:
        print("\n--- full JSON ---")
        print(json.dumps(result, indent=2, default=str))
    print()


def handle_link(text, show_image, as_json):
    payload, note = prepare_payload(text)
    if note:
        print(f"[note] {note}")
    path = make_qr(payload)
    print(f"[QR saved] {path}")
    run(path, show_image, as_json)


def main():
    ap = argparse.ArgumentParser(description="Generate a QR code for a link and analyze it.")
    ap.add_argument("link", nargs="?", help="URL (or other QR payload) to encode and analyze")
    ap.add_argument("--image", help="analyze an existing QR image instead")
    ap.add_argument("--show", action="store_true", help="open the QR image on screen (Windows)")
    ap.add_argument("--json", action="store_true", help="also print the full JSON result")
    args = ap.parse_args()

    print(f"Stage 2 connected: {pipeline.STAGE2_READY}   Stage 3 connected: {pipeline.STAGE3_READY}")

    if args.image:
        run(args.image, args.show, args.json)
    elif args.link:
        handle_link(args.link, args.show, args.json)
    else:
        print("Type a link and press Enter. Type q to quit.")
        while True:
            try:
                text = input("\nlink> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            if text:
                handle_link(text, args.show, args.json)


if __name__ == "__main__":
    main()
