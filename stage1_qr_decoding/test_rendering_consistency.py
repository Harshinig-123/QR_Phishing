"""
test_rendering_consistency.py
-----------------------------
Decodes every image from generate_varied_qr.py and checks that the decoded URL
equals the original. Prints success rate per variant and per group.

This gives two report-ready results:
  1. Decoding success by rendering style (how robust is Stage 1?)
  2. Whether a successfully decoded code always yields the SAME URL, so any
     downstream verdict cannot depend on how the QR code was drawn.

Run:
    python test_rendering_consistency.py                   # auto backend
    python test_rendering_consistency.py --backend opencv  # force OpenCV only
    python test_rendering_consistency.py --backend zbar    # force pyzbar only
"""

import argparse
import csv
import os
from collections import defaultdict

import qr_decoder

DIR = "test_images_varied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "zbar", "opencv"], default="auto")
    args = ap.parse_args()

    if args.backend == "opencv":
        qr_decoder.HAS_ZBAR = False
    elif args.backend == "zbar":
        if not qr_decoder.HAS_ZBAR:
            raise SystemExit("pyzbar is not available on this machine.")
        qr_decoder.HAS_CV2 = False

    manifest = os.path.join(DIR, "manifest.csv")
    if not os.path.exists(manifest):
        raise SystemExit("Run generate_varied_qr.py first.")

    by_variant = defaultdict(lambda: [0, 0])   # [decoded_correctly, total]
    by_group = defaultdict(lambda: [0, 0])
    wrong_url = []                              # decoded, but a different URL (should be empty)
    failed = []

    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = qr_decoder.process_qr_image(os.path.join(DIR, row["filename"]))
            ok = r.status == "ok" and r.url == row["expected_url"]
            for table, key in ((by_variant, row["variant"]), (by_group, row["group"])):
                table[key][1] += 1
                table[key][0] += int(ok)
            if r.status == "ok" and r.url != row["expected_url"]:
                wrong_url.append((row["filename"], r.url))
            if not ok:
                failed.append((row["filename"], r.status))

    backend = "auto" if args.backend == "auto" else args.backend
    print(f"Backend: {backend}   (pyzbar available: {qr_decoder.HAS_ZBAR}, OpenCV available: {qr_decoder.HAS_CV2})\n")
    print(f"{'Variant':<18}{'Decoded':>10}")
    for v, (a, n) in by_variant.items():
        print(f"{v:<18}{a:>4}/{n:<4} {'' if a == n else '  <-- some failures'}")
    print("\nBy group:")
    for g, (a, n) in by_group.items():
        print(f"  {g:<10} {a}/{n}  ({100 * a / n:.0f}%)")
    total_ok = sum(a for a, _ in by_group.values())
    total = sum(n for _, n in by_group.values())
    print(f"\nOverall: {total_ok}/{total} ({100 * total_ok / total:.0f}%)")
    print(f"Decoded but WRONG url: {len(wrong_url)}  (must be 0 for the 'rendering-invariant' claim)")
    for fn, u in wrong_url:
        print("   ", fn, "->", u)
    if failed:
        print(f"\nNot decoded ({len(failed)}):")
        for fn, st in failed:
            print("   ", fn, f"[{st}]")


if __name__ == "__main__":
    main()
