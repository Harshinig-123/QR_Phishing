"""
demo_cli.py
-----------
Quick command-line demo: point it at any QR code image and see the result.

Usage:
    python demo_cli.py path/to/qr_image.png
"""

import sys
import json

from qr_decoder import process_qr_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python demo_cli.py <path_to_qr_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = process_qr_image(image_path)

    print(json.dumps(result.to_dict(), indent=2))

    if result.status == "ok":
        print(f"\n✅ Decoded URL ready for Stage 2: {result.url}")
    else:
        print(f"\n⚠️  Stage 1 did not produce a URL. Reason: {result.message}")


if __name__ == "__main__":
    main()
