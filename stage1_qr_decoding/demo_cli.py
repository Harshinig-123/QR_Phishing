"""
demo_cli.py - Stage 1 only demo. Point it at any QR image.

Usage:  python demo_cli.py path/to/qr_image.png
(For the full pipeline use: python pipeline.py path/to/qr_image.png)
"""

import json
import sys

from qr_decoder import process_qr_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python demo_cli.py <path_to_qr_image>")
        sys.exit(1)

    result = process_qr_image(sys.argv[1])
    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(f"Payload type: {result.payload_type}")
    if result.status == "ok":
        print(f"URL ready for Stage 2: {result.url}")
    else:
        print(f"Stage 1 did not produce a URL. {result.message}")
    if result.sensitive:
        print(f"WARNING: {result.sensitivity_note}")


if __name__ == "__main__":
    main()
