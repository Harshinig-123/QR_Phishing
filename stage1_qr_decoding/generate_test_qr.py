"""
generate_test_qr.py
--------------------
Creates a handful of sample QR code images in ./test_images/ so you (and
the rest of the team) have something concrete to test the decoder against
before real "phishing-style" QR codes are prepared for the full evaluation
set in Week/Day 5.

Run:
    python generate_test_qr.py
"""

import os
import qrcode

OUTPUT_DIR = "test_images"

TEST_CASES = {
    # filename                 : encoded content
    "legit_url.png":            "https://www.google.com",
    "legit_url_subpage.png":    "https://docs.python.org/3/library/urllib.parse.html",
    "phishing_style_ip.png":    "http://192.168.1.50/login/secure-update",
    "phishing_style_at.png":    "http://google.com@evil-site.example/login",
    "shortened_url.png":        "https://bit.ly/3exampleShortLink",
    "no_scheme_text.png":       "www.not-a-valid-scheme-example.com",   # should fail URL validation (no http/https scheme)
    "plain_text.png":           "This is just plain text, not a URL at all.",
    "wifi_config.png":          "WIFI:T:WPA;S:MyNetwork;P:mypassword;;",  # realistic non-URL QR payload
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, content in TEST_CASES.items():
        img = qrcode.make(content)
        path = os.path.join(OUTPUT_DIR, filename)
        img.save(path)
        print(f"Created {path}  ->  {content!r}")

    print(f"\nDone. {len(TEST_CASES)} test QR codes written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
