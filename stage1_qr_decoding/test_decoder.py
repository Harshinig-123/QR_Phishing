"""
test_decoder.py
----------------
Simple test suite for qr_decoder.py. Run after generate_test_qr.py.

Run:
    python test_decoder.py
"""

import os
import sys

from qr_decoder import process_qr_image, is_valid_url

TEST_DIR = "test_images"

# (filename, expected_status)
EXPECTED = [
    ("legit_url.png",         "ok"),
    ("legit_url_subpage.png", "ok"),
    ("phishing_style_ip.png", "ok"),        # it IS a URL — Stage 2 decides it's risky, not Stage 1
    ("phishing_style_at.png", "ok"),
    ("shortened_url.png",     "ok"),
    ("no_scheme_text.png",    "not_a_url"),
    ("plain_text.png",        "not_a_url"),
    ("wifi_config.png",       "not_a_url"),
]


def test_url_validation_unit_cases():
    assert is_valid_url("https://example.com") is True
    assert is_valid_url("http://example.com/path?q=1") is True
    assert is_valid_url("ftp://example.com") is False          # wrong scheme
    assert is_valid_url("example.com") is False                # missing scheme
    assert is_valid_url("not a url at all") is False
    assert is_valid_url("") is False
    assert is_valid_url(None) is False
    print("[PASS] is_valid_url unit cases")


def test_missing_file():
    result = process_qr_image("this_file_does_not_exist.png")
    assert result.status == "unreadable"
    print("[PASS] missing file handled as 'unreadable'")


def test_corrupt_image(tmp_path="corrupt_test.png"):
    # Write garbage bytes with a .png extension to simulate a corrupt upload
    with open(tmp_path, "wb") as f:
        f.write(b"this is not actually a png file")
    result = process_qr_image(tmp_path)
    assert result.status == "unreadable"
    os.remove(tmp_path)
    print("[PASS] corrupt image handled as 'unreadable'")


def test_generated_samples():
    if not os.path.isdir(TEST_DIR):
        print(f"[SKIP] '{TEST_DIR}/' not found — run generate_test_qr.py first.")
        return

    failures = []
    for filename, expected_status in EXPECTED:
        path = os.path.join(TEST_DIR, filename)
        if not os.path.exists(path):
            failures.append(f"{filename}: file missing")
            continue

        result = process_qr_image(path)
        if result.status != expected_status:
            failures.append(
                f"{filename}: expected status '{expected_status}', got '{result.status}' "
                f"(message: {result.message})"
            )
        else:
            extra = f"-> url={result.url!r}" if result.status == "ok" else f"-> {result.message}"
            print(f"[PASS] {filename}: status='{result.status}' {extra}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\nAll {len(EXPECTED)} sample QR codes decoded as expected.")


if __name__ == "__main__":
    test_url_validation_unit_cases()
    test_missing_file()
    test_corrupt_image()
    test_generated_samples()
