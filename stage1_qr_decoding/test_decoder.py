"""
test_decoder.py - tests for qr_decoder.py. Run after generate_test_qr.py.

Run:  python test_decoder.py
"""

import os
import sys

from qr_decoder import process_qr_image, is_valid_url, classify_payload

TEST_DIR = "test_images"

# (filename, expected_status, expected_payload_type, expected_sensitive)
EXPECTED = [
    ("legit_url.png",         "ok",        "url",                   False),
    ("legit_url_subpage.png", "ok",        "url",                   False),
    ("phishing_style_ip.png", "ok",        "url",                   False),  # Stage 2/3 judge risk, not Stage 1
    ("phishing_style_at.png", "ok",        "url",                   False),
    ("shortened_url.png",     "ok",        "url",                   False),
    ("uppercase_scheme.png",  "ok",        "url",                   False),
    ("no_scheme_text.png",    "not_a_url", "domain_without_scheme", False),
    ("plain_text.png",        "not_a_url", "plain_text",            False),
    ("wifi_config.png",       "not_a_url", "wifi",                  True),
    ("sms_payload.png",       "not_a_url", "sms",                   True),
    ("phone_payload.png",     "not_a_url", "phone",                 True),
    ("email_payload.png",     "not_a_url", "email",                 True),
    ("upi_payment.png",       "not_a_url", "payment_upi",           True),
    ("crypto_payment.png",    "not_a_url", "payment_crypto",        True),
    ("app_link.png",          "not_a_url", "app_link",              True),
    ("script_payload.png",    "not_a_url", "script",                True),
    ("ftp_link.png",          "not_a_url", "other_scheme_link",     True),
    ("vcard_contact.png",     "not_a_url", "contact",               False),
    ("geo_location.png",      "not_a_url", "geolocation",           False),
]


def test_url_validation():
    assert is_valid_url("https://example.com")
    assert is_valid_url("http://example.com/path?q=1")
    assert is_valid_url("HTTP://EXAMPLE.COM")
    assert not is_valid_url("ftp://example.com")
    assert not is_valid_url("example.com")
    assert not is_valid_url("http://")
    assert not is_valid_url("http://exa mple.com")
    assert not is_valid_url("not a url at all")
    assert not is_valid_url("")
    assert not is_valid_url(None)
    print("[PASS] is_valid_url unit cases")


def test_classify_payload():
    cases = {
        "https://a.com": ("url", False),
        "WIFI:S:x;T:WPA;P:y;;": ("wifi", True),
        "wifi:S:x;;": ("wifi", True),
        "smsto:123:hi": ("sms", True),
        "tel:+1555": ("phone", True),
        "upi://pay?pa=a@b": ("payment_upi", True),
        "javascript:alert(1)": ("script", True),
        "custom-app://open": ("other_scheme_link", True),
        "www.example.com/login": ("domain_without_scheme", False),
        "hello world": ("plain_text", False),
        "": ("empty", False),
    }
    for text, (ptype, sens) in cases.items():
        got = classify_payload(text)
        assert got[0] == ptype and got[1] == sens, f"{text!r}: expected {(ptype, sens)}, got {got[:2]}"
    print("[PASS] classify_payload unit cases")


def test_wifi_password_masked():
    from qr_decoder import _safe_preview
    assert "secret123" not in _safe_preview("WIFI:T:WPA;S:Net;P:secret123;;")
    print("[PASS] WiFi password masked in message preview")


def test_missing_file():
    assert process_qr_image("this_file_does_not_exist.png").status == "unreadable"
    print("[PASS] missing file -> 'unreadable'")


def test_corrupt_image(tmp_path="corrupt_test.png"):
    with open(tmp_path, "wb") as f:
        f.write(b"this is not actually a png file")
    assert process_qr_image(tmp_path).status == "unreadable"
    os.remove(tmp_path)
    print("[PASS] corrupt image -> 'unreadable'")


def test_blank_image_no_qr():
    from PIL import Image
    Image.new("RGB", (200, 200), "white").save("blank_test.png")
    assert process_qr_image("blank_test.png").status == "no_qr"
    os.remove("blank_test.png")
    print("[PASS] blank image -> 'no_qr'")


def test_assume_scheme_option():
    path = os.path.join(TEST_DIR, "no_scheme_text.png")
    if not os.path.exists(path):
        return
    r = process_qr_image(path, assume_scheme="https")
    assert r.status == "ok" and r.url.startswith("https://www.not-a-valid")
    print("[PASS] assume_scheme='https' turns a bare domain into a URL")


def test_generated_samples():
    if not os.path.isdir(TEST_DIR):
        print(f"[SKIP] '{TEST_DIR}/' not found. Run generate_test_qr.py first.")
        return
    failures = []
    for filename, status, ptype, sensitive in EXPECTED:
        path = os.path.join(TEST_DIR, filename)
        if not os.path.exists(path):
            failures.append(f"{filename}: file missing (re-run generate_test_qr.py)")
            continue
        r = process_qr_image(path)
        if (r.status, r.payload_type, r.sensitive) != (status, ptype, sensitive):
            failures.append(f"{filename}: expected {(status, ptype, sensitive)}, "
                            f"got {(r.status, r.payload_type, r.sensitive)}")
        else:
            flag = " [SENSITIVE]" if r.sensitive else ""
            print(f"[PASS] {filename}: {r.status} / {r.payload_type}{flag}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"\nAll {len(EXPECTED)} sample QR codes decoded and typed as expected.")


if __name__ == "__main__":
    test_url_validation()
    test_classify_payload()
    test_wifi_password_masked()
    test_missing_file()
    test_corrupt_image()
    test_blank_image_no_qr()
    test_generated_samples()
    test_assume_scheme_option()
