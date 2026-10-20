"""
generate_test_qr.py
--------------------
Creates sample QR images in ./test_images/ covering URLs, URL tricks and
non-URL payload types.

Run:  python generate_test_qr.py
"""

import os
import qrcode

OUTPUT_DIR = "test_images"

TEST_CASES = {
    # --- URLs (Stage 1 passes these on; Stage 2/3 judge them) ---
    "legit_url.png":            "https://www.google.com",
    "legit_url_subpage.png":    "https://docs.python.org/3/library/urllib.parse.html",
    "phishing_style_ip.png":    "http://192.168.1.50/login/secure-update",
    "phishing_style_at.png":    "http://google.com@evil-site.example/login",
    "shortened_url.png":        "https://bit.ly/3exampleShortLink",
    "uppercase_scheme.png":     "HTTPS://EXAMPLE.COM/Login",
    # --- Not web URLs ---
    "no_scheme_text.png":       "www.not-a-valid-scheme-example.com",
    "plain_text.png":           "This is just plain text, not a URL at all.",
    "wifi_config.png":          "WIFI:T:WPA;S:MyNetwork;P:mypassword;;",
    "sms_payload.png":          "SMSTO:+15550100:You won! Reply YES to claim",
    "phone_payload.png":        "tel:+15550100",
    "email_payload.png":        "mailto:support@example.com?subject=Verify",
    "upi_payment.png":          "upi://pay?pa=shop@examplebank&pn=Shop&am=500&cu=INR",
    "crypto_payment.png":       "bitcoin:bc1qexampleaddress0000000000000000000?amount=0.01",
    "app_link.png":             "market://details?id=com.example.app",
    "script_payload.png":       "javascript:alert(document.cookie)",
    "ftp_link.png":             "ftp://files.example.com/update.exe",
    "vcard_contact.png":        "BEGIN:VCARD\nVERSION:3.0\nFN:Example Person\nTEL:+15550100\nEND:VCARD",
    "geo_location.png":         "geo:17.3850,78.4867",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, content in TEST_CASES.items():
        qrcode.make(content).save(os.path.join(OUTPUT_DIR, filename))
        print(f"Created {os.path.join(OUTPUT_DIR, filename)}")
    print(f"\nDone. {len(TEST_CASES)} test QR codes written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
