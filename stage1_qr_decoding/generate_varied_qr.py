"""
generate_varied_qr.py
---------------------
Renders the SAME URLs as QR codes in many different styles, so the team can
show that the verdict does not depend on how the QR code looks, and measure
how well Stage 1 decodes harder images.

Groups:
    clean     : different versions, error-correction levels, sizes, colors
    hard      : no quiet-zone border, inverted colors
    degraded  : JPEG compression, blur, rotation, small size, low contrast,
                "screenshot-like" (QR pasted on a busy background)

Run:  python generate_varied_qr.py
Out:  test_images_varied/*.png|jpg  and  test_images_varied/manifest.csv
"""

import csv
import io
import os

import qrcode
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

OUT_DIR = "test_images_varied"

URLS = {
    "legit_google":   "https://www.google.com/",
    "legit_docs":     "https://docs.python.org/3/library/urllib.parse.html",
    "phish_ip":       "http://192.168.1.50/login/secure-update",
    "phish_at":       "http://google.com@evil-site.example/login",
    "short_bitly":    "https://bit.ly/3exampleShortLink",
    "long_query":     "https://secure-login.example.com/account/verify?session=abc123&redirect=https%3A%2F%2Fexample.com%2Fhome",
}


def render(url, version=None, ec=ERROR_CORRECT_M, box=10, border=4, fill="black", back="white"):
    qr = qrcode.QRCode(version=version, error_correction=ec, box_size=box, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color=fill, back_color=back).convert("RGB")


def base(url):
    return render(url)


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def screenshot_like(img):
    bg = Image.new("RGB", (900, 700), (232, 234, 240))
    d = ImageDraw.Draw(bg)
    for y in range(0, 700, 40):
        d.line([(0, y), (900, y)], fill=(215, 218, 226), width=2)
    d.rectangle([40, 40, 860, 90], fill=(60, 90, 160))
    qr = img.resize((380, 380))
    bg.paste(qr, (260, 180))
    return jpeg(bg.filter(ImageFilter.GaussianBlur(0.8)), 70)


# name -> (group, function(url) -> PIL image, file extension)
VARIANTS = {
    "base":            ("clean",    lambda u: base(u), "png"),
    "ec_L":            ("clean",    lambda u: render(u, ec=ERROR_CORRECT_L), "png"),
    "ec_Q":            ("clean",    lambda u: render(u, ec=ERROR_CORRECT_Q), "png"),
    "ec_H":            ("clean",    lambda u: render(u, ec=ERROR_CORRECT_H), "png"),
    "box4_border1":    ("clean",    lambda u: render(u, box=4, border=1), "png"),
    "box20":           ("clean",    lambda u: render(u, box=20), "png"),
    "version10":       ("clean",    lambda u: render(u, version=10), "png"),
    "version20":       ("clean",    lambda u: render(u, version=20), "png"),
    "navy_on_cream":   ("clean",    lambda u: render(u, fill="#1a237e", back="#fff8e1"), "png"),
    "green_on_white":  ("clean",    lambda u: render(u, fill="#1b5e20"), "png"),
    "no_border":       ("hard",     lambda u: render(u, border=0), "png"),
    "inverted":        ("hard",     lambda u: render(u, fill="white", back="black"), "png"),
    "jpeg_q30":        ("degraded", lambda u: jpeg(base(u), 30), "jpg"),
    "blur_1_5":        ("degraded", lambda u: base(u).filter(ImageFilter.GaussianBlur(1.5)), "png"),
    "rotate_20":       ("degraded", lambda u: base(u).rotate(20, expand=True, fillcolor="white", resample=Image.BICUBIC), "png"),
    "small_120px":     ("degraded", lambda u: base(u).resize((120, 120), Image.LANCZOS), "png"),
    "low_contrast":    ("degraded", lambda u: ImageEnhance.Contrast(base(u)).enhance(0.35), "png"),
    "screenshot_like": ("degraded", lambda u: screenshot_like(base(u)), "jpg"),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for key, url in URLS.items():
        for vname, (group, fn, ext) in VARIANTS.items():
            fname = f"{key}__{vname}.{ext}"
            fn(url).save(os.path.join(OUT_DIR, fname))
            rows.append([fname, url, vname, group])
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "expected_url", "variant", "group"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} images ({len(URLS)} URLs x {len(VARIANTS)} variants) to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
