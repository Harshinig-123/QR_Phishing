"""
qr_decoder.py
-------------
Member 1 - Stage 1: QR Decoding and Payload Typing

What this module does:
    1. Accepts a QR image (file path OR raw uploaded bytes).
    2. Decodes it locally (pyzbar, with an OpenCV fallback).
    3. Identifies the PAYLOAD TYPE (url, wifi, sms, payment, ...).
    4. Validates URL payloads and flags sensitive non-URL payloads.
    5. Returns one structured QRDecodeResult for Stage 2 / Stage 3.

Nothing here touches the network. Decoding and typing are purely local.

Integration contract
--------------------
    result.status         "ok" | "no_qr" | "unreadable" | "not_a_url"
    result.url            validated URL (only when status == "ok")
    result.payload_type   "url", "wifi", "sms", "phone", "email",
                          "payment_upi", "payment_crypto", "app_link",
                          "script", "other_scheme_link", "contact",
                          "calendar", "geolocation",
                          "domain_without_scheme", "plain_text", "empty", "none"
    result.sensitive      True if the payload type deserves a warning
                          (WiFi join, payment, SMS/phone, app link, script...)
    result.sensitivity_note  human-readable reason for the warning

Example:
    from qr_decoder import process_qr_image
    r = process_qr_image("qr.png")
    if r.status == "ok":
        ...                      # send r.url to Stage 2
    elif r.sensitive:
        print(r.sensitivity_note)  # e.g. "Joins a WiFi network..."
"""

import re
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

# pyzbar needs a native DLL that fails to load on some Windows machines
# (missing Visual C++ 2013 runtime). It is optional: we fall back to OpenCV.
try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except Exception:
    zbar_decode = None
    HAS_ZBAR = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

if not HAS_ZBAR and not HAS_CV2:
    raise ImportError("No QR backend available. Run: pip install opencv-python-headless")


# --------------------------------------------------------------------------
# Result object: the "contract" the rest of the team codes against
# --------------------------------------------------------------------------

@dataclass
class QRDecodeResult:
    status: str                          # "ok" | "no_qr" | "unreadable" | "not_a_url"
    url: Optional[str] = None            # set only when status == "ok"
    raw_data: Optional[str] = None       # decoded text, even if not a URL
    message: str = ""                    # human-readable explanation
    all_detected: Optional[List[str]] = None   # every payload found in the image
    payload_type: str = "none"           # see module docstring
    sensitive: bool = False              # True -> show a warning to the user
    sensitivity_note: str = ""           # why it is sensitive

    def to_dict(self) -> dict:
        return asdict(self)

    def __bool__(self) -> bool:
        return self.status == "ok"


# --------------------------------------------------------------------------
# Step 1: load the image
# --------------------------------------------------------------------------

def _load_image_from_path(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def _load_image_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        return None


# --------------------------------------------------------------------------
# Step 2: decode (pyzbar first, OpenCV fallback)
# --------------------------------------------------------------------------

def _decode_with_zbar(img: Image.Image) -> List[str]:
    if not HAS_ZBAR:
        return []
    for candidate in (img, img.convert("L")):
        try:
            results = zbar_decode(candidate)
        except Exception:
            results = []
        if results:
            return [r.data.decode("utf-8", errors="replace") for r in results]
    return []


def _opencv_detectors():
    """Aruco-based detector (OpenCV >= 4.8) is more robust; classic one as backup."""
    dets = []
    if hasattr(cv2, "QRCodeDetectorAruco"):
        dets.append(cv2.QRCodeDetectorAruco())
    dets.append(cv2.QRCodeDetector())
    return dets


def _decode_with_opencv(img: Image.Image) -> List[str]:
    """Try each OpenCV detector at several scales, with and without a white border
    (denser or smaller codes often only decode after resizing)."""
    if not HAS_CV2:
        return []
    try:
        base = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        for detector in _opencv_detectors():
            for scale in (1.0, 2.0, 0.5, 3.0):
                arr = base if scale == 1.0 else cv2.resize(
                    base, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
                for pad in (0, 40):
                    work = arr if pad == 0 else cv2.copyMakeBorder(
                        arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    data, _, _ = detector.detectAndDecode(work)
                    if data:
                        return [data]
        return []
    except Exception:
        return []


def _decode_qr(img: Image.Image) -> List[str]:
    """Try pyzbar, then OpenCV; if both fail, retry on the colour-inverted image
    (white-on-black QR codes are not read by many decoders)."""
    found = _decode_with_zbar(img) or _decode_with_opencv(img)
    if found:
        return found
    try:
        inverted = ImageOps.invert(img.convert("RGB"))
    except Exception:
        return []
    return _decode_with_zbar(inverted) or _decode_with_opencv(inverted)


# --------------------------------------------------------------------------
# Step 3: URL validation
# --------------------------------------------------------------------------

def is_valid_url(candidate: str) -> bool:
    """True only for http/https URLs with a host and no whitespace."""
    if not candidate or not isinstance(candidate, str):
        return False
    candidate = candidate.strip()
    if re.search(r"\s", candidate):
        return False
    try:
        parsed = urlparse(candidate)
        host = parsed.hostname
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc) and bool(host)


def normalize_url(candidate: str) -> str:
    """Strip surrounding whitespace only; never alter the URL's meaning."""
    return candidate.strip()


# --------------------------------------------------------------------------
# Step 4: payload typing (NEW)
# --------------------------------------------------------------------------

# (prefixes, payload_type, sensitive, note). Matched case-insensitively, in order.
_PAYLOAD_RULES = [
    (("wifi:",), "wifi", True,
     "Joins a WiFi network. A rogue network can intercept your traffic."),
    (("upi://",), "payment_upi", True,
     "Opens a payment request. Check the payee and amount before approving."),
    (("bitcoin:", "ethereum:", "litecoin:", "monero:", "dogecoin:"), "payment_crypto", True,
     "Starts a cryptocurrency payment to an address chosen by the code's creator."),
    (("sms:", "smsto:", "mms:", "mmsto:"), "sms", True,
     "Pre-fills a text message to a number chosen by the code's creator (premium-rate/smishing risk)."),
    (("tel:",), "phone", True,
     "Prepares a phone call to a number chosen by the code's creator (vishing/premium-rate risk)."),
    (("mailto:", "matmsg:"), "email", True,
     "Pre-fills an email to an address chosen by the code's creator."),
    (("intent:", "market:", "itms-apps:", "itms-services:"), "app_link", True,
     "Opens or installs an app. Can be used to push malicious apps."),
    (("javascript:", "data:", "vbscript:"), "script", True,
     "Contains executable script or inline data. Very unusual for a legitimate QR code."),
    (("begin:vcard", "mecard:"), "contact", False,
     "Adds a contact card. Check the details before saving."),
    (("begin:vevent", "begin:vcalendar"), "calendar", False,
     "Adds a calendar event."),
    (("geo:",), "geolocation", False,
     "Opens a map location."),
]

_OTHER_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_DOMAIN_LIKE = re.compile(r"^(?:www\.)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)+(?:[/:?#]\S*)?$", re.IGNORECASE)


def classify_payload(payload: str) -> Tuple[str, bool, str]:
    """Return (payload_type, sensitive, note) for decoded QR text."""
    s = (payload or "").strip()
    if not s:
        return ("empty", False, "")
    low = s.lower()

    if low.startswith(("http://", "https://")):
        return ("url", False, "")

    for prefixes, ptype, sensitive, note in _PAYLOAD_RULES:
        if low.startswith(prefixes):
            return (ptype, sensitive, note)

    if _OTHER_SCHEME.match(s):
        return ("other_scheme_link", True,
                "Uses a non-web link scheme that may open an app or protocol handler.")

    if _DOMAIN_LIKE.match(s) and " " not in s:
        return ("domain_without_scheme", False,
                "Looks like a web address without http/https. Many phones will still open it in a browser.")

    return ("plain_text", False, "")


def _safe_preview(payload: str, limit: int = 80) -> str:
    """Short preview for messages; masks WiFi passwords."""
    text = payload
    if text.lower().startswith("wifi:"):
        text = re.sub(r"(P:)([^;]*)", r"\1****", text, flags=re.IGNORECASE)
    return text[:limit] + ("..." if len(text) > limit else "")


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def process_qr_image(path: str, assume_scheme: Optional[str] = None) -> QRDecodeResult:
    """QR code available as a file on disk."""
    img = _load_image_from_path(path)
    if img is None:
        return QRDecodeResult(status="unreadable",
                              message=f"Could not open or read image file: {path}")
    return _process_image(img, assume_scheme)


def process_qr_bytes(data: bytes, assume_scheme: Optional[str] = None) -> QRDecodeResult:
    """QR code as raw bytes, e.g. Flask: request.files['qr'].read()."""
    img = _load_image_from_bytes(data)
    if img is None:
        return QRDecodeResult(status="unreadable",
                              message="Could not decode the uploaded data as an image.")
    return _process_image(img, assume_scheme)


def _process_image(img: Image.Image, assume_scheme: Optional[str] = None) -> QRDecodeResult:
    """
    assume_scheme: optional, e.g. "https". If set, payloads that look like a
    bare domain ("www.example.com/login") are treated as URLs with that scheme.
    Off by default so behaviour stays strict.
    """
    payloads = _decode_qr(img)
    if not payloads:
        return QRDecodeResult(status="no_qr",
                              message="No QR code could be detected in this image.")

    typed = [(p,) + classify_payload(p) for p in payloads]  # (payload, type, sensitive, note)

    # Sensitive payloads elsewhere in the image must not be silently lost.
    sensitive_notes = [f"{t}: {n}" for (_, t, s, n) in typed if s]

    # 1) Prefer a valid URL
    for payload, ptype, _, _ in typed:
        if is_valid_url(payload):
            return QRDecodeResult(
                status="ok", url=normalize_url(payload), raw_data=payload,
                message="QR code decoded and validated successfully.",
                all_detected=payloads, payload_type="url",
                sensitive=bool(sensitive_notes),
                sensitivity_note=("Another QR code in this image carries a sensitive payload ("
                                  + "; ".join(sensitive_notes) + ")") if sensitive_notes else "",
            )

    # 2) Optional: bare domains become URLs if the caller opts in
    if assume_scheme:
        for payload, ptype, _, _ in typed:
            if ptype == "domain_without_scheme":
                candidate = f"{assume_scheme}://{payload.strip()}"
                if is_valid_url(candidate):
                    return QRDecodeResult(
                        status="ok", url=candidate, raw_data=payload,
                        message=f"No scheme in QR code; assumed {assume_scheme}://.",
                        all_detected=payloads, payload_type="domain_without_scheme",
                        sensitive=bool(sensitive_notes),
                        sensitivity_note="; ".join(sensitive_notes),
                    )

    # 3) Decoded fine, but not a web URL. Report the most security-relevant payload.
    primary = next((t for t in typed if t[2]), typed[0])
    payload, ptype, sensitive, note = primary
    msg = (f"QR code was read successfully, but its payload type is '{ptype}', "
           f"not a web URL (preview: {_safe_preview(payload)!r}). "
           "This tool analyzes URL-based QR codes.")
    return QRDecodeResult(
        status="not_a_url", raw_data=payload, message=msg, all_detected=payloads,
        payload_type=ptype, sensitive=sensitive, sensitivity_note=note,
    )
