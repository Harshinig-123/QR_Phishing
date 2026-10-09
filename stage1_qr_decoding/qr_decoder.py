"""
qr_decoder.py
-------------
Member 1 — Stage 1: QR Decoding

Responsibilities covered in this module:
    1. Handle QR image input (from a file path OR raw uploaded bytes).
    2. Decode the QR code using pyzbar.
    3. Extract the embedded data.
    4. Validate whether the decoded content is actually a URL.
    5. Handle invalid / unreadable / non-URL QR codes gracefully.
    6. Return a clean, structured result that Stage 2 (Member 2) and
       Stage 3 (Member 3) can consume directly.

Integration contract
---------------------
Every public function in this module returns a QRDecodeResult object
(see below). Downstream stages should check `result.status` first:

    result.status == "ok"       -> result.url is a validated URL, safe to pass on
    result.status == "no_qr"    -> no QR code was found in the image
    result.status == "unreadable" -> image could not be opened/processed
    result.status == "not_a_url"  -> QR decoded fine, but content isn't a URL

Example (what Member 2 / Member 3 will do):

    from qr_decoder import process_qr_image

    result = process_qr_image("uploaded_qr.png")
    if result.status == "ok":
        url = result.url
        # ... pass url into Stage 2 (heuristics) ...
    else:
        # show result.message to the user / log it
        ...
"""

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Optional, List
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

# pyzbar needs a native DLL that fails to load on some Windows machines
# (missing Visual C++ 2013 runtime). We treat it as optional and fall back
# to OpenCV's built-in QR detector, which has no such dependency.
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
    raise ImportError(
        "No QR backend available. Run: pip install opencv-python-headless"
    )


# --------------------------------------------------------------------------
# Result object — this is the "contract" the rest of the team codes against
# --------------------------------------------------------------------------

@dataclass
class QRDecodeResult:
    status: str                 # "ok" | "no_qr" | "unreadable" | "not_a_url"
    url: Optional[str] = None   # populated only when status == "ok"
    raw_data: Optional[str] = None  # whatever was decoded, even if not a URL
    message: str = ""           # human-readable explanation, for logs/UI
    all_detected: Optional[List[str]] = None  # raw payloads of every QR code found

    def to_dict(self) -> dict:
        """Convenience for passing this across API/UI boundaries as JSON."""
        return asdict(self)

    def __bool__(self) -> bool:
        # lets callers do `if result:` as a shorthand for "status == ok"
        return self.status == "ok"


# --------------------------------------------------------------------------
# Step 1: Load the image (from a file path or raw bytes, e.g. a web upload)
# --------------------------------------------------------------------------

def _load_image_from_path(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()  # force-read the file now so corrupt files fail here
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
# Step 2: Decode the QR code(s) in the image
# --------------------------------------------------------------------------

def _decode_with_zbar(img: Image.Image) -> List[str]:
    if not HAS_ZBAR:
        return []
    for candidate in (img, img.convert("L")):  # grayscale retry helps on photos
        try:
            results = zbar_decode(candidate)
        except Exception:
            results = []
        if results:
            return [r.data.decode("utf-8", errors="replace") for r in results]
    return []


def _decode_with_opencv(img: Image.Image) -> List[str]:
    if not HAS_CV2:
        return []
    try:
        arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(arr)
        if data:
            return [data]
        # QR codes with no quiet zone sometimes decode after adding a border
        arr = cv2.copyMakeBorder(arr, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
        data, _, _ = detector.detectAndDecode(arr)
        return [data] if data else []
    except Exception:
        return []


def _decode_qr(img: Image.Image) -> List[str]:
    """Try pyzbar first (if it loaded), then fall back to OpenCV."""
    return _decode_with_zbar(img) or _decode_with_opencv(img)


# --------------------------------------------------------------------------
# Step 3: Validate that the decoded content is actually a URL
# --------------------------------------------------------------------------

def is_valid_url(candidate: str) -> bool:
    """
    Lightweight URL validation (no external dependency needed).
    Requires: a scheme (http/https) and a non-empty network location.
    A QR code can encode plain text, WiFi configs (WIFI:...), contact
    cards (vCard), etc. — this filters those out.
    """
    if not candidate or not isinstance(candidate, str):
        return False

    candidate = candidate.strip()

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc:
        return False

    return True


def normalize_url(candidate: str) -> str:
    """
    Minor cleanup before handing the URL to Stage 2.
    Strips surrounding whitespace; does NOT change the URL's meaning
    (no lowercasing of the path, since paths can be case-sensitive).
    """
    return candidate.strip()


# --------------------------------------------------------------------------
# Public entry points — these are what Members 2 and 3 will actually call
# --------------------------------------------------------------------------

def process_qr_image(path: str) -> QRDecodeResult:
    """Main entry point when the QR code is available as a file on disk."""
    img = _load_image_from_path(path)
    if img is None:
        return QRDecodeResult(
            status="unreadable",
            message=f"Could not open or read image file: {path}",
        )
    return _process_image(img)


def process_qr_bytes(data: bytes) -> QRDecodeResult:
    """
    Main entry point when the QR code arrives as raw bytes — e.g. a file
    uploaded through a web form (Flask's `request.files['qr'].read()`).
    """
    img = _load_image_from_bytes(data)
    if img is None:
        return QRDecodeResult(
            status="unreadable",
            message="Could not decode the uploaded data as an image.",
        )
    return _process_image(img)


def _process_image(img: Image.Image) -> QRDecodeResult:
    payloads = _decode_qr(img)

    if not payloads:
        return QRDecodeResult(
            status="no_qr",
            message="No QR code could be detected in this image.",
        )

    # If multiple QR codes are present, we validate each and use the first
    # valid URL found; this keeps Stage 2/3 simple (one URL in, one verdict out)
    # while still surfacing everything that was detected for debugging/report use.
    for payload in payloads:
        if is_valid_url(payload):
            return QRDecodeResult(
                status="ok",
                url=normalize_url(payload),
                raw_data=payload,
                message="QR code decoded and validated successfully.",
                all_detected=payloads,
            )

    # QR code(s) decoded fine, but none of them contain a URL
    return QRDecodeResult(
        status="not_a_url",
        raw_data=payloads[0],
        message=(
            "QR code was read successfully, but its content is not a URL "
            f"(got: {payloads[0][:80]!r}). This tool only analyzes "
            "URL-based QR codes."
        ),
        all_detected=payloads,
    )