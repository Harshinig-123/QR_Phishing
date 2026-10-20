# Member 1: QR Decoding and Payload Typing (Stage 1)

Decodes a QR image **locally** (no network), identifies what the QR code
contains, validates URLs, and flags sensitive non-URL payloads before handing
off to Stage 2 (heuristics) and Stage 3 (ML).

## Setup

```bash
pip install -r requirements.txt
```

`pyzbar` is optional. If it fails to load on Windows (`libzbar-64.dll` /
`libiconv.dll` error), the module falls back to OpenCV automatically.
To fix pyzbar itself, install the Microsoft Visual C++ 2013 Redistributable
(x64). Linux: `sudo apt-get install libzbar0`. macOS: `brew install zbar`.

## Files

| File | Purpose |
|---|---|
| `qr_decoder.py` | Stage 1 module (decode, validate, payload typing) |
| `pipeline.py` | Chains Stage 1 -> 2 -> 3 (**put in repo root**) |
| `generate_test_qr.py` | 19 sample QR codes (URLs and non-URL payload types) |
| `test_decoder.py` | Tests for Stage 1 |
| `generate_varied_qr.py` | Same 6 URLs in 18 rendering styles (108 images) |
| `test_rendering_consistency.py` | Decode success per style; checks no URL is decoded wrongly |
| `test_pipeline.py` | Checks pipeline wiring with fake Stage 2/3 (before integration) |
| `demo_cli.py` | Stage 1 only demo |

## Quick start

```bash
python generate_test_qr.py
python test_decoder.py
python generate_varied_qr.py
python test_rendering_consistency.py            # or: --backend opencv / --backend zbar
python test_pipeline.py
python pipeline.py test_images/legit_url.png
```

## Integration contract (Members 2 and 3)

```python
from qr_decoder import process_qr_image, process_qr_bytes
result = process_qr_image("qr.png")          # or process_qr_bytes(file_bytes)
```

| Field | Meaning |
|---|---|
| `status` | `"ok"`, `"no_qr"`, `"unreadable"`, `"not_a_url"` |
| `url` | Validated URL (only when `status == "ok"`) |
| `payload_type` | `url`, `wifi`, `sms`, `phone`, `email`, `payment_upi`, `payment_crypto`, `app_link`, `script`, `other_scheme_link`, `contact`, `calendar`, `geolocation`, `domain_without_scheme`, `plain_text`, `empty`, `none` |
| `sensitive` | `True` if the payload deserves a user warning |
| `sensitivity_note` | Why (WiFi join, payment, SMS/phone, app link, script, ...) |
| `message`, `raw_data`, `all_detected` | Explanation, decoded text, every payload found |

Always check `status` before using `url`. Stage 1 only checks that something
**is** a URL; it never judges risk. IP-based and `@` URLs correctly return
`ok`; Stage 2/3 decide they are risky.

Optional: `process_qr_image(path, assume_scheme="https")` treats a bare domain
(`www.example.com/login`) as a URL. Off by default.

## Pipeline verdicts (`pipeline.py`)

`phishing`, `legitimate`, `caution` (sensitive non-URL payload),
`not_supported` (non-URL, non-sensitive), `no_qr`, `unreadable`,
`undetermined` (Stage 2/3 not connected or both failed).

Stage 2 must expose `heuristics.analyze_url(url)` and Stage 3
`ml_predict.predict_url(url)` (see their folder READMEs). Until they exist,
placeholders are used and clearly labelled.

## Measured results (for the report)

Decoding the same 6 URLs rendered in 18 styles (clean, hard, degraded):

| Backend | Decoded correctly | Wrong URL decoded |
|---|---|---|
| auto (pyzbar, then OpenCV) | 108 / 108 | 0 |
| OpenCV only | 108 / 108 | 0 |
| pyzbar only | 107 / 108 | 0 |

Re-run on your own machine and report your numbers. Two fixes came from this
test: an invert-and-retry step for white-on-black codes (0/6 before) and
multi-scale OpenCV retries (94% before). Results are on synthetic images with
mild degradation, not real-world photos of printed codes.
