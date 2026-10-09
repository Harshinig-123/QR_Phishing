# Member 1 — QR Decoding (Stage 1)

Decodes an uploaded QR code image, extracts the embedded content, and
validates that it's a URL before handing it off to Stage 2 (heuristics)
and Stage 3 (ML classification).

## Setup

```bash
pip install -r requirements.txt
```

### Windows: "Could not find module libzbar-64.dll / libiconv.dll"

`pyzbar` is **optional** — the module automatically falls back to OpenCV,
so after `pip install -r requirements.txt` everything works even if you see
this error elsewhere. If you still want pyzbar working, install the
**Microsoft Visual C++ Redistributable 2013 (x64)** (`vcredist_x64.exe`)
from Microsoft's site, then restart your terminal.

Linux: `sudo apt-get install libzbar0` · macOS: `brew install zbar`

## Files

| File | Purpose |
|---|---|
| `qr_decoder.py` | Main module — all decoding/validation logic lives here |
| `generate_test_qr.py` | Creates sample QR images in `test_images/` to test against |
| `test_decoder.py` | Automated test suite |
| `demo_cli.py` | Quick manual demo: `python demo_cli.py some_qr.png` |

## Quick start

```bash
python generate_test_qr.py   # creates test_images/ with 8 sample QR codes
python test_decoder.py       # runs the full test suite
python demo_cli.py test_images/legit_url.png   # try it on one image
```

## Integration contract (for Member 2 & Member 3)

Call one of these two functions depending on how the QR image arrives:

```python
from qr_decoder import process_qr_image, process_qr_bytes

# If you have a file path (e.g. CLI, or a file already saved to disk)
result = process_qr_image("uploaded_qr.png")

# If you have raw bytes (e.g. Flask: request.files['qr'].read())
result = process_qr_bytes(uploaded_file_bytes)
```

Both return a `QRDecodeResult` object:

```python
result.status      # "ok" | "no_qr" | "unreadable" | "not_a_url"
result.url         # the validated URL (only set when status == "ok")
result.raw_data    # whatever was decoded, even if it wasn't a URL
result.message     # human-readable explanation (good for UI/logs)
result.all_detected  # list of every payload found, if multiple QR codes present
```

**Always check `result.status` before using `result.url`.** Example:

```python
result = process_qr_image(path)

if result.status == "ok":
    url = result.url
    # pass `url` into Stage 2: heuristic_check(url), then Stage 3: ml_predict(url)
else:
    # don't continue the pipeline — show result.message to the user instead
    print(f"Could not process this QR code: {result.message}")
```

You can also use `if result:` as shorthand for `if result.status == "ok":`.

### Status meanings

| Status | Meaning | What downstream stages should do |
|---|---|---|
| `"ok"` | Valid URL found | Proceed to Stage 2/3 with `result.url` |
| `"no_qr"` | No QR code detected in the image | Stop; ask user to re-upload a clearer image |
| `"unreadable"` | File isn't a valid/openable image | Stop; ask user to re-upload |
| `"not_a_url"` | QR decoded fine, but isn't a URL (e.g. WiFi config, plain text) | Stop; this tool only analyzes URL-based QR codes |

### Note on scope

Stage 1 only validates that the content **is structurally a URL** (has a
scheme + domain). It does **not** judge whether the URL looks suspicious —
that's intentionally left to Stage 2 (heuristics) and Stage 3 (ML), so
`phishing_style_ip.png` and `phishing_style_at.png` in the test set both
correctly return `status="ok"` even though they look risky; it's Stage
2/3's job to flag that, not Stage 1's.