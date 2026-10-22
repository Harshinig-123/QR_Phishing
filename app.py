"""
app.py - small local web UI for Quishing Shield.

Put this file in the repo root, next to pipeline.py.

    pip install flask
    python app.py
    then open  http://127.0.0.1:5000  in your browser

Two ways to use it:
  1. Type a link: the page draws its QR code, decodes it, and runs all three stages.
  2. Upload a QR image: it runs the same pipeline on your image.

It runs only on your own computer (127.0.0.1), works offline (no external fonts or
scripts), and never opens the link. Stage 2 may contact a URL shortener service
for known shorteners; everything else stays local.
"""

import base64
import io
import re
import warnings

import qrcode
from flask import Flask, render_template_string, request
from PIL import Image

import pipeline

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024  # 3 MB upload limit

NON_URL_PREFIXES = ("wifi:", "tel:", "sms:", "smsto:", "mms:", "mailto:", "upi:", "bitcoin:",
                    "ethereum:", "geo:", "begin:vcard", "javascript:", "market:", "intent:", "data:")

VERDICT_TEXT = {
    "phishing":      "Likely phishing. Do not open this link.",
    "suspicious":    "Suspicious. Flagged for review; treat with caution.",
    "legitimate":    "No red flags found. This is not a guarantee of safety.",
    "caution":       "Sensitive (non-URL) content. Check it before acting on it.",
    "not_supported": "Readable QR code, but not a web link, so it was not analyzed.",
    "no_qr":         "No QR code could be found in this image.",
    "unreadable":    "This file could not be read as an image.",
    "undetermined":  "No detection stage produced a verdict.",
}

PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quishing Shield</title>
<style>
  :root { --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#1d4ed8; }
  * { box-sizing: border-box; }
  body { font-family: Segoe UI, Arial, sans-serif; color: var(--ink); margin: 0; background:#f8fafc; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 24px 16px 60px; }
  h1 { margin: 0 0 4px; font-size: 28px; } .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:14px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; } .row > .card { flex:1 1 380px; margin-bottom:0; }
  label { font-weight:600; display:block; margin-bottom:6px; }
  input[type=text], input[type=file] { width:100%; padding:10px; border:1px solid #cbd5e1; border-radius:8px; font-size:15px; }
  button { margin-top:10px; background:var(--accent); color:#fff; border:0; padding:10px 18px; border-radius:8px; font-size:15px; cursor:pointer; }
  .banner { border-radius:12px; padding:16px 18px; margin:14px 0; color:#fff; }
  .banner h2 { margin:0; font-size:26px; letter-spacing:.5px; text-transform:uppercase; } .banner p { margin:4px 0 0; }
  .v-phishing{background:#b91c1c} .v-suspicious{background:#c2410c} .v-legitimate{background:#15803d}
  .v-caution{background:#a16207} .v-not_supported,.v-no_qr,.v-unreadable,.v-undetermined{background:#475569}
  table { width:100%; border-collapse:collapse; } td { padding:5px 4px; border-bottom:1px solid var(--line); vertical-align:top; }
  td:first-child { color:var(--muted); width:38%; } code { background:#f1f5f9; padding:1px 5px; border-radius:4px; word-break:break-all; }
  .pill { display:inline-block; padding:1px 9px; border-radius:999px; font-size:13px; font-weight:600; background:#e2e8f0; }
  .p-phishing{background:#fecaca} .p-suspicious{background:#fed7aa} .p-legitimate{background:#bbf7d0}
  img.qr { width:170px; height:170px; image-rendering:pixelated; border:1px solid var(--line); border-radius:8px; }
  ul { margin:6px 0 0 18px; padding:0; } .note { color:var(--muted); font-size:13px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap:14px; }
</style></head><body><div class="wrap">
<h1>Quishing Shield</h1>
<p class="sub">QR code phishing detection: decode, heuristic rules, ML. Everything runs on this computer.</p>

<div class="row">
  <form class="card" method="post" action="/" >
    <label for="link">Type a link</label>
    <input type="text" id="link" name="link" placeholder="e.g. http://secure-bank-login.example/verify" value="{{ typed or '' }}">
    <button type="submit">Make QR and analyze</button>
    <p class="note">Use made-up domains (like .example) for phishing demos. The link is never opened.</p>
  </form>
  <form class="card" method="post" action="/" enctype="multipart/form-data">
    <label for="qr">Or upload a QR image</label>
    <input type="file" id="qr" name="qr" accept="image/*">
    <button type="submit">Analyze image</button>
    <p class="note">PNG or JPG, up to 3 MB.</p>
  </form>
</div>

{% if error %}<div class="card"><b>{{ error }}</b></div>{% endif %}

{% if r %}
  <div class="banner v-{{ r.final_verdict }}">
    <h2>{{ r.final_verdict.replace('_', ' ') }}</h2>
    <p>{{ verdict_text }}</p>
  </div>

  <div class="grid">
    <div class="card">
      <b>QR code analyzed</b><br>
      {% if preview %}<img class="qr" alt="QR code" src="data:image/png;base64,{{ preview }}">{% else %}<span class="note">No preview.</span>{% endif %}
      {% if note %}<p class="note">{{ note }}</p>{% endif %}
    </div>

    <div class="card">
      <b>Stage 1: decode</b>
      <table>
        <tr><td>Status</td><td><span class="pill">{{ r.stage1.status }}</span></td></tr>
        <tr><td>Payload type</td><td>{{ r.stage1.payload_type }}</td></tr>
        <tr><td>URL</td><td>{% if r.stage1.url %}<code>{{ r.stage1.url }}</code>{% else %}-{% endif %}</td></tr>
        <tr><td>Sensitive</td><td>{{ 'yes' if r.stage1.sensitive else 'no' }}</td></tr>
      </table>
      {% if r.stage1.sensitivity_note %}<p class="note">{{ r.stage1.sensitivity_note }}</p>{% endif %}
    </div>

    {% if r.stage2 %}
    <div class="card">
      <b>Stage 2: heuristic rules</b>
      {% if r.stage2.placeholder %}<p class="note">Not connected.</p>
      {% elif r.stage2.error %}<p class="note">Error: {{ r.stage2.error }}</p>
      {% else %}
      <table>
        <tr><td>Verdict</td><td><span class="pill p-{{ r.stage2.verdict }}">{{ r.stage2.verdict }}</span></td></tr>
        <tr><td>Risk score</td><td>{{ r.stage2.score }}</td></tr>
        <tr><td>Shortened link</td><td>{{ 'yes' if r.stage2.is_shortener else 'no' }}</td></tr>
        {% if r.stage2.is_shortener %}<tr><td>Resolved to</td><td><code>{{ r.stage2.resolved_url }}</code></td></tr>{% endif %}
      </table>
      {% if r.stage2.triggered_rules %}<ul>{% for t in r.stage2.triggered_rules %}<li>{{ t }}</li>{% endfor %}</ul>
      {% else %}<p class="note">No rules triggered.</p>{% endif %}
      {% endif %}
    </div>
    {% endif %}

    {% if r.stage3 %}
    <div class="card">
      <b>Stage 3: ML classifier</b>
      {% if r.stage3.placeholder %}<p class="note">Not connected.</p>
      {% elif r.stage3.error %}<p class="note">Error: {{ r.stage3.error }}</p>
      {% else %}
      <table>
        <tr><td>Prediction</td><td><span class="pill p-{{ r.stage3.prediction }}">{{ r.stage3.prediction }}</span></td></tr>
        <tr><td>Confidence</td><td>{{ r.stage3.probability }}</td></tr>
      </table>
      {% if r.stage3.top_factors %}<p class="note">Top factors:</p>
      <ul>{% for f in r.stage3.top_factors[:4] %}<li>{{ f.feature }} (pushes toward {{ f.direction }})</li>{% endfor %}</ul>{% endif %}
      {% endif %}
    </div>
    {% endif %}
  </div>

  {% if r.reasons %}
  <div class="card"><b>Why this verdict</b><ul>{% for x in r.reasons %}<li>{{ x }}</li>{% endfor %}</ul></div>
  {% endif %}
  <p class="note">Rules are the primary signal; the ML layer alone can only raise a link to "suspicious".
  Results on a small synthetic test set are indicative, not a guarantee.</p>
{% endif %}
</div></body></html>
"""


def prepare_payload(text):
    text = text.strip()
    low = text.lower()
    if "://" in text or low.startswith(NON_URL_PREFIXES):
        return text, ""
    if " " not in text and "." in text:
        return "https://" + text, "No scheme typed, so https:// was added."
    return text, ""


def qr_png_bytes(payload):
    buf = io.BytesIO()
    qrcode.make(payload).save(buf, format="PNG")
    return buf.getvalue()


def preview_from_bytes(data):
    """Re-encode any uploaded image as PNG for display; None if it is not an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


@app.errorhandler(413)
def too_large(_):
    return render_template_string(PAGE, r=None, error="File too large (limit 3 MB).", typed=""), 413


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(r=None, error=None, typed="", preview=None, note="", verdict_text="")
    if request.method == "POST":
        link = (request.form.get("link") or "").strip()
        upload = request.files.get("qr")
        data, note = None, ""
        if link:
            payload, note = prepare_payload(link)
            data = qr_png_bytes(payload)
            ctx["typed"] = link
        elif upload and upload.filename:
            data = upload.read()
        else:
            ctx["error"] = "Type a link or choose a QR image first."
        if data:
            r = pipeline.run_pipeline(data, from_bytes=True)
            ctx.update(r=r, preview=preview_from_bytes(data), note=note,
                       verdict_text=VERDICT_TEXT.get(r["final_verdict"], ""))
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    print(f"Stage 2 connected: {pipeline.STAGE2_READY}   Stage 3 connected: {pipeline.STAGE3_READY}")
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)
