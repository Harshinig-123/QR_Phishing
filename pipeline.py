"""
pipeline.py
-----------
Chains Stage 1 (QR decoding) -> Stage 2 (heuristics + shortener) -> Stage 3 (ML).

PLACE THIS FILE IN THE REPO ROOT:

    QR_Phishing/
    |-- pipeline.py              <- this file
    |-- stage1_qr_decoding/qr_decoder.py
    |-- stage2_heuristics/heuristics.py     must define analyze_url(url) -> dict
    |-- stage3_ml/ml_predict.py             must define predict_url(url) -> dict

It also works from inside the stage1 folder (stages 2 and 3 are then
placeholders). Until Members 2 and 3 push their modules, placeholder stages
are used. They are clearly marked and never produce a verdict.

Expected interfaces (see each stage folder's README):
    analyze_url(url) -> {"resolved_url": str, "is_shortener": bool,
                         "triggered_rules": [...], "score": number,
                         "verdict": "phishing" | "legitimate", ...}
    predict_url(url) -> {"prediction": "phishing" | "legitimate",
                         "probability": float, ...}

Usage:
    python pipeline.py path/to/qr.png
"""

import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "stage1_qr_decoding", "stage2_heuristics", "stage3_ml"):
    _path = os.path.join(ROOT, _sub)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from stage1_qr_decoding.qr_decoder import process_qr_bytes, process_qr_image, QRDecodeResult  # noqa: E402
except ImportError:
    from qr_decoder import process_qr_bytes, process_qr_image, QRDecodeResult  # noqa: E402


def _try_import(module_candidates, func_name):
    """Return (function, True) if the teammate's module exists, else (None, False).
    Supports checking package path (stageX_...) before flat module.
    Real bugs inside their module are NOT hidden: only 'module not found' is."""
    if isinstance(module_candidates, str):
        module_candidates = [module_candidates]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, func_name), True
        except ModuleNotFoundError as e:
            if e.name == module_name or ("." in module_name and e.name == module_name.split(".")[0]):
                continue
            raise
    return None, False


# Direct data and analysis imports from Stage 2 and Stage 3 with dynamic fallback
try:
    from stage2_heuristics.heuristics import (
        analyze_url as _analyze_url,
        SHORTENER_DOMAINS,
        SUSPICIOUS_TLDS,
        PHISHING_KEYWORDS,
    )
    STAGE2_READY = True
except ImportError:
    _analyze_url, STAGE2_READY = _try_import(["stage2_heuristics.heuristics", "heuristics"], "analyze_url")
    SHORTENER_DOMAINS = frozenset()
    SUSPICIOUS_TLDS = frozenset()
    PHISHING_KEYWORDS = ()

try:
    from stage3_ml.ml_predict import predict_url as _predict_url, should_run_stage3
    STAGE3_READY = True
except ImportError:
    _predict_url, STAGE3_READY = _try_import(["stage3_ml.ml_predict", "ml_predict"], "predict_url")
    should_run_stage3 = None


def _placeholder(stage):
    return {"placeholder": True, "verdict": "unknown",
            "note": f"{stage} is not connected yet (module not found)."}


def _run_stage(func, ready, name, *args):
    """Run a stage; a crash in one stage must not kill the whole demo."""
    if not ready:
        return _placeholder(name)
    try:
        return func(*args)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "verdict": "unknown"}


def _normalize_label(label):
    """Map whatever a stage calls its verdict onto: phishing / suspicious / legitimate / None."""
    text = str(label or "").strip().lower()
    if not text:
        return None
    if any(k in text for k in ("phish", "malicious", "high")):
        return "phishing"
    if any(k in text for k in ("suspic", "medium", "warn", "caution")):
        return "suspicious"
    if any(k in text for k in ("legit", "benign", "safe", "low")):
        return "legitimate"
    return None


def _stage_vote(stage_result):
    """'phishing' / 'suspicious' / 'legitimate' / None (no usable opinion)."""
    if not stage_result or stage_result.get("placeholder") or "error" in stage_result:
        return None
    return _normalize_label(stage_result.get("verdict") or stage_result.get("prediction"))


def run_pipeline(source, from_bytes=False, assume_scheme=None):
    """
    source: file path (default) or raw image bytes (from_bytes=True).
    Returns a dict with every stage's output plus final_verdict and reasons.
    """
    s1 = (process_qr_bytes(source, assume_scheme) if from_bytes
          else process_qr_image(source, assume_scheme))

    out = {"stage1": s1.to_dict(), "stage2": None, "stage3": None,
           "stage2_connected": STAGE2_READY, "stage3_connected": STAGE3_READY,
           "final_verdict": None, "reasons": []}

    # ---- Stage 1 stopped the pipeline ----
    if s1.status != "ok":
        if s1.status == "not_a_url" and s1.sensitive:
            out["final_verdict"] = "caution"
            out["reasons"].append(f"Sensitive payload type '{s1.payload_type}': {s1.sensitivity_note}")
        elif s1.status == "not_a_url":
            out["final_verdict"] = "not_supported"
            out["reasons"].append(s1.message)
        else:
            out["final_verdict"] = s1.status          # "no_qr" or "unreadable"
            out["reasons"].append(s1.message)
        return out

    # ---- Stage 2: heuristics + safe shortener resolution ----
    out["stage2"] = _run_stage(_analyze_url, STAGE2_READY, "Stage 2", s1.url)

    # Stage 3 should see the destination of a shortened link, if Stage 2 found one
    url_for_ml = (out["stage2"] or {}).get("resolved_url") or s1.url

    # ---- Stage 3: ML ----
    out["stage3"] = _run_stage(_predict_url, STAGE3_READY, "Stage 3", url_for_ml)

    # ---- Combine ----
    # Stage 2 (explainable rules) is the primary signal. The ML layer can only raise a
    # URL to "suspicious" by itself; it never produces a final "phishing" on its own.
    for name, r in (("Stage 2", out["stage2"]), ("Stage 3", out["stage3"])):
        raw = (r or {}).get("verdict") or (r or {}).get("prediction")
        if r and not r.get("placeholder") and "error" not in r and raw \
                and _normalize_label(raw) is None and str(raw).lower() != "unknown":
            out["reasons"].append(f"{name} returned an unrecognized verdict label: {raw!r}")

    v2, v3 = _stage_vote(out["stage2"]), _stage_vote(out["stage3"])
    if v2 == "phishing":
        out["final_verdict"] = "phishing"
        if v3 == "phishing":
            out["reasons"].append("Both heuristic and ML layers flag this URL.")
    elif v2 == "suspicious":
        out["final_verdict"] = "suspicious"
        out["reasons"].append("Heuristic layer rates this URL as suspicious"
                              + (" and the ML layer agrees." if v3 == "phishing" else "."))
    elif v3 == "phishing":
        out["final_verdict"] = "suspicious"
        out["reasons"].append(
            "ML layer flags this URL as phishing"
            + (", but the heuristic layer found no strong red flags." if v2 == "legitimate"
               else " (heuristic layer unavailable)."))
    elif v2 or v3:
        out["final_verdict"] = "legitimate"
    else:
        out["final_verdict"] = "undetermined"
        out["reasons"].append("No detection stage produced a verdict (Stage 2/3 missing or failed).")

    if s1.sensitive:
        out["reasons"].append(s1.sensitivity_note)
    for rule in (out["stage2"] or {}).get("triggered_rules", []) or []:
        out["reasons"].append(f"Heuristic: {rule}")
    return out


def summarize(result):
    s1 = result["stage1"]
    lines = [f"Stage 1: {s1['status']} (type: {s1['payload_type']})"]
    if s1.get("url"):
        lines.append(f"  URL: {s1['url']}")
    for key, label in (("stage2", "Stage 2"), ("stage3", "Stage 3")):
        r = result[key]
        if r is None:
            continue
        if r.get("placeholder"):
            lines.append(f"{label}: not connected yet")
        elif "error" in r:
            lines.append(f"{label}: ERROR {r['error']}")
        else:
            lines.append(f"{label}: {r.get('verdict') or r.get('prediction')}")
    lines.append(f"FINAL VERDICT: {result['final_verdict'].upper()}")
    for reason in result["reasons"]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        stage1_sample = os.path.join(ROOT, "stage1_qr_decoding", "test_images", "phishing_style_ip.png")
        if os.path.exists(stage1_sample):
            print(f"[*] No image specified. Importing test QR data directly from Stage 1: {os.path.relpath(stage1_sample, ROOT)}\n")
            target = stage1_sample
        else:
            print("Usage: python pipeline.py <path_to_qr_image>")
            sys.exit(1)
    else:
        target = sys.argv[1]
    res = run_pipeline(target)
    print(summarize(res))
    print("\n--- full JSON ---")
    print(json.dumps(res, indent=2))
