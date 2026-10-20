"""
test_pipeline.py - checks pipeline.py logic using FAKE Stage 2/3 modules,
so you can verify wiring before your teammates finish their code.

Run:  python test_pipeline.py
"""

import os
import sys
import tempfile
import textwrap

# --- create throwaway fake teammate modules and put them on the path ---
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "heuristics.py"), "w") as f:
    f.write(textwrap.dedent('''
        def analyze_url(url):
            risky = "192.168" in url or "@" in url
            return {"resolved_url": url, "is_shortener": False,
                    "triggered_rules": ["IP address in URL"] if risky else [],
                    "score": 5 if risky else 0,
                    "verdict": "phishing" if risky else "legitimate"}
    '''))
with open(os.path.join(tmp, "ml_predict.py"), "w") as f:
    f.write(textwrap.dedent('''
        def predict_url(url):
            return {"url": url, "prediction": "legitimate", "probability": 0.9}
    '''))
sys.path.insert(0, tmp)

import pipeline  # noqa: E402

# Once the real Stage 2/3 modules exist in the repo they take precedence over
# these fakes, so this test no longer applies. Skip it then.
_real = [m for m in ("heuristics", "ml_predict")
         if m in sys.modules and not str(getattr(sys.modules[m], "__file__", "")).startswith(tmp)]
if _real:
    print(f"[SKIP] Real teammate modules found ({', '.join(_real)}); fake-stage test not needed anymore.")
    sys.exit(0)

D = "test_images"


def check(name, got, want):
    assert got == want, f"{name}: expected {want!r}, got {got!r}"
    print(f"[PASS] {name}: {got}")


def test_wiring_with_fake_stages():
    assert pipeline.STAGE2_READY and pipeline.STAGE3_READY
    check("legit URL", pipeline.run_pipeline(f"{D}/legit_url.png")["final_verdict"], "legitimate")
    check("IP-based URL", pipeline.run_pipeline(f"{D}/phishing_style_ip.png")["final_verdict"], "phishing")
    check("WiFi payload", pipeline.run_pipeline(f"{D}/wifi_config.png")["final_verdict"], "caution")
    check("plain text", pipeline.run_pipeline(f"{D}/plain_text.png")["final_verdict"], "not_supported")
    check("missing file", pipeline.run_pipeline("nope.png")["final_verdict"], "unreadable")


def test_bytes_input():
    with open(f"{D}/legit_url.png", "rb") as f:
        r = pipeline.run_pipeline(f.read(), from_bytes=True)
    check("bytes input", r["final_verdict"], "legitimate")


def test_stage_crash_is_contained():
    original = pipeline._analyze_url
    pipeline._analyze_url = lambda url: 1 / 0
    r = pipeline.run_pipeline(f"{D}/legit_url.png")
    pipeline._analyze_url = original
    assert "error" in r["stage2"]
    check("Stage 2 crash does not kill pipeline", r["final_verdict"], "legitimate")  # Stage 3 still votes


def test_placeholders_when_stages_missing():
    saved = (pipeline.STAGE2_READY, pipeline.STAGE3_READY)
    pipeline.STAGE2_READY = pipeline.STAGE3_READY = False
    r = pipeline.run_pipeline(f"{D}/legit_url.png")
    pipeline.STAGE2_READY, pipeline.STAGE3_READY = saved
    check("placeholders give 'undetermined'", r["final_verdict"], "undetermined")


if __name__ == "__main__":
    test_wiring_with_fake_stages()
    test_bytes_input()
    test_stage_crash_is_contained()
    test_placeholders_when_stages_missing()
    print("\nAll pipeline tests passed.")
