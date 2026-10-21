"""
evaluate_pipeline.py
--------------------
Builds a labeled test set of 32 QR codes (16 legitimate, 16 phishing-style),
runs the FULL pipeline on each one, and reports:

  * metrics for Stage 2 alone, Stage 3 alone, and the hybrid verdict
  * which layer catches which phishing URL (both / only heuristics / only ML / neither)
  * every false alarm and every miss, so you can discuss them on the slide

Put this file in the repo root, next to pipeline.py.

    python evaluate_pipeline.py                 # generate eval_set/ if missing, then evaluate
    python evaluate_pipeline.py --regenerate    # rebuild the QR images first

Output: eval_set/ (QR images + manifest.csv) and eval_results.csv

About the test set
  * Legitimate URLs are real, well-known sites with paths, subdomains, query
    strings and login pages, which are the hard cases for URL-based detectors.
  * Phishing-style URLs are SYNTHETIC. They use reserved test domains
    (.example) and reserved IP ranges, so they do not point at real sites.
  * Nothing here is fetched from the network except by Stage 2's shortener
    resolution, and no URL in this set is a shortener.
  * 32 URLs is a small set. Report results as indicative, not definitive.
"""

import argparse
import csv
import os

import qrcode

import pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "eval_set")
MANIFEST = os.path.join(DIR, "manifest.csv")

LEGIT = [
    "https://www.google.com/",
    "https://www.wikipedia.org/",
    "https://en.wikipedia.org/wiki/QR_code",
    "https://github.com/python/cpython/issues",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.amazon.in/gp/help/customer/display.html?nodeId=201889410",
    "https://www.bbc.com/news/technology",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.microsoft.com/en-us/security/blog/",
    "https://accounts.google.com/signin/v2/identifier",       # legit, but looks like a login page
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.irctc.co.in/nget/train-search",
    "https://www.nptel.ac.in/courses",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.linkedin.com/in/example-profile/",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP",
]

PHISH = [
    "http://192.168.1.50/login/secure-update",
    "http://203.0.113.45/paypal/signin.php",
    "http://google.com@evil-site.example/login",
    "http://secure-paypal-login.example/verify/account",
    "http://paypa1.example/signin?session=84738",
    "https://secure-login.bank-update.example/account/verify?id=12345&redirect=home",
    "http://login.microsoft.com.account-verify.example/oauth2/authorize",
    "http://amazon-account-suspended.example/confirm-identity",
    "http://www.netflix-billing-update.example/payment/update.html",
    "https://appleid-locked.example/unlock?user=victim%40mail.com",
    "http://sbi-netbanking-secure.example/retail/login.htm",
    "http://update-your-upi-kyc.example/verify",
    "http://198.51.100.7:8080/webscr?cmd=_login",
    "http://free-gift-card-claim.example/win?token=abc123",
    "https://dropbox.com-secure-share.example/doc/view",
    "http://bank-alert-verify.example/",
]


def generate():
    os.makedirs(DIR, exist_ok=True)
    rows = []
    for label, urls in (("legitimate", LEGIT), ("phishing", PHISH)):
        for i, url in enumerate(urls, 1):
            name = f"{label[:5]}_{i:02d}.png"
            qrcode.make(url).save(os.path.join(DIR, name))
            rows.append({"filename": name, "url": url, "label": label})
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "url", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"Generated {len(rows)} QR codes in {DIR}")


def ratio(a, b):
    return a / b if b else 0.0


def metrics(rows, predicted_positive):
    """Positive class = phishing. predicted_positive(row) -> bool."""
    tp = sum(1 for r in rows if r["label"] == "phishing" and predicted_positive(r))
    fn = sum(1 for r in rows if r["label"] == "phishing" and not predicted_positive(r))
    fp = sum(1 for r in rows if r["label"] == "legitimate" and predicted_positive(r))
    tn = sum(1 for r in rows if r["label"] == "legitimate" and not predicted_positive(r))
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision, "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
    }


def show_metrics(name, m):
    print(f"{name:<34} TP={m['TP']:>2} FP={m['FP']:>2} FN={m['FN']:>2} TN={m['TN']:>2}  "
          f"acc={m['accuracy']:.2f} prec={m['precision']:.2f} rec={m['recall']:.2f} f1={m['f1']:.2f}")


def evaluate():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    rows = []
    for m in manifest:
        res = pipeline.run_pipeline(os.path.join(DIR, m["filename"]))
        s1, s2, s3 = res["stage1"], res["stage2"] or {}, res["stage3"] or {}
        rows.append({
            "filename": m["filename"], "url": m["url"], "label": m["label"],
            "decoded_ok": s1["status"] == "ok" and s1.get("url") == m["url"],
            "s2_verdict": s2.get("verdict", ""), "s2_score": s2.get("score", ""),
            "s3_prediction": s3.get("prediction", ""), "s3_probability": s3.get("probability", ""),
            "final_verdict": res["final_verdict"],
        })

    decode_fail = [r for r in rows if not r["decoded_ok"]]
    scored = [r for r in rows
              if r["decoded_ok"] and r["s2_verdict"] in ("phishing", "legitimate")
              and r["s3_prediction"] in ("phishing", "legitimate")]
    n_legit = sum(1 for r in scored if r["label"] == "legitimate")
    n_phish = len(scored) - n_legit

    print(f"\nImages: {len(rows)}   decoded correctly: {len(rows) - len(decode_fail)}   "
          f"scored: {len(scored)} ({n_phish} phishing, {n_legit} legitimate)\n")
    if len(scored) < len(rows):
        print("WARNING: some images were not scored (decode failure or a stage returned no verdict).\n")

    print("Metrics (positive class = phishing):\n")
    show_metrics("Stage 2 (heuristics) alone", metrics(scored, lambda r: r["s2_verdict"] == "phishing"))
    show_metrics("Stage 3 (ML) alone", metrics(scored, lambda r: r["s3_prediction"] == "phishing"))
    show_metrics("Hybrid: only 'phishing' counts", metrics(scored, lambda r: r["final_verdict"] == "phishing"))
    show_metrics("Hybrid: phishing OR suspicious", metrics(scored, lambda r: r["final_verdict"] in ("phishing", "suspicious")))

    ph = [r for r in scored if r["label"] == "phishing"]
    both = sum(1 for r in ph if r["s2_verdict"] == "phishing" and r["s3_prediction"] == "phishing")
    only2 = sum(1 for r in ph if r["s2_verdict"] == "phishing" and r["s3_prediction"] != "phishing")
    only3 = sum(1 for r in ph if r["s2_verdict"] != "phishing" and r["s3_prediction"] == "phishing")
    neither = len(ph) - both - only2 - only3
    print(f"\nWhich layer catches the {len(ph)} phishing URLs?")
    print(f"  both layers: {both}   only heuristics: {only2}   only ML: {only3}   neither: {neither}")

    legit = [r for r in scored if r["label"] == "legitimate"]
    print(f"\nFalse alarms on the {len(legit)} legitimate URLs: "
          f"heuristics {sum(r['s2_verdict'] == 'phishing' for r in legit)}, "
          f"ML {sum(r['s3_prediction'] == 'phishing' for r in legit)}, "
          f"hybrid-phishing {sum(r['final_verdict'] == 'phishing' for r in legit)}, "
          f"hybrid-suspicious {sum(r['final_verdict'] == 'suspicious' for r in legit)}")

    print("\nLegitimate URLs that any layer flagged:")
    flagged = [r for r in legit if r["s2_verdict"] == "phishing" or r["s3_prediction"] == "phishing"]
    for r in flagged or []:
        print(f"  S2={r['s2_verdict']:<10} S3={r['s3_prediction']:<10} final={r['final_verdict']:<10} {r['url']}")
    if not flagged:
        print("  none")

    print("\nPhishing URLs the hybrid ('phishing' only) missed:")
    missed = [r for r in ph if r["final_verdict"] != "phishing"]
    for r in missed:
        print(f"  S2={r['s2_verdict']:<10} S3={r['s3_prediction']:<10} final={r['final_verdict']:<10} {r['url']}")
    if not missed:
        print("  none")

    if decode_fail:
        print("\nNot decoded correctly:")
        for r in decode_fail:
            print("  ", r["filename"], r["url"])

    out = os.path.join(ROOT, "eval_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")
    print("Reminder: a 32-URL synthetic set gives indicative results only; say so on the slide.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="rebuild the QR images")
    args = ap.parse_args()
    print(f"Stage 2 connected: {pipeline.STAGE2_READY}   Stage 3 connected: {pipeline.STAGE3_READY}")
    if args.regenerate or not os.path.exists(MANIFEST):
        generate()
    evaluate()
