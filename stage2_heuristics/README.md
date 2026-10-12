# Member 2 — Heuristic + Shortener Analysis (Stage 2)

Performs rule-based heuristic checks and URL shortener resolution on validated URLs passed from Stage 1 (`qr_decoder.py`), producing a risk score, verdict, and feature dictionary ready for Stage 3 ML classification.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Files

| File | Purpose |
|---|---|
| `heuristics.py` | Main module — shortener resolution, heuristic checks, scoring & feature extraction |
| `test_heuristics.py` | Automated test suite verifying rules, scoring, and resilience |
| `demo_cli.py` | Interactive CLI for testing URLs directly or chaining from a Stage 1 QR code image |
| `requirements.txt` | Dependencies (`requests`, `urllib3`) |

---

## Quick Start

### 1. Run Automated Test Suite
```bash
python test_heuristics.py
```

### 2. Test a URL directly with the CLI
```bash
python demo_cli.py "http://192.168.1.50/login/secure-update"
```

### 3. Chain Stage 1 (QR image) directly into Stage 2
```bash
python demo_cli.py --qr ../stage1_qr_decoding/test_images/phishing_style_ip.png
```

---

## Heuristic Rule Engine

Each rule evaluates a classic phishing indicator and contributes weighted risk points to the cumulative score (0 to 100):

| Rule / Feature | Weight | Indicator / Rationale |
|---|---|---|
| **Raw IP Address Host** | +35 | Using an IP directly (e.g., `http://192.168.1.50/login`) instead of a registered domain to bypass DNS domain reputation. |
| **`@` Symbol Obfuscation** | +30 | Standard URL parsing ignores authority before `@` (e.g., `google.com@evil.com`), tricking victims into landing on the attacker site. |
| **URL Shortener** | +15 | Services like `bit.ly`, `tinyurl` conceal the true landing destination. Stage 2 resolves them safely via HTTP HEAD/GET redirects. |
| **Excessive URL Length** | +10 to +20 | Length > 75 (+20) or >= 54 (+10) chars hides host info on mobile displays. |
| **Hyphen in Domain (`-`)** | +15 | Brand impersonation in domain names (e.g., `paypal-verify.com`). |
| **Deep Subdomain Nesting** | +15 | 2 or more subdomains (e.g., `login.secure.bank.com`). |
| **Insecure HTTP** | +10 | Plain HTTP transmission for sensitive pages. |
| **Deceptive `https` in Domain** | +20 | Deceptive use of the string `https` inside domain name when served over HTTP (e.g., `http://https-bank.com`). |
| **Double Slash Redirect (`//`)** | +15 | Open redirect pattern inside URL path. |
| **Non-standard Port** | +15 | Connection over atypical ports (other than 80 or 443). |
| **High-Abuse TLD** | +15 | Free/low-reputation top-level domains (`.tk`, `.ml`, `.xyz`, `.top`, etc.). |
| **Credential / Phishing Keywords** | +10 each (max +25) | High-risk strings (`login`, `signin`, `verify`, `account`, `banking`, `secure`, `password`, `wallet`, etc.). |

---

## Verdict Categories

- **`0 – 24`**: `"legitimate"`
- **`25 – 54`**: `"suspicious"`
- **`55 – 100`**: `"phishing"`

---

## Integration Contract (for Member 3 & Main Pipeline)

Call `analyze_url()` passing `result.url` from Stage 1:

```python
from heuristics import analyze_url

# Pass URL from Stage 1
analysis = analyze_url(stage1_result.url)

print(analysis["verdict"])       # "phishing" | "suspicious" | "legitimate"
print(analysis["score"])         # Risk score: 0 to 100
print(analysis["resolved_url"])  # The expanded destination URL (after following shorteners)
print(analysis["features"])      # Feature dictionary ready for Stage 3 ML model
```

### Full Output Schema:
```json
{
  "original_url": "https://bit.ly/3example",
  "is_shortener": true,
  "resolved_url": "http://192.168.1.50/login/secure-update",
  "redirect_chain": [
    "https://bit.ly/3example",
    "http://192.168.1.50/login/secure-update"
  ],
  "features": {
    "url_length": 39,
    "having_ip_address": 1,
    "having_at_symbol": 0,
    "double_slash_redirecting": 0,
    "prefix_suffix_hyphen": 1,
    "subdomain_count": 0,
    "is_https": 0,
    "https_token_in_domain": 0,
    "shortening_service": 1,
    "non_standard_port": 0,
    "suspicious_tld": 0,
    "suspicious_keyword_count": 2,
    "matched_keywords": ["login", "secure"],
    "count_dots": 3,
    "count_hyphens": 1,
    "count_question": 0,
    "count_equal": 0,
    "count_percent": 0
  },
  "triggered_rules": [
    "Uses raw IP address as hostname instead of a registered domain (+35)",
    "Uses URL shortening service to conceal the target destination (+15)",
    "Domain contains hyphen '-' (frequent brand impersonation tactic) (+15)",
    "Uses unencrypted HTTP connection (+10)",
    "Contains credential/security keywords: 'login', 'secure' (+20)"
  ],
  "score": 95,
  "verdict": "phishing",
  "resolution_note": null
}
```

---

## What Stage 3 (ML) Should Consume
Member 3 can directly use:
- **`analysis["resolved_url"]`**: To ensure ML features are derived from the true destination rather than a shortener redirect wrapper.
- **`analysis["features"]`**: Extracted numerical features directly matching the UCI Phishing Websites Dataset attributes.
