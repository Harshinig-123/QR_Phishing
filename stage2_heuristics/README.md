# Stage 2: Heuristic + Shortener Analysis (Member 2)

Put all Stage 2 code in this folder.

## Your tasks
- Rule-based checks: URL length, IP address, `@` symbol, suspicious keywords,
  number of subdomains, HTTPS, special characters, suspicious domain patterns
- Detect known URL shorteners (bit.ly, tinyurl, cutt.ly, ...)
- Resolve shortened URLs with HTTP requests (set a timeout, handle failures)
- Run the heuristic checks on the final/resolved URL
- Generate heuristic features/results

## Input (from Stage 1)
A validated URL string: `result.url` from `stage1_qr_decoding/qr_decoder.py`.

## Suggested interface (so Stage 3 and the pipeline can plug in)
Create `heuristics.py` with:

```python
def analyze_url(url: str) -> dict:
    return {
        "original_url": url,
        "is_shortener": False,
        "resolved_url": url,        # same as original if not a shortener
        "features": {...},          # e.g. {"url_length": 54, "has_ip": False, ...}
        "triggered_rules": [...],   # human-readable reasons
        "score": 0,                 # risk score
        "verdict": "legitimate",    # "phishing" or "legitimate"
    }
```

Member 3 needs `resolved_url` and `features` for the ML stage.

## Report sections you own
Methodology Stage 2, heuristic techniques, URL shortener analysis.
