"""
feature_extractor.py
--------------------
Stage 3: Purely Lexical/Structural Feature Extraction for URL Phishing Classification.

Calculates 30 static, URL-computable features directly from the raw string.
Guarantees 100% static execution:
  - ZERO network calls (no DNS, no WHOIS, no HTTP requests, no sockets)
  - Safe against malformed/empty/non-string inputs (never raises exceptions)
  - No domain identity or TLD categorical features (prevents dataset memorization bias)
  - Directly reuses Stage 2's canonical keyword & shortener definitions to maintain consistency
"""

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlsplit

# Locate Stage 2 directory to reuse canonical shortener and keyword definitions
REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2_DIR = REPO_ROOT / "stage2_heuristics"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

try:
    from heuristics import SHORTENER_DOMAINS, PHISHING_KEYWORDS
except ImportError:
    # Safe standalone fallback in isolated environments
    SHORTENER_DOMAINS = frozenset({
        "bit.ly", "bitly.com", "j.mp", "tinyurl.com", "tiny.cc", "t.co", "goo.gl",
        "ow.ly", "is.gd", "v.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
        "rb.gy", "bl.ink", "lnkd.in", "t.ly", "s.id", "soo.gd", "qrco.de", "qr.net",
        "1url.com", "adf.ly", "shorte.st", "x.co", "snip.ly", "short.io", "clck.ru",
        "u.to", "trib.al", "dlvr.it", "fb.me", "tr.im", "urlz.fr", "shorturl.com",
        "mcaf.ee", "po.st", "tny.im", "chilp.it", "bc.vc", "ouo.io", "gg.gg",
    })
    PHISHING_KEYWORDS = (
        "login", "signin", "verify", "account", "banking", "secure", "password",
        "wallet", "confirm", "webscr", "credential", "authenticate", "suspend",
        "unlock", "billing",
    )

SECOND_LEVEL_LABELS = frozenset(
    {"co", "com", "org", "net", "gov", "edu", "ac", "ne", "or", "go"}
)

FEATURE_ORDER: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "count_dots",
    "count_hyphens",
    "count_digits",
    "count_underscores",
    "count_slashes",
    "count_at",
    "count_question",
    "count_equal",
    "count_ampersand",
    "count_percent",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "is_ip_host",
    "has_at_symbol",
    "is_https",
    "num_subdomains",
    "path_depth",
    "has_port",
    "non_standard_port",
    "has_punycode",
    "hostname_entropy",
    "longest_token_length",
    "suspicious_keyword_count",
    "is_known_shortener",
    "has_double_slash_in_path",
]


def _is_ip(hostname: str) -> bool:
    """Pure static check for IPv4 or IPv6 literals, including hex/integer formats."""
    h = hostname.strip("[]").lower()
    if not h:
        return False
    parts = h.split(":")
    clean_host = parts[0]
    # Check standard IPv4 octets
    octets = clean_host.split(".")
    if len(octets) == 4:
        try:
            if all(0 <= int(x) <= 255 for x in octets):
                return True
        except ValueError:
            pass
    # Check hex or integer notation (e.g. 0x7f000001 or 2130706433)
    if re.fullmatch(r"0x[0-9a-f]+", clean_host) or (clean_host.isdigit() and len(clean_host) >= 8):
        return True
    # Standard IPv6 colon check
    if ":" in h and not re.search(r"[^0-9a-f:]", h):
        return True
    return False


def _calculate_entropy(text: str) -> float:
    """Computes Shannon entropy of character distribution in text."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return round(entropy, 4)


def _count_subdomains(hostname: str) -> int:
    """Calculates number of subdomain labels, handling ccTLD second level domains."""
    if not hostname or _is_ip(hostname):
        return 0
    labels = [p for p in hostname.split(".") if p]
    if labels and labels[0] == "www":
        labels = labels[1:]
    if len(labels) < 2:
        return 0
    base = 2
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL_LABELS:
        base = 3  # e.g., co.uk, com.au
    return max(0, len(labels) - base)


def _safe_default_features() -> Dict[str, Any]:
    """Default feature vector for empty or completely unparseable input."""
    return {name: 0 if not name.endswith("_ratio") and not name.endswith("_entropy") else 0.0
            for name in FEATURE_ORDER}


def extract_ml_features(url: str) -> Dict[str, Any]:
    """
    Extracts the 30 lexical/structural URL features.
    
    Guarantees:
      - Never performs any network I/O
      - Never raises exceptions on malformed strings or invalid types
      - Returns numerical values matching FEATURE_ORDER
    """
    if not url or not isinstance(url, str):
        return _safe_default_features()

    clean_url = url.strip()
    if not clean_url:
        return _safe_default_features()

    try:
        # Prepend scheme if missing so urlsplit parses host and path reliably
        parse_target = clean_url
        if "://" not in clean_url and not re.match(r"^[a-zA-Z]+:", clean_url):
            parse_target = "http://" + clean_url

        parsed = urlsplit(parse_target)
        hostname = (parsed.hostname or "").lower().strip()
        path = parsed.path or ""
        query = parsed.query or ""
        scheme = parsed.scheme.lower()
        port = parsed.port

        url_len = len(clean_url)
        host_len = len(hostname)
        path_len = len(path)
        query_len = len(query)

        # Character counts across full URL
        count_dots = clean_url.count(".")
        count_hyphens = clean_url.count("-")
        count_digits = sum(c.isdigit() for c in clean_url)
        count_underscores = clean_url.count("_")
        count_slashes = clean_url.count("/")
        count_at = clean_url.count("@")
        count_question = clean_url.count("?")
        count_equal = clean_url.count("=")
        count_ampersand = clean_url.count("&")
        count_percent = clean_url.count("%")

        # Ratios (normalized by URL length)
        digit_ratio = round(count_digits / url_len, 4) if url_len > 0 else 0.0
        letter_count = sum(c.isalpha() for c in clean_url)
        letter_ratio = round(letter_count / url_len, 4) if url_len > 0 else 0.0
        special_count = url_len - count_digits - letter_count
        special_char_ratio = round(special_count / url_len, 4) if url_len > 0 else 0.0

        # Host and structural flags
        is_ip_host = 1 if _is_ip(hostname) else 0
        has_at_symbol = 1 if "@" in clean_url else 0
        is_https = 1 if scheme == "https" else 0
        num_subdomains = _count_subdomains(hostname)

        # Path depth: number of non-empty directory levels in the path
        path_segments = [s for s in path.split("/") if s]
        path_depth = len(path_segments)

        # Port indicators
        has_port = 1 if port is not None else 0
        non_standard_port = 1 if (port is not None and port not in (80, 443)) else 0

        # Punycode / IDN homograph attack indicator
        has_punycode = 1 if "xn--" in hostname else 0

        # Hostname entropy (randomness of domain characters)
        hostname_entropy = _calculate_entropy(hostname)

        # Longest token in hostname
        host_tokens = re.split(r"[\.\-_]", hostname) if hostname else []
        longest_token_length = max(len(t) for t in host_tokens) if host_tokens else 0

        # Phishing keywords in URL (reusing Stage 2's list)
        lowered_url = clean_url.lower()
        suspicious_keyword_count = sum(1 for kw in PHISHING_KEYWORDS if kw in lowered_url)

        # URL shortener service
        cleaned_host = hostname.rstrip(".")
        is_known_shortener = 1 if any(
            cleaned_host == s or cleaned_host.endswith("." + s) for s in SHORTENER_DOMAINS
        ) else 0

        # Open redirect / double slash pattern inside path
        has_double_slash_in_path = 1 if "//" in path else 0

        return {
            "url_length": url_len,
            "hostname_length": host_len,
            "path_length": path_len,
            "query_length": query_len,
            "count_dots": count_dots,
            "count_hyphens": count_hyphens,
            "count_digits": count_digits,
            "count_underscores": count_underscores,
            "count_slashes": count_slashes,
            "count_at": count_at,
            "count_question": count_question,
            "count_equal": count_equal,
            "count_ampersand": count_ampersand,
            "count_percent": count_percent,
            "digit_ratio": digit_ratio,
            "letter_ratio": letter_ratio,
            "special_char_ratio": special_char_ratio,
            "is_ip_host": is_ip_host,
            "has_at_symbol": has_at_symbol,
            "is_https": is_https,
            "num_subdomains": num_subdomains,
            "path_depth": path_depth,
            "has_port": has_port,
            "non_standard_port": non_standard_port,
            "has_punycode": has_punycode,
            "hostname_entropy": hostname_entropy,
            "longest_token_length": longest_token_length,
            "suspicious_keyword_count": suspicious_keyword_count,
            "is_known_shortener": is_known_shortener,
            "has_double_slash_in_path": has_double_slash_in_path,
        }

    except Exception:
        # Failsafe default on any unexpected edge case
        return _safe_default_features()


def extract_feature_vector(url: str) -> List[float]:
    """Extracts features in fixed order as a numeric list for model ingestion."""
    feats = extract_ml_features(url)
    return [float(feats[name]) for name in FEATURE_ORDER]
