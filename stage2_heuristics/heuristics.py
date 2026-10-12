"""
heuristics.py
-------------
Member 2 — Stage 2: Heuristic + Shortener Analysis

Responsibilities:
    1. Detect known URL shorteners (bit.ly, tinyurl, cutt.ly, etc.).
    2. Safely resolve shortened URLs using lightweight HTTP HEAD/GET requests.
    3. Analyze the resolved URL against rule-based heuristic checks
       (IP address, @ symbol, length, subdomains, prefix/suffix, keywords, etc.).
    4. Calculate a cumulative risk score and assign a verdict
       ('legitimate', 'suspicious', or 'phishing').
    5. Extract structured lexical/structural features that Member 3 (Stage 3 ML)
       can consume directly for model training and inference.
"""

import ipaddress
import re
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


# --------------------------------------------------------------------------
# Configuration & Knowledge Bases
# --------------------------------------------------------------------------

# Curated list of common URL shortening services
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bitly.com", "cutt.ly", "rb.gy", "shorturl.at",
    "tiny.cc", "lnkd.in", "rebrand.ly", "v.gd", "qr.ae", "clck.ru",
    "rotf.lol", "t.ly", "bc.vc", "s.id", "bl.ink", "trib.al"
}

# High-risk keywords commonly found in credential harvesting attacks
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "banking", "authenticate", "confirm", "password",
    "credential", "wallet", "support", "billing", "recover", "validate"
]

# TLDs historically associated with high abuse rates in free/cheap domains
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click", "loan", "cam"
}


# --------------------------------------------------------------------------
# Shortener Detection & Resolution
# --------------------------------------------------------------------------

def is_url_shortener(url: str) -> bool:
    """
    Checks if the URL's hostname belongs to a known URL shortener service.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().strip()
        # Direct match or subdomain match (e.g., custom.bit.ly)
        for shortener in KNOWN_SHORTENERS:
            if hostname == shortener or hostname.endswith("." + shortener):
                return True
        return False
    except Exception:
        return False


def resolve_shortener(
    url: str,
    timeout: float = 4.0,
    max_redirects: int = 5
) -> Tuple[str, List[str], Optional[str]]:
    """
    Safely resolves a shortened URL by following redirects without downloading
    large or potentially malicious payload bodies.

    Returns:
        (resolved_url, redirect_chain, error_message)
    """
    if not is_url_shortener(url):
        return url, [url], None

    if not HAS_REQUESTS:
        return url, [url], "requests library not installed; skipped HTTP unshortening"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    }

    redirect_chain = [url]
    current_url = url

    try:
        # Use a session with max redirects bound to prevent redirect loops
        session = requests.Session()
        session.max_redirects = max_redirects

        # Attempt HEAD request first to avoid downloading body
        response = session.head(
            current_url,
            allow_redirects=True,
            timeout=timeout,
            headers=headers
        )

        # Some servers reject HEAD requests with 405 Method Not Allowed
        if response.status_code == 405:
            response = session.get(
                current_url,
                allow_redirects=True,
                stream=True,
                timeout=timeout,
                headers=headers
            )
            response.close()

        # Collect history of redirects
        if response.history:
            redirect_chain = [r.url for r in response.history] + [response.url]
        else:
            redirect_chain = [response.url]

        return response.url, redirect_chain, None

    except requests.exceptions.Timeout:
        return url, redirect_chain, f"Resolution timed out after {timeout}s (link may be offline)"
    except requests.exceptions.RequestException as e:
        return url, redirect_chain, f"Resolution network error: {type(e).__name__}"
    except Exception as e:
        return url, redirect_chain, f"Unexpected resolution error: {str(e)}"


# --------------------------------------------------------------------------
# Heuristic Feature Check Functions
# --------------------------------------------------------------------------

def check_ip_address(hostname: str) -> bool:
    """
    Checks if hostname is an IPv4 or IPv6 address.
    Phishing attacks frequently use raw IP addresses to bypass domain reputation.
    """
    if not hostname:
        return False
    # Strip port if present
    host = hostname.split(":")[0].strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def check_at_symbol(url: str) -> bool:
    """
    Checks if '@' is present in the URL.
    In standard URL syntax, everything prior to '@' is treated as userinfo
    and ignored by the browser, tricking users into visiting an attacker domain.
    """
    return "@" in url


def check_double_slash_redirect(url: str) -> bool:
    """
    Checks if '//' appears in the URL path, often used for open-redirect tricks.
    """
    parsed = urlparse(url)
    return "//" in parsed.path


def check_prefix_suffix_hyphen(hostname: str) -> bool:
    """
    Checks if a hyphen '-' is present in the domain name.
    Phishers commonly register domain names with hyphens to mimic legitimate brands
    (e.g., paypal-update.com, apple-verify.com).
    """
    if not hostname:
        return False
    # Strip TLD and check core domain
    parts = hostname.split(".")
    domain_body = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
    return "-" in domain_body


def count_subdomains(hostname: str) -> int:
    """
    Counts the number of subdomains in the host.
    A standard domain like example.com has 0 subdomains (2 parts).
    www.example.com has 1 subdomain (3 parts).
    """
    if not hostname or check_ip_address(hostname):
        return 0
    parts = [p for p in hostname.split(".") if p]
    # Subtract 2 for domain + TLD (simplification for common TLDs)
    return max(0, len(parts) - 2)


def check_suspicious_tld(hostname: str) -> bool:
    """
    Checks if the domain uses a high-abuse top-level domain.
    """
    if not hostname:
        return False
    tld = hostname.split(".")[-1].lower()
    return tld in SUSPICIOUS_TLDS


def extract_matched_keywords(url: str) -> List[str]:
    """
    Finds credential harvesting keywords in the domain or path.
    """
    url_lower = url.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]


# --------------------------------------------------------------------------
# Main Analysis & Feature Extraction Function
# --------------------------------------------------------------------------

def extract_features(url: str, is_shortener: bool) -> Dict[str, Any]:
    """
    Extracts lexical and structural features from the URL.
    These features serve a dual purpose:
      1. They feed into the Stage 2 heuristic rule scoring engine.
      2. They are structured for Member 3 (Stage 3 ML) feature extraction.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    url_len = len(url)

    has_ip = 1 if check_ip_address(hostname) else 0
    has_at = 1 if check_at_symbol(url) else 0
    double_slash = 1 if check_double_slash_redirect(url) else 0
    prefix_suffix = 1 if check_prefix_suffix_hyphen(hostname) else 0
    subdomains = count_subdomains(hostname)
    is_https = 1 if parsed.scheme.lower() == "https" else 0
    https_in_domain = 1 if "https" in hostname else 0
    has_suspicious_tld = 1 if check_suspicious_tld(hostname) else 0
    matched_kw = extract_matched_keywords(url)
    kw_count = len(matched_kw)

    # Character counts
    count_dots = url.count(".")
    count_hyphens = url.count("-")
    count_question = url.count("?")
    count_equal = url.count("=")
    count_percent = url.count("%")

    # Non-standard port check
    has_port = 0
    if parsed.port and parsed.port not in (80, 443):
        has_port = 1

    return {
        "url_length": url_len,
        "having_ip_address": has_ip,
        "having_at_symbol": has_at,
        "double_slash_redirecting": double_slash,
        "prefix_suffix_hyphen": prefix_suffix,
        "subdomain_count": subdomains,
        "is_https": is_https,
        "https_token_in_domain": https_in_domain,
        "shortening_service": 1 if is_shortener else 0,
        "non_standard_port": has_port,
        "suspicious_tld": has_suspicious_tld,
        "suspicious_keyword_count": kw_count,
        "matched_keywords": matched_kw,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_question": count_question,
        "count_equal": count_equal,
        "count_percent": count_percent,
    }


def analyze_url(url: str, resolve_redirects: bool = True) -> Dict[str, Any]:
    """
    Main entry point for Stage 2.

    Args:
        url: The validated URL string from Stage 1 (result.url).
        resolve_redirects: Whether to perform HTTP unshortening on known shorteners.

    Returns:
        Structured dictionary matching the Stage 2 & 3 integration contract:
        {
            "original_url": str,
            "is_shortener": bool,
            "resolved_url": str,
            "redirect_chain": List[str],
            "features": Dict[str, Any],
            "triggered_rules": List[str],
            "score": int (0 - 100),
            "verdict": "legitimate" | "suspicious" | "phishing",
            "resolution_note": Optional[str]
        }
    """
    if not url or not isinstance(url, str):
        return {
            "original_url": url,
            "is_shortener": False,
            "resolved_url": url,
            "redirect_chain": [],
            "features": {},
            "triggered_rules": ["Invalid or empty URL provided"],
            "score": 0,
            "verdict": "legitimate",
            "resolution_note": "No URL provided"
        }

    url = url.strip()
    is_short = is_url_shortener(url)

    # 1. Resolve shorteners if requested
    resolution_note = None
    if is_short and resolve_redirects:
        resolved_url, redirect_chain, err = resolve_shortener(url)
        if err:
            resolution_note = err
    else:
        resolved_url = url
        redirect_chain = [url]

    # 2. Extract features on the final (resolved) target URL
    features = extract_features(resolved_url, is_shortener=is_short)

    # 3. Rule Evaluation & Scoring Engine
    triggered_rules = []
    score = 0

    # Rule: Raw IP Address Host (+35 pts)
    if features["having_ip_address"] == 1:
        score += 35
        triggered_rules.append("Uses raw IP address as hostname instead of a registered domain (+35)")

    # Rule: @ Symbol Obfuscation (+30 pts)
    if features["having_at_symbol"] == 1:
        score += 30
        triggered_rules.append("Contains '@' symbol used to disguise destination URL (+30)")

    # Rule: URL Shortening (+15 pts)
    if features["shortening_service"] == 1:
        score += 15
        triggered_rules.append("Uses URL shortening service to conceal the target destination (+15)")

    # Rule: URL Length (>75 high, 54-75 moderate)
    if features["url_length"] > 75:
        score += 20
        triggered_rules.append(f"Excessive URL length ({features['url_length']} chars > 75) (+20)")
    elif features["url_length"] >= 54:
        score += 10
        triggered_rules.append(f"Moderate URL length ({features['url_length']} chars >= 54) (+10)")

    # Rule: Hyphen in Domain (+15 pts)
    if features["prefix_suffix_hyphen"] == 1:
        score += 15
        triggered_rules.append("Domain contains hyphen '-' (frequent brand impersonation tactic) (+15)")

    # Rule: Subdomain Depth (>= 2 subdomains: +15 pts)
    if features["subdomain_count"] >= 2:
        score += 15
        triggered_rules.append(f"Deep subdomain nesting ({features['subdomain_count']} subdomains) (+15)")

    # Rule: Insecure HTTP (+10 pts)
    if features["is_https"] == 0:
        score += 10
        triggered_rules.append("Uses unencrypted HTTP connection (+10)")

    # Rule: HTTPS token in domain (+20 pts)
    if features["https_token_in_domain"] == 1 and features["is_https"] == 0:
        score += 20
        triggered_rules.append("Deceptive 'https' keyword found inside the domain name (+20)")

    # Rule: Double slash redirect in path (+15 pts)
    if features["double_slash_redirecting"] == 1:
        score += 15
        triggered_rules.append("Double slash '//' redirect pattern found in URL path (+15)")

    # Rule: Non-standard port (+15 pts)
    if features["non_standard_port"] == 1:
        score += 15
        triggered_rules.append("Connects via an unusual non-standard port (+15)")

    # Rule: High-abuse TLD (+15 pts)
    if features["suspicious_tld"] == 1:
        score += 15
        triggered_rules.append("Registered under high-abuse free/low-reputation TLD (+15)")

    # Rule: Phishing keywords (+10 pts each, capped at 25 pts)
    if features["suspicious_keyword_count"] > 0:
        kw_pts = min(25, features["suspicious_keyword_count"] * 10)
        score += kw_pts
        kws_str = ", ".join(repr(k) for k in features["matched_keywords"])
        triggered_rules.append(f"Contains credential/security keywords: {kws_str} (+{kw_pts})")

    # Rule: Unresolved shortener warning
    if resolution_note:
        triggered_rules.append(f"Shortener notice: {resolution_note}")

    # Clamp score to [0, 100]
    score = min(100, score)

    # 4. Final Verdict Determination
    if score >= 55:
        verdict = "phishing"
    elif score >= 25:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    return {
        "original_url": url,
        "is_shortener": is_short,
        "resolved_url": resolved_url,
        "redirect_chain": redirect_chain,
        "features": features,
        "triggered_rules": triggered_rules,
        "score": score,
        "verdict": verdict,
        "resolution_note": resolution_note,
    }
