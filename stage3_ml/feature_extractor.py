"""
feature_extractor.py
--------------------
Stage 3: Feature Extractor for URL-based Phishing Classification.

Extracts the 10 URL-computable features corresponding to the UCI Phishing
Websites Dataset (UCI ID 327) using exact UCI numerical encodings (-1, 0, 1):
  1  = Legitimate
  0  = Suspicious
 -1  = Phishing

Where applicable, helper functions from Stage 2 (`stage2_heuristics.heuristics`)
are imported and reused to avoid code duplication across project stages.
"""

import os
import sys
from typing import Dict, List, Any
from urllib.parse import urlparse

# Add Stage 2 to sys.path to reuse existing heuristic helpers
STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage2_heuristics"))
if STAGE2_DIR not in sys.path:
    sys.path.insert(0, STAGE2_DIR)

try:
    from heuristics import (
        is_url_shortener,
        check_ip_address,
        check_at_symbol,
        check_double_slash_redirect,
        check_prefix_suffix_hyphen,
        count_subdomains,
    )
except ImportError:
    # Standalone fallback if Stage 2 is not in path
    import ipaddress

    KNOWN_SHORTENERS = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "bitly.com", "cutt.ly", "rb.gy", "shorturl.at",
        "tiny.cc", "lnkd.in", "rebrand.ly", "v.gd", "qr.ae", "clck.ru",
        "rotf.lol", "t.ly", "bc.vc", "s.id", "bl.ink", "trib.al"
    }

    def is_url_shortener(url: str) -> bool:
        try:
            h = (urlparse(url).hostname or "").lower().strip()
            return any(h == s or h.endswith("." + s) for s in KNOWN_SHORTENERS)
        except Exception:
            return False

    def check_ip_address(hostname: str) -> bool:
        if not hostname:
            return False
        host = hostname.split(":")[0].strip("[]")
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    def check_at_symbol(url: str) -> bool:
        return "@" in url

    def check_double_slash_redirect(url: str) -> bool:
        return "//" in urlparse(url).path

    def check_prefix_suffix_hyphen(hostname: str) -> bool:
        if not hostname:
            return False
        parts = hostname.split(".")
        body = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
        return "-" in body

    def count_subdomains(hostname: str) -> int:
        if not hostname or check_ip_address(hostname):
            return 0
        parts = [p for p in hostname.split(".") if p]
        return max(0, len(parts) - 2)


# Canonical order of features expected by the trained ML models
FEATURE_ORDER: List[str] = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "sslfinal_state",
    "port",
    "https_token",
]


def extract_ml_features(url: str) -> Dict[str, int]:
    """
    Extracts the 10 URL-computable features and maps them to the exact UCI
    encoding (-1 for phishing, 0 for suspicious, 1 for legitimate).

    Rules derived from UCI ID 327 specification:
      1. having_ip_address:
         - Phishing (-1): Hostname is an IPv4 or IPv6 address.
         - Legitimate (1): Hostname is a standard domain name.
      2. url_length:
         - Legitimate (1): Length < 54 characters.
         - Suspicious (0): 54 <= Length <= 75 characters.
         - Phishing (-1): Length > 75 characters.
      3. shortining_service:
         - Phishing (-1): Hostname matches a known shortening service.
         - Legitimate (1): Standard non-shortened domain.
      4. having_at_symbol:
         - Phishing (-1): URL contains '@' (disguising actual destination).
         - Legitimate (1): URL does not contain '@'.
      5. double_slash_redirecting:
         - Phishing (-1): '//' appears after index 7 (e.g. in path or query).
         - Legitimate (1): Standard protocol '//' only.
      6. prefix_suffix:
         - Phishing (-1): Hyphen '-' present in domain name.
         - Legitimate (1): No hyphen in domain name.
      7. having_sub_domain:
         - Legitimate (1): 0 subdomains (dots <= 1 after stripping TLD).
         - Suspicious (0): 1 subdomain.
         - Phishing (-1): >= 2 subdomains (or IP address).
      8. sslfinal_state:
         - Legitimate (1): URL scheme is HTTPS.
         - Phishing (-1): URL scheme is unencrypted HTTP or missing.
      9. port:
         - Phishing (-1): Explicit non-standard port (not 80 or 443).
         - Legitimate (1): Standard port (80/443) or omitted.
      10. https_token:
         - Phishing (-1): Token 'https' is part of domain name body.
         - Legitimate (1): Token 'https' not present in domain body.

    Args:
        url: The raw URL string to extract features from.

    Returns:
        Dict mapping each feature name in FEATURE_ORDER to its integer encoding.
    """
    # Safe default: return all 1s (legitimate) on invalid/empty URL inputs
    default_features = {feat: 1 for feat in FEATURE_ORDER}

    if not url or not isinstance(url, str):
        return default_features

    try:
        clean_url = url.strip()
        parsed = urlparse(clean_url)
        hostname = (parsed.hostname or "").lower().strip()

        # 1. having_ip_address
        has_ip = check_ip_address(hostname)
        f_ip = -1 if has_ip else 1

        # 2. url_length
        url_len = len(clean_url)
        if url_len < 54:
            f_length = 1
        elif url_len <= 75:
            f_length = 0
        else:
            f_length = -1

        # 3. shortining_service
        f_shortener = -1 if is_url_shortener(clean_url) else 1

        # 4. having_at_symbol
        f_at = -1 if check_at_symbol(clean_url) else 1

        # 5. double_slash_redirecting
        # In standard URLs, "//" appears at index 5 (http://) or 6 (https://).
        # Any occurrence after index 7 or inside the parsed path indicates redirecting.
        has_redirect_slash = (clean_url.rfind("//") > 7) or check_double_slash_redirect(clean_url)
        f_double_slash = -1 if has_redirect_slash else 1

        # 6. prefix_suffix (hyphen in domain)
        f_prefix_suffix = -1 if check_prefix_suffix_hyphen(hostname) else 1

        # 7. having_sub_domain
        # Subdomain count: 0 -> 1 (legit), 1 -> 0 (suspicious), >= 2 -> -1 (phishing)
        if has_ip:
            f_subdomain = -1
        else:
            sub_count = count_subdomains(hostname)
            if sub_count == 0:
                f_subdomain = 1
            elif sub_count == 1:
                f_subdomain = 0
            else:
                f_subdomain = -1

        # 8. sslfinal_state (approximated from scheme)
        f_ssl = 1 if parsed.scheme.lower() == "https" else -1

        # 9. port
        has_non_standard_port = False
        if parsed.port and parsed.port not in (80, 443):
            has_non_standard_port = True
        f_port = -1 if has_non_standard_port else 1

        # 10. https_token in domain
        # Checks if token 'https' is deceptively placed within the domain name
        # e.g., 'http://https-paypal-verify.com' or 'http://login.https.example.com'
        f_https_token = -1 if "https" in hostname else 1

        return {
            "having_ip_address": f_ip,
            "url_length": f_length,
            "shortining_service": f_shortener,
            "having_at_symbol": f_at,
            "double_slash_redirecting": f_double_slash,
            "prefix_suffix": f_prefix_suffix,
            "having_sub_domain": f_subdomain,
            "sslfinal_state": f_ssl,
            "port": f_port,
            "https_token": f_https_token,
        }

    except Exception:
        # Guarantee that extraction never raises
        return default_features


def extract_feature_vector(url: str) -> List[int]:
    """
    Extracts features in the exact list order specified by FEATURE_ORDER,
    suitable for passing to scikit-learn models.
    """
    feat_dict = extract_ml_features(url)
    return [feat_dict[k] for k in FEATURE_ORDER]
