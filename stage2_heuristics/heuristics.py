"""
Stage 2 - Heuristic + URL-shortener analysis.

Takes a validated URL from Stage 1 (qr_decoder.py), safely expands URL
shorteners, runs rule-based phishing heuristics and returns a risk score,
a verdict and a feature dictionary for Stage 3 (ML classification).

Safety design (motivated by Trad & Chehab, "Detecting Quishing Attacks with
Machine Learning Techniques Through QR Code Analysis", arXiv:2505.03451):
resolving a QR payload can itself expose the user or the analysis host to a
malicious site. Therefore:

  * Only hosts on the known-shortener list are ever contacted.
  * Each hop is requested with HEAD (GET fallback, body never read) and
    redirects are NOT followed automatically.
  * As soon as a Location header points to a non-shortener host, resolution
    stops. The final destination is recorded but never connected to.
  * Shortener hosts that resolve to private / loopback addresses are refused
    (SSRF guard), hops are capped and redirect loops are detected.

Public API:
    analyze_url(url, resolve=True, timeout=5.0, max_hops=5) -> dict
    resolve_shortener(url, timeout=5.0, max_hops=5) -> (resolved, chain, note)
    extract_features(url, shortener=None) -> dict
    to_feature_vector(features) -> list[float]
"""
from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import SplitResult, urljoin, urlsplit

import requests

__all__ = [
    "analyze_url",
    "resolve_shortener",
    "extract_features",
    "to_feature_vector",
    "verdict_for_score",
    "FEATURE_ORDER",
]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_HOPS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
USER_AGENT = "Mozilla/5.0 (compatible; QR-Phishing-Analyzer/1.0; +stage2)"

SHORTENER_DOMAINS = frozenset({
    "bit.ly", "bitly.com", "j.mp", "tinyurl.com", "tiny.cc", "t.co", "goo.gl",
    "ow.ly", "is.gd", "v.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
    "rb.gy", "bl.ink", "lnkd.in", "t.ly", "s.id", "soo.gd", "qrco.de", "qr.net",
    "1url.com", "adf.ly", "shorte.st", "x.co", "snip.ly", "short.io", "clck.ru",
    "u.to", "trib.al", "dlvr.it", "fb.me", "tr.im", "urlz.fr", "shorturl.com",
    "mcaf.ee", "po.st", "tny.im", "chilp.it", "bc.vc", "ouo.io", "gg.gg",
})

SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work", "zip", "mov",
    "loan", "country", "kim", "men", "party", "review", "science", "stream",
    "download", "racing", "win", "bid", "icu", "cyou", "buzz", "rest",
    "monster", "su",
})

PHISHING_KEYWORDS = (
    "login", "signin", "verify", "account", "banking", "secure", "password",
    "wallet", "confirm", "webscr", "credential", "authenticate", "suspend",
    "unlock", "billing",
)

_SECOND_LEVEL_LABELS = frozenset(
    {"co", "com", "org", "net", "gov", "edu", "ac", "ne", "or", "go"}
)

# Verdict thresholds (inclusive lower bounds), see README.
SUSPICIOUS_THRESHOLD = 25
PHISHING_THRESHOLD = 55

# Numeric features in a fixed order, for Stage 3 models (XGBoost/LightGBM...).
FEATURE_ORDER: Tuple[str, ...] = (
    "url_length", "having_ip_address", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix_hyphen", "subdomain_count",
    "is_https", "https_token_in_domain", "shortening_service",
    "non_standard_port", "suspicious_tld", "suspicious_keyword_count",
    "count_dots", "count_hyphens", "count_question", "count_equal",
    "count_percent",
)


# --------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------
_NON_HTTP_SCHEME = re.compile(
    r"^(javascript|data|mailto|tel|sms|file|vbscript):", re.IGNORECASE
)


def _normalize(url: str) -> str:
    """Strip whitespace; assume http:// when the scheme is missing."""
    url = url.strip()
    if "://" not in url and not _NON_HTTP_SCHEME.match(url):
        url = "http://" + url
    return url


def _parse_valid(url: str) -> Optional[SplitResult]:
    """Return the parsed URL if it is a well-formed http(s) URL, else None."""
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises ValueError on an invalid port
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host:
        return None
    return parts


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_shortener_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in SHORTENER_DOMAINS)


def _is_ip_host(host: str) -> bool:
    """True for IPv4/IPv6 literals and integer/hex/octal obfuscated IPv4."""
    h = host.strip("[]").lower()
    if not h:
        return False
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    if re.fullmatch(r"0x[0-9a-f]+", h) or re.fullmatch(r"\d{8,10}", h):
        return True
    return bool(re.fullmatch(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){3}", h))


def _subdomain_count(host: str) -> int:
    """Number of subdomain labels, ignoring a leading 'www'."""
    if not host or _is_ip_host(host):
        return 0
    labels = [label for label in host.split(".") if label]
    if labels and labels[0] == "www":
        labels = labels[1:]
    if len(labels) < 2:
        return 0
    base = 2
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL_LABELS:
        base = 3  # e.g. example.co.uk
    return max(0, len(labels) - base)


def verdict_for_score(score: int) -> str:
    if score >= PHISHING_THRESHOLD:
        return "phishing"
    if score >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "legitimate"


# --------------------------------------------------------------------------
# Safe shortener resolution
# --------------------------------------------------------------------------
def _is_public_host(host: str) -> bool:
    """SSRF guard: every address the host resolves to must be globally routable."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def _fetch_redirect(url: str, timeout: float) -> Tuple[int, Optional[str]]:
    """
    Request ONE hop without following redirects or reading any body.
    Returns (status_code, Location header or None).
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    with requests.Session() as session:
        resp = session.head(url, allow_redirects=False, timeout=timeout, headers=headers)
        try:
            if resp.status_code not in REDIRECT_STATUSES:
                # Some shorteners reject HEAD; retry with GET but never read the body.
                resp.close()
                resp = session.get(
                    url, allow_redirects=False, timeout=timeout,
                    headers=headers, stream=True,
                )
            return resp.status_code, resp.headers.get("Location")
        finally:
            resp.close()


def resolve_shortener(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> Tuple[str, List[str], Optional[str]]:
    """
    Expand a shortened URL without ever contacting the final destination.

    Returns (resolved_url, redirect_chain, note). `note` is None on a clean
    resolution and a short explanation when resolution stopped early.
    """
    chain = [url]
    current = url
    hops = 0

    while _is_shortener_host(_host_of(current)):
        if hops >= max_hops:
            return current, chain, f"Stopped after max redirect hops ({max_hops})"
        host = _host_of(current)
        if not _is_public_host(host):
            return current, chain, f"Refused to contact '{host}': not a public address"
        try:
            status, location = _fetch_redirect(current, timeout)
        except requests.Timeout:
            return current, chain, f"Timeout while resolving '{host}'"
        except requests.RequestException as exc:
            return current, chain, f"Network error while resolving '{host}': {type(exc).__name__}"
        hops += 1

        if status not in REDIRECT_STATUSES or not location:
            return current, chain, f"Shortener returned HTTP {status} without a redirect"

        nxt = urljoin(current, location.strip())
        if nxt in chain:
            chain.append(nxt)
            return nxt, chain, "Redirect loop detected"
        chain.append(nxt)
        current = nxt
        if _parse_valid(nxt) is None:
            return nxt, chain, "Redirect target is not a valid HTTP(S) URL"

    return current, chain, None


# --------------------------------------------------------------------------
# Feature extraction (UCI Phishing Websites style)
# --------------------------------------------------------------------------
def _empty_features(length: int = 0) -> Dict[str, Any]:
    feats: Dict[str, Any] = {name: 0 for name in FEATURE_ORDER}
    feats["url_length"] = length
    feats["matched_keywords"] = []
    return feats


def extract_features(url: str, shortener: Optional[bool] = None) -> Dict[str, Any]:
    """Extract the Stage 3 feature dictionary from a (resolved) http(s) URL."""
    parts = _parse_valid(url)
    if parts is None:
        raise ValueError(f"Not a valid http(s) URL: {url!r}")

    host = parts.hostname.lower()
    port = parts.port
    lowered = url.lower()
    matched = [k for k in PHISHING_KEYWORDS if k in lowered]
    is_short = _is_shortener_host(host) if shortener is None else bool(shortener)
    tld = host.rsplit(".", 1)[-1] if "." in host and not _is_ip_host(host) else ""

    return {
        "url_length": len(url),
        "having_ip_address": int(_is_ip_host(host)),
        "having_at_symbol": int("@" in url),
        "double_slash_redirecting": int(url.rfind("//") > 7),
        "prefix_suffix_hyphen": int("-" in host),
        "subdomain_count": _subdomain_count(host),
        "is_https": int(parts.scheme.lower() == "https"),
        "https_token_in_domain": int("https" in host),
        "shortening_service": int(is_short),
        "non_standard_port": int(port is not None and port not in (80, 443)),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "suspicious_keyword_count": len(matched),
        "matched_keywords": matched,
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_percent": url.count("%"),
    }


def to_feature_vector(features: Dict[str, Any]) -> List[float]:
    """Numeric vector in FEATURE_ORDER, ready for an sklearn/XGBoost/LightGBM model."""
    return [float(features[name]) for name in FEATURE_ORDER]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _score(features: Dict[str, Any], is_http: bool) -> Tuple[int, List[str]]:
    score = 0
    rules: List[str] = []

    def add(points: int, text: str) -> None:
        nonlocal score
        score += points
        rules.append(f"{text} (+{points})")

    if features["having_ip_address"]:
        add(35, "Uses raw IP address as hostname instead of a registered domain")
    if features["having_at_symbol"]:
        add(30, "Contains '@' symbol that can hide the real destination")
    if features["shortening_service"]:
        add(15, "Uses URL shortening service to conceal the target destination")

    length = features["url_length"]
    if length > 75:
        add(20, f"Excessive URL length ({length} chars)")
    elif length >= 54:
        add(10, f"Long URL ({length} chars)")

    if features["prefix_suffix_hyphen"]:
        add(15, "Domain contains hyphen '-' (frequent brand impersonation tactic)")
    if features["subdomain_count"] >= 2:
        add(15, f"Deep subdomain nesting ({features['subdomain_count']} subdomains)")
    if is_http:
        add(10, "Uses unencrypted HTTP connection")
        if features["https_token_in_domain"]:
            add(20, "Deceptive 'https' token inside domain name while served over HTTP")
    if features["double_slash_redirecting"]:
        add(15, "Contains '//' redirect pattern inside the URL")
    if features["non_standard_port"]:
        add(15, "Uses a non-standard port")
    if features["suspicious_tld"]:
        add(15, "Uses a high-abuse top-level domain")

    kw = features["matched_keywords"]
    if kw:
        points = min(10 * len(kw), 25)
        names = ", ".join(f"'{k}'" for k in kw)
        add(points, f"Contains credential/security keywords: {names}")

    return min(score, 100), rules


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def _malformed_result(url: str) -> Dict[str, Any]:
    score = 30
    return {
        "original_url": url,
        "is_shortener": False,
        "resolved_url": url,
        "redirect_chain": [url],
        "features": _empty_features(len(url)),
        "triggered_rules": ["URL is malformed or not a valid HTTP(S) URL (+30)"],
        "score": score,
        "verdict": verdict_for_score(score),
        "resolution_note": None,
    }


def analyze_url(
    url: str,
    resolve: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> Dict[str, Any]:
    """
    Analyze a URL from Stage 1 and return the schema documented in README.md.

    Raises TypeError for non-string input and ValueError for empty input.
    Malformed URLs and network failures never raise; they are reflected in the
    returned score / `resolution_note`.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    if not url.strip():
        raise ValueError("url must not be empty")

    candidate = _normalize(url)
    parts = _parse_valid(candidate)
    if parts is None:
        return _malformed_result(candidate)

    is_short = _is_shortener_host(parts.hostname.lower())
    resolved, chain, note = candidate, [candidate], None

    if is_short and resolve:
        resolved, chain, note = resolve_shortener(candidate, timeout, max_hops)
    elif is_short:
        note = "Shortener resolution disabled; only the short link was analysed"

    target = resolved
    extra_points = 0
    extra_rules: List[str] = []
    resolved_parts = _parse_valid(resolved)
    if resolved_parts is None:
        # Shortener pointed at something that is not http(s): analyse the short link
        # itself and penalise the unusable destination.
        target = candidate
        resolved_parts = parts
        extra_points = 30
        extra_rules.append("Shortener redirects to a non-HTTP(S) destination (+30)")

    features = extract_features(target, shortener=is_short)
    score, rules = _score(features, is_http=resolved_parts.scheme.lower() == "http")
    score = min(score + extra_points, 100)
    rules.extend(extra_rules)

    return {
        "original_url": candidate,
        "is_shortener": is_short,
        "resolved_url": resolved,
        "redirect_chain": chain,
        "features": features,
        "triggered_rules": rules,
        "score": score,
        "verdict": verdict_for_score(score),
        "resolution_note": note,
    }