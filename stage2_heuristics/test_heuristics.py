"""
test_heuristics.py
------------------
Automated test suite for stage2_heuristics/heuristics.py.

Tests:
    1. Shortener detection (known vs standard domains)
    2. IP address detection (IPv4 / IPv6)
    3. @ symbol detection
    4. Domain hyphens, subdomains, and keywords
    5. Scoring engine and verdict categories ('legitimate', 'suspicious', 'phishing')
    6. Feature extraction dictionary schema for Member 3
    7. Offline / graceful failure behavior

Run:
    python test_heuristics.py
"""

import sys
from heuristics import (
    is_url_shortener,
    check_ip_address,
    check_at_symbol,
    check_prefix_suffix_hyphen,
    count_subdomains,
    analyze_url,
    extract_features
)


def test_shortener_detection():
    assert is_url_shortener("https://bit.ly/3example") is True
    assert is_url_shortener("http://tinyurl.com/abc1234") is True
    assert is_url_shortener("https://cutt.ly/test") is True
    assert is_url_shortener("https://t.co/abc") is True
    assert is_url_shortener("https://www.google.com") is False
    assert is_url_shortener("https://github.com/torvalds") is False
    print("[PASS] test_shortener_detection")


def test_ip_address_detection():
    assert check_ip_address("192.168.1.50") is True
    assert check_ip_address("10.0.0.1:8080") is True
    assert check_ip_address("127.0.0.1") is True
    assert check_ip_address("google.com") is False
    assert check_ip_address("paypal.com.attacker.com") is False
    print("[PASS] test_ip_address_detection")


def test_at_symbol_detection():
    assert check_at_symbol("http://google.com@evil-site.example/login") is True
    assert check_at_symbol("https://example.com/search?q=test@home") is True
    assert check_at_symbol("https://example.com/login") is False
    print("[PASS] test_at_symbol_detection")


def test_prefix_suffix_hyphen():
    assert check_prefix_suffix_hyphen("paypal-update.com") is True
    assert check_prefix_suffix_hyphen("apple-verify-account.org") is True
    assert check_prefix_suffix_hyphen("google.com") is False
    assert check_prefix_suffix_hyphen("docs.python.org") is False
    print("[PASS] test_prefix_suffix_hyphen")


def test_subdomains():
    assert count_subdomains("google.com") == 0
    assert count_subdomains("www.google.com") == 1
    assert count_subdomains("login.secure.bank.com") == 2
    assert count_subdomains("a.b.c.d.example.com") == 4
    print("[PASS] test_subdomains")


def test_legitimate_url_verdict():
    res = analyze_url("https://www.google.com", resolve_redirects=False)
    assert res["verdict"] == "legitimate", f"Expected legitimate, got {res['verdict']}"
    assert res["score"] < 25
    assert res["features"]["is_https"] == 1
    assert res["features"]["having_ip_address"] == 0
    print(f"[PASS] test_legitimate_url_verdict (score={res['score']}, verdict='{res['verdict']}')")


def test_phishing_ip_url():
    # IP address + HTTP + login/secure keywords
    res = analyze_url("http://192.168.1.50/login/secure-update", resolve_redirects=False)
    assert res["verdict"] == "phishing", f"Expected phishing, got {res['verdict']}"
    assert res["features"]["having_ip_address"] == 1
    assert res["features"]["is_https"] == 0
    assert len(res["features"]["matched_keywords"]) >= 2
    assert res["score"] >= 55
    print(f"[PASS] test_phishing_ip_url (score={res['score']}, verdict='{res['verdict']}')")


def test_phishing_at_symbol_url():
    # @ symbol + HTTP
    res = analyze_url("http://google.com@evil-site.example/login", resolve_redirects=False)
    assert res["verdict"] in ("phishing", "suspicious")
    assert res["features"]["having_at_symbol"] == 1
    assert res["score"] >= 35
    print(f"[PASS] test_phishing_at_symbol_url (score={res['score']}, verdict='{res['verdict']}')")


def test_feature_contract_for_member3():
    features = extract_features("https://paypal-verify.account-update.xyz/login", is_shortener=False)
    required_keys = [
        "url_length", "having_ip_address", "having_at_symbol",
        "double_slash_redirecting", "prefix_suffix_hyphen", "subdomain_count",
        "is_https", "shortening_service", "suspicious_keyword_count",
        "count_dots", "count_hyphens"
    ]
    for key in required_keys:
        assert key in features, f"Missing required ML feature: {key}"
    print("[PASS] test_feature_contract_for_member3")


def test_resilience_to_invalid_input():
    res1 = analyze_url("")
    assert res1["verdict"] == "legitimate"
    assert "triggered_rules" in res1

    res2 = analyze_url(None)
    assert res2["verdict"] == "legitimate"
    print("[PASS] test_resilience_to_invalid_input")


if __name__ == "__main__":
    print("Running Stage 2 Heuristics Test Suite...\n" + "=" * 45)
    test_shortener_detection()
    test_ip_address_detection()
    test_at_symbol_detection()
    test_prefix_suffix_hyphen()
    test_subdomains()
    test_legitimate_url_verdict()
    test_phishing_ip_url()
    test_phishing_at_symbol_url()
    test_feature_contract_for_member3()
    test_resilience_to_invalid_input()
    print("=" * 45 + "\nAll Stage 2 heuristic tests PASSED successfully!")
