"""
Automated tests for Stage 2. Fully offline: all network calls are mocked.

Run:  python test_heuristics.py
"""
import json
import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heuristics as h  # noqa: E402
from heuristics import analyze_url  # noqa: E402


def run(url):
    """Analyse without any network."""
    return analyze_url(url, resolve=False)


class TestRules(unittest.TestCase):
    def test_clean_https_site_is_legitimate(self):
        r = run("https://www.google.com")
        self.assertEqual(r["score"], 0)
        self.assertEqual(r["verdict"], "legitimate")
        self.assertEqual(r["triggered_rules"], [])

    def test_raw_ip_host(self):
        r = run("http://192.168.1.50/login/secure-update")
        self.assertEqual(r["features"]["having_ip_address"], 1)
        # IP 35 + http 10 + keywords(login, secure) 20
        self.assertEqual(r["score"], 65)
        self.assertEqual(r["verdict"], "phishing")
        self.assertEqual(r["features"]["matched_keywords"], ["login", "secure"])

    def test_obfuscated_ip_forms(self):
        for u in ("http://3232235777/", "http://0xC0A80132/", "http://[::1]/x"):
            self.assertEqual(run(u)["features"]["having_ip_address"], 1, u)

    def test_at_symbol(self):
        r = run("http://google.com@evil.com/")
        self.assertEqual(r["features"]["having_at_symbol"], 1)
        self.assertEqual(r["score"], 40)  # @ 30 + http 10
        self.assertEqual(r["verdict"], "suspicious")

    def test_hyphen_only_counts_in_domain(self):
        self.assertEqual(run("https://example.com/a-b-c")["features"]["prefix_suffix_hyphen"], 0)
        r = run("https://paypal-verify.com")
        self.assertEqual(r["features"]["prefix_suffix_hyphen"], 1)
        self.assertEqual(r["score"], 25)  # hyphen 15 + keyword 'verify' 10
        self.assertEqual(r["verdict"], "suspicious")  # boundary: 25 is suspicious

    def test_subdomain_nesting(self):
        r = run("https://login.secure.bank.com/")
        self.assertEqual(r["features"]["subdomain_count"], 2)
        self.assertEqual(r["score"], 35)  # 15 + keywords 20
        self.assertEqual(run("https://www.example.com")["features"]["subdomain_count"], 0)
        self.assertEqual(run("https://mail.example.co.uk")["features"]["subdomain_count"], 1)

    def test_https_token_in_domain_over_http(self):
        r = run("http://https-bank.com")
        self.assertEqual(r["score"], 45)  # http 10 + token 20 + hyphen 15
        self.assertEqual(run("https://https-bank.com")["score"], 15)  # no http / token penalty

    def test_double_slash(self):
        r = run("https://example.com//redirect")
        self.assertEqual(r["features"]["double_slash_redirecting"], 1)
        self.assertEqual(r["score"], 15)
        self.assertEqual(run("https://example.com/a")["features"]["double_slash_redirecting"], 0)

    def test_non_standard_port(self):
        self.assertEqual(run("https://example.com:8443/")["score"], 15)
        self.assertEqual(run("https://example.com:443/")["score"], 0)
        self.assertEqual(run("http://example.com:80/")["features"]["non_standard_port"], 0)

    def test_suspicious_tld(self):
        self.assertEqual(run("https://example.tk")["score"], 15)
        self.assertEqual(run("https://example.com")["features"]["suspicious_tld"], 0)

    def test_keyword_points_are_capped(self):
        r = run("https://x.com/login/signin/verify/account/banking")
        self.assertEqual(r["features"]["suspicious_keyword_count"], 5)
        self.assertEqual(r["score"], 25)  # capped at +25

    def test_length_tiers(self):
        base = "https://example.com/"  # 20 chars
        self.assertEqual(run(base + "a" * 33)["score"], 0)    # 53
        self.assertEqual(run(base + "a" * 34)["score"], 10)   # 54
        self.assertEqual(run(base + "a" * 55)["score"], 10)   # 75
        self.assertEqual(run(base + "a" * 56)["score"], 20)   # 76

    def test_score_is_capped_at_100(self):
        r = run("http://paypal.com@192.168.1.1:8080//login/verify/secure/account-x.tk")
        self.assertEqual(r["score"], 100)
        self.assertEqual(r["verdict"], "phishing")

    def test_verdict_boundaries(self):
        self.assertEqual(h.verdict_for_score(0), "legitimate")
        self.assertEqual(h.verdict_for_score(24), "legitimate")
        self.assertEqual(h.verdict_for_score(25), "suspicious")
        self.assertEqual(h.verdict_for_score(54), "suspicious")
        self.assertEqual(h.verdict_for_score(55), "phishing")
        self.assertEqual(h.verdict_for_score(100), "phishing")


class TestFeatures(unittest.TestCase):
    def test_counts(self):
        f = h.extract_features("https://a-b.example.com/p?x=1&y=%20")
        self.assertEqual(f["count_question"], 1)
        self.assertEqual(f["count_equal"], 2)
        self.assertEqual(f["count_percent"], 1)
        self.assertEqual(f["count_hyphens"], 1)
        self.assertEqual(f["count_dots"], 2)

    def test_feature_vector_matches_order(self):
        f = h.extract_features("https://example.com/login")
        vec = h.to_feature_vector(f)
        self.assertEqual(len(vec), len(h.FEATURE_ORDER))
        self.assertTrue(all(isinstance(v, float) for v in vec))

    def test_schema_keys(self):
        r = run("https://example.com")
        self.assertEqual(
            set(r),
            {"original_url", "is_shortener", "resolved_url", "redirect_chain",
             "features", "triggered_rules", "score", "verdict", "resolution_note"},
        )
        self.assertEqual(set(h.FEATURE_ORDER) | {"matched_keywords"}, set(r["features"]))

    def test_json_serialisable(self):
        json.dumps(run("http://192.168.1.50/login"))


@mock.patch("heuristics._is_public_host", return_value=True)
class TestShortenerResolution(unittest.TestCase):
    def test_multi_hop_chain_never_contacts_destination(self, _pub):
        routes = {
            "http://bit.ly/abc": (301, "https://tinyurl.com/xyz"),
            "https://tinyurl.com/xyz": (302, "http://evil-login.tk/verify"),
        }
        calls = []

        def fake(url, timeout):
            calls.append(url)
            return routes[url]

        with mock.patch("heuristics._fetch_redirect", side_effect=fake):
            r = analyze_url("http://bit.ly/abc")

        self.assertTrue(r["is_shortener"])
        self.assertEqual(r["resolved_url"], "http://evil-login.tk/verify")
        self.assertEqual(r["redirect_chain"][-1], "http://evil-login.tk/verify")
        self.assertEqual(len(r["redirect_chain"]), 3)
        self.assertEqual(calls, ["http://bit.ly/abc", "https://tinyurl.com/xyz"])
        self.assertIsNone(r["resolution_note"])
        self.assertEqual(r["features"]["shortening_service"], 1)
        self.assertTrue(any("shortening" in x for x in r["triggered_rules"]))
        # features are derived from the *resolved* URL
        self.assertEqual(r["features"]["url_length"], len("http://evil-login.tk/verify"))
        self.assertEqual(r["features"]["suspicious_tld"], 1)

    def test_shortener_to_ip_matches_readme_flow(self, _pub):
        with mock.patch("heuristics._fetch_redirect",
                        return_value=(301, "http://192.168.1.50/login/secure-update")):
            r = analyze_url("https://bit.ly/3example")
        self.assertEqual(r["resolved_url"], "http://192.168.1.50/login/secure-update")
        self.assertEqual(r["score"], 80)  # IP 35 + shortener 15 + http 10 + keywords 20
        self.assertEqual(r["verdict"], "phishing")

    def test_relative_location_is_joined(self, _pub):
        routes = {
            "http://bit.ly/a": (302, "/b"),
            "http://bit.ly/b": (302, "https://example.com/done"),
        }
        with mock.patch("heuristics._fetch_redirect", side_effect=lambda u, t: routes[u]):
            r = analyze_url("http://bit.ly/a")
        self.assertEqual(r["resolved_url"], "https://example.com/done")

    def test_redirect_loop(self, _pub):
        routes = {
            "http://bit.ly/a": (301, "https://tinyurl.com/b"),
            "https://tinyurl.com/b": (301, "http://bit.ly/a"),
        }
        with mock.patch("heuristics._fetch_redirect", side_effect=lambda u, t: routes[u]):
            r = analyze_url("http://bit.ly/a")
        self.assertIn("loop", r["resolution_note"].lower())

    def test_max_hops_is_enforced(self, _pub):
        counter = {"n": 0}

        def endless(url, timeout):
            counter["n"] += 1
            return 301, f"https://bit.ly/{counter['n']}"

        with mock.patch("heuristics._fetch_redirect", side_effect=endless):
            r = analyze_url("http://bit.ly/start", max_hops=5)
        self.assertEqual(counter["n"], 5)
        self.assertIn("max", r["resolution_note"].lower())

    def test_timeout_is_handled(self, _pub):
        with mock.patch("heuristics._fetch_redirect", side_effect=requests.Timeout()):
            r = analyze_url("http://bit.ly/slow")
        self.assertEqual(r["resolved_url"], "http://bit.ly/slow")
        self.assertIn("timeout", r["resolution_note"].lower())
        self.assertEqual(r["features"]["shortening_service"], 1)

    def test_connection_error_is_handled(self, _pub):
        with mock.patch("heuristics._fetch_redirect",
                        side_effect=requests.ConnectionError("boom")):
            r = analyze_url("http://bit.ly/down")
        self.assertIn("ConnectionError", r["resolution_note"])

    def test_non_redirect_response(self, _pub):
        with mock.patch("heuristics._fetch_redirect", return_value=(200, None)):
            r = analyze_url("http://bit.ly/page")
        self.assertIn("200", r["resolution_note"])
        self.assertEqual(r["resolved_url"], "http://bit.ly/page")

    def test_redirect_to_non_http_scheme(self, _pub):
        with mock.patch("heuristics._fetch_redirect", return_value=(302, "javascript:alert(1)")):
            r = analyze_url("http://bit.ly/x")
        self.assertIn("not a valid", r["resolution_note"].lower())
        self.assertTrue(any("non-HTTP" in x for x in r["triggered_rules"]))

    def test_no_network_for_regular_urls(self, _pub):
        with mock.patch("heuristics._fetch_redirect") as fetch:
            analyze_url("https://example.com/login")
        fetch.assert_not_called()

    def test_resolve_false_makes_no_requests(self, _pub):
        with mock.patch("heuristics._fetch_redirect") as fetch:
            r = analyze_url("http://bit.ly/x", resolve=False)
        fetch.assert_not_called()
        self.assertTrue(r["is_shortener"])
        self.assertIsNotNone(r["resolution_note"])


class TestSafetyGuards(unittest.TestCase):
    def test_private_shortener_host_is_refused(self):
        with mock.patch("heuristics._is_public_host", return_value=False), \
             mock.patch("heuristics._fetch_redirect") as fetch:
            r = analyze_url("http://bit.ly/x")
        fetch.assert_not_called()
        self.assertIn("public", r["resolution_note"].lower())

    def test_shortener_matching_is_exact(self):
        self.assertTrue(h._is_shortener_host("bit.ly"))
        self.assertTrue(h._is_shortener_host("www.bit.ly"))
        self.assertFalse(h._is_shortener_host("notbit.ly"))
        self.assertFalse(h._is_shortener_host("microsoft.co"))
        self.assertFalse(h._is_shortener_host("bit.ly.evil.com"))


class TestResilience(unittest.TestCase):
    def test_bad_input_types(self):
        with self.assertRaises(TypeError):
            analyze_url(None)
        with self.assertRaises(TypeError):
            analyze_url(123)
        with self.assertRaises(ValueError):
            analyze_url("   ")

    def test_malformed_urls_do_not_crash(self):
        for u in ("javascript:alert(1)", "http://", "http://[::1", "ftp://example.com",
                  "http://example.com:99999/"):
            r = run(u)
            self.assertEqual(r["verdict"], "suspicious", u)
            self.assertEqual(r["score"], 30, u)

    def test_missing_scheme_is_assumed_http(self):
        r = run("example.com/login")
        self.assertTrue(r["original_url"].startswith("http://"))
        self.assertEqual(r["features"]["is_https"], 0)

    def test_very_long_and_unicode_urls(self):
        r = run("https://example.com/" + "a" * 10000)
        self.assertEqual(r["score"], 20)
        r = run("https://exämple.com/路径?q=ü")
        self.assertIn(r["verdict"], {"legitimate", "suspicious", "phishing"})

    def test_whitespace_is_stripped(self):
        self.assertEqual(run("  https://example.com  ")["original_url"], "https://example.com")


if __name__ == "__main__":
    unittest.main(verbosity=2)