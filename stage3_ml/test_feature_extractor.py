"""
test_feature_extractor.py
-------------------------
Unit tests for Stage 3 feature extraction logic:
  1. IP address URLs (is_ip_host = 1)
  2. URLs containing '@' symbol (has_at_symbol = 1, count_at >= 1)
  3. Long URLs (url_length accurately measured)
  4. Punycode IDN URLs (has_punycode = 1)
  5. Shortener URLs (is_known_shortener = 1)
  6. Normal legitimate HTTPS URLs (is_https = 1, is_ip_host = 0)
  7. Malformed, empty, and garbage inputs (safe defaults, never raises)
  8. Static safety verification: monkeypatches socket to raise an exception,
     proving feature extraction makes zero network calls.
"""

import socket
import sys
import unittest
from pathlib import Path

# Ensure stage3_ml directory is in sys.path so tests can run from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor import extract_ml_features, extract_feature_vector, FEATURE_ORDER


class TestFeatureExtractor(unittest.TestCase):

    def test_normal_legitimate_https_url(self):
        """Standard HTTPS URL should produce expected structural values."""
        url = "https://www.google.com"
        features = extract_ml_features(url)
        self.assertEqual(features["is_https"], 1)
        self.assertEqual(features["is_ip_host"], 0)
        self.assertEqual(features["has_at_symbol"], 0)
        self.assertEqual(features["num_subdomains"], 0)
        self.assertEqual(features["has_punycode"], 0)
        self.assertEqual(features["has_port"], 0)
        self.assertEqual(features["is_known_shortener"], 0)
        self.assertGreater(features["letter_ratio"], 0.5)

    def test_ip_address_url(self):
        """URL with IP address host must be flagged with is_ip_host = 1."""
        url = "http://192.168.1.50:8080/login/update"
        features = extract_ml_features(url)
        self.assertEqual(features["is_ip_host"], 1)
        self.assertEqual(features["is_https"], 0)
        self.assertEqual(features["has_port"], 1)
        self.assertEqual(features["non_standard_port"], 1)
        self.assertEqual(features["path_depth"], 2)
        self.assertGreaterEqual(features["suspicious_keyword_count"], 1)

    def test_at_symbol_url(self):
        """URL with '@' symbol must be detected in both boolean and count features."""
        url = "http://legit-service.com@evil-phish.net/login"
        features = extract_ml_features(url)
        self.assertEqual(features["has_at_symbol"], 1)
        self.assertEqual(features["count_at"], 1)

    def test_long_url(self):
        """Length features must accurately measure long URLs."""
        url = "https://example.com/" + "subpath/" * 15 + "?param=value&token=123456789"
        features = extract_ml_features(url)
        self.assertEqual(features["url_length"], len(url))
        self.assertGreater(features["path_depth"], 10)
        self.assertEqual(features["count_ampersand"], 1)

    def test_punycode_url(self):
        """IDN homograph URLs containing 'xn--' must be flagged."""
        url = "http://xn--pple-43d.com/secure-signin"
        features = extract_ml_features(url)
        self.assertEqual(features["has_punycode"], 1)

    def test_shortener_url(self):
        """Known URL shortener domains must be identified."""
        url = "https://bit.ly/secure-redirect-3x"
        features = extract_ml_features(url)
        self.assertEqual(features["is_known_shortener"], 1)

    def test_double_slash_in_path(self):
        """URL containing '//' in path must be flagged for open redirect risk."""
        url = "https://bank.com//redirect?target=http://evil.com"
        features = extract_ml_features(url)
        self.assertEqual(features["has_double_slash_in_path"], 1)

    def test_malformed_and_empty_inputs(self):
        """Empty, non-string, and garbage inputs must never raise exceptions."""
        bad_inputs = ["", "   ", None, 12345, "not_a_url_at_all!!@#$%", "http://", "://broken"]
        for bad in bad_inputs:
            feats = extract_ml_features(bad)
            self.assertIsInstance(feats, dict)
            self.assertEqual(len(feats), len(FEATURE_ORDER))
            vec = extract_feature_vector(bad)
            self.assertEqual(len(vec), len(FEATURE_ORDER))
            self.assertTrue(all(isinstance(v, (int, float)) for v in vec))

    def test_zero_network_calls_guarantee(self):
        """
        Monkeypatches socket.socket and socket.getaddrinfo to raise errors.
        Proves that feature extraction is 100% static and performs ZERO network I/O.
        """
        def forbidden_call(*args, **kwargs):
            raise AssertionError("CRITICAL VIOLATION: Network socket operation attempted during static extraction!")

        orig_socket = socket.socket
        orig_getaddrinfo = socket.getaddrinfo
        try:
            socket.socket = forbidden_call
            socket.getaddrinfo = forbidden_call

            test_urls = [
                "https://www.google.com",
                "http://192.168.1.1/admin",
                "https://bit.ly/test",
                "http://fake-domain-that-does-not-exist-anywhere-in-dns.xyz/login",
            ]
            for u in test_urls:
                feats = extract_ml_features(u)
                self.assertEqual(len(feats), len(FEATURE_ORDER))
        finally:
            socket.socket = orig_socket
            socket.getaddrinfo = orig_getaddrinfo


if __name__ == "__main__":
    unittest.main()
