"""
test_feature_extractor.py
-------------------------
Unit tests for Stage 3 feature extraction logic against known URL patterns:
  1. IP address URL
  2. URL containing '@' symbol
  3. Long URL (>75 characters)
  4. Shortener URL (bit.ly / tinyurl)
  5. Normal legitimate HTTPS URL
  6. URL with double-slash redirect in path
  7. URL with non-standard port
  8. Deceptive HTTPS token in domain
  9. Malformed / empty / invalid URLs
"""

import unittest
from feature_extractor import extract_ml_features, extract_feature_vector, FEATURE_ORDER


class TestFeatureExtractor(unittest.TestCase):

    def test_normal_legitimate_url(self):
        """Standard clean HTTPS URL should evaluate as legitimate (all 1s)."""
        url = "https://www.google.com"
        features = extract_ml_features(url)
        self.assertEqual(features["having_ip_address"], 1)
        self.assertEqual(features["url_length"], 1)  # < 54 chars
        self.assertEqual(features["shortining_service"], 1)
        self.assertEqual(features["having_at_symbol"], 1)
        self.assertEqual(features["double_slash_redirecting"], 1)
        self.assertEqual(features["prefix_suffix"], 1)
        self.assertEqual(features["sslfinal_state"], 1)
        self.assertEqual(features["port"], 1)
        self.assertEqual(features["https_token"], 1)

    def test_ip_address_url(self):
        """URL with IP address as hostname should be flagged with having_ip_address = -1."""
        url = "http://192.168.1.50/login/secure-update"
        features = extract_ml_features(url)
        self.assertEqual(features["having_ip_address"], -1)
        self.assertEqual(features["sslfinal_state"], -1)  # http -> -1

    def test_at_symbol_url(self):
        """URL with '@' symbol should be flagged with having_at_symbol = -1."""
        url = "http://legit-service.com@evil-phish.net/login"
        features = extract_ml_features(url)
        self.assertEqual(features["having_at_symbol"], -1)

    def test_long_url(self):
        """URL with > 75 characters should be flagged with url_length = -1."""
        url = "https://example.com/very/long/path/with/lots/of/parameters/and/tokens/that/exceeds/seventy-five/characters"
        self.assertGreater(len(url), 75)
        features = extract_ml_features(url)
        self.assertEqual(features["url_length"], -1)

    def test_medium_url_length(self):
        """URL between 54 and 75 characters should evaluate to 0 (suspicious)."""
        url = "https://example.com/" + "a" * 40  # 20 + 40 = 60 chars
        self.assertTrue(54 <= len(url) <= 75)
        features = extract_ml_features(url)
        self.assertEqual(features["url_length"], 0)

    def test_shortener_url(self):
        """Known URL shortener should be flagged with shortining_service = -1."""
        url = "https://bit.ly/3xPh1sh"
        features = extract_ml_features(url)
        self.assertEqual(features["shortining_service"], -1)

    def test_prefix_suffix_hyphen(self):
        """Domain with hyphen should be flagged with prefix_suffix = -1."""
        url = "https://paypal-security-update.com/signin"
        features = extract_ml_features(url)
        self.assertEqual(features["prefix_suffix"], -1)

    def test_subdomains(self):
        """Multiple subdomains should be flagged with having_sub_domain = -1."""
        url = "https://login.verify.account.chase.com"
        features = extract_ml_features(url)
        self.assertEqual(features["having_sub_domain"], -1)

    def test_double_slash_redirect(self):
        """URL with '//' in the path should be flagged with double_slash_redirecting = -1."""
        url = "http://example.com//redirect-path"
        features = extract_ml_features(url)
        self.assertEqual(features["double_slash_redirecting"], -1)

    def test_non_standard_port(self):
        """URL specifying an unusual port should be flagged with port = -1."""
        url = "http://example.com:8080/portal"
        features = extract_ml_features(url)
        self.assertEqual(features["port"], -1)

    def test_https_token_in_domain(self):
        """URL embedding 'https' into its domain body should be flagged with https_token = -1."""
        url = "http://https-paypal.com"
        features = extract_ml_features(url)
        self.assertEqual(features["https_token"], -1)

    def test_malformed_and_empty_urls(self):
        """Malformed or empty inputs must never crash and must return valid vector."""
        for bad_input in ["", "   ", "not a url", "http://", None, 12345]:
            features = extract_ml_features(bad_input)
            self.assertIsInstance(features, dict)
            self.assertEqual(len(features), len(FEATURE_ORDER))
            vec = extract_feature_vector(bad_input)
            self.assertEqual(len(vec), len(FEATURE_ORDER))


if __name__ == "__main__":
    unittest.main()
