# Stage 3 Feature Selection & Excluded Feature Rationale

This document details the **23 static lexical and structural URL features** extracted in Stage 3 (`feature_extractor.py`), their alignment with Stage 2 heuristic rules, the empirical distribution analysis that led to dropping dataset shortcut features, and the strict technical rationale for excluding network, DOM, and reputation features.

---

## 1. Empirical Training Data Distribution & Dropped Shortcut Features

During full pipeline evaluation, Stage 3 initially exhibited false positives on legitimate URLs containing paths or trailing slashes:
- `https://www.google.com` $\rightarrow$ `legitimate` (0.9946)
- `https://www.google.com/` $\rightarrow$ `phishing` (1.0000)
- `https://docs.python.org` $\rightarrow$ `phishing` (0.6840)
- `https://docs.python.org/3/library/urllib.parse.html` $\rightarrow$ `phishing` (1.0000)

### Empirical Distribution in Training Data (`urls_labeled.csv` / PhiUSIIL)
An empirical distribution analysis of the 235,370 labeled URLs revealed severe collection bias in the raw benchmark dataset:

| Feature / Attribute | Legitimate Class ($N = 134,850$) | Phishing Class ($N = 100,520$) | Dataset Artifact Consequence |
|---|---|---|---|
| `count_slashes` | **Mean: 2.0000, Std: 0.0000** (100% == 2) | Mean: 3.0165, Min: 2, Max: 68 | Any URL with $\ge 3$ slashes was split as 100% phishing |
| `path_depth` | **Mean: 0.0000, Std: 0.0000** (100% == 0) | Mean: 0.5870, Max: 33 | Any non-empty path was split as 100% phishing |
| `path_length` | **Mean: 0.0000, Std: 0.0000** (100% == 0) | Mean: 8.7132, Max: 4,248 | Any URL with a path length > 0 was split as phishing |
| `has_trailing_slash` | **0 out of 134,850 (0.00%)** | 36,524 out of 100,520 (36.3%) | `https://google.com/` treated as phishing |
| `has_query` | **0 out of 134,850 (0.00%)** | 6,134 out of 100,520 (6.1%) | Any URL with `?param=value` treated as phishing |
| `has_www` prefix | **134,850 out of 134,850 (100.0%)** | 41,716 out of 100,520 (41.5%) | Naked domains (`github.com`) split as phishing |

Because legitimate URLs in PhiUSIIL were gathered solely from root-domain homepages (`https://www.domain.com`), decision trees learned that **the mere presence of a path, trailing slash, or query parameter was a shortcut indicator of phishing**.

### Architectural Solution
To eliminate these false positives while preserving robust adversarial detection:
1. **URL Normalization (`normalize_url`)**:
   Static canonicalization strips trailing slashes on authority/root URLs (`https://www.google.com/` $\rightarrow$ `https://www.google.com`).
2. **Dropped Shortcut Features**:
   Completely removed `count_slashes`, `path_depth`, `path_length`, `query_length`, `count_question`, `count_equal`, and `count_ampersand`.
3. **Refocused on Hostname Metrics**:
   Replaced whole-URL character counts with `hostname_dots`, `hostname_hyphens`, and `hostname_digits`. Attackers abuse hyphens, digits, and dots in hostnames (`paypal-security-update.com`), whereas legitimate URLs naturally have hyphens and dots in paths/slugs (`/3/library/urllib.parse.html`).
4. **Aligned Subdomain Rule with Stage 2**:
   `num_subdomains` only flags excessive subdomain nesting ($\ge 2$ subdomains). Single subdomains (`docs.python.org`, `mail.google.com`, `en.wikipedia.org`) are recognized as standard legitimate web architecture.
5. **Benign Path & Subdomain Augmentation**:
   During training in `train_model.py`, 40% of legitimate training samples are augmented with realistic benign paths, subdomains, and naked domains, teaching the model that paths and subdomains are standard web structures.

---

## 2. Static URL-Only Feature Set (23 Features)

All 23 features are computed **100% statically** from the raw URL string using Python's standard library (`urllib.parse`, `re`, `math`). Zero network calls, socket connections, DNS requests, or external lookups are ever performed.

| # | Feature Name | Computation / Extraction Method | Related Stage 2 Heuristic Rule |
|---|---|---|---|
| 1 | `url_length` | Total character count `len(url)` | Long URL check (>75 / >=54 chars) |
| 2 | `hostname_length` | Character count of parsed hostname | Excessive domain length obfuscation |
| 3 | `hostname_dots` | Count of `.` strictly within hostname | Hostname nesting & subdomain stacking |
| 4 | `hostname_hyphens` | Count of `-` strictly within hostname | Stage 2 Brand spoofing hyphen rule |
| 5 | `hostname_digits` | Numeric characters `[0-9]` in hostname | Numeric domain / spoofing identification |
| 6 | `count_underscores` | Count of `_` across URL | Non-standard URL structure |
| 7 | `count_at` | Count of `@` in URL | Stage 2 `@` symbol obfuscation |
| 8 | `count_percent` | Count of `%` in URL | Hex / percent URL-encoding |
| 9 | `digit_ratio` | Total digits / `url_length` | Automated / algorithmic URL generation |
| 10 | `letter_ratio` | Letter count / `url_length` | Lexical naturalness metric |
| 11 | `special_char_ratio` | Special character count / `url_length` | Punctuation / obfuscation density |
| 12 | `is_ip_host` | Binary (1 if IPv4/IPv6 literal, 0 else) | Stage 2 Raw IP Host rule (+35 pts) |
| 13 | `has_at_symbol` | Binary (1 if `@` in URL, 0 else) | Stage 2 `@` Symbol rule (+30 pts) |
| 14 | `is_https` | Binary (1 if scheme == `https`, 0 else) | Stage 2 Insecure HTTP rule (+10 pts) |
| 15 | `num_subdomains` | Excessive subdomains beyond base ($\ge 2$) | Stage 2 Deep Subdomain Nesting (+15 pts) |
| 16 | `has_port` | Binary (1 if explicit `:port` present, 0 else) | Stage 2 Port presence check |
| 17 | `non_standard_port` | Binary (1 if port not in {80, 443}, 0 else) | Stage 2 Non-standard Port rule (+15 pts) |
| 18 | `has_punycode` | Binary (1 if `xn--` in hostname, 0 else) | IDN homograph attack detection |
| 19 | `hostname_entropy` | Shannon character entropy: $-\sum p \log_2 p$ | Algorithmic domain generation (DGA) |
| 20 | `longest_token_length` | Max length of token split by `.` / `-` / `_` | Long pseudo-brand token obfuscation |
| 21 | `suspicious_keyword_count` | Count of matched keywords from Stage 2 | Stage 2 Credential Keywords (+25 pts) |
| 22 | `is_known_shortener` | Binary (1 if host matches Stage 2 shorteners) | Stage 2 Shortener Detection (+15 pts) |
| 23 | `has_double_slash_in_path`| Binary (1 if `//` appears inside path) | Stage 2 Double Slash Redirect (+15 pts) |

---

## 3. Deliberately Excluded Data Sources & Features

In academic literature (e.g. models trained on the original 30-feature UCI dataset), classifiers claim 95–97% accuracy by relying heavily on external web requests and page inspection. In the **Quishing Shield** project, all such features are **strictly and deliberately excluded**:

### A. Webpage Content & DOM Features (Excluded)
* **Excluded:** `LineOfCode`, `LargestLineLength`, `HasTitle`, `TitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasHiddenFields`, `HasPasswordField`, `Bank`/`Pay`/`Crypto` content tags, `NoOfImage`, `NoOfCSS`, `NoOfJS`.
* **Reason for Exclusion:** 
  1. **Drive-by Malware & Zero-Day Safety:** QR code scanning is an untrusted endpoint action. Fetching HTML/JavaScript from an unverified adversary server can trigger browser exploits, session hijacking, or malicious payload execution.
  2. **Latency:** QR scanning requires instantaneous classification (<15ms). Downloading megabytes of DOM assets over mobile cellular connections introduces unacceptable latency (2–5 seconds).
  3. **Ephemeral Landing Pages:** Phishing campaigns frequently employ cloaking (returning 404 or blank pages to crawlers while serving phishing kits to mobile user agents).

### B. Network, DNS & WHOIS Features (Excluded)
* **Excluded:** `dnsrecord`, `age_of_domain`, `domain_registration_length`, reverse DNS lookups, registrar details.
* **Reason for Exclusion:**
  1. **Privacy:** Performing DNS and WHOIS lookups exposes user browsing habits and location to network intermediaries and recursive resolvers.
  2. **DNS Rate Limiting & Failures:** Public WHOIS servers rate-limit or block bulk queries; network timeouts can stall the classification pipeline.

### C. Third-Party Traffic & PageRank Features (Excluded)
* **Excluded:** `web_traffic`, `page_rank`, `google_index`, `links_pointing_to_page`.
* **Reason for Exclusion:**
  1. **Service Deprecation:** Alexa Rank and Google PageRank APIs have been discontinued or paywalled.
  2. **Vendor Lock-in & Online Dependency:** An air-gapped or offline mobile security scanner must function reliably without external API subscription keys.

### D. TLD and Domain Identity Memorization (Excluded)
* **Excluded:** Categorical one-hot encodings of domain names (`Domain`) or top-level domains (`TLD`).
* **Reason for Exclusion:**
  1. **Dataset Shortcut / Memorization Bias:** Including categorical domain names causes decision trees to memorize benign domain strings (`google.com`, `wikipedia.org`) rather than learning structural attack signatures, resulting in poor out-of-sample generalization.
