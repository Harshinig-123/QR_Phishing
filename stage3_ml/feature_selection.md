# Stage 3 Feature Selection & Excluded Feature Rationale

This document details the **30 lexical and structural URL features** extracted in Stage 3 (`feature_extractor.py`), their alignment with Stage 2 heuristic rules, and the strict technical rationale for excluding network, DOM, and reputation features.

---

## 1. Static URL-Only Feature Set (30 Features)

All 30 features are computed **100% statically** from the raw URL string using Python's standard library (`urllib.parse`, `re`, `math`). No network calls, socket connections, DNS requests, or external lookups are ever performed.

| # | Feature Name | Computation / Extraction Method | Related Stage 2 Heuristic Rule |
|---|---|---|---|
| 1 | `url_length` | Total character count `len(url)` | Long URL check (>75 / >=54 chars) |
| 2 | `hostname_length` | Character count of parsed hostname | Excessive domain length obfuscation |
| 3 | `path_length` | Character count of URL path component | Deep directory camouflage |
| 4 | `query_length` | Character count of query string | Parameter stuffing & token hiding |
| 5 | `count_dots` | Count of `.` in entire URL | Domain & file extension obfuscation |
| 6 | `count_hyphens` | Count of `-` in entire URL | Brand spoofing hyphen check |
| 7 | `count_digits` | Total numeric characters `[0-9]` | Random subdomain / ID generation |
| 8 | `count_underscores` | Count of `_` in URL | Non-standard URL structure |
| 9 | `count_slashes` | Count of `/` across URL | Directory nesting & redirection |
| 10 | `count_at` | Count of `@` in URL | Stage 2 `@` symbol obfuscation |
| 11 | `count_question` | Count of `?` in URL | Dynamic query manipulation |
| 12 | `count_equal` | Count of `=` in URL | Parameter assignment frequency |
| 13 | `count_ampersand` | Count of `&` in URL | Multi-parameter payload staging |
| 14 | `count_percent` | Count of `%` in URL | Hex / percent URL-encoding |
| 15 | `digit_ratio` | `count_digits / url_length` | Automated / algorithmic URL generation |
| 16 | `letter_ratio` | `letter_count / url_length` | Lexical naturalness metric |
| 17 | `special_char_ratio` | `(url_len - digits - letters) / url_len` | Punctuation / obfuscation density |
| 18 | `is_ip_host` | Binary (1 if IPv4/IPv6 literal or hex/int IP, 0 else) | Stage 2 Raw IP Host rule (+35 pts) |
| 19 | `has_at_symbol` | Binary (1 if `@` in URL, 0 else) | Stage 2 `@` Symbol rule (+30 pts) |
| 20 | `is_https` | Binary (1 if scheme == `https`, 0 else) | Stage 2 Insecure HTTP rule (+10 pts) |
| 21 | `num_subdomains` | Subdomain count excluding `www` and base domain | Stage 2 Deep Subdomain Nesting (+15 pts) |
| 22 | `path_depth` | Count of directory levels in path | Directory depth / fake path hierarchy |
| 23 | `has_port` | Binary (1 if explicit `:port` present, 0 else) | Stage 2 Port presence check |
| 24 | `non_standard_port` | Binary (1 if port not in {80, 443}, 0 else) | Stage 2 Non-standard Port rule (+15 pts) |
| 25 | `has_punycode` | Binary (1 if `xn--` in hostname, 0 else) | IDN homograph attack detection |
| 26 | `hostname_entropy` | Shannon character entropy: $-\sum p \log_2 p$ | Algorithmic domain generation (DGA) |
| 27 | `longest_token_length` | Max length of token split by `.` / `-` / `_` | Long pseudo-brand token obfuscation |
| 28 | `suspicious_keyword_count` | Count of matched keywords from Stage 2 | Stage 2 Credential Keywords (+25 pts) |
| 29 | `is_known_shortener` | Binary (1 if host matches Stage 2 shorteners) | Stage 2 Shortener Detection (+15 pts) |
| 30 | `has_double_slash_in_path`| Binary (1 if `//` appears inside path) | Stage 2 Double Slash Redirect (+15 pts) |

---

## 2. Deliberately Excluded Data Sources & Features

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
