# Stage 3: Feature Analysis & Selection

## 1. Overview
The **UCI Phishing Websites Dataset (UCI ID 327)** contains **30 categorical features** extracted from phishing and legitimate websites. In the original research by Mohammad, Thabtah, and McCluskey, these features are divided into four primary groups:
1. **Address Bar Based Features** (lexical and URL syntactic properties)
2. **Abnormal Based Features** (WHOIS, host identity verification)
3. **HTML and JavaScript Based Features** (DOM structure, scripts, form actions)
4. **Domain Based Features** (external third-party reputations, Alexa traffic, PageRank)

### QR Phishing Detection Constraint
In a real-world QR code security scanner:
- The system frequently operates in low-latency, privacy-preserving, or offline/restricted-connectivity environments.
- Fetching live web pages introduces security risks (drive-by malware downloads, tracking beacons, triggering one-time attacker tokens).
- Querying external third-party services (WHOIS, Alexa, Google Index) introduces rate limits, network latency, and privacy leakage.

Therefore, Stage 3 ML classification strictly utilizes **only features that can be computed deterministically from the raw URL string**.

---

## 2. Comprehensive Analysis of all 30 UCI Features

| # | Feature Name | Computable from URL Alone? | Computation Method / Mapping Rule | Reason for Exclusion (if Not Computable) |
|---|---|:---:|---|---|
| 1 | `having_ip_address` | **Yes** | Check if hostname is an IPv4 or IPv6 address (using Python `ipaddress` module or hex/octal pattern).<br>**UCI Encoding**: `-1` if IP used (phishing), `1` if domain used (legitimate). | *Included* |
| 2 | `url_length` | **Yes** | Character count of the entire URL string.<br>**UCI Encoding**: `1` if length < 54 (legitimate), `0` if 54 ≤ length ≤ 75 (suspicious), `-1` if length > 75 (phishing). | *Included* |
| 3 | `shortining_service` | **Yes** | Check hostname against known shortening domains (`bit.ly`, `tinyurl.com`, `t.co`, etc.).<br>**UCI Encoding**: `-1` if shortener used (phishing), `1` otherwise (legitimate). | *Included* |
| 4 | `having_at_symbol` | **Yes** | Check if `@` character is present anywhere in the URL string.<br>**UCI Encoding**: `-1` if `@` present (phishing), `1` otherwise (legitimate). | *Included* |
| 5 | `double_slash_redirecting` | **Yes** | Check if `//` appears after the protocol scheme delimiter (index > 7, e.g. `http://example.com//path`).<br>**UCI Encoding**: `-1` if redirecting `//` present (phishing), `1` otherwise (legitimate). | *Included* |
| 6 | `prefix_suffix` | **Yes** | Check if a hyphen (`-`) appears in the registered domain part of hostname.<br>**UCI Encoding**: `-1` if hyphen in domain body (phishing), `1` otherwise (legitimate). | *Included* |
| 7 | `having_sub_domain` | **Yes** | Count dots in hostname after stripping `www` and TLD.<br>**UCI Encoding**: `1` if dots ≤ 1 / no subdomains (legitimate), `0` if dots = 2 / 1 subdomain (suspicious), `-1` if dots ≥ 3 / multi-subdomains (phishing). | *Included* |
| 8 | `sslfinal_state` | **Yes (Approximated)** | In UCI, checks SSL certificate issuer and validity. Approximated from URL scheme: `https` indicates SSL presence.<br>**UCI Encoding**: `1` if `https://` (legitimate), `-1` if `http://` (phishing). | *Included (Approximated)* |
| 9 | `domain_registration_length` | **No** | Requires querying WHOIS registry database for domain expiration date. | External network/WHOIS lookup required; slow and unreliable for real-time QR scanning. |
| 10 | `favicon` | **No** | Requires fetching page HTML and verifying whether `<link rel="icon">` loads from an external domain. | Requires downloading external webpage HTML payload. |
| 11 | `port` | **Yes** | Extract port number from URL (via `urllib.parse.urlparse`).<br>**UCI Encoding**: `1` if standard port (80, 443, or omitted), `-1` if explicit non-standard port specified (phishing). | *Included* |
| 12 | `https_token` | **Yes** | Check if string token `https` is present inside the hostname/domain component (e.g. `http://https-secure-bank.com`).<br>**UCI Encoding**: `-1` if token in domain (phishing), `1` otherwise (legitimate). | *Included* |
| 13 | `request_url` | **No** | Measures percentage of external assets (images, multimedia) loaded in webpage DOM. | Requires downloading and parsing complete HTML DOM. |
| 14 | `url_of_anchor` | **No** | Measures percentage of `<a>` tags with empty, foreign, or `#` href values. | Requires downloading and parsing complete HTML DOM. |
| 15 | `links_in_tags` | **No** | Measures percentage of `<meta>`, `<script>`, and `<link>` tags pointing to other domains. | Requires downloading and parsing complete HTML DOM. |
| 16 | `sfh` (Server Form Handler) | **No** | Checks `<form action="...">` attributes for blank strings or foreign domains. | Requires downloading and parsing complete HTML DOM. |
| 17 | `submitting_to_email` | **No** | Checks if page uses `mailto:` or server-side mail functions in forms. | Requires downloading and parsing webpage content. |
| 18 | `abnormal_url` | **No** | Compares host identity against WHOIS registrant name. | Requires external WHOIS database query. |
| 19 | `redirect` | **No** | Measures number of HTTP 301/302 redirects (0-1 vs 2-4 vs 4+). | Requires active HTTP connection and network traversal; not static URL computable. |
| 20 | `on_mouseover` | **No** | Checks JavaScript `window.status` event modifications. | Requires downloading and parsing JavaScript execution. |
| 21 | `rightclick` | **No** | Checks if JavaScript disables right-click context menu (`event.button==2`). | Requires downloading and parsing JavaScript code. |
| 22 | `popupwindow` | **No** | Checks if webpage triggers JavaScript popup dialogs with input fields. | Requires executing and rendering browser DOM. |
| 23 | `iframe` | **No** | Checks for invisible or borderless `<iframe>` elements in HTML. | Requires downloading and parsing complete HTML DOM. |
| 24 | `age_of_domain` | **No** | Computes duration since domain creation date via WHOIS. | Requires external WHOIS database query. |
| 25 | `dnsrecord` | **No** | Validates if domain has active DNS A / SOA records. | Requires live DNS lookup query. |
| 26 | `web_traffic` | **No** | Looks up Alexa / SimilarWeb rank of the website. | Requires external commercial traffic ranking API. |
| 27 | `page_rank` | **No** | Looks up Google PageRank score of the domain. | Requires external PageRank API (now deprecated). |
| 28 | `google_index` | **No** | Checks whether the URL is indexed by Google Search. | Requires scraping or querying Google search index. |
| 29 | `links_pointing_to_page` | **No** | Computes external backlink count to the URL. | Requires external backlink/SEO database. |
| 30 | `statistical_report` | **No** | Checks whether host matches PhishTank or StopBadware blacklists. | Requires external threat intelligence API lookup. |

---

## 3. Selected Feature Set for Model Training

The following **10 features** are strictly computable from the URL string and correspond directly to the UCI dataset columns:

1. `having_ip_address`
2. `url_length`
3. `shortining_service`
4. `having_at_symbol`
5. `double_slash_redirecting`
6. `prefix_suffix`
7. `having_sub_domain`
8. `sslfinal_state`
9. `port`
10. `https_token`

### Benchmark Context for University Report
In academic literature, training on all 30 UCI features often achieves **~95% to 97% accuracy** because features like `sslfinal_state` (full cert check), `url_of_anchor`, `web_traffic`, and `prefix_suffix` provide strong signals from the rendered webpage and commercial services. When constrained to **purely URL-based features** (10 features), the model is lightweight, fast, and does not need to download potentially malicious web content. We report the performance honestly in our evaluation.
