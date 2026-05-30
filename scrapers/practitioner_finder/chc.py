"""Council for Homeopathic Certification (CHC) practitioner-directory scraper.

CHC certifies the CCH credential (Certified Classical Homeopath). Its public
"Find a Homeopath" directory is a WordPress Business Directory Plugin (wpbdp)
listing at https://homeopathcertification.org/find-a-homeopath/.

Discovery (2026-05-29)
----------------------
- Plain ``requests`` (browser UA) works — no Cloudflare, no JS. (The default
  ``python-requests`` UA gets a 406, so the UA must be overridden.)
- The public ``?wpbdp_view=all_listings`` excerpt view paginates in a
  **random order per request** (verified: two walks of the same pages returned
  ~30% different rosters), so it cannot be paginated to completeness.
- Instead we use the WordPress REST API, which IS deterministic + complete:
  ``/wp-json/wp/v2/wpbdp_listing?per_page=100&page=N`` returns every listing
  (X-WP-Total = 582 as of 2026-05-29) with title (name), slug, and the
  find-a-homeopath permalink (the stable source_url).
- The REST ``content.rendered`` is only the bio narrative, so contact/location
  come from each listing's detail page, which carries clean LABELED wpbdp
  fields: ``Phone:``, ``Website:``, ``Address:`` (Street, City, ST), and
  ``ZIP Code:``. The ZIP gives precise geocoding via the global sweep.

**Terms check (the reason CHC, not its peers, is ingested):** CHC's site has NO
Terms of Use, Privacy Policy, or anti-scraping / anti-commercial-use language —
verified 2026-05-29. Its homeopathy-peer NASH, plus AHNA / BPS / NAMA, all
carry explicit anti-harvest/commercial-use terms and were dropped; CHC is the
lone clean source. See memory [[practitioner-panel]].

CHC publishes no practitioner email. Output rows: tier='org_member',
source_org='CHC', specialties=['homeopathy','holistic_health'],
credentials='CCH', fellowship_level=False (CHC has no fellow tier).
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15"
)
BASE = "https://homeopathcertification.org"
LISTINGS_PATH = "/find-a-homeopath/"
REST_INDEX = "/wp-json/wp/v2/wpbdp_listing"

LOCKED_SPECIALTIES = ["homeopathy", "holistic_health"]
CREDENTIAL = "CCH"  # Certified Classical Homeopath — the cert CHC issues.

# Canadian province codes — when the trailing 2-letter token is one of these the
# row is Canadian (country 'CA'); otherwise treated as a US state ('US').
CA_PROVINCES = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT",
}

REST_PER_PAGE = 100
REST_MAX_PAGES = 50  # 582 listings / 100 = 6 pages; bound is defensive.


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# A labeled detail field: <span class="field-label">Phone:</span>
#                         <div class="value">...</div>
_DETAIL_FIELD_RE = re.compile(
    r'<span class="field-label">(.*?)</span>\s*<div class="value">(.*?)</div>',
    re.S | re.I,
)
_HREF_RE = re.compile(r'href="([^"]+)"', re.I)
_STATE_TAIL_RE = re.compile(r'^[A-Z]{2}$')
_TLD_RE = re.compile(
    r'^(https?://.+?\.(?:com|org|net|info|biz|us|ca|co|edu|health|clinic|care|'
    r'life|io|me|tv|wellness|coach))',
    re.I,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _text(fragment: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    if not fragment:
        return ""
    out = re.sub(r"<[^>]+>", " ", fragment)
    out = html_module.unescape(out)
    out = out.replace("\xa0", " ")
    return re.sub(r"\s+", " ", out).strip()


def _clean_website(url: str) -> Optional[str]:
    """Trim a website href, cutting a mashed-in second URL after the first TLD."""
    if not url:
        return None
    url = html_module.unescape(url).strip()
    m = _TLD_RE.match(url)
    if not m:
        return url or None
    head = m.group(1)
    rest = url[len(head):]
    if rest.startswith(("www.", "http")) or "www." in rest:
        return head
    return url


def _parse_location(text: str) -> dict:
    """Parse 'Street, City, ST' / 'City, ST' into address1/city/state/country.

    Right-anchored off the trailing 2-letter state/province token. CA provinces
    set country 'CA', everything else 'US'. Returns all-None when there is no
    trailing 2-letter token.
    """
    out = {"address1": None, "city": None, "state": None, "country": None}
    if not text:
        return out
    # Drop a trailing ZIP if the address carries one (the ZIP also has its own
    # field): "..., PA 19438" -> strip the "19438".
    text = re.sub(r",?\s*\d{5}(?:-\d{4})?\s*$", "", text.strip())
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return out
    tail = parts[-1]
    if not _STATE_TAIL_RE.match(tail):
        return out
    out["state"] = tail
    out["country"] = "CA" if tail in CA_PROVINCES else "US"
    out["city"] = parts[-2] or None
    street = ", ".join(parts[:-2]).strip()
    out["address1"] = street or None
    return out


# ---------------------------------------------------------------------------
# REST index parser (PURE)
# ---------------------------------------------------------------------------

def parse_index_json(items: list) -> list[dict]:
    """REST ``wpbdp_listing`` items -> [{name, slug, url}] (one per listing)."""
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        name = _text((it.get("title") or {}).get("rendered", ""))
        slug = it.get("slug")
        url = it.get("link")
        if not name or not url:
            continue
        out.append({"name": name, "slug": slug, "url": url})
    return out


# ---------------------------------------------------------------------------
# Detail-page parser (PURE)
# ---------------------------------------------------------------------------

def _detail_fields(html: str) -> dict:
    """Map lowercased field label (no trailing colon) -> raw value HTML."""
    fields: dict = {}
    for m in _DETAIL_FIELD_RE.finditer(html or ""):
        label = _text(m.group(1)).rstrip(":").strip().lower()
        if label and label not in fields:
            fields[label] = m.group(2)
    return fields


def parse_detail_html(
    html: str, name: str, source_url: str,
) -> Optional[NormalizedPractitionerRow]:
    """Build a row from a listing detail page + its REST name/permalink.

    Reads the labeled wpbdp fields: Phone, Website, Address (Street, City, ST),
    ZIP Code. Missing fields are left None (e.g. "Profile Not Public" listings).
    """
    if not name or not source_url:
        return None
    fields = _detail_fields(html or "")

    phone = _text(fields.get("phone", "")) or None

    website = None
    web_m = _HREF_RE.search(fields.get("website", ""))
    if web_m:
        website = _clean_website(web_m.group(1))

    loc = _parse_location(_text(fields.get("address", "")))
    postal = _text(fields.get("zip code", "")) or None

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="CHC",
        source_url=source_url,
        fellowship_level=False,
        practice_name=None,
        credentials=CREDENTIAL,
        phone=phone,
        email=None,  # CHC publishes no practitioner emails.
        website=website,
        address1=loc["address1"],
        city=loc["city"],
        state=loc["state"],
        postal=postal,
        country=loc["country"] or "US",
    )


# ---------------------------------------------------------------------------
# Live fetch (plain requests — REST index + per-detail).
# ---------------------------------------------------------------------------

def _session(session=None) -> requests.Session:
    sess = session or requests.Session()
    # requests.Session() ships a default 'python-requests/...' UA the host 406s.
    sess.headers["User-Agent"] = USER_AGENT
    sess.headers["Accept"] = (
        "text/html,application/xhtml+xml,application/json,*/*;q=0.8"
    )
    return sess


def fetch_listing_index(session=None, per_page: int = REST_PER_PAGE) -> list[dict]:
    """Paginate the WP REST API for the complete deterministic listing index."""
    sess = _session(session)
    out: list[dict] = []
    for page in range(1, REST_MAX_PAGES + 1):
        url = (f"{BASE}{REST_INDEX}?per_page={per_page}&page={page}"
               f"&_fields=slug,link,title&orderby=title&order=asc")
        resp = sess.get(url, timeout=30)
        if resp.status_code == 400:  # page past the end -> rest_post_invalid_page
            break
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        out.extend(parse_index_json(items))
        total_pages = int(resp.headers.get("X-WP-TotalPages", 0) or 0)
        if total_pages and page >= total_pages:
            break
    return out


def fetch_all_directory_rows(
    session=None, sleep_s: float = 0.2,
) -> list[NormalizedPractitionerRow]:
    """Full deterministic scrape: REST index, then parse each detail page.

    Dedups by source_url. Per-detail fetch failures are skipped (the index is
    authoritative; a single 5xx shouldn't abort the whole run).
    """
    sess = _session(session)
    index = fetch_listing_index(sess)
    out: list[NormalizedPractitionerRow] = []
    seen: set[str] = set()
    for entry in index:
        url = entry["url"]
        if url in seen:
            continue
        seen.add(url)
        try:
            html = sess.get(url, timeout=30).text
        except requests.RequestException:
            continue
        row = parse_detail_html(html, entry["name"], url)
        if row is not None:
            out.append(row)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out
