"""Institute for Functional Medicine (IFM) practitioner-directory scraper.

IFM is the canonical cross-disciplinary functional-medicine body. Its public
"Find a Practitioner" directory lives at:

    https://www.ifm.org/find-a-practitioner/

Discovery (2026-05-29)
----------------------
The landing page hosts a single ``GET`` search form
(``#practitioner-landing-search-form``) whose only meaningful input is a
Google-Places-autocomplete ``location`` field plus a hidden ``radius``
(default 40). There is no public JSON / AJAX listings endpoint: the site is
a Drupal "view" that server-renders result cards. Submitting the form
navigates to a path-encoded results URL of the shape:

    /practitioner-listings/<lat,lng<=Rkm>/<lat,lng<=Rkm>/<country>/<count>
        ?location=<text>&geocode=<lat,lng>&radius=<R>&visit_type=All

So the live driver must (a) warm the browser context on the landing page,
(b) geocode a location (or reuse Google's autocomplete) to get lat/lng,
then (c) GET the ``/practitioner-listings/...`` URL with ``&page=N`` for
each Drupal pager page (10 cards per page). This module's parser is PURE
and operates on whatever rendered results HTML the driver captures.

Cloudflare
----------
The site sits behind Cloudflare but a HEADLESS Playwright session
(``playwright_session(headless=True)``) loads the directory without hitting
a challenge interstitial (verified 2026-05-29). A plain ``requests`` GET
returns 403. Headless works — no need to escalate to the visible-window
NCCAOM pattern.

Result-card structure
---------------------
Each result is a ``<li class="practitioner-card">``:

  - ``<button id="practitioner-id-<N>" ...>`` — stable numeric practitioner
    id present on EVERY card (the dedup anchor when the profile slug is
    absent).
  - ``<h3 class="name">First Last, MD, IFMCP</h3>`` — name + comma-separated
    credential post-nominals.
  - ``<div class="ifm-certified-label"><img class="ibfmc_image"
    src=".../IFM-FMCP-M-Button.png"></div>`` — the IFM Functional Medicine
    Certified Practitioner badge. Two image variants are in use
    (``IFM-FMCP-Button.png`` and ``IFM-FMCP-M-Button.png``); BOTH denote
    IFMCP/FMCP certification. The badge is the sole reliable certification
    signal (the credential string sometimes spells it ``IFMCP`` or
    ``FMCP-M`` but often omits it entirely).
  - ``<a href="mailto:...">`` — practitioner email (in the card-bottom).
  - ``<a href="https://...">`` — practice website (the first external,
    non-mailto, non-ifm.org anchor in the card-bottom left column;
    occasionally pollution like "siteA and siteB" appears — trimmed).
  - ``Offers Telehealth`` text — informational; not stored.
  - ``<a href="/practitioners/<slug>?distance_primary=...">See full
    details</a>`` — canonical profile permalink (slug). Present on most but
    NOT all cards.

The list cards do NOT carry phone / street / city / state / postal. Those
fields live ONLY on the per-practitioner profile page
(``/practitioners/<slug>``). A full enrichment pass would fetch each
profile; this adapter leaves those fields None on the list-derived rows and
documents the per-profile fetch as future work (see the migrate runner I is
NOT building here — central integration owns that).

Directory is NOT IFMCP-only
---------------------------
The Drupal view exposes a "Functional Medicine Certified" sort option and,
across metros, mixes certified and non-certified members. Verified samples
(2026-05-29): NYC 40km = all certified; LA 40km = ~1/3 non-certified;
Chicago 40km = ~1/2 non-certified. So the fellowship flag is real and
discriminating.

FELLOWSHIP RULE
---------------
``fellowship_level=True`` iff the card carries the ``ifm-certified-label``
badge (the IFM-FMCP button image). Non-badged cards -> False. This is a
structural signal, independent of how the credential string is spelled.

Output rows have tier='org_member', source_org='IFM',
specialties=['functional_medicine', 'holistic_health']. The per-practitioner
``source_url`` is the absolute profile permalink
``https://www.ifm.org/practitioners/<slug>`` when present, else an
id-anchored fallback ``https://www.ifm.org/find-a-practitioner/#practitioner-<id>``
(stable + unique across re-runs since the numeric id is IFM-assigned).
"""
import html as html_module
import re
from typing import Optional

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://www.ifm.org"
FIND_URL = f"{BASE}/find-a-practitioner/"

LOCKED_SPECIALTIES = ["functional_medicine", "holistic_health"]

# Drupal pager page size (result cards per page). A page with fewer than this
# many cards is the last page; an empty page means we've walked off the end.
PAGE_SIZE = 10

# Default radius for the centroid fan-out. Larger than IFM's 40km form default
# so the metro grid bridges most inter-metro gaps; the per-practitioner profile
# enrichment carries exact coordinates regardless, so radius only bounds (a) the
# number of pager pages per centroid and (b) the centroid-fallback accuracy for
# slug-less rows. ~50km keeps fallback error modest.
DEFAULT_RADIUS_KM = 50

# Per-centroid pager ceiling — a defensive bound so a pathological pager can't
# loop forever. 10 cards/page x 60 pages = 600 cards per metro, far above any
# real 50km result count.
DEFAULT_MAX_PAGES = 60

# geocode_quality is the Postgres enum practitioner_geocode_quality
# (full|city|zip|state_only) — NOT free text. Centroid-fallback rows get the
# surfacing metro's coords, which is city/metro-level precision -> 'city'.
# Profile-enriched rows get exact street coords -> 'full'.
FALLBACK_GEOCODE_QUALITY = "city"


# ---------------------------------------------------------------------------
# Regexes (module-level, compiled once)
# ---------------------------------------------------------------------------

# A single result card is <li class="practitioner-card"> ... </li>. We split
# on the opening tag and take everything up to the first closing </li>.
_CARD_SPLIT = '<li class="practitioner-card">'

# Stable numeric practitioner id on the toggle button.
_PID_RE = re.compile(r'practitioner-id-(\d+)', re.I)

# Profile permalink slug (strip any ?distance_primary=... query).
_SLUG_RE = re.compile(r'href="(/practitioners/[^"?#]+)', re.I)

# Name + credential post-nominals in the card heading.
_NAME_RE = re.compile(r'<h3\s+class="name">(.*?)</h3>', re.S | re.I)

# IFM certified badge (the FMCP button image). Either variant qualifies.
_BADGE_RE = re.compile(r'class="ibfmc_image"\s+src="[^"]*IFM-FMCP[^"]*\.png"', re.I)

# Email — first mailto in the card.
_MAILTO_RE = re.compile(r'href="mailto:([^"?]+)"', re.I)

# External website anchors (not mailto, not ifm.org internal links).
_HREF_RE = re.compile(r'href="(https?://[^"]+)"', re.I)

_EMAIL_SHAPE_RE = re.compile(
    r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$'
)

# US phone on a card, e.g. "+1 (818) 550-1113", "808-555-0199", "(212) 555.1234".
# Optional leading +1/1, area code (parenthesized or not), then 3-4 with
# space/dot/hyphen separators.
_PHONE_RE = re.compile(
    r'(\+?1[\s.\-]?)?'           # optional country code
    r'(\(?\d{3}\)?[\s.\-]?)'     # area code
    r'(\d{3}[\s.\-]?\d{4})'      # exchange + line
    r'(?!\d)'                    # not part of a longer digit run
)

# Per-practitioner marker coordinates on a profile page live in the embedded
# Google-Maps "Open this area in Google Maps" link: ...maps?ll=<lat>,<lng>&...
_PROFILE_LL_RE = re.compile(
    r'maps\.google\.[a-z.]+/maps\?ll=(-?\d+\.\d+),(-?\d+\.\d+)', re.I
)

# The visible "Address:" line, e.g.
#   Address: 10645 Riverside Dr, Toluca Lake, CA 91602, US
# We anchor on the US "<ST> <ZIP>" component and capture from the label to the
# end of that address. Country (the trailing ", US"/", USA") is optional.
_PROFILE_ADDR_RE = re.compile(
    r'Address:\s*(.+?,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?(?:,\s*[A-Za-z.]{2,})?)\b'
)
_STATE_ZIP_RE = re.compile(r'\b([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b')


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Stripped, HTML-unescaped string or None for empty/missing values."""
    if val is None:
        return None
    if not isinstance(val, str):
        val = str(val)
    s = html_module.unescape(val)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _strip_tags(s: str) -> str:
    """Drop HTML tags from a snippet and collapse whitespace."""
    if not s:
        return ""
    out = re.sub(r"<[^>]+>", " ", s)
    out = html_module.unescape(out)
    out = out.replace("\xa0", " ")
    return re.sub(r"\s+", " ", out).strip()


def _split_name_credentials(heading: str) -> tuple[str, Optional[str]]:
    """Split 'First Last, MD, IFMCP' into (name, credentials).

    Credentials are everything after the FIRST comma. Honorifics stay with
    the name. IFM headings are clean comma-delimited strings; some carry no
    credential at all (no comma -> credentials None).
    """
    s = _strip_tags(heading)
    if not s:
        return "", None
    parts = s.split(",", 1)
    name = parts[0].strip()
    creds = parts[1].strip().rstrip(",").strip() if len(parts) > 1 else None
    return name, (creds or None)


def _is_certified(card_html: str) -> bool:
    """True when the card carries the IFM Functional Medicine Certified
    Practitioner badge (the FMCP button image)."""
    return bool(_BADGE_RE.search(card_html or ""))


def _extract_email(card_html: str) -> Optional[str]:
    """First shape-valid mailto address in the card."""
    for m in _MAILTO_RE.finditer(card_html or ""):
        raw = html_module.unescape(m.group(1)).strip()
        if _EMAIL_SHAPE_RE.match(raw):
            return raw
    return None


def _extract_website(card_html: str) -> Optional[str]:
    """First external practice-website anchor in the card.

    Skips ifm.org internal links and the 'Schedule a Visit' booking button
    only when no other site exists (booking links are still a usable
    website, but we prefer the practice site). The IFM cards occasionally
    pollute the field with 'siteA and siteB' free text — we truncate at the
    first whitespace-delimited ' and ' / ' or ' connector and at any
    trailing space.
    """
    candidates: list[str] = []
    for m in _HREF_RE.finditer(card_html or ""):
        url = m.group(1).strip()
        low = url.lower()
        if "ifm.org" in low:
            continue
        candidates.append(url)
    if not candidates:
        return None
    site = candidates[0]
    # Trim free-text pollution: the href value sometimes contains
    # "https://a.com and Lifespanmedicine.com" — cut at the first space.
    site = re.split(r"\s+", site, maxsplit=1)[0].strip()
    site = site.rstrip(".,;")
    return site or None


def _extract_phone(card_html: str) -> Optional[str]:
    """First US phone number in the card text, returned verbatim (trimmed).

    Cards render the practitioner's phone as visible text (e.g.
    ``+1 (818) 550-1113``). We preserve the original formatting — downstream
    normalization is the portal's job, and keeping the human format matches the
    other adapters' light-touch contact handling.
    """
    if not card_html:
        return None
    m = _PHONE_RE.search(card_html)
    if not m:
        return None
    return m.group(0).strip()


def _build_source_url(slug: Optional[str], pid: Optional[str]) -> Optional[str]:
    """Stable, unique per-practitioner URL.

    Prefer the canonical profile permalink. Fall back to an id-anchored
    find-a-practitioner URL (the numeric id is IFM-assigned and stable).
    Returns None only when both are missing (defensive — never seen live).
    """
    if slug:
        return f"{BASE}{slug}"
    if pid:
        return f"{FIND_URL}#practitioner-{pid}"
    return None


# ---------------------------------------------------------------------------
# Card -> row
# ---------------------------------------------------------------------------

def _card_to_row(card_html: str) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: one practitioner-card <li> body -> row.

    Returns None when no usable name is present. List cards carry name,
    credentials, certification badge, email, and website; phone / street /
    city / state / postal are profile-only and left None here.
    """
    name_m = _NAME_RE.search(card_html)
    if not name_m:
        return None
    name, credentials = _split_name_credentials(name_m.group(1))
    if not name:
        return None

    pid_m = _PID_RE.search(card_html)
    pid = pid_m.group(1) if pid_m else None
    slug_m = _SLUG_RE.search(card_html)
    slug = slug_m.group(1) if slug_m else None
    source_url = _build_source_url(slug, pid)
    if not source_url:
        return None

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="IFM",
        source_url=source_url,
        fellowship_level=_is_certified(card_html),
        practice_name=None,
        credentials=credentials,
        phone=_extract_phone(card_html),
        email=_extract_email(card_html),
        website=_extract_website(card_html),
        address1=None,
        city=None,
        state=None,
        postal=None,
        country="US",
    )


# ---------------------------------------------------------------------------
# Public parser (PURE — no I/O)
# ---------------------------------------------------------------------------

def parse_listings_html(html: str) -> list[NormalizedPractitionerRow]:
    """Parse a rendered ``/practitioner-listings/...`` results page into rows.

    Iterates every ``<li class="practitioner-card">`` block. Deduplicates by
    ``source_url`` within the page (the Drupal pager occasionally re-emits a
    card at a page boundary). No network I/O — the live driver feeds this
    whatever HTML the Playwright session captured.
    """
    if not isinstance(html, str):
        return []
    rows: list[NormalizedPractitionerRow] = []
    seen: set[str] = set()
    for chunk in html.split(_CARD_SPLIT)[1:]:
        card_html = chunk.split("</li>", 1)[0]
        row = _card_to_row(card_html)
        if row is None:
            continue
        if row.source_url in seen:
            continue
        seen.add(row.source_url)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Live fetch (Playwright-backed). Headless works; no Cloudflare escalation.
# ---------------------------------------------------------------------------

def _listings_url(lat: float, lng: float, radius_km: int = 40, page: int = 0) -> str:
    """Build the path-encoded results URL for a geocoded location + pager page.

    Mirrors the live form-submission URL shape. ``geocode`` query param is
    what the Drupal view actually reads; the path segments are display
    breadcrumbs. ``visit_type=All`` includes both in-person and telehealth.
    """
    geo = f"{lat},{lng}"
    seg = f"{geo}<={radius_km}km"
    from urllib.parse import quote
    seg_e = quote(seg, safe="")
    qs = (
        f"geocode={quote(geo, safe='')}"
        f"&radius={radius_km}&visit_type=All&page={page}"
    )
    return f"{FIND_URL[:-1].replace('/find-a-practitioner', '')}/practitioner-listings/{seg_e}/{seg_e}/no-country/0?{qs}"


def fetch_listings_html(
    lat: float,
    lng: float,
    radius_km: int = 40,
    page: int = 0,
    fetcher=None,
) -> str:
    """Fetch one results page via the Playwright shim (headless).

    Warms the context on the landing page on first use when opening a
    one-shot session. ``fetcher`` lets the central runner pass a long-lived
    session so the cf_clearance cookie persists across the whole walk.
    Headless is sufficient — IFM does not gate with a Turnstile interstitial.
    """
    url = _listings_url(lat, lng, radius_km=radius_km, page=page)
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector="ul.practitioner-list, .practitioner-list__no-results", sleep_s=2.0)
    # Import lazily so the pure parser doesn't drag Playwright in at import.
    from scrapers.practitioner_finder.playwright_fetch import playwright_session
    with playwright_session(headless=True) as f:
        f.get(FIND_URL, sleep_s=1.0)
        return f.get(url, wait_for_selector="ul.practitioner-list, .practitioner-list__no-results", sleep_s=2.0)


# ---------------------------------------------------------------------------
# Per-profile enrichment parser (PURE — no I/O)
#
# List cards carry no street/city/state/postal/coordinates. The finder searches
# purely by lat/lng radius, so rows need coordinates or they are invisible. The
# per-practitioner profile page (/practitioners/<slug>) carries BOTH the exact
# marker coordinates (in the embedded Google-Maps "open in Google Maps" link)
# and a visible "Address:" line. This parser pulls both from rendered profile
# HTML. Slug-less (id-anchored) practitioners have no profile page and fall back
# to the surfacing centroid's coordinates (see scrape_grid / enrich_rows).
# ---------------------------------------------------------------------------

def _split_address(addr: str) -> dict:
    """Split 'street, city, ST ZIP[, country]' right-anchored off the ST+ZIP.

    Right-anchoring survives suites/commas in the street ('123 Main St, Suite
    400, Denver, CO 80202, US') and a missing trailing country
    ('456 Oak Ave, Austin, TX 78701' -> country defaults US).
    """
    out = {"address1": None, "city": None, "state": None,
           "postal": None, "country": None}
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    sz_idx = None
    for i, p in enumerate(parts):
        m = _STATE_ZIP_RE.search(p)
        if m:
            sz_idx = i
            out["state"] = m.group(1)
            out["postal"] = m.group(2)
            break
    if sz_idx is None:
        return out
    if sz_idx - 1 >= 0:
        out["city"] = parts[sz_idx - 1] or None
    street = ", ".join(parts[: max(sz_idx - 1, 0)]).strip()
    out["address1"] = street or None
    country = parts[sz_idx + 1] if sz_idx + 1 < len(parts) else "US"
    if country and country.upper().replace(".", "") in ("US", "USA"):
        country = "US"
    out["country"] = country or "US"
    return out


def parse_profile_html(html: str) -> dict:
    """Extract lat/lng + postal address from a rendered profile page.

    Returns a dict with keys lat, lng, address1, city, state, postal, country
    (any of which may be None). No network I/O — the driver feeds it whatever
    the Playwright session rendered.
    """
    out = {"lat": None, "lng": None, "address1": None, "city": None,
           "state": None, "postal": None, "country": None}
    if not isinstance(html, str) or not html:
        return out
    m = _PROFILE_LL_RE.search(html_module.unescape(html))
    if m:
        try:
            out["lat"] = float(m.group(1))
            out["lng"] = float(m.group(2))
        except ValueError:
            pass
    text = _strip_tags(html)
    am = _PROFILE_ADDR_RE.search(text)
    if am:
        out.update(_split_address(am.group(1).strip()))
    return out


# ---------------------------------------------------------------------------
# US metro/ZIP centroid grid for the geo-radius fan-out.
#
# IFM's directory is geo-radius-search-only (no list-all endpoint). We walk a
# grid of major US metro centroids at DEFAULT_RADIUS_KM and dedup by source_url.
# Functional-medicine practitioners concentrate in metros, so a metro grid
# captures the vast majority; sparse rural gaps are accepted (and logged by the
# runner, not silently dropped). Coords are 2-decimal centroids; the profile
# enrichment supplies exact coordinates for slug-bearing practitioners, so this
# grid only needs to ensure discovery coverage + fallback positioning.
# ---------------------------------------------------------------------------

US_METRO_CENTROIDS: list[tuple[str, float, float]] = [
    # Northeast / Mid-Atlantic
    ("New York NY", 40.71, -74.01), ("Boston MA", 42.36, -71.06),
    ("Philadelphia PA", 39.95, -75.16), ("Pittsburgh PA", 40.44, -79.99),
    ("Buffalo NY", 42.89, -78.88), ("Providence RI", 41.82, -71.41),
    ("Hartford CT", 41.76, -72.69), ("Albany NY", 42.65, -73.75),
    ("Washington DC", 38.90, -77.04), ("Baltimore MD", 39.29, -76.61),
    ("Richmond VA", 37.54, -77.44), ("Virginia Beach VA", 36.85, -76.29),
    ("Portland ME", 43.66, -70.26),
    # Southeast
    ("Charlotte NC", 35.23, -80.84), ("Raleigh NC", 35.78, -78.64),
    ("Asheville NC", 35.60, -82.55), ("Atlanta GA", 33.75, -84.39),
    ("Savannah GA", 32.08, -81.09), ("Jacksonville FL", 30.33, -81.66),
    ("Orlando FL", 28.54, -81.38), ("Tampa FL", 27.95, -82.46),
    ("Miami FL", 25.76, -80.19), ("Fort Myers FL", 26.64, -81.87),
    ("Nashville TN", 36.16, -86.78), ("Memphis TN", 35.15, -90.05),
    ("Knoxville TN", 35.96, -83.92), ("Louisville KY", 38.25, -85.76),
    ("Lexington KY", 38.04, -84.50), ("Birmingham AL", 33.52, -86.80),
    ("New Orleans LA", 29.95, -90.07), ("Baton Rouge LA", 30.45, -91.19),
    ("Columbia SC", 34.00, -81.03), ("Charleston SC", 32.78, -79.93),
    ("Greenville SC", 34.85, -82.39), ("Jackson MS", 32.30, -90.18),
    # Midwest
    ("Chicago IL", 41.88, -87.63), ("Detroit MI", 42.33, -83.05),
    ("Grand Rapids MI", 42.96, -85.67), ("Indianapolis IN", 39.77, -86.16),
    ("Columbus OH", 39.96, -82.99), ("Cleveland OH", 41.50, -81.69),
    ("Cincinnati OH", 39.10, -84.51), ("Milwaukee WI", 43.04, -87.91),
    ("Madison WI", 43.07, -89.40), ("Minneapolis MN", 44.98, -93.27),
    ("St. Louis MO", 38.63, -90.20), ("Kansas City MO", 39.10, -94.58),
    ("Omaha NE", 41.26, -95.93), ("Des Moines IA", 41.59, -93.62),
    ("Wichita KS", 37.69, -97.34), ("Fargo ND", 46.88, -96.79),
    ("Sioux Falls SD", 43.55, -96.73),
    # South Central
    ("Dallas TX", 32.78, -96.80), ("Fort Worth TX", 32.76, -97.33),
    ("Houston TX", 29.76, -95.37), ("San Antonio TX", 29.42, -98.49),
    ("Austin TX", 30.27, -97.74), ("El Paso TX", 31.76, -106.49),
    ("Oklahoma City OK", 35.47, -97.52), ("Tulsa OK", 36.15, -95.99),
    ("Little Rock AR", 34.75, -92.29),
    # Mountain / Southwest
    ("Denver CO", 39.74, -104.99), ("Colorado Springs CO", 38.83, -104.82),
    ("Salt Lake City UT", 40.76, -111.89), ("Boise ID", 43.62, -116.21),
    ("Albuquerque NM", 35.08, -106.65), ("Santa Fe NM", 35.69, -105.94),
    ("Phoenix AZ", 33.45, -112.07), ("Tucson AZ", 32.22, -110.97),
    ("Las Vegas NV", 36.17, -115.14), ("Reno NV", 39.53, -119.81),
    ("Billings MT", 45.78, -108.50), ("Cheyenne WY", 41.14, -104.82),
    # West Coast / Pacific
    ("Los Angeles CA", 34.05, -118.24), ("San Diego CA", 32.72, -117.16),
    ("San Francisco CA", 37.77, -122.42), ("San Jose CA", 37.34, -121.89),
    ("Sacramento CA", 38.58, -121.49), ("Fresno CA", 36.74, -119.79),
    ("Santa Barbara CA", 34.42, -119.70), ("Portland OR", 45.52, -122.68),
    ("Eugene OR", 44.05, -123.09), ("Seattle WA", 47.61, -122.33),
    ("Spokane WA", 47.66, -117.43), ("Honolulu HI", 21.31, -157.86),
    ("Anchorage AK", 61.22, -149.90),
]


# ---------------------------------------------------------------------------
# Fan-out runner (fetcher-injected so it's unit-testable).
# ---------------------------------------------------------------------------

def iter_centroid_pages(
    fetcher,
    lat: float,
    lng: float,
    radius_km: int = DEFAULT_RADIUS_KM,
    max_pages: int = DEFAULT_MAX_PAGES,
):
    """Yield fresh-to-this-centroid row batches, paging until exhausted.

    Stops on: an empty page (walked off the end), a partial page
    (<PAGE_SIZE cards -> last page), or a page that introduces no new
    source_url for this centroid (stuck-pager guard). Each yield is the list of
    rows new to THIS centroid; cross-centroid dedup is the caller's job.
    """
    centroid_seen: set = set()
    for page in range(max_pages):
        html = fetch_listings_html(
            lat, lng, radius_km=radius_km, page=page, fetcher=fetcher
        )
        rows = parse_listings_html(html)
        if not rows:
            return
        fresh = [r for r in rows if r.source_url not in centroid_seen]
        if not fresh:
            return
        for r in fresh:
            centroid_seen.add(r.source_url)
        yield fresh
        if len(rows) < PAGE_SIZE:
            return


def scrape_grid(
    fetcher,
    centroids: Optional[list] = None,
    radius_km: int = DEFAULT_RADIUS_KM,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[NormalizedPractitionerRow]:
    """Walk the centroid grid, global-dedup by source_url, return rows.

    On first sighting a row is tagged with the surfacing centroid's coordinates
    as a provisional geocode. The geocode_quality column is the Postgres enum
    practitioner_geocode_quality (full|city|zip|state_only) — metro-centroid
    precision maps to FALLBACK_GEOCODE_QUALITY='city' (city/metro level).
    enrich_rows() later overrides slug-bearing rows with their exact profile
    coordinates ('full'); slug-less rows keep this fallback.
    """
    if centroids is None:
        centroids = US_METRO_CENTROIDS
    seen: set = set()
    out: list[NormalizedPractitionerRow] = []
    for label, clat, clng in centroids:
        for fresh in iter_centroid_pages(
            fetcher, clat, clng, radius_km=radius_km, max_pages=max_pages
        ):
            for r in fresh:
                if r.source_url in seen:
                    continue
                seen.add(r.source_url)
                r.lat = clat
                r.lng = clng
                r.geocode_quality = FALLBACK_GEOCODE_QUALITY
                out.append(r)
    return out


def enrich_rows(fetcher, rows: list, profile_sleep_s: float = 1.5) -> int:
    """For slug-bearing rows, fetch the profile + set exact coords/address.

    Rows whose source_url is a /practitioners/<slug> permalink get their
    coordinates + postal address replaced from the profile page
    (geocode_quality='full'). Id-anchored (slug-less) rows have no profile page
    and keep the centroid fallback from scrape_grid. Returns the number of rows
    enriched. Per-profile failures are swallowed (the row keeps its fallback).
    """
    prefix = f"{BASE}/practitioners/"
    enriched = 0
    for r in rows:
        url = r.source_url or ""
        if not url.startswith(prefix):
            continue
        try:
            html = fetcher.get(
                url, wait_for_selector="body", sleep_s=profile_sleep_s
            )
        except Exception:  # noqa: BLE001 - best-effort; keep the fallback
            continue
        d = parse_profile_html(html)
        if d.get("lat") is not None and d.get("lng") is not None:
            r.lat = d["lat"]
            r.lng = d["lng"]
            r.geocode_quality = "full"
        for k in ("address1", "city", "state", "postal", "country"):
            if d.get(k):
                setattr(r, k, d[k])
        enriched += 1
    return enriched
