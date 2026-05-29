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
        phone=None,
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
