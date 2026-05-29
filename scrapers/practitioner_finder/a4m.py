"""American Academy of Anti-Aging Medicine (A4M) "Find a Provider" scraper.

A4M publishes its provider directory at https://www.a4m.com/find-a-doctor.html.
The page itself is a heavy Google-Maps front end behind Cloudflare; the
actual data is served by three lightweight JSON endpoints that the page's
``listing-lookup`` JS module (``/view/core/js/directory/listing-lookup/``)
calls via ``fetch`` (see ``shared/Requests.js``):

    GET /listing-search/coordinates?lat=<lat>&lng=<lng>&radius=<miles>[&<form fields>]
        -> {"error": false, "message": [{"id": "<listingId>", "latitude": ..,
            "longitude": .., "sortIndex": .., "sortDistance": .., "sortAlpha": ..}, ...]}
        A radius search around a coordinate. Returns ONLY listing ids +
        sort metadata (no detail). The default unfiltered search the page
        runs on first load centers on the geographic center of the
        contiguous US (lat=38.341656, lng=-96.69342) — see
        ``shared/FirstSearchRunner.js`` / ``shared/SearchRunner.js``. A
        radius of 3000 miles from that center blankets the entire US (plus
        the small international fringe) in a SINGLE request, so we do not
        need a per-state walk.

    GET /listing-search/listings?id[]=<id>&id[]=<id>&...
        -> {"error": false, "message": [<full listing record>, ...]}
        Bulk detail fetch for a set of listing ids. This is where every
        field we care about lives.

    GET /listing-search/keyword-coordinates?keyword=..&lat=..&lng=..&radius=..
        (keyword search variant — unused here.)

So the scrape is two-stage: (1) one coordinates call to collect every
listing id nationwide, (2) batched listings calls to hydrate the records.
The migrate runner (not built here, out of scope) drives both stages
through the Playwright shim; this module owns the URL builders, the live
fetch helpers, and the PURE parser that turns a ``/listing-search/listings``
response into ``NormalizedPractitionerRow`` objects.

Cloudflare (IMPORTANT — verified 2026-05-29)
--------------------------------------------
``www.a4m.com`` sits behind Cloudflare. Two findings from live testing:

  1. HEADLESS Playwright is BLOCKED. ``playwright_session(headless=True)``
     warms the ``find-a-doctor.html`` page but the JSON endpoints come
     back as the ``"Just a moment..."`` JS-challenge interstitial — the
     headless fingerprint never earns a usable ``cf_clearance`` for the
     XHR origin. We therefore default the live helpers to
     ``headless=False`` (a visible Chromium window, same escalation the
     NCBAHM/Turnstile runner uses). Headed mode warms the cookie and the
     subsequent endpoint calls return real JSON.

  2. The JSON endpoints must be hit as an IN-PAGE ``fetch`` (via
     ``page.evaluate``), NOT a top-level ``page.goto``. The endpoints are
     XHR routes the directory JS calls with same-origin credentials; a
     bare navigation to them re-triggers the Cloudflare challenge, but an
     in-page ``fetch`` after warming carries the ``cf_clearance`` cookie +
     the right ``X-Requested-With`` / referer context and sails through.
     ``_endpoint_fetch`` implements this against the shared shim's
     ``_page`` (we do not modify the shim).

A cold ``requests`` GET sometimes works (when CF's challenge is passive)
and sometimes returns 403 / a transient 520 origin error — it is NOT
reliable, hence the Playwright path. Field-fill from a live headed run
(40-id sample): name/credentials/city/state/zip 100%, phone ~97%,
address ~89%, practice ~74%, website ~45%, EMAIL 0% (A4M never exposes it).

Listing types
-------------
The directory mixes PROVIDERS (physicians) and PRODUCTS/SERVICES (labs,
compounding pharmacies, supplement companies) in one dataset. Each record
carries a ``type`` hash:

    fd6334c22b25d2dbd8b4f0ab910395fd  -> provider (doctor)   <-- KEEP
    ac38be7f7ff5bc932826db127e1b7f23  -> product/service     <-- DROP

(The provider hash matches the first ``listing_type[]`` hidden input on
the search form.) We keep ONLY the provider type — product/service rows
have no practitioner identity and would pollute the directory.

Record shape (provider)
-----------------------
    listingId   -> opaque stable id (dedup key; also embedded in ``url``)
    fname/lname -> given / family name
    degrees     -> comma-separated post-nominals, e.g. "MD, FAARFM, ABAARM"
    company     -> practice / clinic name
    phone       -> formatted phone ("(800) 590-7459")
    address1/2  -> street lines
    city/state/zip/country
    website     -> a4m.com redirect path (NOT the real site) — we keep the
                   human-readable ``websiteLabel`` as the website instead,
                   normalized to an https URL.
    url         -> canonical a4m.com detail page — our stable source_url
    email       -> always null in the live data (A4M never exposes it)
    properties  -> list of {id, name, ...} tags: membership / certification
                   / fellowship / specialty markers. The FELLOWSHIP signal
                   lives BOTH here (e.g. {"name": "FAARFM Fellowship"}) and
                   in ``degrees`` — see the fellowship rule below.

Fellowship rule (LOCKED)
------------------------
A4M's credential ladder, surfaced in the search form's certification
filter, is:

    A4M Membership            (plain membership)
    ABAARM Certification      (American Board of Anti-Aging & Regenerative
                               Medicine — BOARD CERTIFICATION, not a fellowship)
    ABAAHP Certification      (Anti-Aging Health Practitioner — board cert)
    FAARM  Fellowship         (Fellow, Anti-Aging & Regenerative Medicine)  <-- fellowship
    FAARFM Fellowship         (Fellow, Anti-Aging, Regenerative & Functional
                               Medicine)                                    <-- fellowship
    FAAMFM / FAAMM / FMNM / FLM / Stem Cell / etc. (OTHER fellowships)

``fellowship_level=True`` IFF the token ``FAARM`` or ``FAARFM`` appears, on
a word boundary, anywhere in the combined ``degrees`` string OR the
``properties[].name`` list. Everything else — ``ABAARM`` (board cert only),
``ABAAHP``, plain ``A4M Membership``, or the OTHER fellowships
(``FAAMFM``/``FAAMM``/``FMNM``/``FLM``) — is False.

Word boundaries are essential: ``FAARM`` and ``FAARFM`` must each match as
whole tokens while ``FAAMFM``/``FAAMM`` must NOT (they share a prefix).
Live data confirms providers whose ``degrees`` say ``FAAMFM`` or only
``ABAARM`` but whose ``properties`` carry ``"FAARFM Fellowship"`` — these
correctly resolve to True via the property signal.

Output: tier='org_member', source_org='A4M',
specialties=['anti_aging_regenerative', 'holistic_health'], source_url =
the canonical ``url`` detail page (stable, unique per listing). lat/lng,
photo_url, bio are left None (the shared geocoder owns lat/lng).
"""
import html as html_module
import json
import re
import time
from typing import Optional
from urllib.parse import urlencode

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://www.a4m.com"
FIND_URL = f"{BASE}/find-a-doctor.html"
COORDS_URL = f"{BASE}/listing-search/coordinates"
LISTINGS_URL = f"{BASE}/listing-search/listings"

# Unfiltered nationwide search center (geographic center of the contiguous
# US) + a radius that blankets the whole country in one request. Mirrors
# the page's own default-search constants in FirstSearchRunner.js /
# SearchRunner.js (which use this same center).
US_CENTER_LAT = 38.341656
US_CENTER_LNG = -96.69342
US_WIDE_RADIUS_MILES = 3000

# Provider (doctor) listing type. The directory also returns product /
# service listings under a different type hash — we keep only providers.
PROVIDER_TYPE = "fd6334c22b25d2dbd8b4f0ab910395fd"

LOCKED_SPECIALTIES = ["anti_aging_regenerative", "holistic_health"]

# Fellowship marker: the bare token FAARM or FAARFM, on word boundaries so
# we don't false-match the sibling fellowships FAAMFM / FAAMM. Case-
# insensitive defensively (the live data is upper-case).
_FELLOWSHIP_RE = re.compile(r"\b(?:FAARM|FAARFM)\b", re.IGNORECASE)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": FIND_URL,
        }
    )
    return s


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def build_coordinates_url(
    lat: float = US_CENTER_LAT,
    lng: float = US_CENTER_LNG,
    radius: int = US_WIDE_RADIUS_MILES,
) -> str:
    """URL for the listing-id search around a coordinate. Defaults to the
    nationwide unfiltered walk (US center + 3000mi)."""
    qs = urlencode({"lat": lat, "lng": lng, "radius": radius})
    return f"{COORDS_URL}?{qs}"


def build_listings_url(ids: list[str]) -> str:
    """URL for the bulk detail fetch. Repeats the ``id[]`` param once per
    listing id (the exact shape Requests.js#getListings builds via
    FormData ``id[]`` appends)."""
    pairs = [("id[]", str(i)) for i in ids]
    return f"{LISTINGS_URL}?{urlencode(pairs)}"


# ---------------------------------------------------------------------------
# Live fetch helpers (Playwright-backed — kept thin; the migrate runner
# owns orchestration). These accept an optional warmed PlaywrightFetcher;
# when None they open a one-shot session.
#
# HEADLESS_OK is False: A4M's Cloudflare gate blocks headless. The live
# helpers open a HEADED session (headless=False) by default.
# ---------------------------------------------------------------------------

HEADLESS_OK = False


def _warm(fetcher) -> None:
    """Prime the Cloudflare cf_clearance cookie by visiting the directory
    page once. Best-effort — failures are swallowed (the heavy page can be
    slow; the cookie is granted on the challenge solve regardless)."""
    try:
        fetcher.get(FIND_URL, sleep_s=2.0)
    except Exception:  # noqa: BLE001 - best-effort warm-up
        pass


def _endpoint_fetch(fetcher, url: str) -> str:
    """Fetch a JSON endpoint as an IN-PAGE ``fetch`` from the warmed page
    context. This carries the ``cf_clearance`` cookie + same-origin XHR
    headers, which a bare ``page.goto`` does NOT (goto re-triggers the
    Cloudflare challenge on the XHR origin). Returns the response body
    text.

    Uses the shim's ``_page`` directly via ``page.evaluate``; we do not
    modify ``playwright_fetch.py``.
    """
    script = """
    async (u) => {
        const r = await fetch(u, {
            headers: {'X-Requested-With': 'XMLHttpRequest',
                      'Accept': 'application/json, text/plain, */*'},
            credentials: 'include'
        });
        return await r.text();
    }
    """
    return fetcher._page.evaluate(script, url)


def fetch_listing_ids(fetcher=None) -> list[str]:
    """Fetch every provider/product listing id nationwide (single
    coordinates call). Returns the list of id strings in the directory's
    sort order. Provider/product filtering happens later, at detail time.

    A4M's Cloudflare gate blocks headless — opens a HEADED session when
    ``fetcher`` is None.
    """
    if fetcher is None:
        from scrapers.practitioner_finder.playwright_fetch import (
            playwright_session,
        )
        with playwright_session(headless=HEADLESS_OK) as f:
            _warm(f)
            return fetch_listing_ids(fetcher=f)

    raw = _endpoint_fetch(fetcher, build_coordinates_url())
    return parse_coordinates_ids(raw)


def fetch_listings(ids: list[str], fetcher=None) -> list[dict]:
    """Fetch the full detail records for a set of listing ids (one
    ``/listing-search/listings`` call). Returns the raw record dicts."""
    if not ids:
        return []
    if fetcher is None:
        from scrapers.practitioner_finder.playwright_fetch import (
            playwright_session,
        )
        with playwright_session(headless=HEADLESS_OK) as f:
            _warm(f)
            return fetch_listings(ids, fetcher=f)

    raw = _endpoint_fetch(fetcher, build_listings_url(ids))
    return extract_listing_records(raw)


# ---------------------------------------------------------------------------
# Pure JSON extraction
# ---------------------------------------------------------------------------

def _strip_json_envelope(payload):
    """Some rendered Playwright responses wrap the JSON body in
    ``<html><body><pre>...</pre></body></html>`` (Chromium's JSON viewer)
    or surrounding HTML. Pull the JSON object out defensively, accepting:

      - a dict / list already parsed
      - a raw JSON string
      - an HTML string with the JSON embedded in a <pre> or as the body

    Returns the parsed Python object, or None on failure.
    """
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if not isinstance(payload, str):
        return None

    s = payload.strip()
    if not s:
        return None

    # Fast path: it's already JSON.
    if s[0] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    # Chromium JSON viewer / wrapped HTML: pull the first {...} object.
    # Prefer a <pre> block when present.
    m = re.search(r"<pre[^>]*>(.*?)</pre>", s, re.DOTALL | re.IGNORECASE)
    if m:
        inner = html_module.unescape(m.group(1)).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # Last resort: grab the outermost {...} that parses.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = html_module.unescape(s[start : end + 1])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def parse_coordinates_ids(payload) -> list[str]:
    """Pure: extract the list of listing ids from a ``coordinates``
    response. Returns ``[]`` defensively when the shape is unexpected."""
    data = _strip_json_envelope(payload)
    if not isinstance(data, dict):
        return []
    msg = data.get("message")
    if not isinstance(msg, list):
        return []
    ids: list[str] = []
    for item in msg:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.append(str(item["id"]))
    return ids


def extract_listing_records(payload) -> list[dict]:
    """Pure: pull the list of raw record dicts out of a ``listings``
    response envelope (``{"error": false, "message": [...]}``)."""
    data = _strip_json_envelope(payload)
    if not isinstance(data, dict):
        return []
    msg = data.get("message")
    if not isinstance(msg, list):
        return []
    return [r for r in msg if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Stripped string or None for missing/empty/whitespace values."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.replace("\xa0", " ").strip()
        return s or None
    s = str(val).strip()
    return s or None


def _is_provider(rec: dict) -> bool:
    """True when the record is a provider (doctor), not a product/service."""
    return rec.get("type") == PROVIDER_TYPE


def _build_name(rec: dict) -> Optional[str]:
    """Join fname + lname into a display name. Returns None when neither
    is present (no usable identity -> drop the row)."""
    fname = _coerce_str(rec.get("fname"))
    lname = _coerce_str(rec.get("lname"))
    parts = [p for p in (fname, lname) if p]
    if not parts:
        return None
    return " ".join(parts)


def _property_names(rec: dict) -> list[str]:
    """Extract the ``properties[].name`` strings (membership / cert /
    fellowship / specialty tags)."""
    props = rec.get("properties")
    if not isinstance(props, list):
        return []
    out: list[str] = []
    for p in props:
        if isinstance(p, dict):
            n = _coerce_str(p.get("name"))
            if n:
                out.append(n)
    return out


def _has_fellowship(rec: dict) -> bool:
    """LOCKED: True iff FAARM or FAARFM appears (word-bounded) in the
    ``degrees`` string OR any ``properties[].name``. ABAARM / ABAAHP /
    plain membership / other fellowships (FAAMFM/FAAMM/FMNM) -> False."""
    blob = " ".join(
        [_coerce_str(rec.get("degrees")) or ""] + _property_names(rec)
    )
    return bool(_FELLOWSHIP_RE.search(blob))


def _build_address1(rec: dict) -> Optional[str]:
    """Join address1 + address2 (suite/unit) into a single street line."""
    a1 = _coerce_str(rec.get("address1"))
    a2 = _coerce_str(rec.get("address2"))
    parts = [p for p in (a1, a2) if p]
    if not parts:
        return None
    return ", ".join(parts)


def _country_iso2(rec: dict) -> str:
    """Map the free-text country to ISO2. A4M is overwhelmingly US; the
    handful of others are mapped best-effort, defaulting to 'US'."""
    raw = (_coerce_str(rec.get("country")) or "").lower()
    table = {
        "united states": "US",
        "united states of america": "US",
        "usa": "US",
        "us": "US",
        "canada": "CA",
        "mexico": "MX",
        "united kingdom": "GB",
        "uk": "GB",
        "australia": "AU",
        "united arab emirates": "AE",
        "uae": "AE",
        "saudi arabia": "SA",
        "singapore": "SG",
        "india": "IN",
        "brazil": "BR",
    }
    return table.get(raw, "US")


def _normalize_website(rec: dict) -> Optional[str]:
    """Resolve the real practice website.

    The ``website`` field is an internal a4m.com redirect path
    (``/directory-url-redirect;...``) — useless as an external link. The
    human-readable destination is in ``websiteLabel`` (e.g.
    ``www.power2practice.com``); we promote that to an https URL. If the
    label is missing, fall back to None (we do NOT surface the internal
    redirect path)."""
    label = _coerce_str(rec.get("websiteLabel"))
    if not label:
        return None
    if label.startswith("http://") or label.startswith("https://"):
        return label
    if label.lower() in {"none", "n/a", "null"}:
        return None
    return f"https://{label}"


def _build_source_url(rec: dict) -> Optional[str]:
    """Stable, unique per-practitioner source_url: the canonical a4m.com
    detail page (``url``). Falls back to a listingId-anchored find-a-doctor
    URL when ``url`` is absent so the dedup layer still has a unique key."""
    url = _coerce_str(rec.get("url"))
    if url:
        return url
    lid = _coerce_str(rec.get("listingId"))
    if lid:
        return f"{FIND_URL}#listing-{lid}"
    return None


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure: one A4M listing record -> NormalizedPractitionerRow.

    Returns None for non-provider records (products/services) and for
    provider records with no usable name or no stable source_url.
    """
    if not _is_provider(rec):
        return None

    name = _build_name(rec)
    if not name:
        return None

    source_url = _build_source_url(rec)
    if not source_url:
        return None

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="A4M",
        source_url=source_url,
        fellowship_level=_has_fellowship(rec),
        practice_name=_coerce_str(rec.get("company")),
        credentials=_coerce_str(rec.get("degrees")),
        phone=_coerce_str(rec.get("phone")),
        email=_coerce_str(rec.get("email")),
        website=_normalize_website(rec),
        address1=_build_address1(rec),
        city=_coerce_str(rec.get("city")),
        state=_coerce_str(rec.get("state")),
        postal=_coerce_str(rec.get("zip")),
        country=_country_iso2(rec),
    )


def parse_listings_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser. Accepts the ``/listing-search/listings`` response in
    any of these shapes:

      - the full envelope dict ``{"error": false, "message": [...]}``
      - a bare list of record dicts
      - a JSON string of either
      - an HTML-wrapped JSON string (Chromium JSON viewer)

    Returns one NormalizedPractitionerRow per PROVIDER record; product/
    service rows and records with no usable name are silently dropped.
    """
    data = _strip_json_envelope(payload)
    if isinstance(data, dict):
        records = extract_listing_records(data)
    elif isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]
    else:
        records = []

    rows: list[NormalizedPractitionerRow] = []
    for rec in records:
        row = _record_to_row(rec)
        if row is not None:
            rows.append(row)
    return rows
