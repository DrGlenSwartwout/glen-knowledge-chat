"""Neuro-Optometric Rehabilitation Association (NORA) scraper.

NORA publishes its directory on MemberClicks (Joomla CMS + an Angular SPA at
``/find-a-provider`` powered by the ``ui-directory-search/v2`` API). The
public locator surface is two thin JSON endpoints:

    POST https://nora.memberclicks.net/ui-directory-search/v2/search-directory/
         body: {"form": {"directory_search_id": 12133, "elements": []}}
         -> first page of results + a search_id + a pagination data_url.

    POST https://nora.memberclicks.net/ui-directory-search/v2/search-directory-paged/
         body: {"url": "<data_url>?pageSize=10&pageNumber=N"}
         -> page N of the same search.

Discovery 2026-05-27 found 429 total practitioners across 43 pages of
pageSize=10. (Larger page sizes are rejected by the upstream search-results
service with HTTP 400; the SPA itself ships pageSize=10 hardcoded so we
match that contract.) ``directory_search_id`` is constant (=12133) and
discoverable from the public ``get-directory-search-form/find-a-provider``
endpoint.

NORA's MemberClicks account is HTTP-cookie gated — both API endpoints
return ``{"status":401}`` without a session cookie. A single GET against
the public ``/find-a-provider`` page mints the required JSESSIONID-style
cookie (`0012f0e1bd...` + `serviceID`); requests.Session() persists it
for every subsequent call without any login.

Each search result is a "display element" with this shape (after JSON
parsing, before our normalization):

    {
      "id": 1002328432,                       # stable MemberClicks profile id
      "avatar_url": "/membership/profile/.../avatar.jpg",  # ignored (member-only)
      "title": "Julie A. Steinhauer",         # the practitioner name (may
                                              # include trailing credentials,
                                              # comma- or space-delimited)
      "top": [
        {"display_order": 0, "html": "<street>\\n<line2>,\\n<city>, <state> <postal>"},
        {"display_order": 1, "html": "<profession>"},  # e.g. "Occupational Therapist", usually empty
        {"display_order": 2, "html": "<phone>"},       # OR practice name when no phone
        {"display_order": 3, "html": "<website>"}
      ],
      "bottom": [], "left": [], "right": [],   # always empty in NORA's payloads
      "distance": 0
    }

The fields in ``top`` are positional-by-``display_order``, not by index in
the list (the SPA sorts client-side), so the parser always indexes off
``display_order`` rather than list position.

Output rows have tier='org_member', source_org='NORA', and
specialties=['rehabilitation', 'eye_care'].

Fellowship detection
--------------------
NORA's credential ladder, per
https://noravisionrehab.org/healthcare-professionals/clinical-skills-fellowship :

  "Upon completion of all three levels of the program, participants will
   be presented with the designation Fellow of the Neuro-Optometric
   Rehabilitation Association (FNORA)."

NORA does NOT have a separate "Diplomate" tier — the term appears
nowhere in their public materials and matches zero records in the live
directory (vs. 7 records carrying FNORA). FNORA is therefore the
unambiguous elite tier and is what we use for ``fellowship_level=True``.

The public Find-a-Provider payload exposes credentials only inside the
``title`` field — e.g. ``"Briana Larson, OD, FNORA, FOVDR, FAAO"`` or
the space-delimited variant ``"Kauser Sharieff OD FCOVD FNORA"``. We
look for the literal token ``FNORA`` (case-insensitive, word-boundary)
anywhere in the title to set the flag.

Per-practitioner source_url
---------------------------
MemberClicks profile detail pages are member-only (HTTP 403 unauth) and
do not have a stable public URL. We synthesize a stable, deterministic
URL from the public find-a-provider page + the MemberClicks profile id:

    https://nora.memberclicks.net/find-a-provider#/profile/<id>

The fragment is invariant across re-runs (id is stable) so it works as
the upsert dedup key, even though the fragment is currently only used by
the SPA for in-page state.
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://nora.memberclicks.net"
FIND_A_PROVIDER_URL = f"{BASE}/find-a-provider"
SEARCH_FORM_URL = f"{BASE}/ui-directory-search/v2/get-directory-search-form/find-a-provider"
SEARCH_DIRECTORY_URL = f"{BASE}/ui-directory-search/v2/search-directory/"
SEARCH_PAGED_URL = f"{BASE}/ui-directory-search/v2/search-directory-paged/"
PAGE_SIZE = 10  # The upstream search-results service rejects larger sizes.

LOCKED_SPECIALTIES = ["rehabilitation", "eye_care"]

# US state name -> 2-letter abbreviation. NORA puts the full state name in
# the address (e.g. "Glen Carbon, Illinois 62034") — we leave it as-is so
# the row state matches what the user sees on NORA's site; this map is
# unused for now but kept here as documentation of the data shape.
# Country detection happens via address-block heuristic + the state map.
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}

# Canadian provinces (full names appear in NORA addresses) -> CA.
_CANADIAN_PROVINCES = {
    "alberta", "british columbia", "manitoba", "new brunswick",
    "newfoundland and labrador", "nova scotia", "ontario",
    "prince edward island", "quebec", "saskatchewan",
    "northwest territories", "nunavut", "yukon",
    # NORA also files "NSW" (Australia) as a state — we don't add it here
    # because that's an Australian-not-Canadian province; it falls through
    # to the international branch.
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: HTTP fetch
# ---------------------------------------------------------------------------

def fetch_directory_search_id(session: Optional[requests.Session] = None) -> int:
    """Hit the public form endpoint, return the ``directory_search_id``.

    Also primes the session cookie jar (required for every subsequent
    search call). 2026-05-27 the live id is 12133, but reading it from
    the form response makes the scraper self-healing if NORA ever
    re-publishes the locator under a new id.
    """
    s = session or _session()
    # Prime the JSESSIONID-style cookies.
    s.get(FIND_A_PROVIDER_URL, timeout=20)
    time.sleep(0.5)
    r = s.get(
        SEARCH_FORM_URL,
        headers={"Referer": FIND_A_PROVIDER_URL},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    sid = payload.get("directory_search_id")
    if not isinstance(sid, int):
        raise ValueError(f"unexpected search-form response: {payload!r}")
    time.sleep(0.5)
    return sid


def fetch_first_page(directory_search_id: int, session: requests.Session) -> dict:
    """POST to search-directory/ with an empty element list (= all members).

    Returns the full JSON response, including the upstream ``data_url``
    that subsequent pages re-use.
    """
    body = {
        "form": {
            "directory_search_id": directory_search_id,
            "elements": [],
        }
    }
    r = session.post(
        SEARCH_DIRECTORY_URL,
        json=body,
        headers={
            "Referer": FIND_A_PROVIDER_URL,
            "Origin": BASE,
        },
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def fetch_page(data_url: str, page_number: int, session: requests.Session) -> dict:
    """POST to search-directory-paged/ with a re-quoted data_url + page n.

    The Angular SPA sends pageNumber that's already 1-based when paging
    forward; we mirror that contract exactly (the upstream service
    expects the literal pageNumber to be looked up).
    """
    paged_url = f"{data_url}?pageSize={PAGE_SIZE}&pageNumber={page_number}"
    r = session.post(
        SEARCH_PAGED_URL,
        json={"url": paged_url},
        headers={
            "Referer": FIND_A_PROVIDER_URL,
            "Origin": BASE,
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def fetch_all_directory_records() -> list[dict]:
    """Walk every page of the NORA directory, return concatenated raw
    result dicts. Safe to call without arguments; manages its own session
    + cookies."""
    s = _session()
    sid = fetch_directory_search_id(session=s)
    first = fetch_first_page(directory_search_id=sid, session=s)
    out: list[dict] = list(first.get("results") or [])
    data_url = first.get("data_url")
    total_pages = int(first.get("total_page_count") or 0)
    if not data_url or total_pages <= 1:
        return out
    for page in range(2, total_pages + 1):
        page_resp = fetch_page(data_url=data_url, page_number=page, session=s)
        batch = page_resp.get("results") or []
        if not batch:
            break
        out.extend(batch)
    return out


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Stripped string or None for empty/null. NORA serves strings
    directly (no GeoDirectory raw/rendered wrapping), but we still
    guard for None defensively."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


# Tokens we treat as a "credential" when stripping them off a title.
# Includes the ones we've seen live (OD/MD/DO/FCOVD/FNORA/FOVDR/FAAO/
# FCSO/CBIS/CSRS/OTR/L/PT/MOT/MSOT/RD/CES/USAW) plus a generic upper-case
# acronym pattern. We do NOT depend on this list for fellowship
# detection — that uses a literal FNORA word-boundary scan.
_CREDENTIAL_TOKENS = {
    "OD", "DO", "MD", "DC", "DPT", "MSPT", "PT", "OT", "OTR", "OTR/L",
    "MOT", "MSOT", "RD", "CBIS", "CSRS", "CES", "USAW", "ABO", "ABOC",
    "FAAO", "FCOVD", "FNORA", "FOVDR", "FCSO", "FACBO", "VT", "BS",
    "MS", "PHD", "PH.D", "MA", "DR", "DR.", "ND", "MBA", "MEd",
    "BCSc", "BSC", "BCS", "BCS.", "FCSc",
}


def _looks_like_credential(chunk: str) -> bool:
    """True when a comma-separated chunk looks like a credential token
    (mostly uppercase, short, no internal spaces). Used to walk the
    title from the right and chop off the trailing credential block.

    Heuristic: a credential is 1-12 chars, no internal whitespace, made
    of letters + dots + slashes + digits, with at least one uppercase
    letter and no more than one lowercase run (e.g. ``OD``, ``Ph.D``,
    ``OTR/L``, ``PN1/2``, ``CEDH``). Anything with multiple words or
    title-cased English (e.g. ``Occupational Therapist``) is rejected.
    """
    if not chunk:
        return False
    c = chunk.strip()
    if len(c) > 12 or len(c) < 1:
        return False
    if " " in c:
        return False
    # Allow letters, digits, dot, slash, hyphen.
    if not re.match(r"^[A-Za-z0-9./\-]+$", c):
        return False
    # Must contain at least one uppercase letter (rules out lowercase
    # English fragments like "phd" that come from typos).
    if not any(ch.isupper() for ch in c):
        return False
    # Reject title-case English words like "Inc" — credentials are
    # almost all-uppercase, or letter+digits.
    letters = [ch for ch in c if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio < 0.5:
            return False
    return True


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split 'Briana Larson, OD, FNORA, ...' or 'Kauser Sharieff OD FCOVD FNORA'
    into (clean_name, credentials).

    NORA titles come in two shapes:
      * comma-delimited: ``Name, OD, FNORA, FOVDR``
      * space-delimited: ``Name OD FCOVD FNORA``

    We first try the comma form (canonical); if no comma exists but the
    trailing tokens are all in ``_CREDENTIAL_TOKENS`` we treat the run
    of trailing credential tokens as the credential block.
    """
    if not name:
        return "", None
    s = html_module.unescape(name).strip()
    # Paren form (rare on NORA but cheap to support).
    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

    # Comma form. Split on commas and walk from the right — every
    # trailing token that "looks like a credential" (mostly uppercase,
    # short, no spaces) is part of the credential block; the first
    # non-credential token from the right is the boundary.
    if "," in s:
        chunks = [c.strip() for c in s.split(",")]
        # Pop trailing credential-looking chunks.
        creds_block: list[str] = []
        while len(chunks) > 1:
            tail = chunks[-1]
            if _looks_like_credential(tail):
                creds_block.insert(0, chunks.pop())
            else:
                break
        clean = ", ".join(chunks).strip()
        if creds_block:
            return clean, ", ".join(creds_block)
        return clean, None

    # Space form: walk tokens from the right; everything that's a known
    # credential token (case-insensitive, with trailing dot/comma
    # stripped) is part of the credential block.
    tokens = s.split()
    creds_block: list[str] = []
    while tokens:
        tail = tokens[-1].rstrip(".,").upper()
        if tail in _CREDENTIAL_TOKENS:
            creds_block.insert(0, tokens.pop().rstrip(".,"))
        else:
            break
    if creds_block:
        clean = " ".join(tokens).strip()
        return clean, ", ".join(creds_block)
    return s, None


def _name_from_title(title: str) -> str:
    name, _ = _strip_credentials(title or "")
    return name


def _credentials_from_title(title: str) -> Optional[str]:
    _, creds = _strip_credentials(title or "")
    return creds


# NORA's space-credentialed records (e.g. "Kauser Sharieff OD FCOVD FNORA")
# need the literal-token scan too. We match FNORA only as a full word so
# we don't accidentally hit a substring of something else.
_FNORA_RE = re.compile(r"(?<![A-Za-z])FNORA(?![A-Za-z])", re.IGNORECASE)


def _is_fellowship(record: dict) -> bool:
    """True when the record's title carries the FNORA credential.

    NORA's "Fellow of the Neuro-Optometric Rehabilitation Association"
    is the elite tier — it requires completion of a three-level clinical
    skills program. The public Find-a-Provider payload exposes credentials
    only via the ``title`` field, so that's where we look. NORA has no
    Diplomate tier (zero records, zero mentions in their materials), so
    FNORA is the unambiguous fellowship marker.
    """
    title = _coerce_str(record.get("title")) or ""
    if not title:
        return False
    return bool(_FNORA_RE.search(title))


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"^[+()\-\d.\s]+$")

# Canadian postal code: "A1A 1A1". No anchors — we want to find it at
# the end of a "<state> <postal>" string like "Ontario N4W 1B4".
_CA_POSTAL_RE = re.compile(r"\b[A-Za-z]\d[A-Za-z]\s+\d[A-Za-z]\d\b")

# UK-style outward+inward postal: "IG8 8LL", "SW1A 1AA", "M1 1AA".
_UK_POSTAL_RE = re.compile(
    r"\b[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s+\d[A-Za-z]{2}\b"
)

# Plain US zip: "62034" or "62034-1234".
_US_ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")


def _split_state_postal(rest: str) -> tuple[Optional[str], Optional[str]]:
    """Split a `rest` chunk ('<state>? <postal>?') into (state, postal).

    Handles four regimes:
      1. Canadian postal codes — "Ontario N4W 1B4" or bare "N4W 1B4" (a
         3+3 alphanum pair with a space).
      2. UK postal codes — "IG8 8LL" or "SW1A 1AA" (outward+inward
         halves, 2-7 chars total including the space).
      3. US-style "State Zip" — last whitespace-run separates state
         from a numeric zip.
      4. Single token — entire `rest` is treated as a postal code.

    State is None for international records that omit it.
    """
    rest = rest.strip()
    if not rest:
        return None, None

    # Regime 1: Canadian postal — the canonical "A1A 1A1" shape.
    m = _CA_POSTAL_RE.search(rest)
    if m:
        postal = m.group(0)
        state = rest[: m.start()].strip() or None
        return state, postal

    # Regime 2: UK postal anywhere in `rest`.
    m = _UK_POSTAL_RE.search(rest)
    if m:
        postal = m.group(0)
        state = rest[: m.start()].strip() or None
        return state, postal

    # Regime 3/4: US "State Zip" — last token is digits-only / hyphen-digits.
    tokens = rest.rsplit(" ", 1)
    if len(tokens) == 2:
        head, tail = tokens[0].strip(), tokens[1].strip()
        if _US_ZIP_RE.match(tail):
            return (head or None), tail
        # Tail looks like a postal but not US-zip (rare); keep state=head.
        if tail and re.match(r"^[A-Za-z0-9\-]+$", tail) and head:
            # Could be Australian-style "NSW 2103" — `tail` is digits-only.
            if tail.isdigit():
                return (head or None), tail
            # Otherwise treat the whole thing as state.
            return rest, None
        return rest, None

    # Single token in `rest` — most likely a bare postal code (UK / IL / DK).
    if re.match(r"^[A-Za-z0-9\-]+$", rest):
        return None, rest
    return rest, None


def _split_address(block: Optional[str]) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[str], str
]:
    """Parse NORA's ``top[0]`` HTML chunk.

    The string shape is::

        '<line1>\\n<line2>,\\n<city>, <state> <postal>'

    Any of the parts can be empty. NORA stores international addresses
    in the same shape but with a blank state and the postal code in the
    last whitespace-separated slot (e.g. ``'London,  IG8 8LL'``).

    Returns ``(address1, city, state, postal, country_iso2)``.

    Country detection rules (deterministic, no geocoder call):
      - US state name in the state slot                       -> 'US'
      - Canadian province name in the state slot              -> 'CA'
      - state slot blank + city/postal still present          -> 'US' is
        WRONG here (defaults would over-claim US); we instead leave the
        country at 'US' default but only when nothing in the block hints
        otherwise. If the block is essentially empty
        ('\\n,\\n,  ' / '\\n,\\n,'), we still default to 'US' because the
        downstream geocoder can fill in nothing useful anyway and the
        practitioner is most likely US (the directory is US-centered).
    """
    if not block:
        return None, None, None, None, "US"
    raw = block.replace("\r\n", "\n")
    # Three logical pieces separated by newlines: line1, line2, city-line.
    parts = raw.split("\n")
    if len(parts) < 3:
        # Defensive: pad to 3.
        parts = parts + [""] * (3 - len(parts))
    line1 = parts[0].strip().rstrip(",")
    line2 = parts[1].strip().rstrip(",")
    city_line = parts[2].strip()
    # city_line shape: "City, State Postal" — comma separates city from the
    # state+postal block. International records omit the state (e.g.
    # "London,  IG8 8LL"); we tolerate the double-space.
    city: Optional[str] = None
    state: Optional[str] = None
    postal: Optional[str] = None
    if "," in city_line:
        city_part, rest = city_line.split(",", 1)
        city = city_part.strip() or None
        rest = rest.strip()
        if rest:
            state, postal = _split_state_postal(rest)
    else:
        # No comma in the city-line — treat the whole thing as city.
        city = city_line or None

    # Build address1: join non-empty line1 + line2.
    addr1_bits = [p for p in (line1, line2) if p]
    address1 = ", ".join(addr1_bits) if addr1_bits else None

    # Country detection.
    state_norm = (state or "").lower()
    country = "US"
    if state_norm in _US_STATE_NAMES:
        country = "US"
    elif state_norm in _CANADIAN_PROVINCES:
        country = "CA"
    elif state and state_norm not in _US_STATE_NAMES:
        # We have a non-US, non-Canadian state name — it's international.
        # Without a country map keyed off arbitrary region names we leave
        # country at None-by-omission (i.e. "" which Python falsy) so the
        # downstream geocoder uses just city + postal. We can't return None
        # because the dataclass field has a "US" default. The pragmatic
        # call is: if state is "NSW" / "New South Wales", that's Australia;
        # if it's a UK / Israel / Denmark address with empty state, country
        # stays US-default which the geocoder can override.
        # We err toward NOT mislabeling international rows as US by clearing
        # to None. ``None`` here causes the dataclass to_dict to drop the
        # field — but then the DB row carries the schema default 'US' which
        # is no improvement. So we surface the state name as a hint and
        # leave the geocoder responsible for the country override.
        # (This mirrors how iaomt.py treats non-US states: it keeps "US"
        # as the default and lets the geocoder reconcile.)
        country = "US"
    return address1, city, state, postal, country


def _looks_like_phone(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return bool(_PHONE_RE.match(s)) and len(digits) >= 7


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if NORA's website field is a bare domain."""
    s = _coerce_str(raw)
    if not s:
        return None
    # Some NORA records leak "Www.example.com" or " www.example.com" — fix capitalization.
    if s.lower().startswith("www."):
        s = "www." + s[4:]
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://{s}"


def _top_item(record: dict, display_order: int) -> Optional[str]:
    """Pull the ``html`` field of the top-row item at ``display_order`` —
    NORA's display elements are positional-by-display_order, not by list
    index."""
    for it in record.get("top") or []:
        if isinstance(it, dict) and it.get("display_order") == display_order:
            return _coerce_str(it.get("html"))
    return None


def _build_source_url(rec: dict) -> str:
    """Synthesize the per-practitioner URL.

    MemberClicks profile detail pages are member-only (403 unauth), so
    no public detail URL exists. We append the stable profile id to the
    Find-a-Provider URL as a fragment — fragments are invariant across
    re-runs and uniquely identify the practitioner.
    """
    rid = _coerce_str(rec.get("id"))
    if not rid:
        return f"{FIND_A_PROVIDER_URL}#/profile/unknown"
    return f"{FIND_A_PROVIDER_URL}#/profile/{rid}"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: NORA display-element dict -> NormalizedPractitionerRow.

    Returns None if no usable name can be recovered."""
    title = _coerce_str(rec.get("title"))
    if not title:
        return None
    name = _name_from_title(title)
    credentials = _credentials_from_title(title)
    if not name:
        return None

    # top[1] is "profession" — append it to credentials if present and
    # not already there. Most records have it blank; the 10 records with
    # values like "Occupational Therapist" / "Physical Therapist" use it
    # as the practitioner's role.
    profession = _top_item(rec, 1)
    if profession:
        existing = (credentials or "").lower()
        if profession.lower() not in existing:
            credentials = (
                f"{credentials}, {profession}" if credentials else profession
            )

    address_block = _top_item(rec, 0)
    address1, city, state, postal, country = _split_address(address_block)

    # top[2] is phone OR practice_name (or blank). We classify by
    # digit density — anything with 7+ digits + only digits/+/()/-/space
    # is a phone, anything else is the practice name.
    contact = _top_item(rec, 2)
    phone: Optional[str] = None
    practice_name: Optional[str] = None
    if contact:
        if _looks_like_phone(contact):
            phone = contact
        else:
            practice_name = contact

    website = _normalize_website(_top_item(rec, 3))

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="NORA",
        source_url=_build_source_url(rec),
        fellowship_level=_is_fellowship(rec),
        practice_name=practice_name,
        credentials=credentials,
        phone=phone,
        email=None,  # NORA never publishes member emails in the public locator.
        website=website,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a NORA search response (dict with a ``results``
    list, or just the results list itself, or a JSON string of either)
    and returns one NormalizedPractitionerRow per usable record. No I/O.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        import json
        payload = json.loads(payload)

    if isinstance(payload, dict):
        records = payload.get("results") or []
    elif isinstance(payload, list):
        records = payload
    else:
        return []

    rows: list[NormalizedPractitionerRow] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        row = _record_to_row(rec)
        if row is not None:
            rows.append(row)
    return rows
