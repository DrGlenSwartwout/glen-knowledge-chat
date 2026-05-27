"""American Board of Functional Medicine (ABFM) scraper.

Disambiguation note (2026-05-27 discovery)
==========================================

There are several entities whose acronym or name collides with "ABFM":

  * ``theabfm.org``  -> American Board of Family Medicine. *Wrong* board —
    family-practice specialty board. NOT what this adapter targets.
  * ``abfm.org``     -> Association for Budgeting and Financial Management
    (a public-administration org). Not medical at all.
  * Institute for Functional Medicine (IFM, ``ifm.org``) — adjacent but a
    separate org with its own practitioner directory. NOT scraped here.

The ABFM we want is the **American Board of Functional Medicine**, the
certifying body behind the CFMP (Certified Functional Medicine
Practitioner) designation. Its public-facing practitioner directory
lives at ``https://www.functionalmedicinedoctors.com``, which 301s to
``https://www.functionalmedicineuniversity.com/public/find-Functional-Medicine-Clinicians.cfm``
— i.e. ABFM / Functional Medicine University (FMU) share the same
Sequoia Education Systems infrastructure, and FMU's "Find a Clinician"
page IS the canonical ABFM-certified-practitioner directory.

How the data is served
======================

The directory page is a MemberGate CFM template with a Google-Maps-backed
listing. After accepting the Terms of Use page (``/public/1793.cfm``)
the front-end fires an AJAX POST to the MemberGate map endpoint:

    POST https://www.functionalmedicineuniversity.com/mgtags/mgMap.cfm
    Content-Type: application/x-www-form-urlencoded
    body: mapaction=markers
          &id=B35CE045-9DA2-2BB9-50904B67B2027C4E   <- directory id
          &southWestLat=-90 &southWestLng=-180
          &northEastLat=90 &northEastLng=180
          &state= &city= &country= &zipCode=
          &zipCodeProx=  &stateCity= &stateCanada=

A single world-bounds request returns the entire directory (~1,005
practitioners as of 2026-05-27) — no pagination needed. The response
is a single JSON document of shape::

    {"locations": [ {<flat dict per practitioner>}, ... ]}

Each record carries the practitioner's name, license info, address,
phone/email, pre-geocoded lat/lng, and an HTML ``template`` field used
to render the map info-window. We pull only the structured fields and
drop the HTML.

Relevant per-record keys::

  USERID                            stable account id (dedup key)
  MEMBER_NUMBER                     stable numeric member id
  FIRST_NAME                        practitioner first name
  LAST_NAME                         "Lastname, DC, CFMP, ..." — last
                                    name with trailing credentials in
                                    one comma-joined string
  COMPANY                           practice / office name
  ADDRESS / ADDRESS2                street; suppressed when
                                    MGCF_MAPPING_HIDE_ADDRESS == 1
  CITY / STATE / POSTAL_CODE
  COUNTRY                           "United States", "Canada", "United
                                    Kingdom", ...  (free-text)
  MGCF_MAPPING_COUNTRY              user-edited country variant
                                    ("USA", "Canada"). Falls back to
                                    COUNTRY when blank.
  WORK_PHONE / CELLPHONE
  EMAIL / MGCF_MAPPING_EMAIL
  MGCF_MAPPING_WEBSITE              practitioner site
  MGCF_DEGREES_HELD                 "Doctor of Chiropractic", "MD", ...
                                    Many records have placeholder "X"
                                    or "1" — those are silently
                                    suppressed during credential build.
  MGCF_PROFESSIONAL_DEGREECERTIFICATION_  short cred ("DC", "MD")
  MGCF_HEALTHCARE_SPECIALTY         specialty hint (informational)
  MGCF_LICENSE_NUMBER               state license number
  MGCF_LICENSED_IN                  US-state abbreviation licensed in
  MGCF_MAPPING_HIDE_ADDRESS         1 = practitioner asked to hide
                                    street; we suppress address1 in
                                    that case (city / state / postal
                                    are still public per the directory
                                    UI's behavior)
  latitude / longitude              pre-geocoded by FMU (ignored;
                                    shared geocoder owns lat/lng to
                                    keep geocode_quality consistent
                                    across adapters)

Fellowship rule
===============

The task spec says "all ABFM-board-certified practitioners are elite
by definition." The directory exposes exactly one tiering signal: the ``CFMP``
credential token. About 818 of the ~1,005 records (~81%) carry CFMP —
these are the board-certified ones. The remaining ~187 are FMU
training-program graduates who are not (yet) ABFM-board-certified at
the CFMP level. Therefore::

    fellowship_level = "CFMP" token appears in LAST_NAME OR in
                       MGCF_PROFESSIONAL_DEGREECERTIFICATION_

We check both fields because some practitioners only list CFMP in the
short-form cert field while their LAST_NAME-embedded creds stop at
their primary degree(s) (e.g. "Lewis, DC, ..." with CFMP only in the
professional-cert column).

This matches the IAOMT pattern (Accredited Member tier qualifies) and
the IABDM pattern (Certified Member tier qualifies) — board-certified
status = entry-level vetted tier = fellowship-level for cross-adapter
panel ranking.

Row contract
============

  tier              "org_member"
  source_org        "ABFM"
  specialties       ["functional_medicine", "holistic_health"]  (locked)
  source_url        stable per-practitioner URL based on USERID
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://www.functionalmedicineuniversity.com"
DIRECTORY_PAGE = f"{BASE}/public/find-Functional-Medicine-Clinicians.cfm"
API_MARKERS_URL = f"{BASE}/mgtags/mgMap.cfm"
# Stable directory id baked into the front-end JS, found at the AJAX
# POST site in ``/public/1793.cfm`` source.
DIRECTORY_ID = "B35CE045-9DA2-2BB9-50904B67B2027C4E"

LOCKED_SPECIALTIES = ["functional_medicine", "holistic_health"]

# Country-name -> ISO2 (same convention used by iabdm.py / iaomt.py). FMU
# data carries free-text country strings that vary in capitalization and
# include some MGCF_MAPPING_COUNTRY variants like "USA"/"Columbia"
# (sic — common user-edit typo for Colombia).
_COUNTRY_NAME_TO_ISO2 = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "canada": "CA",
    "ca": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "australia": "AU",
    "new zealand": "NZ",
    "ireland": "IE",
    "germany": "DE",
    "france": "FR",
    "spain": "ES",
    "italy": "IT",
    "netherlands": "NL",
    "belgium": "BE",
    "denmark": "DK",
    "sweden": "SE",
    "norway": "NO",
    "finland": "FI",
    "switzerland": "CH",
    "austria": "AT",
    "portugal": "PT",
    "poland": "PL",
    "czech republic": "CZ",
    "slovakia": "SK",
    "hungary": "HU",
    "romania": "RO",
    "bulgaria": "BG",
    "greece": "GR",
    "estonia": "EE",
    "mexico": "MX",
    "brazil": "BR",
    "chile": "CL",
    "argentina": "AR",
    "colombia": "CO",
    "columbia": "CO",  # common user typo seen in FMU data
    "costa rica": "CR",
    "panama": "PA",
    "peru": "PE",
    "ecuador": "EC",
    "venezuela": "VE",
    "uruguay": "UY",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "philippines": "PH",
    "india": "IN",
    "pakistan": "PK",
    "china": "CN",
    "hong kong": "HK",
    "taiwan": "TW",
    "indonesia": "ID",
    "vietnam": "VN",
    "united arab emirates": "AE",
    "uae": "AE",
    "saudi arabia": "SA",
    "qatar": "QA",
    "kuwait": "KW",
    "bahrain": "BH",
    "israel": "IL",
    "turkey": "TR",
    "lebanon": "LB",
    "jordan": "JO",
    "egypt": "EG",
    "south africa": "ZA",
    "morocco": "MA",
    "kenya": "KE",
    "nigeria": "NG",
    "ghana": "GH",
    "africa": None,  # too vague to map; falls through to free-text below
    "russia": "RU",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": DIRECTORY_PAGE,
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: world-bounds POST against the MemberGate marker endpoint
# ---------------------------------------------------------------------------

def fetch_all_directory_records() -> list[dict]:
    """POST world-bounds to the ABFM markers endpoint and return ALL records.

    The FMU front-end uses a Google-Maps viewport-bounded query (the
    ``southWest``/``northEast`` lat/lng pairs come from the visible map
    viewport). Passing world bounds (-90..90 / -180..180) returns the
    entire directory in one shot — there is no pagination on the wire.
    Static UA + 20s timeout + 0.5s post-call sleep (rate-friendly).
    """
    s = _session()
    payload = {
        "mapaction": "markers",
        "id": DIRECTORY_ID,
        "southWestLat": "-90",
        "southWestLng": "-180",
        "northEastLat": "90",
        "northEastLng": "180",
        "minLat": "-90",
        "maxLat": "90",
        "minLng": "-180",
        "maxLng": "180",
        # All filter fields blank = "no filter" — directory returns world.
        "zipCode": "",
        "zipCodeProx": "",
        "state": "",
        "city": "",
        "stateCity": "",
        "stateCanada": "",
        "country": "",
    }
    r = s.post(API_MARKERS_URL, data=payload, timeout=20)
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict):
        return []
    locations = body.get("locations")
    if not isinstance(locations, list):
        return []
    time.sleep(0.5)
    return locations


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty/null values.

    Accepts already-string values as well as the rare wrapped dict shape
    some MemberGate variants emit (``{"raw": ..., "rendered": ...}``)
    — we always prefer ``raw``."""
    if val is None:
        return None
    if isinstance(val, dict):
        return _coerce_str(val.get("raw"))
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


# Set of placeholder values FMU users typed into the degree field. Treat
# these as missing data — they aren't real credentials.
_DEGREE_PLACEHOLDERS = {"x", "1", "0", "n/a", "na", "none", "-", "."}


def _is_placeholder(val: Optional[str]) -> bool:
    """True if the value is a recognized 'no data' placeholder."""
    if val is None:
        return True
    return val.strip().lower() in _DEGREE_PLACEHOLDERS


# Cred-token recognizer: tokenized comma-separated credentials inside
# LAST_NAME. We strip dots and whitespace then look for the literal CFMP
# token. (Robust to "Walton  MSc. BSc. CFMP" -> trailing "CFMP" wins,
# and to "Kim Martin, DC, FASA, BCIM, CGP, CFMP, CCIP" -> embedded
# "CFMP" wins.)
_CFMP_TOKEN_RE = re.compile(r"(?:^|[,\s.])CFMP(?:[,\s.]|$)", re.IGNORECASE)


def _has_cfmp_credential(*values) -> bool:
    """True when ANY of the supplied credential strings carries the
    CFMP token.

    CFMP = Certified Functional Medicine Practitioner, ABFM's board
    designation. We check ALL credential-bearing fields (LAST_NAME plus
    MGCF_PROFESSIONAL_DEGREECERTIFICATION_) because some practitioners
    list CFMP only in the short-form credential field while their
    LAST_NAME-embedded creds are limited to their primary degree(s).
    Used to set ``fellowship_level``.
    """
    for raw in values:
        s = _coerce_str(raw)
        if s and _CFMP_TOKEN_RE.search(s):
            return True
    return False


def _split_bare_credential_suffix(
    s: str,
) -> tuple[str, Optional[str]]:
    """Strip a trailing whitespace-delimited credential token from a name.

    Recognize "Weintraub ND" / "Garcia Monroy D.O." / "Dempster ND" —
    the LAST whitespace-separated chunk is treated as a credential when
    it strips down to a 2..8-character all-uppercase ASCII abbreviation
    (with optional interior dots).

    Returns (clean_name, credential_or_None). When no trailing
    credential is found, returns (s, None) unchanged.
    """
    parts = s.split()
    if len(parts) < 2:
        return s, None
    last_token = parts[-1]
    compact = re.sub(r"[^A-Za-z]", "", last_token)
    if 2 <= len(compact) <= 8 and compact.isupper():
        return " ".join(parts[:-1]).strip(), last_token.rstrip(".") or None
    return s, None


def _split_last_name_and_credentials(
    raw_last: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split "Lastname, DC, CFMP, MIABDM" -> ("Lastname", "DC, CFMP, MIABDM").

    Also handles "Lastname DC" / "Lastname D.C." / "Lastname, D.C." and
    the no-comma "Weintraub ND" form, plus the mixed form "Dempster ND,
    FAARFM" where the pre-comma part STILL carries a trailing credential
    that needs to be peeled off the bare last name and merged into the
    credentials tail.
    """
    s = _coerce_str(raw_last)
    if not s:
        return None, None

    # Mixed form first: "Dempster ND, FAARFM" -> pre-comma still has a
    # trailing credential. Peel the bare last name off the pre-comma
    # part, then merge any peeled-off cred into the post-comma cred list.
    if "," in s:
        head, _, tail = s.partition(",")
        head_clean, head_cred = _split_bare_credential_suffix(head.strip())
        tail_clean = tail.strip().rstrip(",") or None
        if head_cred:
            tail_clean = f"{head_cred}, {tail_clean}" if tail_clean else head_cred
        return head_clean.rstrip(",") or None, tail_clean

    # No-comma form: "Weintraub ND" / "Garcia Monroy D.O.".
    clean, cred = _split_bare_credential_suffix(s)
    return (clean or None), cred


def _strip_placeholder_credentials(creds: Optional[str]) -> Optional[str]:
    """Drop any credential tokens that are FMU placeholders ('x', '1', ...).

    Returns the cleaned comma-joined credential string, or None if no
    real tokens remain.
    """
    if not creds:
        return None
    out: list[str] = []
    for tok in creds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if _is_placeholder(tok):
            continue
        out.append(tok)
    return ", ".join(out) if out else None


def _build_full_name(first: Optional[str], last_bare: Optional[str]) -> str:
    parts = [p.strip() for p in (first, last_bare) if p and p.strip()]
    return " ".join(parts)


def _normalize_website(raw) -> Optional[str]:
    """Add an https:// scheme if the entry lists a bare domain. Drops
    obvious placeholders. Recognizes the rare wrapped-dict shape."""
    s = _coerce_str(raw)
    if not s:
        return None
    if _is_placeholder(s):
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # Bare 'www.example.com' or 'example.com' -> add scheme. Skip strings
    # that don't look like a domain (no dot, or contains spaces).
    if " " in s or "." not in s:
        return None
    return f"https://{s}"


def _country_iso2(raw) -> Optional[str]:
    """Map a free-text country name to ISO2; None if unrecognized."""
    s = _coerce_str(raw)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _normalize_state(raw, country_iso2: Optional[str]) -> Optional[str]:
    """Clean up the STATE field.

    US records use mixed-case 2-letter codes ("Pa", "OR", "TX"). We
    upper-case those. Non-US records sometimes have placeholders like
    "UNITED KINGDOM" or "Ireland" copied into STATE; when STATE equals
    the country itself we treat it as missing. Numeric / single-char
    placeholders ('1', 'x') are dropped too — they are FMU's "no data"
    fallback in numeric form (the form input still required SOMETHING).
    """
    s = _coerce_str(raw)
    if not s:
        return None
    if _is_placeholder(s):
        return None
    # Strip out anything that's clearly a country-name dup.
    iso = _country_iso2(s)
    if iso is not None and iso == country_iso2:
        return None
    # 2-letter ASCII state codes -> upper.
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return s


def _select_country(
    country_raw, mapping_country_raw
) -> tuple[Optional[str], str]:
    """Pick the best country signal, return (iso2_or_None, original_string).

    Prefer MGCF_MAPPING_COUNTRY (the user-edited mapping-form value)
    when COUNTRY is blank — the original ``COUNTRY`` field is the
    account-registration country, the mapping field is the practice
    country and is the one shown on the directory.
    """
    primary = _coerce_str(country_raw)
    secondary = _coerce_str(mapping_country_raw)
    chosen = primary or secondary or ""
    iso2 = _country_iso2(chosen)
    return iso2, chosen


def _coalesce_email(rec: dict) -> Optional[str]:
    """EMAIL and MGCF_MAPPING_EMAIL are often duplicates; either works."""
    return _coerce_str(rec.get("EMAIL")) or _coerce_str(rec.get("MGCF_MAPPING_EMAIL"))


def _coalesce_phone(rec: dict) -> Optional[str]:
    """WORK_PHONE > CELLPHONE > MGCF_MAPPING_WORK_PHONE."""
    return (
        _coerce_str(rec.get("WORK_PHONE"))
        or _coerce_str(rec.get("CELLPHONE"))
        or _coerce_str(rec.get("MGCF_MAPPING_WORK_PHONE"))
    )


def _coalesce_address(rec: dict) -> Optional[str]:
    """Compose the street line, respecting the hide-address opt-out.

    When ``MGCF_MAPPING_HIDE_ADDRESS == 1`` the practitioner asked to
    hide their street from the public directory — we drop address1
    accordingly (city/state/postal remain since they're still shown in
    the directory UI). Falls back to MGCF_MAPPING_ADDRESS when the
    legacy ADDRESS field is blank.
    """
    hide_flag = rec.get("MGCF_MAPPING_HIDE_ADDRESS")
    if hide_flag in (1, "1", True):
        return None
    line1 = _coerce_str(rec.get("ADDRESS")) or _coerce_str(rec.get("MGCF_MAPPING_ADDRESS"))
    line2 = _coerce_str(rec.get("ADDRESS2"))
    if line1 and line2:
        return f"{line1}, {line2}"
    return line1 or line2


def _build_source_url(rec: dict) -> str:
    """Stable per-practitioner URL.

    The FMU directory is single-page (AJAX-served), so there is no
    individual detail URL per practitioner. We synthesize a stable URL
    by attaching the USERID (or MEMBER_NUMBER) as a fragment to the
    directory landing — gives a unique-per-practitioner dedup key
    that's also human-meaningful when pasted into a browser.
    """
    userid = _coerce_str(rec.get("USERID"))
    if userid:
        return f"{DIRECTORY_PAGE}#user={userid}"
    member_num = _coerce_str(rec.get("MEMBER_NUMBER"))
    if member_num:
        return f"{DIRECTORY_PAGE}#member={member_num}"
    # Last-resort tiebreak — combine name + city when both IDs missing
    # (defensive; should never happen with real data).
    name_key = (
        f"{_coerce_str(rec.get('FIRST_NAME')) or ''}-"
        f"{_coerce_str(rec.get('LAST_NAME')) or ''}-"
        f"{_coerce_str(rec.get('CITY')) or ''}"
    )
    name_key = re.sub(r"[^A-Za-z0-9-]+", "-", name_key).strip("-").lower()
    return f"{DIRECTORY_PAGE}#unknown-{name_key or 'no-id'}"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: FMU marker dict -> NormalizedPractitionerRow.

    Returns None when no usable practitioner name can be recovered.
    """
    first = _coerce_str(rec.get("FIRST_NAME"))
    last_raw = _coerce_str(rec.get("LAST_NAME"))
    last_bare, last_creds = _split_last_name_and_credentials(last_raw)

    full_name = _build_full_name(first, last_bare)
    if not full_name:
        # Try the title field as last-resort (it's "FIRST LAST, creds...")
        title = _coerce_str(rec.get("title"))
        if title:
            title = html_module.unescape(title)
            full_name, _ = _split_last_name_and_credentials(title)
            full_name = full_name or title
    if not full_name:
        return None

    # Build the credentials string. Sources in priority order:
    #   1. trailing creds carved out of LAST_NAME ("DC, CFMP, ...")
    #   2. MGCF_PROFESSIONAL_DEGREECERTIFICATION_ short code ("DC")
    #   3. MGCF_DEGREES_HELD long form ("Doctor of Chiropractic") -
    #      only added when it would contribute NEW information; a long
    #      form like "Doctor of Chiropractic" is suppressed when "DC"
    #      already appears in the trailing creds (otherwise the row
    #      ends up with "DC, CFMP, Doctor of Chiropractic" — three
    #      slots for the same fact).
    # Placeholders ("x" / "1" / "0") are dropped at each step.
    cred_parts: list[str] = []
    seen_tokens: set[str] = set()  # case-insensitive de-dup memory

    def _add_cred(value: str) -> None:
        compact_lower = re.sub(r"[^a-z]", "", value.lower())
        full_lower = value.strip().lower()
        if not compact_lower and not full_lower:
            return
        if full_lower in seen_tokens or compact_lower in seen_tokens:
            return
        cred_parts.append(value.strip())
        if full_lower:
            seen_tokens.add(full_lower)
        if compact_lower:
            seen_tokens.add(compact_lower)

    if last_creds:
        for chunk in last_creds.split(","):
            chunk = chunk.strip()
            if chunk and not _is_placeholder(chunk):
                _add_cred(chunk)

    # The "short" field is sometimes a comma-separated list itself
    # ("DC, DABAAHP, FAAIM, BCIM, DAAPM, CFMP") and sometimes a single
    # token ("DC"). Tokenize before adding so the dedup catches per-
    # token overlap with what we already pulled from LAST_NAME.
    short = _coerce_str(rec.get("MGCF_PROFESSIONAL_DEGREECERTIFICATION_"))
    if short and not _is_placeholder(short):
        for chunk in short.split(","):
            chunk = chunk.strip()
            if chunk and not _is_placeholder(chunk):
                _add_cred(chunk)

    long_deg = _coerce_str(rec.get("MGCF_DEGREES_HELD"))
    if long_deg and not _is_placeholder(long_deg):
        # Suppress long-form when its leading word matches an already-
        # captured short code (e.g. "Doctor of Chiropractic" when "DC"
        # is present). Compare against the short-form abbreviation
        # heuristic: take first letter of each capitalized word.
        words = long_deg.split()
        if words:
            initials = "".join(
                w[0].upper() for w in words
                if w[:1].isupper() and w.lower() not in {"of", "and", "in"}
            )
            if initials and initials.lower() in seen_tokens:
                pass  # suppress duplicate long form
            else:
                _add_cred(long_deg)
        else:
            _add_cred(long_deg)

    credentials = ", ".join(cred_parts) if cred_parts else None

    # Practice / office name. Suppress when it duplicates the
    # practitioner's name (solo practitioner who filled in their own
    # name in the COMPANY field).
    practice = _coerce_str(rec.get("COMPANY"))
    if practice and practice.lower() == full_name.lower():
        practice = None

    phone = _coalesce_phone(rec)
    email = _coalesce_email(rec)
    website = _normalize_website(rec.get("MGCF_MAPPING_WEBSITE"))

    iso2, country_raw = _select_country(
        rec.get("COUNTRY"), rec.get("MGCF_MAPPING_COUNTRY")
    )
    state = _normalize_state(rec.get("STATE"), iso2)
    postal = _coerce_str(rec.get("POSTAL_CODE")) or _coerce_str(
        rec.get("MGCF_MAPPING_POSTAL_CODE")
    )
    city = _coerce_str(rec.get("CITY")) or _coerce_str(rec.get("MGCF_MAPPING_CITY"))
    address1 = _coalesce_address(rec)

    country = iso2 or (country_raw or "US")

    return NormalizedPractitionerRow(
        tier="org_member",
        name=full_name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="ABFM",
        source_url=_build_source_url(rec),
        fellowship_level=_has_cfmp_credential(
            rec.get("LAST_NAME"),
            rec.get("MGCF_PROFESSIONAL_DEGREECERTIFICATION_"),
        ),
        practice_name=practice,
        credentials=credentials,
        phone=phone,
        email=email,
        website=website,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a markers response (dict with ``locations`` list,
    a raw list, or a JSON string of either) and returns one row per
    usable record. No I/O.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        import json
        payload = json.loads(payload)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("locations") or payload.get("data") or []
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
