"""American Board of Chiropractic Internists (ABCI / CDID / DABCI) scraper.

The DABCI credential (Diplomate of the American Board of Chiropractic
Internists) is the post-doctoral specialty credential for chiropractors
who focus on diagnosis and treatment of internal disorders -- the
functional / internal-medicine chiropractors. The specialty council is
the ACA Council on Diagnosis & Internal Disorders (CDID).

DIRECTORY-URL DISCOVERY (2026-05-29)
------------------------------------
A prior research pass suggested ``aca-cdid.com/.../doctors.php?search=all``
but that domain does NOT resolve (NXDOMAIN / curl 000 over both http and
https, www and bare). ``cdid.org`` 301-redirects to an unrelated
children's hospital. ``dabci.info`` is a live ABCI site but exposes NO
practitioner directory at all (HOME / ABOUT / ELIGIBILITY / TESTING /
CLASSES only).

The ONLY public, login-free, plain-``requests``-scrapable DABCI directory
is the "DABCI Status" table on the archived ABCI site:

    https://www.dabci.org/index.php?id=1&reveal=yes&view_only=yes

This is a single static HTML page (no JS render, no pagination, no login)
returned by a plain GET. Its "DABCI Status" section is described on-page
as "a public directory of all DABCIs, their location, and current status
with the ABCI." It carries ~89 diplomate rows.

CAVEAT (reported up the chain): the page header states the content is
from the site's 2008-2009 archived pages ("CIRCA 2008-2009"). So this is
a historical diplomate roster, not a live-maintained one. It is, however,
the only public scrapable DABCI list that exists -- the current authority
(aca-cdid.com) is offline. Treat the rows as a seed roster; freshness
should be reconciled later if aca-cdid.com ever returns.

PAGE STRUCTURE
--------------
The page has three ``<table border="1" width="100%">`` elements:

  1-2) two continuing-education *seminar* schedules (Dates / Seminar /
       Hrs / Location / Contact) -- these are NOT practitioners and must
       be skipped.
  3)   the practitioner directory, whose header row is exactly
       Doctor / Location / Status / Disciplinary Action.

We therefore identify the directory table by its header row text
("doctor" + "disciplinary action") rather than by position, so a future
re-ordering of the decoy tables won't break the parser.

Each directory row is four ``<td>`` cells:

    Doctor              -> "Last, First[ Suffix]" (e.g. "Zevan III, Alex",
                           "Smith, Todd A.", "Satterwhite, R Vincent").
                           We flip to "First Last" for the name field.
    Location            -> "City, ST" (US only in the live data). Some
                           cells are state-only ("OK"), some are blank
                           ("Hug, Reginald" has no location), and several
                           carry data-entry typos in the state ("IH" for
                           IA, "Il" for IL). We keep city/state when a
                           confident 2-letter US state is present; the
                           raw location always lands in address1 so the
                           shared geocoder can recover.
    Status              -> "Certified" / "Retired" / "Uncredentialed"
                           (with the odd typo, e.g. "Cerifiied"). This is
                           the ABCI *credentialing* status, NOT a
                           diplomate-vs-member distinction -- everyone in
                           the table is a DABCI diplomate. We surface it
                           in the credentials field as "DABCI (<status>)".
    Disciplinary Action -> usually blank; captured into credentials when
                           present.

There is NO per-row profile URL, phone, email, website, practice name,
street address, or postal code in this directory -- only name, city,
state, and status. Those fields stay None; geocoding (lat/lng) is handled
elsewhere.

FELLOWSHIP RULE
---------------
DABCI is itself a diplomate credential, and every row in this table is a
listed diplomate (the page title literally is "DABCI Status... a public
directory of all DABCIs"). So ``fellowship_level=True`` for every row --
same posture as the NCCAOM adapter. The Certified/Retired/Uncredentialed
status does NOT downgrade the diplomate fact; it only reflects current
recertification standing, which we preserve in the credentials string.

SOURCE URL
----------
The directory is a single page with no per-doctor anchor or detail URL.
Mirroring cso.py, we build a stable, unique per-practitioner source_url
by anchoring the directory URL on a slug derived from the doctor's
"Last, First" + location (so a re-run produces identical keys and two
distinct doctors never collide):

    https://www.dabci.org/index.php?id=1&reveal=yes&view_only=yes#dabci-<slug>

Output rows: tier='org_member', source_org='ABCI',
specialties=['chiropractic', 'holistic_health'].
"""
import html as html_module
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://www.dabci.org"
DIRECTORY_URL = f"{BASE}/index.php?id=1&reveal=yes&view_only=yes"

LOCKED_SPECIALTIES = ["chiropractic", "holistic_health"]

_US_STATE_ABBR_SET = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

# A trailing name suffix that should stay glued to the (flipped) name and
# never be mistaken for the given name.
_NAME_SUFFIX_RE = re.compile(r"^(?:Jr\.?|Sr\.?|II|III|IV|V)$", re.IGNORECASE)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: fetch the DABCI Status directory page
# ---------------------------------------------------------------------------

def fetch_directory_page() -> str:
    """Download the full DABCI Status page HTML.

    Single GET -- no pagination; the entire roster ships in one response.
    Static UA + 20s timeout + 0.5s sleep (rate-friendly, per
    feedback_e4l_portal_no_concurrency.md)."""
    s = _session()
    r = s.get(DIRECTORY_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def fetch_all_records() -> list[dict]:
    """Public fetch entry point for run_all.

    Fetches the live page and returns the parsed raw directory records
    (one dict per practitioner: {'doctor', 'location', 'status',
    'disciplinary'}). The pure parser ``parse_directory_html`` is used by
    tests against captured fixtures."""
    html = fetch_directory_page()
    return extract_directory_records(html)


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _clean_cell(text: Optional[str]) -> str:
    """Unescape HTML entities, collapse whitespace (incl. nbsp), strip."""
    if not text:
        return ""
    t = html_module.unescape(text)
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_header_row(cells: list[str]) -> bool:
    """True for the directory table's header row (Doctor / Location /
    Status / Disciplinary Action). Case-insensitive."""
    joined = " ".join(c.lower() for c in cells)
    return "doctor" in joined and "disciplinary" in joined


def _looks_like_directory_table(header_cells: list[str]) -> bool:
    """Identify the practitioner directory among the page's tables by its
    header row, so the two continuing-education seminar tables (which are
    headed Dates / Seminar / Hrs / Location / Contact) are skipped."""
    return _is_header_row(header_cells)


def extract_directory_records(html: str) -> list[dict]:
    """Pull the DABCI Status directory rows out of the page HTML.

    Returns a list of raw record dicts:
        {'doctor': str, 'location': str, 'status': str, 'disciplinary': str}

    Skips the header row, the trailing all-blank row, and the two
    continuing-education seminar tables. Returns [] if the directory
    table is missing (defensive -- would indicate the page was
    restructured)."""
    soup = BeautifulSoup(html, "html.parser")

    directory_table = None
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        first_cells = [_clean_cell(td.get_text(" ")) for td in rows[0].find_all("td")]
        if _looks_like_directory_table(first_cells):
            directory_table = table
            break

    if directory_table is None:
        return []

    records: list[dict] = []
    for tr in directory_table.find_all("tr"):
        cells = [_clean_cell(td.get_text(" ")) for td in tr.find_all("td")]
        if len(cells) < 3:
            continue
        if _is_header_row(cells):
            continue
        doctor = cells[0]
        location = cells[1] if len(cells) > 1 else ""
        status = cells[2] if len(cells) > 2 else ""
        disciplinary = cells[3] if len(cells) > 3 else ""
        # Drop fully-blank trailing/separator rows (no doctor name).
        if not doctor:
            continue
        records.append(
            {
                "doctor": doctor,
                "location": location,
                "status": status,
                "disciplinary": disciplinary,
            }
        )
    return records


def _flip_name(doctor: str) -> str:
    """'Last, First[ M.]' -> 'First[ M.] Last'. A trailing generational
    suffix on the LAST-name side ('Zevan III, Alex') stays attached to the
    surname: 'Alex Zevan III'. If there is no comma, return as-is."""
    s = doctor.strip()
    if "," not in s:
        return s
    last_part, first_part = s.split(",", 1)
    last_part = last_part.strip()
    first_part = first_part.strip()
    if not first_part:
        return last_part
    if not last_part:
        return first_part

    # Detach a generational suffix from the surname segment so it trails
    # the full flipped name: "Zevan III" -> surname "Zevan", suffix "III".
    suffix = ""
    last_tokens = last_part.split()
    if len(last_tokens) > 1 and _NAME_SUFFIX_RE.match(last_tokens[-1]):
        suffix = last_tokens[-1]
        last_part = " ".join(last_tokens[:-1])

    flipped = f"{first_part} {last_part}".strip()
    if suffix:
        flipped = f"{flipped} {suffix}".strip()
    return flipped


def _parse_location(location: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort 'City, ST' split. Returns (address1, city, state).

    - 'Sandwich, IL'            -> ('Sandwich, IL', 'Sandwich', 'IL')
    - 'North Kanas City, MO'    -> (raw, 'North Kanas City', 'MO')
    - 'OK'                      -> ('OK', None, 'OK')  (state-only)
    - 'Prudenville,MI' (no sp.) -> handled (comma split, then trim)
    - '' (blank)                -> (None, None, None)

    State is only set when the trailing token is a confident 2-letter US
    state code. The full raw location always lands in address1 so the
    shared geocoder can resolve typo'd states ('IH', 'Il') and odd
    spellings; we deliberately do not try to correct typos here."""
    loc = location.strip().rstrip(",").strip()
    if not loc:
        return None, None, None

    if "," in loc:
        city, st = loc.rsplit(",", 1)
        city = city.strip() or None
        st = st.strip()
        state = st.upper() if st.upper() in _US_STATE_ABBR_SET else None
        return loc, city, state

    # No comma: either a bare state code or a single token we can't split.
    if loc.upper() in _US_STATE_ABBR_SET:
        return loc, None, loc.upper()
    return loc, None, None


def _slugify(*parts: str) -> str:
    """Stable lowercase slug from the given parts (for the source_url
    anchor). Non-alphanumerics collapse to single hyphens."""
    raw = "-".join(p for p in parts if p)
    raw = raw.lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    return raw.strip("-") or "unknown"


def _build_source_url(doctor: str, location: str) -> str:
    """Stable, unique per-practitioner URL. The directory page has no
    per-doctor detail URL, so we anchor the directory URL on a slug of
    the doctor's 'Last, First' + location. Mirrors cso.py's marker-anchor
    approach; stable across re-runs and unique per (doctor, location)."""
    slug = _slugify(doctor, location)
    return f"{DIRECTORY_URL}#dabci-{slug}"


def _build_credentials(status: str, disciplinary: str) -> str:
    """Compose the credentials string. Always begins with 'DABCI' (every
    row is a diplomate), then appends the ABCI status and any disciplinary
    note. e.g. 'DABCI (Certified)' or 'DABCI (Retired)'."""
    status = status.strip().rstrip(",").strip()
    creds = "DABCI"
    if status:
        creds = f"{creds} ({status})"
    disc = disciplinary.strip().rstrip(",").strip()
    if disc:
        creds = f"{creds}; Disciplinary: {disc}"
    return creds


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: a directory record dict ->
    NormalizedPractitionerRow. Returns None when no usable name exists."""
    doctor = (rec.get("doctor") or "").strip()
    if not doctor:
        return None
    name = _flip_name(doctor)
    if not name:
        return None

    location = (rec.get("location") or "").strip()
    address1, city, state = _parse_location(location)

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="ABCI",
        source_url=_build_source_url(doctor, location),
        fellowship_level=True,  # every listed practitioner is a DABCI diplomate
        practice_name=None,
        credentials=_build_credentials(rec.get("status") or "", rec.get("disciplinary") or ""),
        phone=None,
        email=None,
        website=None,
        address1=address1,
        city=city,
        state=state,
        postal=None,
        country="US",
    )


def parse_directory_records(records: list[dict]) -> list[NormalizedPractitionerRow]:
    """Pure parser: list of raw directory record dicts -> rows. Records
    with no extractable name are silently dropped."""
    rows: list[NormalizedPractitionerRow] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        row = _record_to_row(rec)
        if row is not None:
            rows.append(row)
    return rows


def parse_directory_html(html: str) -> list[NormalizedPractitionerRow]:
    """End-to-end pure parser: page HTML -> rows. Used by tests that
    capture full-page HTML fixtures and by fetch_all_records' callers."""
    return parse_directory_records(extract_directory_records(html))
