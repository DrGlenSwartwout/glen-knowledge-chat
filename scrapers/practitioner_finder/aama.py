"""American Academy of Medical Acupuncture (AAMA) scraper.

**Disambiguation:** AAMA = the medical-doctor acupuncture professional
society (MDs/DOs who practice acupuncture as a medical specialty after
completing the Helms Medical Institute training and, optionally, board
certification via the American Board of Medical Acupuncture / DABMA).
This is NOT the unrelated American Association of Medical Assistants.
The directory page literally reads "American Academy of Medical
Acupuncture (AAMA)" and the membership domain is ``medacu.memberclicks.net``
— verified 2026-05-27.

Discovery 2026-05-27:
AAMA's primary site at ``https://medicalacupuncture.org`` is WordPress
(no GeoDirectory / WP-REST listing — wp/v2 namespaces don't expose any
member endpoint). The public ``/find-an-acupuncturist/`` page links out
to the **MemberClicks** patient-referral directory:

    https://medacu.memberclicks.net/patient-referral-directory#/

This is the same MemberClicks ``ui-directory-search/v2`` API that
backs the NORA adapter — only the directory id and a couple of payload
field positions differ. The two thin JSON endpoints are:

    POST https://medacu.memberclicks.net/ui-directory-search/v2/search-directory/
         body: {"form": {"directory_search_id": 2002770, "elements": []}}
         -> first page + the upstream ``data_url`` for pagination.

    POST https://medacu.memberclicks.net/ui-directory-search/v2/search-directory-paged/
         body: {"url": "<data_url>?pageSize=10&pageNumber=N"}
         -> page N of the same search.

Discovery 2026-05-27 found ``total_count`` = 246 practitioners across 25
pages at ``pageSize=10`` (the SPA-encoded contract; the upstream
search-results service rejects larger page sizes — matches the NORA
behaviour exactly). ``directory_search_id`` = 2002770 is discoverable
from the public ``get-directory-search-form/patient-referral-directory``
endpoint so the scraper is self-healing if AAMA ever re-publishes the
locator under a new id.

MemberClicks is HTTP-cookie gated: a single GET against
``/patient-referral-directory`` mints the session cookie
(``0012f0e1bd...`` + ``serviceID`` / ``Login=1``) which
``requests.Session()`` then re-uses for every subsequent call. No login
is required.

Each search result is a "display element" with this shape (after JSON
parsing, before our normalization):

    {
      "id": 2007179612,                       # stable MemberClicks profile id
      "avatar_url": null,
      "title": "Shiyi Abla-Yao, MD, DABMA",   # name + comma-delimited creds
      "top": [], "bottom": [],                 # always empty on AAMA
      "left": [
        {"display_order": 0, "html": "<line1>, \\n<line2><br />\\n<city>, <ST> <postal>"},
        {"display_order": 1, "html": "<strong>Phone:</strong> <phone>"}
      ],
      "right": [
        {"display_order": 0, "html": "<strong>Specialty:</strong> <specialty list>"},
        {"display_order": 1, "html": "<website_or_blank>"}
      ],
      "distance": 0
    }

Note that AAMA pushes data into ``left`` and ``right`` (NORA used
``top``), so the field-positional helpers are different even though the
SPA framing is the same. Inside ``left[1]`` and ``right[0]`` the values
are prefixed with literal "<strong>Phone:</strong>" / "<strong>Specialty:</strong>"
HTML labels — we strip those to recover the bare value.

Output rows have tier='org_member', source_org='AAMA', and
specialties=['medical_acupuncture', 'acupuncture_tcm', 'holistic_health'].

Fellowship detection
--------------------
AAMA's credential ladder (per https://medicalacupuncture.org/ and the
public ``for-physicians/become-a-fellow`` page) has two named elite
tiers:

  * **DABMA** — Diplomate of the American Board of Medical Acupuncture.
    The ABMA board-certification exam, sat after completing the Helms
    Medical Institute training. This IS the "Board Certified" tier the
    Wave C spec calls out as the elite default.
  * **FAAMA** — Fellow of the American Academy of Medical Acupuncture.
    AAMA's own internal fellow designation, also gated on Helms
    training + clinical experience hours.

Per the cross-adapter convention ("Board Certified / Master / Fellow
qualifies"), AAMA's DABMA is the structural analogue of IAOMT's
Accredited / NORA's FNORA / IABDM's Certified-Member-or-above tier —
the entry-level vetted-credentials gate — and FAAMA is the named
Fellow tier. So **DABMA OR FAAMA** in the title sets
``fellowship_level=True``. Plain "MD" / "DO" without one of these
two markers stays at False — they're AAMA members in good standing
but have not completed the board-certification or fellow pathway.

(The AAMA public directory does NOT expose an "Affiliate" or
"Student" tier — every record carries an MD or DO degree at minimum,
which matches AAMA's bylaws that restrict full membership to
licensed physicians who have completed Helms training. So the
False-default never under-flags an MD record that's actually elite.)

The public Patient-Referral-Directory payload exposes credentials
only via the ``title`` field (e.g. ``"Thomas E. Archie, MD, DABMA"``
or the multi-tier ``"Norman G. Zavela, MD, FAAMA"``). We look for
the literal tokens ``DABMA`` and ``FAAMA`` (case-insensitive,
word-boundary) anywhere in the title to set the flag.

Per-practitioner source_url
---------------------------
MemberClicks profile detail pages are member-only (403 unauth) and
have no stable public URL — exactly the same constraint as NORA. We
synthesize a stable, deterministic URL from the public patient-
referral-directory page + the MemberClicks profile id:

    https://medacu.memberclicks.net/patient-referral-directory#/profile/<id>

The fragment is invariant across re-runs (profile id is stable) so it
works as the upsert dedup key, even though the fragment is currently
only consumed by the SPA for in-page state.
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://medacu.memberclicks.net"
PATIENT_REFERRAL_URL = f"{BASE}/patient-referral-directory"
SEARCH_FORM_URL = (
    f"{BASE}/ui-directory-search/v2/get-directory-search-form/"
    "patient-referral-directory"
)
SEARCH_DIRECTORY_URL = f"{BASE}/ui-directory-search/v2/search-directory/"
SEARCH_PAGED_URL = f"{BASE}/ui-directory-search/v2/search-directory-paged/"
PAGE_SIZE = 10  # Upstream search-results service caps at 10 per page.

LOCKED_SPECIALTIES = ["medical_acupuncture", "acupuncture_tcm", "holistic_health"]


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
    search call). 2026-05-27 the live id is 2002770, but reading it
    from the form response makes the scraper self-healing if AAMA ever
    re-publishes the locator under a new id.
    """
    s = session or _session()
    # Prime the JSESSIONID-style cookies.
    s.get(PATIENT_REFERRAL_URL, timeout=20)
    time.sleep(0.5)
    r = s.get(
        SEARCH_FORM_URL,
        headers={"Referer": PATIENT_REFERRAL_URL},
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
            "Referer": PATIENT_REFERRAL_URL,
            "Origin": BASE,
        },
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def fetch_page(data_url: str, page_number: int, session: requests.Session) -> dict:
    """POST to search-directory-paged/ with a re-quoted data_url + page n.

    Mirrors the SPA contract exactly: pageSize=10, pageNumber is 1-based.
    """
    paged_url = f"{data_url}?pageSize={PAGE_SIZE}&pageNumber={page_number}"
    r = session.post(
        SEARCH_PAGED_URL,
        json={"url": paged_url},
        headers={
            "Referer": PATIENT_REFERRAL_URL,
            "Origin": BASE,
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def fetch_all_directory_records() -> list[dict]:
    """Walk every page of the AAMA directory, return concatenated raw
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
    """Stripped string or None for empty/null."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


# Tokens we treat as a "credential" when stripping them off a title.
# Drawn from a 66-title live scan 2026-05-27 plus standard physician
# credentials. We do NOT depend on this list for fellowship detection —
# that uses literal DABMA / FAAMA word-boundary scans.
_CREDENTIAL_TOKENS = {
    "MD", "DO", "DDS", "DMD", "DC", "ND", "DPT", "PT", "OT", "OTR",
    "LAc", "LAC", "L.AC", "L.AC.", "MAc", "MAC", "MAC.", "MS", "MSc",
    "MPH", "PhD", "PHD", "MDiv", "MA",
    "DABMA", "FAAMA", "FAAFP", "FAAFRM", "FCA", "HMD", "FACP", "FACOG",
    "FAAEM", "FACEP", "FACS", "FAAFM", "FAAPMR", "FAAAAI",
    "BS", "BA", "RN", "NP", "PA", "PA-C", "ABO", "ABOC",
    "DABT", "DABA", "DABFP",
}


def _looks_like_credential(chunk: str) -> bool:
    """True when a comma-separated chunk looks like a credential token.

    Heuristic mirrors the NORA helper: a credential is 1-12 chars,
    no internal whitespace, made of letters + dots + slashes + digits,
    with at least one uppercase letter and a >=50% uppercase letter
    ratio. The known-token set ``_CREDENTIAL_TOKENS`` is the fast-path
    accept; the heuristic catches typo variants like 'Mac.' or 'MAc'."""
    if not chunk:
        return False
    c = chunk.strip()
    if len(c) > 12 or len(c) < 1:
        return False
    if " " in c:
        return False
    if not re.match(r"^[A-Za-z0-9./\-]+$", c):
        return False
    # Known-token fast path (case-insensitive).
    if c.upper().rstrip(".") in {t.upper().rstrip(".") for t in _CREDENTIAL_TOKENS}:
        return True
    if not any(ch.isupper() for ch in c):
        return False
    letters = [ch for ch in c if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio < 0.5:
            return False
    return True


def _dedupe_preserve_order(tokens: list[str]) -> list[str]:
    """De-dupe tokens case-insensitively while preserving first-seen order.

    Some AAMA titles repeat credentials (e.g.
    ``"Elizabeth Chen Christenson, MD, LAc, MD, LAc, FAAMA"``); we want
    a clean ``"MD, LAc, FAAMA"`` out the other side."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        key = t.upper().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split 'First Last, MD, DABMA, FAAMA' into (clean_name, credentials).

    AAMA titles are always comma-delimited (no space-only variant in
    the live data), so the comma walk handles every observed case.
    A few records have an extra space inside the credentials run
    (``"MD,LAc"`` with no space after the comma); ``split(',')``
    handles those cleanly.
    """
    if not name:
        return "", None
    s = html_module.unescape(name).strip()
    # Paren form is rare but cheap to support.
    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

    # AAMA's titles sometimes carry internal double spaces (e.g.
    # "Shiyi  Abla-Yao") from the source CMS — collapse whitespace runs
    # so the practitioner name is clean.
    s = re.sub(r"\s+", " ", s)

    if "," not in s:
        return s, None

    chunks = [c.strip() for c in s.split(",")]
    # Walk from the right; everything that looks like a credential token
    # is part of the credential block. Stop at the first non-credential
    # chunk from the right (which is the last name fragment).
    creds_block: list[str] = []
    while len(chunks) > 1:
        tail = chunks[-1]
        if _looks_like_credential(tail):
            creds_block.insert(0, chunks.pop())
        else:
            break
    clean = ", ".join(chunks).strip().rstrip(",").strip()
    if not creds_block:
        return clean, None
    creds_block = _dedupe_preserve_order(creds_block)
    return clean, ", ".join(creds_block)


def _name_from_title(title: str) -> str:
    name, _ = _strip_credentials(title or "")
    return name


def _credentials_from_title(title: str) -> Optional[str]:
    _, creds = _strip_credentials(title or "")
    return creds


# DABMA = Diplomate American Board of Medical Acupuncture (Board Certified).
# FAAMA = Fellow of the American Academy of Medical Acupuncture.
# Both are AAMA's elite tiers; either sets fellowship_level=True.
_DABMA_RE = re.compile(r"(?<![A-Za-z])DABMA(?![A-Za-z])", re.IGNORECASE)
_FAAMA_RE = re.compile(r"(?<![A-Za-z])FAAMA(?![A-Za-z])", re.IGNORECASE)


def _is_fellowship(record: dict) -> bool:
    """True when the record's title carries the DABMA or FAAMA credential.

    AAMA's two elite tiers are Diplomate of the American Board of
    Medical Acupuncture (DABMA — "Board Certified") and Fellow of the
    American Academy of Medical Acupuncture (FAAMA). Either credential
    in the title qualifies for fellowship_level=True; plain MD/DO
    without one of these markers stays at the default False.

    The public Patient-Referral-Directory payload exposes credentials
    only via the ``title`` field, so that's where we look.
    """
    title = _coerce_str(record.get("title")) or ""
    if not title:
        return False
    return bool(_DABMA_RE.search(title) or _FAAMA_RE.search(title))


# ---------------------------------------------------------------------------
# Address + contact parsing
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"^[+()\-\d.\s]+$")
_US_ZIP_RE = re.compile(r"^\d{4,5}(?:-\d{4})?$")

# Strip wrapping <strong>Label:</strong> blocks (live for left[1]=Phone and
# right[0]=Specialty). Case-insensitive in case AAMA ever changes casing.
_LABEL_PREFIX_RE = re.compile(
    r"^\s*<strong>\s*[A-Za-z ]+\s*:\s*</strong>\s*", re.IGNORECASE
)


def _strip_label_prefix(html_val: Optional[str]) -> Optional[str]:
    """Remove a leading '<strong>Label:</strong>' block + whitespace.

    AAMA prefixes left[1] with 'Phone:' and right[0] with 'Specialty:'.
    The HTML is the raw markup from the API — we don't render anything,
    we just strip the wrapping tag.
    """
    s = _coerce_str(html_val)
    if not s:
        return None
    cleaned = _LABEL_PREFIX_RE.sub("", s).strip()
    return cleaned or None


def _strip_html(html_val: Optional[str]) -> Optional[str]:
    """Strip any remaining HTML tags + collapse whitespace runs.

    Used for the website slot (right[1]) which is sometimes a bare URL
    and sometimes wrapped in an <a> tag. The address block has its own
    parser because the <br /> in it is semantically a line separator.
    """
    s = _coerce_str(html_val)
    if not s:
        return None
    s = re.sub(r"<[^>]+>", "", s)
    s = html_module.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _looks_like_phone(s: str) -> bool:
    if not s:
        return False
    digits = re.sub(r"\D", "", s)
    return bool(_PHONE_RE.match(s)) and len(digits) >= 7


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if AAMA's website field is a bare domain."""
    s = _strip_html(raw)
    if not s:
        return None
    # Defensive: some entries leak 'Www.example.com' with mid-cap W.
    if s.lower().startswith("www."):
        s = "www." + s[4:]
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # Don't accidentally promote a non-URL value (e.g. a description with
    # spaces) to a website.
    if " " in s or "." not in s:
        return None
    return f"https://{s}"


def _split_address(block: Optional[str]) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[str], str
]:
    """Parse AAMA's ``left[0]`` HTML chunk.

    The string shape is one of::

        '<line1>, \\n<br />\\n<city>, <ST> <postal>'
        '<line1>, \\n<line2><br />\\n<city>, <ST> <postal>'
        ', \\n<br />\\n,  '             # fully empty

    Any of the parts can be empty. Returns
    ``(address1, city, state, postal, country)`` with country defaulting
    to 'US' (every AAMA record observed 2026-05-27 was US/PR — AAMA
    bylaws require US/Canada/PR licensure; we leave the country at US
    and rely on the downstream geocoder if a non-US record ever appears).

    The address1 field is the joined non-empty fragments of the
    pre-``<br />`` portion (line1 + optional line2). The post-``<br />``
    portion is parsed as ``"City, ST Zip"``.
    """
    if not block:
        return None, None, None, None, "US"
    raw = html_module.unescape(block)
    # The address has <br /> between the address-lines and the city-line.
    # Split there.
    parts = re.split(r"<br\s*/?>", raw, maxsplit=1)
    if len(parts) == 1:
        # No <br /> — treat the whole thing as the city-line.
        addr_section, city_line = "", parts[0]
    else:
        addr_section, city_line = parts[0], parts[1]

    # Address section: comma + newline-separated line1, optional line2.
    addr_clean = re.sub(r"\s*\n\s*", " ", addr_section).strip()
    # Drop a trailing comma left over from the canonical
    # "<line1>, \n<br />" shape.
    addr_clean = addr_clean.rstrip(",").strip()
    # Comma can also separate line1 from line2 (e.g.
    # "Alligood Medical Acupuncture PLLC, 1801 Charles Blvd, Ste 109").
    # We keep the whole address as a single address1 field — the inner
    # commas are semantically valid address punctuation.
    address1 = addr_clean or None

    city_line = city_line.strip().lstrip(",").strip()
    # Replace newline runs in the city-line with spaces, then collapse.
    city_line = re.sub(r"\s*\n\s*", " ", city_line).strip()

    city: Optional[str] = None
    state: Optional[str] = None
    postal: Optional[str] = None
    if "," in city_line:
        city_part, rest = city_line.split(",", 1)
        city = city_part.strip() or None
        rest = rest.strip()
        if rest:
            # Last whitespace-run separates state from a numeric postal.
            tokens = rest.rsplit(" ", 1)
            if len(tokens) == 2:
                head, tail = tokens[0].strip(), tokens[1].strip()
                if _US_ZIP_RE.match(tail):
                    state = head or None
                    postal = tail
                else:
                    state = rest or None
            else:
                # Single token in `rest` — most likely a 2-letter state
                # with no zip (rare).
                state = rest or None
    else:
        city = city_line or None

    # If every component came back empty (the placeholder ', \n<br />\n,  '
    # case), drop address1 too — it's just a stray "," we already stripped.
    if not city and not state and not postal and not address1:
        return None, None, None, None, "US"

    return address1, city, state, postal, "US"


def _side_item(record: dict, side: str, display_order: int) -> Optional[str]:
    """Pull the ``html`` field of the ``<side>``-row item at ``display_order``.

    AAMA's display elements are positional-by-display_order, not by list
    index (matching MemberClicks contract used by NORA). ``side`` is
    'left' / 'right' / 'top' / 'bottom'.
    """
    for it in record.get(side) or []:
        if isinstance(it, dict) and it.get("display_order") == display_order:
            return _coerce_str(it.get("html"))
    return None


def _build_source_url(rec: dict) -> str:
    """Synthesize the per-practitioner URL.

    MemberClicks profile detail pages are member-only (403 unauth), so
    no public detail URL exists. We append the stable profile id to the
    Patient-Referral-Directory URL as a fragment — fragments are
    invariant across re-runs and uniquely identify the practitioner.
    """
    rid = _coerce_str(rec.get("id"))
    if not rid:
        return f"{PATIENT_REFERRAL_URL}#/profile/unknown"
    return f"{PATIENT_REFERRAL_URL}#/profile/{rid}"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: AAMA display-element dict -> NormalizedPractitionerRow.

    Returns None if no usable name can be recovered from the title.
    """
    title = _coerce_str(rec.get("title"))
    if not title:
        return None
    name = _name_from_title(title)
    credentials = _credentials_from_title(title)
    if not name:
        return None

    # left[0] = address block.
    address1, city, state, postal, country = _split_address(
        _side_item(rec, "left", 0)
    )

    # left[1] = "<strong>Phone:</strong> <number>".
    phone_raw = _strip_label_prefix(_side_item(rec, "left", 1))
    phone = phone_raw if (phone_raw and _looks_like_phone(phone_raw)) else None

    # right[0] = "<strong>Specialty:</strong> <comma list>".  Append it
    # to the credentials field so the downstream search-by-specialty
    # surface has the practitioner's clinical-area string available.
    specialty = _strip_label_prefix(_side_item(rec, "right", 0))
    if specialty:
        existing = (credentials or "").lower()
        # Append the whole "specialty" string only if no fragment is
        # already there — most credentials are tight acronyms, so this
        # near-always appends.
        if specialty.lower() not in existing:
            credentials = (
                f"{credentials}, {specialty}" if credentials else specialty
            )

    # right[1] = website (may be blank, plain URL, or wrapped <a>).
    website = _normalize_website(_side_item(rec, "right", 1))

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="AAMA",
        source_url=_build_source_url(rec),
        fellowship_level=_is_fellowship(rec),
        practice_name=None,  # AAMA doesn't expose a separate practice slot.
        credentials=credentials,
        phone=phone,
        email=None,  # AAMA never publishes member emails in the public locator.
        website=website,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes an AAMA search response (dict with a ``results``
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
