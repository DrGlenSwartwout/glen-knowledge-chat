"""Unit tests for the AAMA (American Academy of Medical Acupuncture) adapter.

AAMA publishes its directory through a MemberClicks
``ui-directory-search/v2`` JSON API (the same vendor pattern NORA uses,
but with different display-element positions — AAMA puts data in
``left`` / ``right`` whereas NORA used ``top``). Fixtures here are real
responses captured 2026-05-27:

- aama_search_form.json         — response from the public
                                  get-directory-search-form endpoint,
                                  carrying the ``directory_search_id``
                                  (=2002770 at capture time).
- aama_results_page_1.json      — search-directory page 1 of 25. Covers
                                  canonical MD, MD+DABMA, MD+DO
                                  records, US addresses with zip+4 and
                                  4-digit zip, multi-line addresses.
- aama_results_page_5.json      — page 5 of 25. Carries the canonical
                                  FAAMA case (Susan Clemens, MD, FAAMA),
                                  the duplicated-credentials case
                                  (Christenson — ``MD, LAc, MD, LAc, FAAMA``),
                                  and the multi-credential
                                  Clearfield (``DO, FAAMA, HMD, FAAFRM``).
- aama_results_page_10.json     — page 10 of 25. Carries the
                                  empty-address record (only one in
                                  the live set is on page 18 actually,
                                  but page 10 has its own credential
                                  matrix variety including LAc and
                                  the Puerto Rico ``PR 00988-8908`` zip).

These three pages cover the credential matrix (DABMA, FAAMA, both,
neither), the address matrix (zip+4, 4-digit RI zip 2806, PR zip),
the website-presence matrix (none / bare domain / scheme'd URL), and
the duplicate-credentials de-dupe case.

AAMA disambiguation — this is the American Academy of Medical
Acupuncture (MDs/DOs doing acupuncture), NOT the unrelated American
Association of Medical Assistants. Verified at capture time via the
public ``/find-an-acupuncturist/`` page on medicalacupuncture.org which
literally reads "American Academy of Medical Acupuncture (AAMA)" and
links to the MemberClicks directory at
``https://medacu.memberclicks.net/patient-referral-directory#/``.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.aama import (  # noqa: E402
    parse_directory_json,
    _build_source_url,
    _credentials_from_title,
    _dedupe_preserve_order,
    _is_fellowship,
    _looks_like_credential,
    _looks_like_phone,
    _name_from_title,
    _normalize_website,
    _side_item,
    _split_address,
    _strip_credentials,
    _strip_label_prefix,
    LOCKED_SPECIALTIES,
)


def _load(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_page_1_returns_full_batch():
    """Page 1 has 10 search results, every one with a non-empty title —
    adapter must yield 10 rows."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    assert len(rows) == 10


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec across every
    record on every fixture page."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    rows += parse_directory_json(_load("aama_results_page_5.json"))
    rows += parse_directory_json(_load("aama_results_page_10.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "AAMA"
        assert r.specialties == ["medical_acupuncture", "acupuncture_tcm", "holistic_health"]
        # source_url is always populated (dedup key for ON CONFLICT upsert).
        assert r.source_url
        assert r.source_url.startswith(
            "https://medacu.memberclicks.net/patient-referral-directory#/profile/"
        )


def test_locked_specialties_module_constant():
    """LOCKED_SPECIALTIES exposes the locked 3-tag list per spec."""
    assert LOCKED_SPECIALTIES == [
        "medical_acupuncture",
        "acupuncture_tcm",
        "holistic_health",
    ]


def test_spot_check_abla_yao_full_fields():
    """First record on page 1 — Shiyi Abla-Yao MD DABMA in Lewisburg PA.
    Validates US address parsing, phone extraction (with the
    ``<strong>Phone:</strong>`` prefix stripped), DABMA fellowship
    detection, and specialty append."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    abla = next(r for r in rows if "Abla-Yao" in r.name)
    assert abla.name == "Shiyi Abla-Yao"
    assert abla.city == "Lewisburg"
    assert abla.state == "PA"
    assert abla.postal == "17837"
    assert abla.country == "US"
    assert abla.phone == "(570) 522-2000"
    assert abla.fellowship_level is True
    # Credentials include the title-degrees + DABMA + the specialty.
    assert abla.credentials is not None
    assert "MD" in abla.credentials
    assert "DABMA" in abla.credentials
    assert "Anesthesiology" in abla.credentials


def test_spot_check_abramson_website_normalized():
    """Robert J. Abramson — has http://www.robertabramsonmd.com in right[1].
    Website with explicit scheme must be kept as-is, not double-prefixed."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    abramson = next(r for r in rows if "Abramson" in r.name)
    assert abramson.name == "Robert J. Abramson"
    assert abramson.website == "http://www.robertabramsonmd.com"
    # No DABMA / FAAMA in the title — fellowship_level stays False.
    assert abramson.fellowship_level is False


def test_bare_domain_website_gets_https_scheme():
    """Diane Alligood has 'www.allimedacu.com' (no scheme) in right[1].
    The normalizer must prepend https://."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    alligood = next(r for r in rows if "Alligood" in r.name)
    assert alligood.website == "https://www.allimedacu.com"


def test_no_website_field_returns_none():
    """Ina J. Amber has no website (right[1] is empty). Row's website
    field must be None, not a stray empty string."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    amber = next(r for r in rows if "Amber" in r.name)
    assert amber.website is None


def test_fellowship_dabma_sets_true():
    """DABMA in the title sets fellowship_level=True. DABMA = Diplomate
    of the American Board of Medical Acupuncture, the "Board Certified"
    tier the Wave C spec calls out."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    dabma_rows = [r for r in rows if "DABMA" in (r.credentials or "")]
    assert dabma_rows
    for r in dabma_rows:
        assert r.fellowship_level is True


def test_fellowship_faama_sets_true():
    """FAAMA in the title (Fellow of AAMA) also sets fellowship_level=True.
    Susan Clemens MD FAAMA is a canonical FAAMA-only record on page 5."""
    rows = parse_directory_json(_load("aama_results_page_5.json"))
    clemens = next(r for r in rows if "Clemens" in r.name)
    assert clemens.fellowship_level is True
    assert clemens.credentials is not None
    assert "FAAMA" in clemens.credentials


def test_fellowship_false_when_no_dabma_or_faama():
    """Records without DABMA or FAAMA anywhere in the title must keep
    fellowship_level=False — that's the default for plain MD/DO members
    who haven't taken the board exam or earned fellow status."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    for r in rows:
        creds = (r.credentials or "").upper()
        if "DABMA" not in creds and "FAAMA" not in creds:
            assert r.fellowship_level is False


def test_duplicate_credentials_deduped():
    """Elizabeth Chen Christenson's title is
    ``"Elizabeth Chen Christenson, MD, LAc, MD, LAc, FAAMA"`` — the
    parser must de-dupe so MD/LAc only appear once each."""
    rows = parse_directory_json(_load("aama_results_page_5.json"))
    christenson = next(r for r in rows if "Christenson" in r.name)
    # Each of MD and LAc must appear exactly once in the credentials
    # string (case-insensitive). FAAMA must be there too.
    creds_upper = (christenson.credentials or "").upper()
    # Split out only the title-credential block (the specialty was
    # appended on the end with comma + value).
    title_part = creds_upper.split(",")
    # Count tokens that exactly equal MD / LAC after stripping spaces.
    md_count = sum(1 for c in title_part if c.strip().rstrip(".") == "MD")
    lac_count = sum(1 for c in title_part if c.strip().rstrip(".") == "LAC")
    assert md_count == 1, f"expected 1 MD, got {md_count} in {creds_upper!r}"
    assert lac_count == 1, f"expected 1 LAc, got {lac_count} in {creds_upper!r}"
    assert "FAAMA" in creds_upper
    assert christenson.fellowship_level is True


def test_multi_line_address_parsed():
    """Diane Alligood's address has a practice name on line1 + street on
    line2, then 'Greenville, NC 27858' as the city-line. Both lines must
    end up in address1, comma-joined."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    alligood = next(r for r in rows if "Alligood" in r.name)
    assert alligood.address1 is not None
    assert "Alligood Medical Acupuncture PLLC" in alligood.address1
    assert "1801 Charles Blvd" in alligood.address1
    assert alligood.city == "Greenville"
    assert alligood.state == "NC"
    assert alligood.postal == "27858"


def test_short_ri_zip_preserved():
    """Susan Clemens' Rhode Island address has the 4-digit zip ``2806``
    (legitimate — RI's leading zero is sometimes dropped in CMS forms).
    The parser must keep it as-is, not pad or drop it."""
    rows = parse_directory_json(_load("aama_results_page_5.json"))
    clemens = next(r for r in rows if "Clemens" in r.name)
    assert clemens.state == "RI"
    assert clemens.postal == "2806"


def test_puerto_rico_zip_plus_four_preserved():
    """Humberto Herrera's PR address has the zip+4 form '00988-8908'.
    The full string must come through intact."""
    rows = parse_directory_json(_load("aama_results_page_10.json"))
    herrera = next(r for r in rows if "Herrera" in r.name)
    assert herrera.state == "PR"
    assert herrera.postal == "00988-8908"


def test_email_always_none():
    """AAMA never publishes member emails in the public locator. Every
    row's email field must therefore be None, not a stray empty string."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    rows += parse_directory_json(_load("aama_results_page_5.json"))
    for r in rows:
        assert r.email is None


def test_practice_name_always_none():
    """AAMA's payload has no separate practice-name slot — the practice
    name is folded into address1's line1. ``practice_name`` must be
    None for every row."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    rows += parse_directory_json(_load("aama_results_page_5.json"))
    for r in rows:
        assert r.practice_name is None


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("aama_results_page_1.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_uses_profile_id_fragment():
    """Source URL must be the public patient-referral-directory page +
    the stable MemberClicks profile id as a fragment."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    abla = next(r for r in rows if "Abla-Yao" in r.name)
    assert (
        abla.source_url
        == "https://medacu.memberclicks.net/patient-referral-directory#/profile/2007179612"
    )


def test_parser_accepts_full_response_dict():
    """The parser must accept the full search-directory response (a dict
    with 'results' inside), not just a bare list of results."""
    payload = _load("aama_results_page_1.json")
    assert isinstance(payload, dict)
    assert "results" in payload
    rows = parse_directory_json(payload)
    assert len(rows) == 10


def test_parser_accepts_bare_results_list():
    """The parser must also accept a bare list of result dicts — that's
    what fetch_all_directory_records() returns after concatenating pages."""
    payload = _load("aama_results_page_1.json")
    rows_from_dict = parse_directory_json(payload)
    rows_from_list = parse_directory_json(payload["results"])
    assert len(rows_from_dict) == len(rows_from_list)
    assert [r.source_url for r in rows_from_dict] == [
        r.source_url for r in rows_from_list
    ]


def test_parser_accepts_json_string():
    """A raw JSON string of the full response is also valid input."""
    raw = (FIXTURE_DIR / "aama_results_page_5.json").read_text()
    rows = parse_directory_json(raw)
    assert len(rows) == 10


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


def test_parser_skips_records_with_empty_title():
    """A record with no title (no name) must be dropped, not converted
    to a name-less row."""
    rows = parse_directory_json([{"id": 999, "title": "", "left": [], "right": []}])
    assert rows == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_is_fellowship_matches_dabma_or_faama_only():
    """_is_fellowship looks for literal DABMA or FAAMA word tokens in the
    title. Must NOT match substrings of other credentials."""
    assert _is_fellowship({"title": "Thomas E. Archie, MD, DABMA"}) is True
    assert _is_fellowship({"title": "Susan Clemens, MD, FAAMA"}) is True
    assert _is_fellowship({"title": "Norman G. Zavela, MD, FAAMA"}) is True
    # Case-insensitive.
    assert _is_fellowship({"title": "dabma lowercase"}) is True
    assert _is_fellowship({"title": "faama lowercase"}) is True
    # Plain MD/DO — no fellowship.
    assert _is_fellowship({"title": "Yang Ahn, MD"}) is False
    assert _is_fellowship({"title": "Diane M. Aslanis, DO"}) is False
    # Other AAMA-adjacent credentials don't qualify (DABMA-or-FAAMA is the rule).
    assert _is_fellowship({"title": "Joe Smith, MD, FAAFP"}) is False
    assert _is_fellowship({"title": "Jane Doe, MD, LAc"}) is False
    # Empty / missing.
    assert _is_fellowship({"title": ""}) is False
    assert _is_fellowship({}) is False


def test_strip_credentials_basic():
    """Standard comma-delimited title: name, then comma-separated creds."""
    name, creds = _strip_credentials("Thomas E. Archie, MD, DABMA")
    assert name == "Thomas E. Archie"
    assert creds is not None
    assert "MD" in creds and "DABMA" in creds


def test_strip_credentials_no_creds():
    """A bare name with no commas — creds is None and name passes through."""
    name, creds = _strip_credentials("John Doe")
    assert name == "John Doe"
    assert creds is None


def test_strip_credentials_dedupes_repeats():
    """Title with repeated tokens (Christenson case) gets a deduped
    credential string."""
    name, creds = _strip_credentials(
        "Elizabeth Chen Christenson, MD, LAc, MD, LAc, FAAMA"
    )
    assert name == "Elizabeth Chen Christenson"
    # Each token appears at most once in the credential block.
    parts = [c.strip().upper().rstrip(".") for c in (creds or "").split(",")]
    assert parts.count("MD") == 1
    assert parts.count("LAC") == 1
    assert "FAAMA" in parts


def test_strip_credentials_empty():
    """Empty input returns empty + None."""
    name, creds = _strip_credentials("")
    assert name == ""
    assert creds is None


def test_dedupe_preserve_order_is_case_insensitive():
    """De-dupe helper drops case-insensitive duplicates while keeping
    the first occurrence's casing + relative order."""
    out = _dedupe_preserve_order(["MD", "LAc", "MD", "LAC", "FAAMA"])
    assert out == ["MD", "LAc", "FAAMA"]
    assert _dedupe_preserve_order([]) == []
    assert _dedupe_preserve_order(["A"]) == ["A"]


def test_looks_like_credential_classifies_short_uppercase_tokens():
    """Credential-detector accepts short uppercase tokens + known
    typed-set members and rejects multi-word title-case English."""
    assert _looks_like_credential("MD") is True
    assert _looks_like_credential("DABMA") is True
    assert _looks_like_credential("FAAMA") is True
    assert _looks_like_credential("LAc") is True  # known-token, mixed case OK
    assert _looks_like_credential("Family Practice") is False  # has space
    assert _looks_like_credential("Internal Medicine") is False
    assert _looks_like_credential("") is False
    # >12 chars — never a credential.
    assert _looks_like_credential("AAAAAAAAAAAAA") is False


def test_normalize_website_adds_scheme():
    """Bare domain in right[1] must get https:// prepended; explicit
    scheme is preserved; junk / empty values return None."""
    assert _normalize_website("www.allimedacu.com") == "https://www.allimedacu.com"
    assert _normalize_website("susanclemensmd.com") == "https://susanclemensmd.com"
    assert _normalize_website("http://example.com") == "http://example.com"
    assert _normalize_website("https://example.com") == "https://example.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None
    # Non-URL strings (with a space and no dot) are rejected so we don't
    # promote stray text to a website.
    assert _normalize_website("not a website") is None


def test_strip_label_prefix_removes_strong_label():
    """The HTML wrapper '<strong>Label:</strong> X' must collapse to 'X'."""
    assert _strip_label_prefix("<strong>Phone:</strong> 555-1234") == "555-1234"
    assert _strip_label_prefix("<strong>Specialty:</strong> Cardiology") == "Cardiology"
    # Case-insensitive on the tag name.
    assert _strip_label_prefix("<STRONG>Phone:</STRONG> 555") == "555"
    # Already-clean strings pass through unchanged.
    assert _strip_label_prefix("plain string") == "plain string"
    # Empty / None.
    assert _strip_label_prefix("") is None
    assert _strip_label_prefix(None) is None


def test_split_address_us_full():
    """US address with state + 5-digit zip — full parse."""
    addr1, city, state, postal, country = _split_address(
        "One Hospital Drive, \n<br />\nLewisburg, PA 17837"
    )
    assert addr1 == "One Hospital Drive"
    assert city == "Lewisburg"
    assert state == "PA"
    assert postal == "17837"
    assert country == "US"


def test_split_address_two_line_with_practice():
    """Multi-line address — line1 is practice name, line2 is street.
    Both must end up in address1, joined."""
    addr1, city, state, postal, country = _split_address(
        "Alligood Medical Acupuncture PLLC, \n1801 Charles Blvd, Ste 109<br />\nGreenville, NC 27858"
    )
    assert addr1 is not None
    assert "Alligood Medical Acupuncture PLLC" in addr1
    assert "1801 Charles Blvd" in addr1
    assert city == "Greenville"
    assert state == "NC"
    assert postal == "27858"


def test_split_address_zip_plus_four():
    """Zip+4 form '06030-2918' must come through intact as the postal."""
    addr1, city, state, postal, country = _split_address(
        "Department of Family Medicine, \n263 Farmington Ave, MC-2918<br />\nFarmington, CT 06030-2918"
    )
    assert state == "CT"
    assert postal == "06030-2918"


def test_split_address_empty_placeholder():
    """The fully-empty placeholder ', \\n<br />\\n,  ' must produce
    all-None address fields and not crash."""
    addr1, city, state, postal, country = _split_address(", \n<br />\n,  ")
    assert addr1 is None
    assert city is None
    assert state is None
    assert postal is None
    assert country == "US"


def test_split_address_handles_none():
    """A None block (no left[0]) must produce all-None fields with
    country defaulting to 'US'."""
    addr1, city, state, postal, country = _split_address(None)
    assert addr1 is None and city is None and state is None and postal is None
    assert country == "US"


def test_looks_like_phone_classifies_numeric_strings():
    """_looks_like_phone gates the phone field — must accept obvious
    phone shapes and reject plain text."""
    assert _looks_like_phone("(570) 522-2000") is True
    assert _looks_like_phone("631 324 9492") is True
    assert _looks_like_phone("252-623-7815") is True
    assert _looks_like_phone("7035209703") is True
    assert _looks_like_phone("Mind Eye Institute") is False
    assert _looks_like_phone("") is False
    # Too short to be a phone (would have <7 digits).
    assert _looks_like_phone("123") is False


def test_side_item_indexes_by_display_order_not_list_position():
    """AAMA's display elements carry explicit display_order — the parser
    must index off display_order, not list position, so a re-ordered
    API response still yields the right field for each slot."""
    rec = {
        "left": [
            {"display_order": 1, "html": "phone-html"},
            {"display_order": 0, "html": "addr-html"},
        ],
        "right": [
            {"display_order": 1, "html": "website-html"},
            {"display_order": 0, "html": "specialty-html"},
        ],
    }
    assert _side_item(rec, "left", 0) == "addr-html"
    assert _side_item(rec, "left", 1) == "phone-html"
    assert _side_item(rec, "right", 0) == "specialty-html"
    assert _side_item(rec, "right", 1) == "website-html"
    assert _side_item(rec, "left", 99) is None
    assert _side_item(rec, "top", 0) is None  # AAMA never populates top


def test_specialty_appended_to_credentials():
    """right[0]='<strong>Specialty:</strong> X' must be appended to the
    credentials field so the search-by-specialty surface has the
    practitioner's clinical-area string available."""
    rows = parse_directory_json(_load("aama_results_page_1.json"))
    abla = next(r for r in rows if "Abla-Yao" in r.name)
    # Title was 'Shiyi Abla-Yao, MD, DABMA' and right[0] specialty was
    # 'Anesthesiology, Pain Management'.
    creds = abla.credentials or ""
    assert "Anesthesiology" in creds
    assert "Pain Management" in creds


def test_build_source_url_uses_profile_id():
    """Source URL is the public patient-referral-directory URL + profile id."""
    url = _build_source_url({"id": 12345})
    assert (
        url
        == "https://medacu.memberclicks.net/patient-referral-directory#/profile/12345"
    )


def test_build_source_url_handles_missing_id():
    """When no id is present, source_url falls back to a deterministic
    placeholder so the upsert doesn't blow up — but in practice every
    record has an id."""
    url = _build_source_url({})
    assert (
        url
        == "https://medacu.memberclicks.net/patient-referral-directory#/profile/unknown"
    )


def test_name_from_title_and_credentials_from_title_are_consistent():
    """The two title-decomposition helpers must agree about where the
    name ends and the credentials begin."""
    name = _name_from_title("Thomas E. Archie, MD, DABMA")
    creds = _credentials_from_title("Thomas E. Archie, MD, DABMA")
    assert name == "Thomas E. Archie"
    assert creds and "MD" in creds and "DABMA" in creds
    assert _name_from_title("") == ""
    assert _credentials_from_title("") is None
