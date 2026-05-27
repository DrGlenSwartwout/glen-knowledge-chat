"""Unit tests for the IABDM (International Academy of Biological Dentistry
and Medicine) adapter.

IABDM publishes its directory through a single GeoDirectory REST endpoint
that returns JSON pages of ``gd_place`` records. Fixtures here are real
responses captured 2026-05-26:

- iabdm_places_page_1.json   — page 1 @ per_page=100 (US-heavy, mix of
                               Master / Fellow / Certified-only dentists;
                               first record is the canonical Teresa Scott
                               Master entry).
- iabdm_places_page_2.json   — page 2 @ per_page=100 (international skew:
                               Spain, UK, Canada, Turkey, UAE, Romania —
                               most lack the structured member_first_name
                               fields and require title-fallback).
- iabdm_places_page_5.json   — page 5 @ per_page=100 (the tail, 71 recs:
                               hygienists with cat='1606', a Canadian
                               RDH, and an Australian).

These three cover the credential matrix (Master / Fellow / Cert-only /
Precertified), the category matrix (Dentist 1605 / Hygienist 1606), the
member-name extraction matrix (structured first/last vs title-fallback),
and the country spread (US with state+zip, Canadian postal codes,
no-state international).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.iabdm import (  # noqa: E402
    parse_directory_json,
    _build_source_url,
    _country_iso2,
    _credentials_from_title,
    _is_fellowship,
    _is_yes,
    _name_from_title,
    _normalize_website,
    _strip_credentials,
)


def _load(name: str) -> list:
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_page_1_returns_full_batch():
    """Page 1 has 100 raw gd_place records, all with a usable name (either
    structured or in the title) — adapter must produce 100 rows."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    assert len(rows) == 100


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    rows += parse_directory_json(_load("iabdm_places_page_2.json"))
    rows += parse_directory_json(_load("iabdm_places_page_5.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "IABDM"
        assert r.specialties == ["biological", "dental"]
        # source_url is always populated (dedup key)
        assert r.source_url
        assert r.source_url.startswith("https://iabdm.org/")


def test_spot_check_teresa_scott_master_full_fields():
    """First record on page 1 — Teresa Scott, Master IABDM, full US
    address. Validates name + address + contact extraction + fellowship
    detection for the canonical Master tier."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    scott = next(r for r in rows if "Teresa Scott" in r.name)

    assert scott.name == "Teresa Scott"
    assert scott.practice_name == "Holistic Dental Associates"
    assert scott.city == "Spring"
    assert scott.state == "Texas"
    assert scott.postal == "77379"
    assert scott.country == "US"
    assert scott.address1 == "6334 Farm to Market 2920 ste 250, Spring Tx 77379"
    assert scott.phone == "281-655-9175"
    assert scott.email == "info@holisticdentalassociates.com"
    assert scott.website == "http://holisticdentalassociates.com"
    # Master = fellowship-tier
    assert scott.fellowship_level is True
    # Credentials must come from the title (DDS, MAIBDM, ...) — NOT from
    # member_position. The raw record lists member_position='RDH' (the
    # account holder is a hygienist registering on behalf of the dentist),
    # but Teresa is a Master MIABDM dentist with DDS. Emitting
    # "RDH, DDS, ..." is anatomically wrong; the title-extracted creds win.
    assert scott.credentials
    assert "RDH" not in scott.credentials


def test_teresa_scott_title_credentials_win_over_member_position():
    """Title carries DDS, MAIBDM, AIAOMT — those must end up in the
    credentials string. member_position='RDH' is the account holder's
    role and must be suppressed because the title already has a dentist
    degree (DDS)."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    scott = next(r for r in rows if "Teresa Scott" in r.name)
    assert scott.credentials
    assert "DDS" in scott.credentials
    assert "MAIBDM" in scott.credentials


def test_hygienist_only_record_keeps_member_position():
    """When the title has NO dentist/doctor degree, member_position is
    the legitimate practitioner credential and must be preserved. Synthetic
    record because the production fixtures all have title-side degrees."""
    rec = {
        "id": 99999,
        "slug": "jane-hygienist",
        "link": "https://iabdm.org/places/hygienist/jane-hygienist/",
        "title": {"raw": "Jane Hygienist", "rendered": "Jane Hygienist"},
        "member_first_name": "Jane",
        "member_last_name": "Hygienist",
        "member_position": "RDH",
        "country": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].credentials is not None
    assert "RDH" in rows[0].credentials


def test_fellowship_detection_fellow_only():
    """A Fellow IABDM record (is_fellow_member=yes, certified_master may
    be 'yes' too) must mark fellowship_level=True. Charles Cuprill on
    page 1 is the canonical Fellow."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    cuprill = next(r for r in rows if "Charles Cuprill" in r.name)
    assert cuprill.fellowship_level is True


def test_fellowship_set_for_certified_member():
    """Certified Member (without Fellow/Master) IS fellowship-tier — IABDM's
    Certified Member is the structural analogue of IAOMT's Accredited
    Member, the entry-level vetted-credentials tier. Bhumija Gupta on
    page 1 is the canonical cert-only dentist (certifiedmember='yes',
    no Fellow, no Master)."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    gupta = next(r for r in rows if "Bhumija Gupta" in r.name)
    assert gupta.fellowship_level is True


def test_fellowship_not_set_when_no_tier_flag():
    """Records with NO tier flags at all (no Master / Fellow / Certified
    Member) are NOT fellowship-tier. Tugba Duymaz on page 2 has all four
    yes/no flags resolving to None."""
    rows = parse_directory_json(_load("iabdm_places_page_2.json"))
    duymaz = next(r for r in rows if "Tugba Duymaz" in r.name)
    assert duymaz.fellowship_level is False


def test_fellowship_not_set_for_precertified():
    """Precertified-only practitioners (precertified='yes' but no
    Fellow/Master/CertifiedMember flag) are NOT fellowship-tier — they
    are on the IABDM ladder below the entry-level vetted tier. The real
    fixtures don't include this state (everyone precertified in the
    real data has since been promoted to certifiedmember=yes too), so
    we use a synthetic record."""
    rec = {
        "id": 66666,
        "slug": "alice-precert",
        "link": "https://iabdm.org/places/dentist-1/alice-precert/",
        "title": {"raw": "Alice Precert, DDS", "rendered": "Alice Precert, DDS"},
        "member_first_name": "Alice",
        "member_last_name": "Precert",
        "precertified": {"raw": "yes", "rendered": "Yes"},
        "certifiedmember": {"raw": None, "rendered": None},
        "is_fellow_member": {"raw": None, "rendered": None},
        "certified_master": {"raw": None, "rendered": None},
        "country": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].fellowship_level is False


def test_fellowship_not_set_for_is_precertified_member_only():
    """is_precertified_member='yes' alone (no Master / Fellow / Certified
    Member) must NOT mark fellowship_level=True. Synthetic record because
    the fixtures don't include the is_precertified_member field name."""
    rec = {
        "id": 88888,
        "slug": "joe-precert",
        "link": "https://iabdm.org/places/dentist-1/joe-precert/",
        "title": {"raw": "Joe Precert, DDS", "rendered": "Joe Precert, DDS"},
        "member_first_name": "Joe",
        "member_last_name": "Precert",
        "is_precertified_member": {"raw": "yes", "rendered": "Yes"},
        "country": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].fellowship_level is False


def test_fellowship_not_set_for_is_hygienist_only():
    """is_hygienist='yes' alone (no clinical-tier flag) must NOT mark
    fellowship_level=True. Hygienist status is a role, not a fellowship
    tier. Synthetic record because the fixtures don't carry an
    is_hygienist field name."""
    rec = {
        "id": 77777,
        "slug": "jane-hyg",
        "link": "https://iabdm.org/places/hygienist/jane-hyg/",
        "title": {"raw": "Jane Hyg, RDH", "rendered": "Jane Hyg, RDH"},
        "member_first_name": "Jane",
        "member_last_name": "Hyg",
        "is_hygienist": {"raw": "yes", "rendered": "Yes"},
        "country": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].fellowship_level is False


def test_international_record_maps_country_to_iso2():
    """Canadian record on page 2 — country mapped to 'CA', Canadian
    postal code preserved as-is in postal."""
    rows = parse_directory_json(_load("iabdm_places_page_2.json"))
    lam = next(r for r in rows if "Jessica Lam" in r.name)
    assert lam.country == "CA"
    assert lam.state == "British Columbia"
    assert lam.postal == "V6J 1P3"
    assert lam.city == "Vancouver"


def test_international_no_state_kept_intact():
    """Spanish record on page 2 has no region — must not crash, country
    mapped to 'ES'."""
    rows = parse_directory_json(_load("iabdm_places_page_2.json"))
    pernas = next(r for r in rows if "Acosta Pernas" in r.name)
    assert pernas.country == "ES"
    assert pernas.state is None
    assert pernas.city == "Madrid"
    assert pernas.postal == "28006"


def test_name_fallback_from_title_when_structured_blank():
    """International records often have blank member_first_name/last_name.
    The adapter must fall back to parsing the post title — UK entry
    'Anca Condur' has no structured fields but the title carries the name."""
    rows = parse_directory_json(_load("iabdm_places_page_2.json"))
    condur = next(r for r in rows if "Anca Condur" in r.name)
    assert condur.name == "Anca Condur"
    assert condur.country == "GB"
    assert condur.city == "London"


def test_credentials_extracted_from_paren_title():
    """A '(DMD)'-style title carries the degree as credentials. The
    Romanian record 'magdalena dina (DMD)' is the canonical case."""
    rows = parse_directory_json(_load("iabdm_places_page_2.json"))
    dina = next(r for r in rows if "magdalena dina" in r.name)
    assert dina.credentials == "DMD"
    assert dina.country == "RO"


def test_hygienist_category_handled():
    """Hygienist records (default_category='1606') are valid practitioners
    and must produce rows. Page 5 contains the canonical Canadian RDH
    Nicole Brunelle."""
    rows = parse_directory_json(_load("iabdm_places_page_5.json"))
    brunelle = next(r for r in rows if "Nicole Brunelle" in r.name)
    assert brunelle.country == "CA"
    assert brunelle.credentials
    assert "RDH" in brunelle.credentials
    # Hygienist isn't a fellowship tier in IABDM — just dental hygiene.
    assert brunelle.fellowship_level is False


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("iabdm_places_page_1.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # All distinct (no two practitioners share a source_url).
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_uses_canonical_link_when_present():
    """When the API record carries a ``link``, source_url derives from it
    so the URL actually resolves at iabdm.org/places/..."""
    rows = parse_directory_json(_load("iabdm_places_page_1.json"))
    scott = next(r for r in rows if "Teresa Scott" in r.name)
    # Canonical link is /places/dentist-1/teresa-scott-dds/ ; we append
    # #<id> as a stable tiebreak fragment.
    assert scott.source_url == "https://iabdm.org/places/dentist-1/teresa-scott-dds/#6031"


def test_parser_accepts_json_string():
    """A raw JSON string of the response list is also valid input."""
    raw = (FIXTURE_DIR / "iabdm_places_page_5.json").read_text()
    rows = parse_directory_json(raw)
    # Page 5 has 71 records, all parseable.
    assert len(rows) == 71


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing — the upstream WP REST API should never send these,
    but the parser is the last line of defense."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_is_yes_unwraps_geodir_dict_shape():
    """GeoDirectory wraps yes/no fields in {'raw': 'yes', 'rendered': 'Yes'}.
    _is_yes must extract the raw form."""
    assert _is_yes({"raw": "yes", "rendered": "Yes"}) is True
    assert _is_yes({"raw": "no", "rendered": "No"}) is False
    assert _is_yes({"raw": None, "rendered": None}) is False
    assert _is_yes("yes") is True
    assert _is_yes("YES") is True  # case-insensitive
    assert _is_yes("") is False
    assert _is_yes(None) is False


def test_is_fellowship_flag_matrix():
    """Master OR Fellow OR Certified Member qualifies; Precertified or
    hygienist-only or empty does not. Certified Member is included
    because it is IABDM's structural analogue of IAOMT's Accredited
    Member tier (entry-level vetted credentials)."""
    yes = {"raw": "yes", "rendered": "Yes"}
    no = {"raw": None, "rendered": None}
    assert _is_fellowship({"certified_master": yes}) is True
    assert _is_fellowship({"is_fellow_member": yes}) is True
    assert _is_fellowship({"certified_master": yes, "is_fellow_member": yes}) is True
    # Cert Member alone NOW qualifies (analogue of IAOMT Accredited Member).
    assert _is_fellowship({"certifiedmember": yes}) is True
    # Both field-name variants are accepted defensively.
    assert _is_fellowship({"certified_member": yes}) is True
    # Precertified-only / hygienist-only / empty / explicit-no still don't.
    assert _is_fellowship({"precertified": yes}) is False
    assert _is_fellowship({"is_precertified_member": yes}) is False
    assert _is_fellowship({"is_hygienist": yes}) is False
    assert _is_fellowship({"certified_master": no, "is_fellow_member": no}) is False
    assert _is_fellowship({}) is False


def test_normalize_website_adds_scheme():
    assert _normalize_website("holisticdentalassociates.com") == "https://holisticdentalassociates.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website("http://x.com") == "http://x.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None
    # GeoDirectory occasionally sends the wrapped-dict shape for url fields
    assert _normalize_website({"raw": "x.com", "rendered": "x.com"}) == "https://x.com"


def test_country_iso2_canonicalizes_common_names():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("united kingdom") == "GB"
    assert _country_iso2("Canada") == "CA"
    # Some IABDM records use "US" / "CA" as the country (raw ISO2 in the
    # country field instead of full name); the map handles those too.
    assert _country_iso2("US") == "US"
    assert _country_iso2("CA") == "CA"
    assert _country_iso2("United Arab Emirates") == "AE"
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None


def test_strip_credentials_handles_paren_form():
    """'(DMD)' or '(DDS)' style suffixes — typical of international entries."""
    name, creds = _strip_credentials("Anca Condur (DDS)")
    assert name == "Anca Condur"
    assert creds == "DDS"

    name, creds = _strip_credentials("magdalena dina (DMD)")
    assert name == "magdalena dina"
    assert creds == "DMD"


def test_strip_credentials_handles_comma_form():
    """Standard 'Name, Cred1, Cred2' titles."""
    name, creds = _strip_credentials("Katie To, DDS, FAGD, DSD, MIABDM")
    assert name == "Katie To"
    assert "DDS" in creds
    assert "MIABDM" in creds


def test_strip_credentials_preserves_honorific():
    name, _creds = _strip_credentials("Dr. Jessica Lam")
    assert name == "Dr. Jessica Lam"


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Anca Condur")
    assert name == "Anca Condur"
    assert creds is None


def test_name_from_title_strips_creds():
    assert _name_from_title("Anca Condur (DDS)") == "Anca Condur"
    assert _name_from_title("Katie To, DDS, FAGD") == "Katie To"
    assert _name_from_title("") == ""


def test_credentials_from_title():
    assert _credentials_from_title("Anca Condur (DDS)") == "DDS"
    assert _credentials_from_title("Anca Condur") is None
    assert _credentials_from_title("") is None


def test_build_source_url_prefers_link():
    """When the API gives us a canonical link, use it (with #id fragment)."""
    url = _build_source_url({
        "link": "https://iabdm.org/places/dentist-1/jane-doe/",
        "slug": "jane-doe",
        "id": 9999,
    })
    assert url == "https://iabdm.org/places/dentist-1/jane-doe/#9999"


def test_build_source_url_falls_back_to_slug():
    """No link -> synthesize /places/<slug>/."""
    url = _build_source_url({"slug": "jane-doe", "id": 9999})
    assert url == "https://iabdm.org/places/jane-doe/#9999"


def test_build_source_url_falls_back_to_id_only():
    """No link, no slug -> /places/place-<id>/ keeps URL stable."""
    url = _build_source_url({"id": 9999})
    assert url == "https://iabdm.org/places/place-9999/#9999"
