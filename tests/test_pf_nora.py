"""Unit tests for the NORA (Neuro-Optometric Rehabilitation Association)
adapter.

NORA publishes its directory through a MemberClicks Angular SPA backed by
the ``ui-directory-search/v2`` JSON API. Fixtures here are real responses
captured 2026-05-27:

- nora_search_form.json         — response from the public
                                  get-directory-search-form endpoint,
                                  including the ``directory_search_id``
                                  (=12133 at capture time).
- nora_results_page_1.json      — search-directory page 1 of 43; covers
                                  the canonical US + Canadian + UK +
                                  Australian addresses.
- nora_results_page_22.json     — search-directory page 22 of 43; contains
                                  Briana Larson, the canonical
                                  comma-delimited FNORA case
                                  (``Briana Larson, OD, FNORA, FOVDR, FAAO``).
- nora_results_page_35.json     — search-directory page 35 of 43; contains
                                  Kauser Sharieff, the canonical
                                  space-delimited FNORA case
                                  (``Kauser Sharieff OD FCOVD FNORA``).

These three pages cover the credential matrix (FNORA, FCOVD, FAAO, none),
the title-format matrix (comma-delimited credentials, space-delimited
credentials, no credentials), the address matrix (US state name, Canadian
province + Canadian postal, UK / Australia with no state, empty-address
``\\n,\\n,``), and the ``top[2]`` ambiguity (phone vs. practice-name vs.
blank).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.nora import (  # noqa: E402
    parse_directory_json,
    _build_source_url,
    _credentials_from_title,
    _is_fellowship,
    _looks_like_phone,
    _name_from_title,
    _normalize_website,
    _split_address,
    _strip_credentials,
    _top_item,
)


def _load(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_page_1_returns_full_batch():
    """Page 1 has 10 search results, every one with a non-empty title —
    adapter must yield 10 rows."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    assert len(rows) == 10


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec across every
    record on every fixture page."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    rows += parse_directory_json(_load("nora_results_page_22.json"))
    rows += parse_directory_json(_load("nora_results_page_35.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "NORA"
        assert r.specialties == ["rehabilitation", "eye_care"]
        # source_url is always populated (dedup key for ON CONFLICT upsert).
        assert r.source_url
        assert r.source_url.startswith(
            "https://nora.memberclicks.net/find-a-provider#/profile/"
        )


def test_spot_check_julie_steinhauer_full_fields():
    """First record on page 1 — Julie A. Steinhauer in Glen Carbon IL.
    Validates US address parsing, phone-vs-practice classification, and
    website normalization for a canonical record."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    steinhauer = next(r for r in rows if "Steinhauer" in r.name)
    assert steinhauer.name == "Julie A. Steinhauer"
    assert steinhauer.city == "Glen Carbon"
    assert steinhauer.state == "Illinois"
    assert steinhauer.postal == "62034"
    assert steinhauer.country == "US"
    assert steinhauer.phone == "6182881489"
    # No practice name (top[2] held the phone).
    assert steinhauer.practice_name is None
    assert steinhauer.website == "https://visionforlifeworks.com"
    # No FNORA in the title -> default false.
    assert steinhauer.fellowship_level is False
    # No credential markers in the title either.
    assert steinhauer.credentials is None


def test_canadian_record_country_mapped_to_ca():
    """Dipty Acharya in Listowel, Ontario — Canadian province name in the
    state slot must map country to 'CA' and preserve the Canadian
    postal code with its space ('N4W 1B4')."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    acharya = next(r for r in rows if "Acharya" in r.name)
    assert acharya.country == "CA"
    assert acharya.state == "Ontario"
    assert acharya.city == "Listowel"
    assert acharya.postal == "N4W 1B4"


def test_top2_practice_name_when_not_phone():
    """top[2] is ambiguous — phone (digits) OR practice name (alpha).
    Carla Adams' record has top[2]='Mind Eye Institute', which is not a
    phone number and must be classified as the practice name."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    adams = next(r for r in rows if r.name == "Carla Adams")
    assert adams.phone is None
    assert adams.practice_name == "Mind Eye Institute"


def test_empty_address_block_handled():
    """When NORA's address block is just '\\n,\\n,  ' (no real data), the
    parser must not crash and must produce a row with no address fields."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    aalberg = next(r for r in rows if "Aalberg" in r.name)
    assert aalberg.address1 is None
    assert aalberg.city is None
    assert aalberg.state is None
    assert aalberg.postal is None


def test_fellowship_comma_form_briana_larson():
    """``Briana Larson, OD, FNORA, FOVDR, FAAO`` — comma-delimited FNORA
    credential in the title must set fellowship_level=True. This is the
    canonical FNORA case in the comma form."""
    rows = parse_directory_json(_load("nora_results_page_22.json"))
    larson = next(r for r in rows if "Briana Larson" in r.name)
    assert larson.fellowship_level is True
    assert larson.credentials is not None
    assert "FNORA" in larson.credentials


def test_fellowship_space_form_kauser_sharieff():
    """``Kauser Sharieff OD FCOVD FNORA`` — space-delimited (no commas)
    FNORA credential in the title. The space-form credential stripper
    must still detect FNORA and set fellowship_level=True."""
    rows = parse_directory_json(_load("nora_results_page_35.json"))
    sharieff = next(r for r in rows if "Sharieff" in r.name)
    assert sharieff.fellowship_level is True
    assert sharieff.credentials is not None
    assert "FNORA" in sharieff.credentials
    # And the credential string captured all three trailing tokens.
    assert "OD" in sharieff.credentials
    assert "FCOVD" in sharieff.credentials


def test_fellowship_false_when_no_fnora_in_title():
    """Records without FNORA anywhere in the title must keep
    fellowship_level=False. NORA's only public credential signal is
    the title — no FNORA in title means no fellowship."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    for r in rows:
        if "FNORA" not in (r.credentials or ""):
            assert r.fellowship_level is False


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("nora_results_page_1.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_uses_profile_id_fragment():
    """Source URL must be the public find-a-provider page + the stable
    MemberClicks profile id as a fragment. The profile id is invariant
    across re-runs and uniquely identifies the practitioner."""
    rows = parse_directory_json(_load("nora_results_page_1.json"))
    steinhauer = next(r for r in rows if "Steinhauer" in r.name)
    assert (
        steinhauer.source_url
        == "https://nora.memberclicks.net/find-a-provider#/profile/1002328432"
    )


def test_parser_accepts_full_response_dict():
    """The parser must accept the full search-directory response (a dict
    with 'results' inside), not just a bare list of results."""
    payload = _load("nora_results_page_1.json")
    assert isinstance(payload, dict)
    assert "results" in payload
    rows = parse_directory_json(payload)
    assert len(rows) == 10


def test_parser_accepts_bare_results_list():
    """The parser must also accept a bare list of result dicts — that's
    what fetch_all_directory_records() returns after concatenating pages."""
    payload = _load("nora_results_page_1.json")
    rows_from_dict = parse_directory_json(payload)
    rows_from_list = parse_directory_json(payload["results"])
    assert len(rows_from_dict) == len(rows_from_list)
    assert [r.source_url for r in rows_from_dict] == [
        r.source_url for r in rows_from_list
    ]


def test_parser_accepts_json_string():
    """A raw JSON string of the full response is also valid input."""
    raw = (FIXTURE_DIR / "nora_results_page_22.json").read_text()
    rows = parse_directory_json(raw)
    assert len(rows) == 10


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


def test_parser_skips_records_with_empty_title():
    """A NORA record with no title (no name) must be dropped, not
    converted to a name-less row."""
    rows = parse_directory_json([{"id": 999, "title": "", "top": []}])
    assert rows == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_is_fellowship_matches_fnora_token_only():
    """_is_fellowship looks for the literal FNORA word in the title and
    only that. It must NOT match substrings (e.g. inside a different
    credential)."""
    assert _is_fellowship({"title": "Briana Larson, OD, FNORA, FOVDR, FAAO"}) is True
    assert _is_fellowship({"title": "Kauser Sharieff OD FCOVD FNORA"}) is True
    assert _is_fellowship({"title": "fnora lowercase"}) is True  # case-insensitive
    assert _is_fellowship({"title": "Jane Doe, OD, FCOVD"}) is False
    assert _is_fellowship({"title": ""}) is False
    assert _is_fellowship({}) is False
    # NORA has no Diplomate tier — the word must NOT trigger fellowship.
    assert _is_fellowship({"title": "Joe Smith, OD, Diplomate ABO"}) is False


def test_strip_credentials_comma_form():
    """Standard comma-delimited title: name, then comma-separated creds."""
    name, creds = _strip_credentials("Briana Larson, OD, FNORA, FOVDR, FAAO")
    assert name == "Briana Larson"
    assert "OD" in creds
    assert "FNORA" in creds
    assert "FAAO" in creds


def test_strip_credentials_space_form():
    """Space-delimited title: name then space-delimited credential tokens.
    The trailing run of known credential tokens is the credential block."""
    name, creds = _strip_credentials("Kauser Sharieff OD FCOVD FNORA")
    assert name == "Kauser Sharieff"
    assert creds is not None
    assert "FNORA" in creds
    assert "FCOVD" in creds
    assert "OD" in creds


def test_strip_credentials_no_credentials():
    """No credentials at all — name stays intact, creds is None."""
    name, creds = _strip_credentials("Julie A. Steinhauer")
    assert name == "Julie A. Steinhauer"
    assert creds is None


def test_strip_credentials_preserves_honorific():
    """'Dr.' / 'Dra.' style prefixes are part of the name, not credentials."""
    name, _ = _strip_credentials("Dr. Bryce Appelbaum")
    assert name == "Dr. Bryce Appelbaum"


def test_normalize_website_adds_scheme_and_fixes_capitalization():
    """NORA records leak 'Www.example.com' with mid-cap W; the
    normalizer must lowercase the scheme part AND add https://."""
    assert _normalize_website("visionforlifeworks.com") == "https://visionforlifeworks.com"
    assert _normalize_website("Www.theeyesite.com.au") == "https://www.theeyesite.com.au"
    assert _normalize_website("https://example.com/") == "https://example.com/"
    assert _normalize_website("http://example.com") == "http://example.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_split_address_us_full():
    """US address with state + 5-digit zip — full parse."""
    addr1, city, state, postal, country = _split_address(
        "2220 South State Rte 157, Suite 350\n,\nGlen Carbon, Illinois 62034"
    )
    assert addr1 == "2220 South State Rte 157, Suite 350"
    assert city == "Glen Carbon"
    assert state == "Illinois"
    assert postal == "62034"
    assert country == "US"


def test_split_address_canadian_with_space_postal():
    """Canadian postal codes have a space in the middle ('N4W 1B4').
    The parser must preserve that space AND map country to 'CA'."""
    addr1, city, state, postal, country = _split_address(
        "770 Main Street W\n,\nListowel, Ontario N4W 1B4"
    )
    assert city == "Listowel"
    assert state == "Ontario"
    assert postal == "N4W 1B4"
    assert country == "CA"


def test_split_address_international_no_state():
    """UK address — no state, just 'City,  Postal'. The parser must
    still produce a city + postal without crashing."""
    addr1, city, state, postal, country = _split_address(
        "EyeTherapy\nWorkz House, 15 Falcon Road,\nLondon,  IG8 8LL"
    )
    assert city == "London"
    # state slot is empty for the UK record.
    assert state is None
    assert postal == "IG8 8LL"


def test_split_address_empty():
    """The fully-empty placeholder '\\n,\\n,  ' must produce all-None
    address fields and not crash."""
    addr1, city, state, postal, country = _split_address("\n,\n,  ")
    assert addr1 is None
    assert city is None
    assert state is None
    assert postal is None


def test_split_address_handles_none():
    """A None block (record with no top[0] item) must produce all-None
    fields with country defaulting to 'US'."""
    addr1, city, state, postal, country = _split_address(None)
    assert addr1 is None and city is None and state is None and postal is None
    assert country == "US"


def test_looks_like_phone_classifies_numeric_strings():
    """_looks_like_phone gates the top[2] phone-vs-practice-name decision —
    must accept obvious phone shapes and reject practice names."""
    assert _looks_like_phone("6182881489") is True
    assert _looks_like_phone("(215) 663-5933") is True
    assert _looks_like_phone("+972-544742173") is True
    assert _looks_like_phone("203.226.2366") is True
    assert _looks_like_phone("Mind Eye Institute") is False
    assert _looks_like_phone("Bright Eyes Vision Clinic") is False
    # Too short to be a phone (would have <7 digits).
    assert _looks_like_phone("123") is False


def test_top_item_indexes_by_display_order_not_list_position():
    """NORA's display elements carry an explicit display_order — the
    parser must index off display_order, not the list position, so a
    re-ordered API response still yields the right field for each slot."""
    rec = {
        "top": [
            {"display_order": 3, "html": "example.com"},
            {"display_order": 0, "html": "addr"},
            {"display_order": 2, "html": "555-1234"},
            {"display_order": 1, "html": "Occupational Therapist"},
        ]
    }
    assert _top_item(rec, 0) == "addr"
    assert _top_item(rec, 1) == "Occupational Therapist"
    assert _top_item(rec, 2) == "555-1234"
    assert _top_item(rec, 3) == "example.com"
    assert _top_item(rec, 4) is None


def test_top1_profession_appended_to_credentials():
    """top[1] holds a profession string ('Occupational Therapist',
    'Physical Therapist', 'VT', ...) for ~10 records. When present it
    must be appended to the credentials field."""
    rec = {
        "id": 12345,
        "title": "Jane Doe",
        "top": [
            {"display_order": 0, "html": "\n,\n,  "},
            {"display_order": 1, "html": "Occupational Therapist"},
            {"display_order": 2, "html": ""},
            {"display_order": 3, "html": ""},
        ],
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].credentials is not None
    assert "Occupational Therapist" in rows[0].credentials


def test_build_source_url_uses_profile_id():
    """Source URL is the public find-a-provider URL + profile id fragment."""
    url = _build_source_url({"id": 12345})
    assert (
        url == "https://nora.memberclicks.net/find-a-provider#/profile/12345"
    )


def test_build_source_url_handles_missing_id():
    """When no id is present, source_url falls back to a deterministic
    placeholder so the upsert doesn't blow up — but in practice every
    record has an id."""
    url = _build_source_url({})
    assert (
        url
        == "https://nora.memberclicks.net/find-a-provider#/profile/unknown"
    )


def test_name_from_title_and_credentials_from_title_are_consistent():
    """The two title-decomposition helpers must agree about where the
    name ends and the credentials begin."""
    name = _name_from_title("Briana Larson, OD, FNORA, FOVDR, FAAO")
    creds = _credentials_from_title("Briana Larson, OD, FNORA, FOVDR, FAAO")
    assert name == "Briana Larson"
    assert creds and "FNORA" in creds
    assert _name_from_title("") == ""
    assert _credentials_from_title("") is None
