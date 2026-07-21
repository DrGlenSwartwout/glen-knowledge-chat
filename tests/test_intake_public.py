import sqlite3
from datetime import datetime, timedelta

from dashboard import intake, intake_public as ip


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    intake.init_intake_table(cx)
    ip.init_intake_sessions_table(cx)
    return cx


NOW = datetime(2026, 7, 20, 12, 0, 0)


# --- scoped session tokens ---------------------------------------------------
def test_session_roundtrip():
    cx = _cx()
    tok = ip.create_session(cx, "Client@Example.com ", "Pat", NOW)
    assert tok and ip.resolve_session(cx, tok, NOW) == "client@example.com"


def test_session_expired():
    cx = _cx()
    tok = ip.create_session(cx, "a@b.com", "A", NOW)
    later = NOW + timedelta(hours=ip.TOKEN_TTL_HOURS + 1)
    assert ip.resolve_session(cx, tok, later) is None


def test_session_bad_or_blank_token():
    cx = _cx()
    assert ip.resolve_session(cx, "", NOW) is None
    assert ip.resolve_session(cx, "nope", NOW) is None


# --- answer merge ------------------------------------------------------------
def test_merge_coerces_scale_and_drops_unknown_and_email():
    merged = ip.merge_answers(
        {"first_name": "Pat"},
        {"terrain": "5", "bogus": "x", "email": "hacker@evil.com", "sleep": "Yes"})
    assert merged["terrain"] == 5           # scale coerced to int
    assert "bogus" not in merged            # unknown dropped
    assert "email" not in merged            # identity never taken from updates
    assert merged["sleep"] == "Yes"         # known text kept
    assert merged["first_name"] == "Pat"    # existing preserved


def test_merge_drops_non_numeric_scale():
    assert "terrain" not in ip.merge_answers({}, {"terrain": "lots"})


def test_merge_passes_table_rows_through():
    rows = [{"concern": "Lyme", "rating": 10, "years_since_onset": 15}]
    assert ip.merge_answers({}, {"health_concerns": rows})["health_concerns"] == rows


# --- public submit / draft (identity + no-clobber) ---------------------------
def _valid_answers():
    return {"first_name": "Pat", "last_name": "Doe", "email": "pat@x.com",
            "dob": "1970-01-01", "terrain": 5, "penetration": 5, "tissue_layer": 2,
            "response": 2, "commitment": 10,
            "terms": {"agreed": True, "signature": "Pat Doe", "date": "2026-07-20"}}


def test_public_submit_keys_by_token_email_not_body():
    cx = _cx()
    # token email differs from the email typed into the form body
    res = ip.public_submit(cx, "owner@x.com", {**_valid_answers(), "email": "victim@y.com"},
                           NOW.isoformat())
    assert res == "submitted"
    assert intake.is_submitted(cx, "owner@x.com")        # written under the token's email
    assert not intake.is_submitted(cx, "victim@y.com")   # never touched the body email
    assert intake.get_response(cx, "owner@x.com")["answers"]["email"] == "owner@x.com"


def test_public_submit_no_clobber_genuine_submission():
    cx = _cx()
    intake.submit(cx, "pat@x.com", {"first_name": "Real", "answered": True}, NOW.isoformat())
    res = ip.public_submit(cx, "pat@x.com", _valid_answers(), NOW.isoformat())
    assert res == "already"
    assert intake.get_response(cx, "pat@x.com")["answers"]["first_name"] == "Real"


def test_public_submit_may_overwrite_external_stub():
    cx = _cx()
    intake.mark_on_file(cx, "pat@x.com", NOW.isoformat())   # _external stub
    assert ip.public_submit(cx, "pat@x.com", _valid_answers(), NOW.isoformat()) == "submitted"
    assert intake.get_response(cx, "pat@x.com")["answers"].get("_external") is None


def test_save_draft_noop_after_submit():
    cx = _cx()
    ip.public_submit(cx, "pat@x.com", _valid_answers(), NOW.isoformat())
    assert ip.save_public_draft(cx, "pat@x.com", {"first_name": "Edit"}, NOW.isoformat()) is False
