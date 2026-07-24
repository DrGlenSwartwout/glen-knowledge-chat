import sqlite3
from dashboard import document_extractions as dx


def _cx():
    cx = sqlite3.connect(":memory:")
    dx.init_table(cx)
    return cx


def test_put_draft_round_trips_decoded_payloads():
    cx = _cx()
    eid = dx.put_draft(
        cx, 7, "c@x.com", "You had a panel done.",
        attributes=[{"field": "conditions", "value": "Glaucoma", "source_quote": "dx: glaucoma"}],
        facts=[{"fact_key": "on_areds2", "value": True, "source_quote": "taking AREDS2"}],
        unstructured=[{"label": "HbA1c", "value": "6.4", "source_quote": "HbA1c 6.4"}],
        model="claude-opus-4-8")
    got = dx.get_for_document(cx, 7)
    assert got["id"] == eid
    assert got["status"] == "ai_draft"
    assert got["email"] == "c@x.com"
    assert got["narrative_md"] == "You had a panel done."
    assert got["model"] == "claude-opus-4-8"
    assert got["attributes"][0]["value"] == "Glaucoma"
    assert got["facts"][0]["fact_key"] == "on_areds2"
    assert got["unstructured"][0]["label"] == "HbA1c"
    assert got["reviewed_at"] is None


def test_get_for_document_returns_none_when_absent():
    assert dx.get_for_document(_cx(), 999) is None


def test_confirm_sets_status_narrative_and_reviewer():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft text", [], [], [], "m")
    assert dx.confirm(cx, eid, "edited text", "glen") is True
    got = dx.get_for_document(cx, 7)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "edited text"
    assert got["reviewed_by"] == "glen"
    assert got["reviewed_at"]


def test_confirm_is_idempotent_second_call_is_a_noop():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.confirm(cx, eid, "first", "glen") is True
    assert dx.confirm(cx, eid, "second", "glen") is False
    assert dx.get_for_document(cx, 7)["narrative_md"] == "first"


def test_reject_sets_status_and_blocks_later_confirm():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.reject(cx, eid, "glen") is True
    assert dx.get_for_document(cx, 7)["status"] == "rejected"
    assert dx.confirm(cx, eid, "x", "glen") is False


def test_put_draft_replaces_a_prior_draft_for_the_same_document():
    """Re-extraction must not leave two competing drafts on one document."""
    cx = _cx()
    dx.put_draft(cx, 7, "c@x.com", "old", [], [], [], "m")
    dx.put_draft(cx, 7, "c@x.com", "new", [], [], [], "m")
    got = dx.get_for_document(cx, 7)
    assert got["narrative_md"] == "new"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1


def test_put_draft_does_not_touch_a_confirmed_document():
    """A confirmed narrative is live in the client's portal and has no
    history table -- re-extraction must leave it completely alone."""
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    dx.confirm(cx, eid, "approved narrative", "glen")

    new_id = dx.put_draft(cx, 7, "c@x.com", "re-extracted text",
                           [{"field": "x"}], [{"fact_key": "y"}],
                           [{"label": "z"}], "m2")

    assert new_id == eid
    got = dx.get_for_document(cx, 7)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "approved narrative"
    assert got["reviewed_by"] == "glen"
    assert got["reviewed_at"]
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1


def test_put_draft_replaces_a_rejected_document():
    """Rejection must not permanently block re-extraction (re-queuing)."""
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    dx.reject(cx, eid, "glen")

    dx.put_draft(cx, 7, "c@x.com", "re-extracted text", [], [], [], "m2")

    got = dx.get_for_document(cx, 7)
    assert got["status"] == "ai_draft"
    assert got["narrative_md"] == "re-extracted text"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1
