import sqlite3
from dashboard import intake


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    intake.init_intake_table(cx)
    return cx


def test_form_structure_integrity():
    form = intake.INTAKE_FORM
    assert form["version"]
    ids = []
    dim_fields = []
    for sec in form["sections"]:
        assert sec["id"] and sec["title"]
        for f in sec["fields"]:
            assert f["id"] and f["type"]
            ids.append(f["id"])
            if f.get("maps_to"):
                dim_fields.append(f["maps_to"])
            if f["type"] == "scale":
                assert f["options"] and all("value" in o and "label" in o for o in f["options"])
            if f["type"] == "table":
                assert f["columns"] and all("id" in c and "type" in c for c in f["columns"])
    assert len(ids) == len(set(ids)), "field ids must be unique"
    assert sorted(dim_fields) == ["commitment", "penetration", "response", "terrain", "tissue_layer"]


def test_validate_missing_required():
    errors = intake.validate_response({})
    for req in ("first_name", "last_name", "email", "dob", "terrain", "terms"):
        assert req in errors


def test_validate_scale_out_of_range():
    errors = intake.validate_response({"terrain": 9})
    assert "terrain" in errors


def test_validate_consent_unsigned():
    errors = intake.validate_response({"terms": {"agreed": False, "signature": "", "date": ""}})
    assert "terms" in errors


def test_validate_valid_minimal():
    answers = {
        "first_name": "Steven", "last_name": "Fox", "email": "s@x.com", "dob": "1960-06-17",
        "terrain": 1, "penetration": 5, "tissue_layer": 3, "response": 3, "commitment": 8,
        "terms": {"agreed": True, "signature": "Steven Fox", "date": "2026-07-02"},
    }
    assert intake.validate_response(answers) == []


def test_draft_then_submit_transitions_status():
    cx = _cx()
    intake.save_draft(cx, "s@x.com", {"first_name": "Steven"}, "2026-07-07T00:00:00")
    assert intake.is_submitted(cx, "s@x.com") is False
    assert intake.get_response(cx, "s@x.com")["status"] == "draft"
    intake.submit(cx, "s@x.com", {"first_name": "Steven"}, "2026-07-07T01:00:00")
    assert intake.is_submitted(cx, "s@x.com") is True
    row = intake.get_response(cx, "s@x.com")
    assert row["status"] == "submitted" and row["submitted_at"] == "2026-07-07T01:00:00"


def test_list_submitted_only_returns_submitted():
    cx = _cx()
    intake.save_draft(cx, "draft@x.com", {"a": 1}, "2026-07-07T00:00:00")
    intake.submit(cx, "done@x.com", {"a": 2}, "2026-07-07T01:00:00")
    rows = intake.list_submitted(cx)
    assert [r["email"] for r in rows] == ["done@x.com"]
    assert rows[0]["answers"] == {"a": 2}


def test_mark_on_file_sets_submitted_with_external_marker():
    cx = _cx()
    intake.mark_on_file(cx, "Ext@X.com", "2026-07-07T00:00:00")
    assert intake.is_submitted(cx, "ext@x.com") is True
    row = intake.get_response(cx, "ext@x.com")
    assert row["answers"]["_external"] is True
    assert row["answers"]["_note"] == "Completed via Practice Better"
    assert row["submitted_at"] == "2026-07-07T00:00:00"


def test_mark_on_file_guard_does_not_overwrite_real_submission():
    cx = _cx()
    intake.submit(cx, "real@x.com", {"first_name": "Real"}, "2026-07-07T00:00:00")
    intake.mark_on_file(cx, "real@x.com", "2026-07-07T01:00:00")
    row = intake.get_response(cx, "real@x.com")
    assert row["answers"] == {"first_name": "Real"}
    assert row["submitted_at"] == "2026-07-07T00:00:00"


def test_mark_on_file_then_clear_intake_removes_row():
    cx = _cx()
    intake.mark_on_file(cx, "gone@x.com", "2026-07-07T00:00:00")
    assert intake.is_submitted(cx, "gone@x.com") is True
    intake.clear_intake(cx, "gone@x.com")
    assert intake.is_submitted(cx, "gone@x.com") is False
    assert intake.get_response(cx, "gone@x.com") is None
