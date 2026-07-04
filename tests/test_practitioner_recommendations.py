import sqlite3
from dashboard import practitioner_recommendations as pr


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_table(cx); return cx


def test_create_and_active():
    cx = _cx()
    rid = pr.create(cx, practitioner_id="prac-42", patient_email="P@x.com",
                    items=[{"slug": "nerve-repair", "qty": 1}], note="stay the course")
    a = pr.active_for_patient(cx, "p@x.com")   # case-insensitive
    assert a["id"] == rid and a["items"][0]["slug"] == "nerve-repair" and a["status"] == "sent"


def test_dismissed_not_active():
    cx = _cx()
    rid = pr.create(cx, practitioner_id="prac-42", patient_email="p@x.com", items=[], note="")
    pr.set_status(cx, rid, "dismissed")
    assert pr.active_for_patient(cx, "p@x.com") is None
