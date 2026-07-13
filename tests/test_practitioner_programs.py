import sqlite3
from dashboard import practitioner_programs as pp

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pp.init_table(cx); return cx

def test_get_none_when_absent():
    assert pp.get(_cx(), "a@b.com") is None

def test_upsert_roundtrip_and_replace():
    cx = _cx()
    pp.upsert(cx, patient_email="A@B.com", practitioner_id="doc1",
              condition_key="dry-amd", items=[{"slug":"wholomega","name":"WholOmega"}], note="start low")
    got = pp.get(cx, "a@b.com")
    assert got["condition_key"] == "dry-amd" and got["note"] == "start low"
    assert got["items"] == [{"slug":"wholomega","name":"WholOmega"}]
    assert got["practitioner_id"] == "doc1"
    pp.upsert(cx, patient_email="a@b.com", practitioner_id="doc1",
              condition_key="dry-amd", items=[{"slug":"ocuheal-eye-drops","name":"OcuHeal Eye Drops"}], note="")
    got2 = pp.get(cx, "a@b.com")
    assert [i["slug"] for i in got2["items"]] == ["ocuheal-eye-drops"]  # replaced, one row per patient
