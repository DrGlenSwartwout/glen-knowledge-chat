"""Increment 4a: writable authoring store. Authored tests render through the same
report shape as the FMP snapshot, so schedule/narrative/audio reuse unchanged."""
import sqlite3
from dashboard.biofield_authoring import (
    init_auth_tables, create_test, add_chain_row, update_chain_row,
    delete_chain_row, update_header, list_authored, authored_report)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_auth_tables(cx)
    return cx


def test_create_author_and_render(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "Jane Doe", "Jane@x.com", "2026-06-23")
    assert tid.startswith("a")
    add_chain_row(cx, tid, 2, "Acid", "Liver", "Sterol Max", "3 caps", "daily", "with food")
    add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    rep = authored_report(cx, tid)
    assert rep["client"] == {"name": "Jane Doe", "email": "jane@x.com"}
    assert rep["date"] == "2026-06-23"
    assert [(l["layer"], l["head"], l["remedy"]) for l in rep["layers"]] == [
        (1, "Night", "TMG"), (2, "Acid", "Sterol Max")]
    slots = {e["name"]: e["slots"] for e in rep["schedule"]["entries"]}
    assert slots["TMG"] == ["Bedtime"] and slots["Sterol Max"] == ["Breakfast"]


def test_list_authored(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "Jane Doe", "jane@x.com", "2026-06-23")
    add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    lst = list_authored(cx)
    assert len(lst) == 1
    assert lst[0]["test_id"] == tid and lst[0]["name"] == "Jane Doe"
    assert lst[0]["layer_count"] == 1 and lst[0]["authored"] is True


def test_update_and_delete_row(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    rid = add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    update_chain_row(cx, rid, layer=3, remedy="TMG Powder")
    rep = authored_report(cx, tid)
    assert rep["layers"][0]["layer"] == 3 and rep["layers"][0]["remedy"] == "TMG Powder"
    delete_chain_row(cx, rid)
    assert authored_report(cx, tid)["layers"] == []


def test_update_header(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    update_header(cx, tid, name="Jane Q", date="2026-07-01")
    rep = authored_report(cx, tid)
    assert rep["client"]["name"] == "Jane Q" and rep["date"] == "2026-07-01"
