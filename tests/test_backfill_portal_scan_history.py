import sqlite3, importlib.util, sys
from pathlib import Path
from dashboard import client_portal as cp, portal_biofield_reports as pbr

def _load():
    p = Path(__file__).resolve().parent.parent / "scripts" / "backfill_portal_scan_history.py"
    spec = importlib.util.spec_from_file_location("bf", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def test_backfill_sets_defaults_but_preserves_optout(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    cp.init_client_portal_table(cx); pbr.init_table(cx)
    cp.upsert_portal(cx, "a@x.com", "A", {})                       # fresh
    cp.upsert_portal(cx, "b@x.com", "B", {"auto_advance": False, "current_scan_date": "2026-07-02"})
    for e in ("a@x.com", "b@x.com"):
        pbr.upsert_report(cx, e, "2026-07-02", "1", {}, "confirmed")
        pbr.upsert_report(cx, e, "2026-07-09", "2", {}, "confirmed")
    bf = _load()
    bf.backfill(cx)
    assert cp.get_current_scan(cx, "a@x.com") == "2026-07-09"      # newest filled in
    assert cp.get_auto_advance(cx, "a@x.com") is True
    assert cp.get_current_scan(cx, "b@x.com") == "2026-07-02"      # opt-out preserved
    # idempotent: second run changes nothing
    bf.backfill(cx)
    assert cp.get_current_scan(cx, "b@x.com") == "2026-07-02"
