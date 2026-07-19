import sqlite3
from dashboard import portal_backfill as pb
from dashboard import client_portal as cp
from dashboard import biofield_reveals as br
from dashboard import portal_biofield_reports as pbr

def _db():
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx); br.init_table(cx); pbr.init_table(cx)
    return cx

def _seed(cx, *emails):
    for i, e in enumerate(emails):
        br.upsert(cx, e, f"2026-07-{10+i:02d}", {}, [], "t")

def test_dry_run_reports_counts_and_writes_nothing():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com")
    cp.ensure_token(cx, "a@x.com", "")                       # a already has a portal
    res = pb.backfill_portals(cx, commit=False)
    assert res == {"reveal_emails": 2, "already": 1, "provisioned": 0,
                   "remaining": 1, "committed": False}
    assert cx.execute("SELECT 1 FROM client_portals WHERE email='b@x.com'").fetchone() is None

def test_commit_provisions_bare_portals_and_no_report():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com")
    res = pb.backfill_portals(cx, commit=True)
    assert res["provisioned"] == 2 and res["already"] == 0 and res["remaining"] == 0
    assert cx.execute("SELECT COUNT(*) FROM client_portals").fetchone()[0] == 2
    assert cx.execute("SELECT COUNT(*) FROM portal_biofield_reports").fetchone()[0] == 0

def test_rerun_is_idempotent():
    cx = _db(); _seed(cx, "a@x.com")
    pb.backfill_portals(cx, commit=True)
    res = pb.backfill_portals(cx, commit=True)
    assert res["provisioned"] == 0 and res["already"] == 1

def test_limit_caps_provisioning_and_reports_remaining():
    cx = _db(); _seed(cx, "a@x.com", "b@x.com", "c@x.com")
    res = pb.backfill_portals(cx, commit=True, limit=2)
    assert res["provisioned"] == 2 and res["remaining"] == 1
