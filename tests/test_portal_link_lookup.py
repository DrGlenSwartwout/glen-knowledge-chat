"""Console portal-link lookup: find/reissue a client's /portal/<token> link by
email, and list all portals for the lookup table. See dashboard/client_portal.py."""
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

BASE = "https://illtowell.com"


def _cx():
    from dashboard import client_portal as cp
    from dashboard import notify_state as ns
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx)
    ns.init_table(cx)
    return cp, cx


def test_link_for_unknown_email_is_none_and_creates_nothing():
    cp, cx = _cx()
    link, reissued = cp.portal_link_for(cx, "nobody@x.com", BASE)
    assert link is None and reissued is False
    # never spawned a portal for the unknown email
    assert cx.execute("SELECT COUNT(*) FROM client_portals").fetchone()[0] == 0


def test_link_reuses_stable_token_and_is_idempotent():
    cp, cx = _cx()
    token, _ = cp.upsert_portal(cx, "a@x.com", "Ann", {"biofield_status": "confirmed"})
    from dashboard import notify_state as ns
    ns.set_token(cx, "a@x.com", token)          # stable raw token on file
    link1, reissued1 = cp.portal_link_for(cx, "A@x.com", BASE)   # case-insensitive
    assert link1 == f"{BASE}/portal/{token}" and reissued1 is False
    link2, reissued2 = cp.portal_link_for(cx, "a@x.com", BASE)
    assert link2 == link1 and reissued2 is False   # same link, not rotated


def test_link_reissues_when_only_hash_stored():
    cp, cx = _cx()
    from dashboard import notify_state as ns
    cp.upsert_portal(cx, "b@x.com", "Bee", {"biofield_status": "confirmed"})
    ns.set_token(cx, "b@x.com", "")   # legacy portal: only the hash survived, no raw token
    link, reissued = cp.portal_link_for(cx, "b@x.com", BASE)
    assert link and link.startswith(f"{BASE}/portal/") and reissued is True
    # and it's stable on the next lookup (now a raw token is on file)
    link2, reissued2 = cp.portal_link_for(cx, "b@x.com", BASE)
    assert link2 == link and reissued2 is False


def test_list_portals_reports_token_state():
    cp, cx = _cx()
    from dashboard import notify_state as ns
    cp.upsert_portal(cx, "has@x.com", "Has", {})    # upsert stores a stable raw token
    cp.upsert_portal(cx, "no@x.com", "No", {})
    ns.set_token(cx, "no@x.com", "")                # clear it -> only hash remains
    rows = {r["email"]: r for r in cp.list_portals(cx)}
    assert rows["has@x.com"]["has_token"] is True
    assert rows["no@x.com"]["has_token"] is False
    assert rows["has@x.com"]["name"] == "Has"
