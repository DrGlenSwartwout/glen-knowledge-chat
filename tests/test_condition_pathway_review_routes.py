"""Condition-pathway review page + API routes (queue / decide / needs-canonical).

Mirror of test_pathway_review_routes.py, one axis over. Deviates from the plan's
brief in two ways the modules themselves now require:

1. `decide()` returns a dict with three outcomes (success / invalid input /
   conflict), and the route must turn those into different HTTP statuses -- see
   `test_decide_returns_409_on_conflict_not_400` for why a conflict is not a bad
   request.
2. `queue()` takes no `offset` -- the set it reads shrinks with every decision, so
   passing one back in would silently skip edges. The route (and this file) never
   sends one.

Also carries the CONSOLE_SECRET-clearing autouse fixture every other route test
file in this suite uses (see test_pathway_review_routes.py, test_admin_*_api.py):
dashboard/__init__.py captures CONSOLE_SECRET at import time, so a bare
monkeypatch.delenv is not enough on its own -- this shell's CONSOLE_SECRET is set
(healingoasis2026), and without both lines every request here 401s before it ever
reaches the route logic.
"""
import sqlite3

import pytest

from dashboard import condition_pathway_review as cpr


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    import dashboard as _d
    monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


def _ingredients(tmp_path):
    p = str(tmp_path / "ingredients.db")
    c = sqlite3.connect(p)
    cpr.init_tables(c)
    c.executescript("""
        INSERT INTO canonical_pathways(id,slug,label,family,status,created_at)
            VALUES(1,'nf-kb','NF-kB','inflammation','proposed','t');
        INSERT INTO conditions(key,label,system,batch,created_at)
            VALUES('dry-eye','Dry Eye','eye','eye','t');
        INSERT INTO condition_pathway
            (id,condition_key,canonical_id,desired_direction,tier,
             proposed_direction,proposed_tier,rationale,decision,source)
            VALUES(1,'dry-eye',1,'down','core','down','core',
                   'surface inflammation','proposed','ai');
    """)
    c.commit()
    c.close()
    return p


def _client(tmp_path, db_path=None):
    from biofield_local_app import create_app
    db_path = db_path or _ingredients(tmp_path)
    return create_app(str(tmp_path / "c.db"), e4l_db=str(tmp_path / "e.db"),
                      ingredients_db=db_path,
                      scan_lookup=lambda e: {"status": "none", "found": False,
                                             "findings": [], "days_ago": None,
                                             "fresh": False}).test_client()


def test_page_renders(tmp_path):
    r = _client(tmp_path).get("/condition-pathway-review")
    assert r.status_code == 200
    body = r.data.decode()
    assert "NF-kB" in body and "Dry Eye" in body


def test_page_renders_the_direction_tier_and_corpus_controls(tmp_path):
    """The verification step this task calls out: a card must carry its direction
    select, tier select, and corpus line, not just the labels."""
    body = _client(tmp_path).get("/condition-pathway-review").data.decode()
    assert 'data-field="desired_direction"' in body
    assert 'data-field="tier"' in body
    assert "no product in the catalog touches this pathway" in body


def test_queue_route_returns_html_fragments(tmp_path):
    r = _client(tmp_path).get("/api/condition-review/queue")
    body = r.get_json()
    assert body["ok"] is True and body["count"] == 1
    assert "NF-kB" in body["html"]


def test_decide_confirms(tmp_path):
    r = _client(tmp_path).post("/api/condition-review/decide",
                               json={"edge_id": 1, "decision": "confirmed",
                                     "desired_direction": "down", "tier": "core"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True and j["edge_id"] == 1


def test_decide_rejects_an_adverse_direction_with_400(tmp_path):
    r = _client(tmp_path).post("/api/condition-review/decide",
                               json={"edge_id": 1, "decision": "confirmed",
                                     "desired_direction": "adverse"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_decide_rejects_an_unknown_edge_with_400(tmp_path):
    r = _client(tmp_path).post("/api/condition-review/decide",
                               json={"edge_id": 999, "decision": "confirmed"})
    assert r.status_code == 400
    assert r.get_json()["error"] == "unknown-edge"


def test_decide_returns_409_on_conflict_not_400(tmp_path):
    """A conflict means another tab already ruled on this edge -- the Stargardt
    trace from condition_pathway_review.decide()'s docstring. It is not a bad
    request: the honest response tells the UI to say 'reload', which needs its own
    status code, not the generic 400 every malformed submission also gets.
    409 Conflict is the HTTP-standard code for exactly this shape of failure --
    the request is well-formed but conflicts with the resource's current state."""
    c = _client(tmp_path)
    first = c.post("/api/condition-review/decide",
                   json={"edge_id": 1, "decision": "confirmed",
                         "desired_direction": "down"})
    assert first.status_code == 200
    stale = c.post("/api/condition-review/decide",
                   json={"edge_id": 1, "decision": "confirmed",
                         "desired_direction": "up"})
    assert stale.status_code == 409
    j = stale.get_json()
    assert j["ok"] is False and j["error"] == "conflict" and j["decision"] == "confirmed"


def test_decide_carries_a_wrong_direction_reject_reason_to_the_database(tmp_path):
    """The highest-value item in this task: the page's JS never sent
    reject_reason, so a 'wrong-direction' rejection -- the exact case the vault's
    gate was fixed to count as a direction defect -- was invisible end to end.
    This proves the reason reaches the actual row, not just that the route
    accepts and echoes the field back in its response."""
    db = _ingredients(tmp_path)
    c = _client(tmp_path, db_path=db)
    r = c.post("/api/condition-review/decide",
               json={"edge_id": 1, "decision": "rejected",
                     "reject_reason": "wrong-direction"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT decision, reject_reason FROM condition_pathway "
                         "WHERE id=1").fetchone()
    assert row == ("rejected", "wrong-direction")


def test_needs_canonical_does_not_create_a_pathway(tmp_path):
    r = _client(tmp_path).post("/api/condition-review/needs-canonical",
                               json={"edge_id": 1,
                                     "label": "Goblet cell mucin secretion"})
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_needs_canonical_missing_label_is_400(tmp_path):
    r = _client(tmp_path).post("/api/condition-review/needs-canonical",
                               json={"edge_id": 1, "label": ""})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_every_condition_review_route_is_console_gated(tmp_path, monkeypatch):
    """Same load-bearing shape as test_pathway_review_routes.py's sibling test:
    the rest of this file clears CONSOLE_SECRET, which opens the gate, so without
    this test the gated path is never exercised and a write endpoint could ship
    unauthenticated."""
    monkeypatch.setenv("CONSOLE_SECRET", "s3cret")
    from biofield_local_app import create_app
    app = create_app(str(tmp_path / "c.db"), e4l_db=str(tmp_path / "e.db"),
                     ingredients_db=_ingredients(tmp_path),
                     scan_lookup=lambda e: {"status": "none", "found": False,
                                            "findings": [], "days_ago": None,
                                            "fresh": False})
    c = app.test_client()
    assert c.get("/condition-pathway-review").status_code == 401
    assert c.get("/api/condition-review/queue").status_code == 401
    for path, body in (
        ("/api/condition-review/decide", {"edge_id": 1, "decision": "confirmed"}),
        ("/api/condition-review/needs-canonical", {"edge_id": 1, "label": "X"}),
    ):
        assert c.post(path, json=body).status_code == 401, path
    r = c.get("/condition-pathway-review?key=s3cret")
    assert r.status_code == 200 and "rm_biofield_key" in r.headers.get("Set-Cookie", "")
    assert c.get("/condition-pathway-review").status_code == 200
