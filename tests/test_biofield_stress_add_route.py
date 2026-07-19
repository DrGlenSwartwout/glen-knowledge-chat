import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _app(db):
    return create_app(db, scan_lookup=lambda e: _NONE)


def _new(client):
    return client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]


def _stress_state(client, tid):
    return client.get(f"/author/{tid}/stresses").get_json()["data"]


def test_add_active_stress_unassigned(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    j = client.post(f"/author/{tid}/stress/add", json={"label": "Geopathic Stress"}).get_json()
    assert j["ok"] and isinstance(j["sid"], int) and j["layer"] is None
    s = _stress_state(client, tid)
    active = {x["label"] for x in s["active"]}
    assert "Geopathic Stress" in active


def test_add_new_term_persisted_to_vocab(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/stress/add", json={"label": "Scalar Interference"})
    cx = sqlite3.connect(db)
    assert cx.execute("SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)='scalar interference'").fetchone()


def test_add_balanced_on_layer(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    # Give layer 1 a remedy row so the stress can be balanced by it.
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver", "remedy": "Liver Support"})
    j = client.post(f"/author/{tid}/stress/add",
                    json={"label": "Liver Congestion", "layer": 1}).get_json()
    assert j["ok"] and j["layer"] == 1
    s = _stress_state(client, tid)
    balanced = {x["label"] for x in s["balanced"]}
    assert "Liver Congestion" in balanced
    assert "Liver Congestion" not in {x["label"] for x in s["active"]}


def test_add_empty_label_rejected(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    resp = client.post(f"/author/{tid}/stress/add", json={"label": "   "})
    assert resp.status_code == 400


def test_add_merges_into_scan_sourced_stress_by_normalized_label(tmp_path):
    """Finding 1 regression: a scan-sourced row stores code=<raw E4L finding code>,
    not _norm(label). Typing a label that normalizes to that row's label must merge
    (add_stress's own dedup) AND stress_id_for must still resolve the id, so the
    requested layer-balancing action (cover_stress/set_manual_balanced) actually runs."""
    db = str(tmp_path / "c.db")
    client = _app(db).test_client()
    tid = _new(client)
    num_tid = int(tid.lstrip("a"))
    from dashboard.biofield_stress import init_stress_tables
    cx = sqlite3.connect(db)
    init_stress_tables(cx)
    cx.execute(
        "INSERT INTO biofield_auth_stress(test_id, code, label, source, balance, "
        "manual_balanced, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
        (num_tid, "E4L-999", "Liver Congestion", "scan", "required", 0,
         "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))
    cx.commit()
    cx.close()

    # Give layer 1 a remedy row so the stress can be balanced by it.
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver", "remedy": "Liver Support"})

    j = client.post(f"/author/{tid}/stress/add",
                    json={"label": "Liver Congestion", "layer": 1}).get_json()
    assert j["ok"] is True
    assert j["sid"] is not None
    assert j["layer"] == 1

    s = _stress_state(client, tid)
    balanced = {x["label"] for x in s["balanced"]}
    active = {x["label"] for x in s["active"]}
    assert "Liver Congestion" in balanced
    assert "Liver Congestion" not in active


def test_add_stress_non_numeric_layer_does_not_500(tmp_path):
    """Finding 2 regression: a non-numeric layer must not raise an uncaught
    ValueError -> 500. It should be handled gracefully (200 with layer=None/coerced,
    or a clean 400), never a 500."""
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    resp = client.post(f"/author/{tid}/stress/add",
                        json={"label": "Adrenal Fatigue", "layer": "not-a-number"})
    assert resp.status_code != 500
