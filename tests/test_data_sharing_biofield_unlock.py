import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed_portal(tmp_db, email):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "M", {})
    return token


def test_tier2_optin_records_biofield_free_unlock(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    email = "member@ex.com"
    token = _seed_portal(tmp_db, email)
    # seed a biofield reveal so there is a reveal_id to unlock
    from dashboard import biofield_reveals as br
    with sqlite3.connect(str(tmp_db)) as cx:
        br.init_table(cx)
        # NOT NULL columns per CREATE TABLE: email, scan_date, created_at, updated_at
        # (interpretation_json/remedies_json/first_approved have DEFAULTs).
        cx.execute(
            "INSERT INTO biofield_reveals (email, scan_date, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (email, "2026-07-01", "2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
        )
        cx.commit()
        rid = cx.execute(
            "SELECT id FROM biofield_reveals WHERE email=? ORDER BY id DESC LIMIT 1", (email,)
        ).fetchone()[0]
    # opt into Tier 2 (research_results toggle) via the sharing endpoint
    r = app.app.test_client().post(
        f"/api/portal/{token}/sharing", json={"toggles": {"research_results": True}}
    )
    assert r.status_code == 200
    with sqlite3.connect(str(tmp_db)) as cx:
        from dashboard import biofield_reveals as br2
        assert br2.free_unlock_reveal_id(cx, email) == rid  # unlock recorded for latest reveal


def test_tier2_optin_no_reveal_is_clean_noop(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db, "noreveal@ex.com")
    r = app.app.test_client().post(
        f"/api/portal/{token}/sharing", json={"toggles": {"research_results": True}}
    )
    assert r.status_code == 200  # succeeds even with no reveal to unlock
