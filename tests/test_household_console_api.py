import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)   # auth open in test
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_console_household_crud_and_reassign(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pbr.init_table(cx)
        pbr.upsert_report(cx, "wrong@x.com", "2026-06-25", "s1", {"n": 1}, "confirmed"); cx.commit()
    c = appmod.app.test_client()
    assert c.post("/api/console/household",
                  json={"primary_email": "p@x.com", "member_email": "wrong@x.com", "label": "W"}).status_code == 200
    assert c.post("/api/console/household",
                  json={"primary_email": "p@x.com", "member_email": "right@x.com", "label": "R"}).status_code == 200
    g = c.get("/api/console/household?primary_email=p@x.com").get_json()
    emails = {m["email"] for m in g["members"]}
    assert emails == {"wrong@x.com", "right@x.com"}
    assert "2026-06-25" in next(m for m in g["members"] if m["email"] == "wrong@x.com")["scan_dates"]
    # reassign wrong→right
    r = c.post("/api/console/household/reassign",
               json={"scan_date": "2026-06-25", "from_email": "wrong@x.com", "to_email": "right@x.com"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert pbr.list_report_dates(sqlite3.connect(appmod.LOG_DB), "right@x.com") == ["2026-06-25"]
    # cross-household reassign refused (400)
    bad = c.post("/api/console/household/reassign",
                 json={"scan_date": "2026-06-25", "from_email": "right@x.com", "to_email": "stranger@x.com"})
    assert bad.status_code == 400 and bad.get_json()["ok"] is False
    # delete a link
    assert c.delete("/api/console/household",
                    json={"primary_email": "p@x.com", "member_email": "wrong@x.com"}).status_code == 200
