# tests/test_ash_ally.py
import sqlite3
import dashboard.ash_ally as aa
import dashboard.ash_map as am


def _seed(db_path, email, summary, dim, cell):
    cx = sqlite3.connect(db_path)
    m = am.get(cx, email)
    m["dimensions"][dim].update(cell)
    am._upsert(cx, email, summary, m["dimensions"])
    cx.close()


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    assert aa.ENABLED() is False
    monkeypatch.setenv("ASH_ALLY_ENABLED", "TRUE")
    assert aa.ENABLED() is True
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    assert aa.ENABLED() is True
    monkeypatch.setenv("ASH_ALLY_ENABLED", "off")
    assert aa.ENABLED() is False


def test_overlay_empty_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    db = str(tmp_path / "t.db")
    _seed(db, "a@b.com", "A summary.", "symptoms", {"state": "explored", "notes": "AM knee"})
    assert aa.ally_overlay(db, "a@b.com") == ""


def test_overlay_empty_for_no_email_or_blank_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    assert aa.ally_overlay(db, "") == ""            # no subject
    assert aa.ally_overlay(db, "never@seen.com") == ""  # unseen → blank memory


def test_overlay_returns_framed_context_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    _seed(db, "a@b.com", "A tired caregiver.", "symptoms",
          {"state": "explored", "notes": "AM knee pain"})
    ov = aa.ally_overlay(db, "a@b.com")
    assert "WHAT YOU ALREADY KNOW ABOUT THIS PERSON" in ov   # frame header
    assert "A tired caregiver." in ov                        # the summary, via context_block
    assert "Never read this back as a list" in ov            # frame footer guidance


def test_overlay_fail_open_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    monkeypatch.setattr(am, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # must swallow and return "" — never raise
    assert aa.ally_overlay(str(tmp_path / "t.db"), "a@b.com") == ""
