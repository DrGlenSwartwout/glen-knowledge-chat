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


import threading


def test_record_turn_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    db = str(tmp_path / "t.db")
    called = {"extract": 0}
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: called.__setitem__("extract", called["extract"] + 1) or {"dimensions": {}, "summary": ""})
    assert aa.record_turn(db, threading.Lock(), "a@b.com", "hi", "") is None
    assert called["extract"] == 0   # never even reached the LLM


def test_record_turn_noop_for_empty_email(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    assert aa.record_turn(db, threading.Lock(), "", "hi", "") is None


def test_record_turn_persists_and_accumulates(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    seq = iter([
        {"dimensions": {"symptoms": {"state": "opened", "excerpt": "knee aches", "notes": "AM"}},
         "summary": "One."},
        {"dimensions": {"symptoms": {"state": "deep", "excerpt": "ignored", "notes": "night pain"}},
         "summary": "Two."},
    ])
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: next(seq))
    aa.record_turn(db, threading.Lock(), "u@x.com", "my knee aches", "")
    cx = sqlite3.connect(db)
    assert am.get(cx, "u@x.com")["dimensions"]["symptoms"]["state"] == "opened"
    aa.record_turn(db, threading.Lock(), "u@x.com", "worse at night", "")
    m = am.get(cx, "u@x.com")
    assert m["dimensions"]["symptoms"]["state"] == "deep"          # deepened
    assert m["dimensions"]["symptoms"]["opened_excerpt"] == "knee aches"  # set-once preserved
    assert "AM" in m["dimensions"]["symptoms"]["notes"] and "night pain" in m["dimensions"]["symptoms"]["notes"]
    assert m["summary"] == "Two."
    cx.close()


def test_record_turn_fail_open_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # extract raising must be swallowed — record_turn returns None, no exception
    assert aa.record_turn(str(tmp_path / "t.db"), threading.Lock(), "u@x.com", "hi", "") is None


def test_record_turn_does_not_hold_lock_during_extract(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    lock = threading.Lock()
    observed = {}

    def fake_extract(memory, user_text, ally_text=""):
        # The lock MUST be free while the (slow) extract runs.
        got = lock.acquire(blocking=False)
        observed["lock_free_during_extract"] = got
        if got:
            lock.release()
        return {"dimensions": {"mind": {"state": "opened", "excerpt": "x", "notes": "y"}}, "summary": "s"}

    monkeypatch.setattr(am, "_haiku_extract", fake_extract)
    aa.record_turn(db, lock, "u@x.com", "hi", "")
    assert observed["lock_free_during_extract"] is True
