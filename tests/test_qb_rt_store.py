import os, threading, pytest
from dashboard import money

pg = bool(os.environ.get("PG_DSN"))

class _FakeResp:
    def __init__(self, rt, at): self._rt, self._at = rt, at
    def raise_for_status(self): pass
    def json(self): return {"refresh_token": self._rt, "access_token": self._at}

def _install_fake_intuit(monkeypatch, counter):
    # Each refresh returns a NEW rotated RT; asserts it was CALLED with the
    # currently-stored RT (proves serialization: never a stale/reused RT).
    def fake_post(url, headers=None, data=None, timeout=None):
        n = counter["n"]; counter["n"] += 1
        counter["seen"].append(data["refresh_token"])
        return _FakeResp(rt=f"rt{n+1}", at=f"at{n}")
    monkeypatch.setattr(money.requests, "post", fake_post)

def test_qb_rt_seed_from_env_then_persist_sqlite(tmp_path, monkeypatch):
    dbp = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(money, "_qb_default_db_path", lambda: dbp)
    monkeypatch.setattr(money, "QB_REFRESH_TOKEN", "rt0")
    monkeypatch.setattr(money, "QB_CLIENT_ID", "c"); monkeypatch.setattr(money, "QB_CLIENT_SECRET", "s")
    counter = {"n": 0, "seen": []}
    _install_fake_intuit(monkeypatch, counter)
    at = money.qb_refresh()
    assert at == "at0"
    assert counter["seen"] == ["rt0"]           # used the env seed
    assert money._qb_rt_read(dbp) == "rt1"      # persisted the rotated RT to DB

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_qb_rt_rotate_race_no_double_spend_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    from dashboard import db
    monkeypatch.setattr(money, "_qb_default_db_path", lambda: "/data/chat_log.db")
    monkeypatch.setattr(money, "QB_CLIENT_ID", "c"); monkeypatch.setattr(money, "QB_CLIENT_SECRET", "s")
    with db.connect("/data/chat_log.db") as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS oauth_tokens (name TEXT PRIMARY KEY, token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.execute("DELETE FROM oauth_tokens WHERE name='qbo_refresh'")
        cx.commit()
    # seed via the production seed helper (assumes caller holds the appropriate lock;
    # single-threaded here, before any workers start)
    money._qb_rt_write_at("/data/chat_log.db", "rt0")
    counter = {"n": 0, "seen": []}
    _install_fake_intuit(monkeypatch, counter)
    errors = []
    def worker():
        try:
            for _ in range(10): money.qb_refresh()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=worker) for _ in range(5)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors, errors[:2]
    # 50 serialized refreshes: every POST saw the RT the prior write produced,
    # never a duplicate -> no double-spend of a rotated token.
    assert len(counter["seen"]) == 50
    assert len(set(counter["seen"])) == 50, "a refresh token was reused (rotate race)"
