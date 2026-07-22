import os, pytest
from dashboard import intelligence as intel

pg = bool(os.environ.get("PG_DSN"))

def test_briefing_roundtrip_sqlite(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    # empty before any write
    assert intel.read_briefing("money-cash", db_path=dbp)["empty"] is True
    intel.write_briefing("money-cash", "# Finance\n[HIGH] pay X", db_path=dbp)
    got = intel.read_briefing("money-cash", db_path=dbp)
    assert got["empty"] is False
    assert got["markdown"] == "# Finance\n[HIGH] pay X"
    assert got["bytes"] == len("# Finance\n[HIGH] pay X")
    assert "generated_at" in got and got["generated_at"]

def test_links_roundtrip_and_missing_sqlite(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    assert intel.read_links("clients-pipeline", db_path=dbp) == {}  # missing -> {}
    reg = {"r1": {"type": "person", "display": "Jane", "url": "/x"}}
    intel.write_links("clients-pipeline", reg, db_path=dbp)
    assert intel.read_links("clients-pipeline", db_path=dbp) == reg
    # links survive a later briefing write, and read_briefing embeds them
    intel.write_briefing("clients-pipeline", "# Clients", db_path=dbp)
    assert intel.read_briefing("clients-pipeline", db_path=dbp)["links"] == reg

def test_bytes_accepts_bytes_and_str(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    intel.write_briefing("signals-patterns", b"# Signals", db_path=dbp)
    assert intel.read_briefing("signals-patterns", db_path=dbp)["markdown"] == "# Signals"

def test_unknown_slug_rejected(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    with pytest.raises(ValueError):
        intel.write_briefing("bogus", "x", db_path=dbp)

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_briefing_roundtrip_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    from dashboard import db
    with db.connect("/data/chat_log.db") as cx:
        cx.execute("DROP TABLE IF EXISTS intelligence_briefings")
        cx.commit()
    dbp = "/data/chat_log.db"
    assert intel.read_briefing("money-cash", db_path=dbp)["empty"] is True
    intel.write_briefing("money-cash", "# Finance\n[HIGH] pay X", db_path=dbp)
    intel.write_links("money-cash", {"r1": {"type": "x"}}, db_path=dbp)
    got = intel.read_briefing("money-cash", db_path=dbp)
    assert got["empty"] is False and got["markdown"].startswith("# Finance")
    assert got["links"] == {"r1": {"type": "x"}}
    assert set(intel.list_all(db_path=dbp)) == {"money-cash", "clients-pipeline", "signals-patterns"}
