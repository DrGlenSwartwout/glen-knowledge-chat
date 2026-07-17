import sqlite3
from dashboard.biofield_stress import init_custom_vocab, add_custom_vocab, vocab_has
from dashboard.biofield_authoring import stress_vocab


def _cx(tmp_path):
    return sqlite3.connect(str(tmp_path / "c.db"))


def _seed_fmp(cx, terms):
    cx.execute("CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT)")
    cx.executemany("INSERT INTO fmp_snap_client_active_main_stress(main_stress) VALUES(?)",
                   [(t,) for t in terms])
    cx.commit()


def test_add_custom_vocab_idempotent_case_insensitive(tmp_path):
    cx = _cx(tmp_path)
    assert add_custom_vocab(cx, "Geopathic Stress") is True
    assert add_custom_vocab(cx, "  geopathic stress ") is False   # case/space dup
    assert cx.execute("SELECT COUNT(*) FROM custom_stress_vocab").fetchone()[0] == 1


def test_stress_vocab_unions_custom(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion", "Adrenal Fatigue"])
    add_custom_vocab(cx, "Geopathic Stress")
    assert "Geopathic Stress" in stress_vocab(cx, "geo")
    assert "Liver Congestion" in stress_vocab(cx, "liver")


def test_stress_vocab_dedupes_across_sources(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "liver congestion")                      # same term, diff case
    got = stress_vocab(cx, "liver")
    assert sum(1 for t in got if t.lower() == "liver congestion") == 1


def test_vocab_has_across_sources(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "Geopathic Stress")
    assert vocab_has(cx, "liver congestion") is True             # FMP
    assert vocab_has(cx, "GEOPATHIC STRESS") is True             # custom
    assert vocab_has(cx, "Nonexistent Term") is False


def test_custom_vocab_survives_fmp_reimport(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "Geopathic Stress")
    # Simulate an FMP snapshot re-import: wipe + rewrite the FMP table.
    cx.execute("DELETE FROM fmp_snap_client_active_main_stress")
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress(main_stress) VALUES('Adrenal Fatigue')")
    cx.commit()
    assert "Geopathic Stress" in stress_vocab(cx, "geo")         # custom intact
