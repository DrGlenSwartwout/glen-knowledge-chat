import sqlite3
from dashboard.canonical_tags import init_tables, resolve, set_attr


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    return cx


def test_discrete_dedup_by_norm(tmp_path):
    cx = _cx(tmp_path)
    assert set_attr(cx, "J@x.com", "conditions", "Eczema", source="manual") is True
    assert set_attr(cx, "j@x.com", "conditions", " eczema ", source="ai") is False   # norm-dup
    rows = cx.execute("SELECT email, value, source FROM person_attributes WHERE field='conditions'").fetchall()
    assert rows == [("j@x.com", "Eczema", "manual")]          # email lowercased, first source kept


def test_scalar_replace(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "goals", "more energy", source="import")
    set_attr(cx, "j@x.com", "goals", "sleep better", source="manual")
    rows = cx.execute("SELECT value FROM person_attributes WHERE field='goals'").fetchall()
    assert rows == [("sleep better",)]                        # single row, replaced


def test_resolve_vocab_alias_discrete_only(tmp_path):
    cx = _cx(tmp_path)
    cx.execute("INSERT INTO canonical_vocab(field,alias_norm,canonical) VALUES('conditions','adrenal exhaustion','Adrenal Fatigue')")
    cx.commit()
    assert resolve(cx, "conditions", "Adrenal  Exhaustion") == "Adrenal Fatigue"   # alias->canonical
    assert resolve(cx, "conditions", "Unmapped Thing") == "Unmapped Thing"          # fallback cleaned
    assert resolve(cx, "goals", "adrenal exhaustion") == "adrenal exhaustion"       # scalar ignores vocab (cleaned)


def test_empty_and_bad_field_noop(tmp_path):
    cx = _cx(tmp_path)
    assert set_attr(cx, "j@x.com", "conditions", "   ", source="manual") is False
    assert set_attr(cx, "", "conditions", "x", source="manual") is False
    assert set_attr(cx, "j@x.com", "not_a_field", "x", source="manual") is False
    assert cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0] == 0
