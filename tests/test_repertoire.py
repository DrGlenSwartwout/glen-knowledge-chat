import sqlite3
from dashboard import repertoire


def _cx():
    cx = sqlite3.connect(":memory:")
    repertoire.init_repertoire_table(cx)
    return cx


def test_add_and_read_skus_lowercased_and_deduped():
    cx = _cx()
    n1 = repertoire.add_skus(cx, "Glen@Example.com", ["Neuro-Mag", "neuro-mag", "terrain-restore"])
    assert n1 == 2  # deduped case-insensitively
    assert repertoire.repertoire_slugs(cx, "glen@example.com") == {"neuro-mag", "terrain-restore"}
    n2 = repertoire.add_skus(cx, "glen@example.com", ["neuro-mag"])  # already present
    assert n2 == 0


def test_seed_from_history_window():
    cx = _cx()
    def fake_history(cx_, email, window_days):
        assert window_days == 90
        return ["a", "b", "b"]  # b duplicated
    added = repertoire.seed_from_history(cx, "x@y.com", 90, order_slugs_fn=fake_history)
    assert added == 2
    assert repertoire.repertoire_slugs(cx, "x@y.com") == {"a", "b"}
