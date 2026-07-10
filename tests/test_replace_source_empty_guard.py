"""`purchase_history.replace_source` must not wipe a slice on an empty extraction.

The old body did an UNCONDITIONAL `DELETE FROM purchase_history WHERE source=?`
and then inserted from `rows`. If the extraction yielded nothing — FMP projection
tables not loaded, Gmail fetch errored, every slug filtered out — the DELETE still
ran and the whole slice was destroyed with nothing to replace it. An empty
extraction almost always means the SOURCE failed, not that the client genuinely
has zero purchases now.

Fix: build the storable set first; if it's empty, SKIP the DELETE and return 0,
leaving the slice intact. `allow_empty=True` is the out-of-band escape for a
deliberate clear.
"""
import sqlite3

import pytest

from dashboard import purchase_history as ph


@pytest.fixture()
def cx():
    c = sqlite3.connect(":memory:")
    ph.init_purchase_history_table(c)
    # A pre-existing slice that a failed rebuild must not destroy.
    c.execute("INSERT INTO purchase_history(email, slug, purchased_at, source, source_ref) "
              "VALUES ('a@b.com','terrain-restore','2026-01-01','fmp','it1')")
    c.commit()
    yield c
    c.close()


def _fmp_rows(c):
    return c.execute("SELECT email, slug FROM purchase_history WHERE source='fmp'").fetchall()


def test_empty_rows_skips_delete_and_preserves_the_slice(cx):
    assert ph.replace_source(cx, "fmp", [], resolve=lambda s: s) == 0
    assert _fmp_rows(cx) == [("a@b.com", "terrain-restore")], "slice was wiped by an empty rebuild"


def test_all_invalid_rows_are_treated_as_empty(cx):
    """A non-empty input that yields zero STORABLE rows (blank email/slug, or a slug
    that resolves to nothing) is still an empty extraction — the count of input rows
    is not the guard, the count of storable rows is."""
    bad = [("", "x", "2026-01-01", "r1"), ("a@b.com", "", "2026-01-01", "r2")]
    assert ph.replace_source(cx, "fmp", bad, resolve=lambda s: s) == 0
    assert _fmp_rows(cx) == [("a@b.com", "terrain-restore")]


def test_slug_resolving_to_nothing_is_treated_as_empty(cx):
    """resolve() returning falsy for every row -> no storable rows -> no wipe."""
    rows = [("a@b.com", "gone", "2026-01-01", "r1")]
    assert ph.replace_source(cx, "fmp", rows, resolve=lambda s: "") == 0
    assert _fmp_rows(cx) == [("a@b.com", "terrain-restore")]


def test_allow_empty_true_clears_the_slice(cx):
    """The out-of-band escape: a caller that genuinely means to empty a slice can."""
    n = ph.replace_source(cx, "fmp", [], resolve=lambda s: s, allow_empty=True)
    assert n == 0
    assert _fmp_rows(cx) == []


def test_allow_empty_only_clears_the_named_source(cx):
    cx.execute("INSERT INTO purchase_history(email, slug, purchased_at, source, source_ref) "
               "VALUES ('a@b.com','wholomega','2026-01-01','groovekart','g1')")
    cx.commit()
    ph.replace_source(cx, "fmp", [], resolve=lambda s: s, allow_empty=True)
    gk = cx.execute("SELECT slug FROM purchase_history WHERE source='groovekart'").fetchall()
    assert gk == [("wholomega",)], "clearing fmp must not touch the groovekart slice"


def test_normal_rebuild_still_replaces(cx):
    """Regression: a non-empty extraction replaces the slice as before."""
    n = ph.replace_source(cx, "fmp",
                          [("c@d.com", "clear-the-way", "2026-02-02", "it9")],
                          resolve=lambda s: s)
    assert n == 1
    assert _fmp_rows(cx) == [("c@d.com", "clear-the-way")]


def test_rebuild_from_fmp_preserves_slice_when_projection_tables_are_empty(cx):
    """The real bite: rebuild against empty fmp_* tables must not wipe the slice."""
    from dashboard import fmp_history as fh
    cx.executescript(
        "CREATE TABLE fmp_clients(id_pk TEXT, email TEXT);"
        "CREATE TABLE fmp_invoices(id_pk TEXT, id_fk_client TEXT, invoice_date TEXT);"
        "CREATE TABLE fmp_invoice_items(id_pk TEXT, id_fk_invoice TEXT, id_fk_product TEXT);")
    result = fh.rebuild_from_fmp(cx, {"resolved": {}})
    assert result["rows"] == 0
    assert _fmp_rows(cx) == [("a@b.com", "terrain-restore")], "empty rebuild wiped the slice"
