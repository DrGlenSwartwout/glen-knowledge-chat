"""`superseded_by` must be applied where slugs are STORED and READ, not only in
`_get_product`.

#763 retired `wholomega-120-capsules`. `_get_product` redirects it, so the storefront
was fine — but two paths kept dead slugs alive:

  * `gk_email_history` scrapes slugs out of storefront URLs (`…/237-wholomega-120-capsules`)
    and filters them against EVERY key in products.json, retired records included. It
    still writes the dead slug, and rebuilding it does not fix that.
  * `repertoire.add_skus` is additive, so a slug seeded before a retirement is never
    removed. `_inhouse_line_unit_cents` tests `slug in repertoire_slugs` against the
    RESOLVED slug, so the member's reorder discount silently never matches.

The redirect now happens at the write boundary (so no new dead slugs are stored) AND at
the repertoire read boundary (so rows stored before the retirement heal themselves,
without a migration).

`resolve` is injectable, but its DEFAULT is the real catalog lookup — not an identity
no-op. A default of "do nothing" is a call site everyone forgets.
"""
import json
import sqlite3

import pytest

from dashboard import products as products_mod
from dashboard import purchase_history as ph
from dashboard import repertoire as rep

DEAD = "wholomega-120-capsules"
LIVE = "wholomega-120-gelcaps"

# The catalog the DEFAULT resolver sees is whatever `DATA_DIR/products.json` holds, and
# other tests in this suite leave DATA_DIR pointing at a two-product fixture. Pin it, so
# these tests assert on the redirect and never on ambient state. The real catalog is
# checked separately, from the repo file, below.
_FIXTURE = {
    DEAD: {"inactive": True, "superseded_by": LIVE},
    LIVE: {"price_cents": 19000},
}


@pytest.fixture(autouse=True)
def _pin_catalog(monkeypatch):
    monkeypatch.setattr(products_mod, "_cached_products", lambda: _FIXTURE)


@pytest.fixture()
def cx():
    c = sqlite3.connect(":memory:")
    ph.init_purchase_history_table(c)
    rep.init_repertoire_table(c)
    yield c
    c.close()


# ── the pure resolver ──
def test_the_real_catalog_redirects_the_retired_record():
    """Against `data/products.json` itself, read explicitly — not via DATA_DIR."""
    catalog = json.load(open("data/products.json"))["products"]
    assert products_mod.superseded_slug(DEAD, catalog) == LIVE
    assert products_mod.superseded_slug(LIVE, catalog) == LIVE


def test_superseded_slug_redirects_a_retired_record():
    assert products_mod.superseded_slug(DEAD) == LIVE


def test_superseded_slug_leaves_a_live_slug_alone():
    assert products_mod.superseded_slug(LIVE) == LIVE


def test_superseded_slug_leaves_an_unknown_slug_alone():
    assert products_mod.superseded_slug("not-a-product") == "not-a-product"


def test_superseded_slug_follows_a_chain_and_survives_a_cycle():
    cat = {
        "a": {"inactive": True, "superseded_by": "b"},
        "b": {"inactive": True, "superseded_by": "c"},
        "c": {},
        "x": {"inactive": True, "superseded_by": "y"},
        "y": {"inactive": True, "superseded_by": "x"},
    }
    assert products_mod.superseded_slug("a", cat) == "c"
    assert products_mod.superseded_slug("x", cat) in ("x", "y")  # terminates, no hang


def test_retired_with_no_successor_stays_put():
    assert products_mod.superseded_slug("a", {"a": {"inactive": True}}) == "a"


def test_app_superseded_delegates_to_the_one_implementation(monkeypatch):
    """Two copies of this redirect would drift apart. `app._superseded` must CALL the
    shared resolver, not re-implement the walk. Asserting only `_superseded(DEAD) == LIVE`
    would pass against a duplicate implementation, so patch the resolver and prove the
    call goes through it."""
    import importlib
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        appmod = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

    calls = []

    def _spy(slug, products=None):
        calls.append((slug, products))
        return "sentinel"

    monkeypatch.setattr(products_mod, "superseded_slug", _spy)
    assert appmod._superseded(DEAD) == "sentinel", "app re-implements the walk"
    assert len(calls) == 1 and calls[0][0] == DEAD
    assert calls[0][1] is not None, "app must pass its own in-memory catalog, not re-read"


# ── write boundary: purchase_history ──
def test_replace_source_stores_the_live_slug(cx):
    ph.replace_source(cx, "groovekart", [("a@b.com", DEAD, "2026-01-01", "ref1")])
    stored = [r[0] for r in cx.execute("SELECT slug FROM purchase_history")]
    assert stored == [LIVE]


def test_replace_source_collapses_a_dead_and_live_pair_in_one_order(cx):
    """Same order listing both slugs is ONE product. The PK (source, source_ref, slug)
    dedups once they resolve to the same thing."""
    n = ph.replace_source(cx, "groovekart", [("a@b.com", DEAD, "2026-01-01", "ref1"),
                                             ("a@b.com", LIVE, "2026-01-01", "ref1")])
    assert n == 1
    assert [r[0] for r in cx.execute("SELECT slug FROM purchase_history")] == [LIVE]


def test_replace_source_resolve_is_injectable(cx):
    ph.replace_source(cx, "fmp", [("a@b.com", "zzz", "2026-01-01", "r")],
                      resolve=lambda s: "swapped")
    assert [r[0] for r in cx.execute("SELECT slug FROM purchase_history")] == ["swapped"]


# ── write + read boundary: repertoire ──
def test_add_skus_stores_the_live_slug(cx):
    rep.add_skus(cx, "a@b.com", [DEAD])
    assert [r[0] for r in cx.execute("SELECT slug FROM repertoire")] == [LIVE]


def test_add_skus_counts_a_dead_and_live_pair_once(cx):
    assert rep.add_skus(cx, "a@b.com", [DEAD, LIVE]) == 1


def test_repertoire_slugs_heals_a_row_stored_before_the_retirement(cx):
    """The rows already in prod. No migration: they resolve on read."""
    cx.execute("INSERT INTO repertoire(email, slug, added_at) VALUES (?,?,?)",
               ("a@b.com", DEAD, "2026-01-01"))
    cx.commit()
    assert rep.repertoire_slugs(cx, "a@b.com") == {LIVE}


def test_repertoire_slugs_does_not_duplicate_when_both_are_stored(cx):
    for s in (DEAD, LIVE):
        cx.execute("INSERT INTO repertoire(email, slug, added_at) VALUES (?,?,?)",
                   ("a@b.com", s, "2026-01-01"))
    cx.commit()
    assert rep.repertoire_slugs(cx, "a@b.com") == {LIVE}


def test_seed_from_history_stores_the_live_slug(cx):
    rep.seed_from_history(cx, "a@b.com", 365, order_slugs_fn=lambda c, e, w: [DEAD])
    assert [r[0] for r in cx.execute("SELECT slug FROM repertoire")] == [LIVE]


# ── the path that actually bit us ──
def test_fmp_rebuild_writes_the_live_slug_even_from_a_stale_slug_map(cx):
    """Belt and braces: data/fmp_slug_map.json now points 448 at the live slug, but the
    write boundary must not depend on that file being right."""
    from dashboard import fmp_history as fh
    cx.executescript(
        "CREATE TABLE fmp_clients(id_pk TEXT, email TEXT);"
        "CREATE TABLE fmp_invoices(id_pk TEXT, id_fk_client TEXT, invoice_date TEXT);"
        "CREATE TABLE fmp_invoice_items(id_pk TEXT, id_fk_invoice TEXT, id_fk_product TEXT);"
        "INSERT INTO fmp_clients VALUES ('c1','a@b.com');"
        "INSERT INTO fmp_invoices VALUES ('i1','c1','2026-01-01');"
        "INSERT INTO fmp_invoice_items VALUES ('it1','i1','448');")
    fh.rebuild_from_fmp(cx, {"resolved": {"448": DEAD}})
    assert [r[0] for r in cx.execute("SELECT slug FROM purchase_history")] == [LIVE]
