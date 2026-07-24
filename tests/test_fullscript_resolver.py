"""candidates_for: unions the drivers, dedupes by product, keeps the
highest-priority origin. Pins are an explicit clinical decision by Glen and
therefore outrank anything derived."""
import sqlite3
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": "Neuro Magnesium", "relation": "substitute",
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
    {"name": "Taurine", "brand": "Montiff", "product_slug": "taurine",
     "external_id": "P2", "best_ff": None, "relation": None,
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Mag Taurate", "rank": 0},
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Taurine", "rank": 1},
  ],
  "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    return cx


def test_scan_only():
    cx = _cx()
    out = fs.candidates_for(cx, "a@b.com", item_codes=["ED4"])
    assert [c["name"] for c in out] == ["Mag Taurate", "Taurine"]
    assert all(c["origin"] == "scan" for c in out)
    assert out[0]["focus_area_name"] == "Nervous System"


def test_pin_outranks_scan_and_dedupes():
    cx = _cx()
    cx.execute("INSERT INTO fullscript_client_pins "
               "(email, fs_product_name, note, pinned_by, pinned_at) VALUES (?,?,?,?,?)",
               ("a@b.com", "Taurine", "for sleep", "glen", "2026-07-23"))
    out = fs.candidates_for(cx, "a@b.com", item_codes=["ED4"])
    names = [c["name"] for c in out]
    assert names.count("Taurine") == 1, "deduped, not listed twice"
    assert names[0] == "Taurine", "pinned sorts first"
    taurine = out[0]
    assert taurine["origin"] == "pinned" and taurine["reason"] == "for sleep"


def test_no_drivers_yields_nothing():
    cx = _cx()
    assert fs.candidates_for(cx, "a@b.com", item_codes=[]) == []


def test_unknown_client_yields_nothing():
    cx = _cx()
    assert fs.candidates_for(cx, "nobody@nowhere.com", item_codes=None) == []


def test_origin_priority_ordering_contract():
    """Pins the ordering contract future drivers depend on. `review` and
    `condition` aren't wired yet (see candidates_for's docstring), so the sort
    that enforces this order can't be exercised behaviorally through
    candidates_for today -- this test is what stands in for that until they
    land. Covers all four origins and fails if any two entries are swapped."""
    assert set(fs.ORIGIN_PRIORITY) == {"pinned", "review", "scan", "condition"}
    assert (fs.ORIGIN_PRIORITY["pinned"] < fs.ORIGIN_PRIORITY["review"]
            < fs.ORIGIN_PRIORITY["scan"] < fs.ORIGIN_PRIORITY["condition"])


def test_all_currently_emitted_origins_have_priority_entries():
    """pinned and scan are the only drivers wired in this phase. If either
    origin string ever drifts out of sync with ORIGIN_PRIORITY, the trailing
    sorted() call in candidates_for raises KeyError at runtime -- guard that
    crash path directly against real output, not just the map in isolation."""
    cx = _cx()
    cx.execute("INSERT INTO fullscript_client_pins "
               "(email, fs_product_name, note, pinned_by, pinned_at) VALUES (?,?,?,?,?)",
               ("a@b.com", "Mag Taurate", "keep", "glen", "2026-07-23"))
    out = fs.candidates_for(cx, "a@b.com", item_codes=["ED4"])
    emitted_origins = {c["origin"] for c in out}
    assert emitted_origins == {"pinned", "scan"}, "only these two drivers are wired in this phase"
    for origin in emitted_origins:
        assert origin in fs.ORIGIN_PRIORITY
