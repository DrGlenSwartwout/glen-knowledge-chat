"""Owned-consumables projection for the "My Healing Oasis" client-portal tile.

`replenish_items(cx, email)` walks a client's order history and projects it
down to the CONSUMABLE products they have ever ordered -- one row per slug
with the most recent order date, how many times it was ordered, and a
lightweight `running_low` hint (days since last order >= `_TYPICAL_BOTTLE_DAYS`).
Devices/tools/books/services/info-only lines are excluded via `_is_consumable`.

Pure-ish: the only I/O is the `cx` read the caller supplies (LOG_DB in prod);
`catalog` and `today` are injectable so this unit-tests without app.py or a
real clock. Never raises on a malformed order row, a malformed line item, or a
malformed (non-dict) catalog entry -- each degrades by skipping just that
item rather than crashing the portal tile (see `_fetch_orders_raw` for why
the order read does not call `dashboard.orders.list_orders_by_email`
directly)."""
import datetime
import json

from dashboard.shipping import is_shippable

_TYPICAL_BOTTLE_DAYS = 30

# Dosed/bottled supplement bottle_types (see dashboard/shipping.py
# PROD_BOTTLE_NAMES for the full prod vocabulary, which also includes many
# device/accessory/book types -- own-box, book, harmony-laser, denas,
# nightlight, dowsing-rods, nasal-clip, toothbrush, handcradle -- that are
# NOT dosed consumables and must stay excluded). This is an ALLOWLIST
# (fail-closed): a bottle_type not listed here is treated as NOT a
# consumable, so a newly-added device/book type never silently passes
# through as "restock this" copy. Normalized lower().strip() before matching
# since the catalog mixes case ("30 Caps" vs "120 caps"). Glen: extend this
# set when a new DOSED bottle_type is added to the catalog (cross-ref:
# adding a bottle_type touches several places -- shipping.py PROD_BOTTLE_NAMES,
# _STANDARD_BOTTLES, and here).
_CONSUMABLE_BOTTLE_TYPES = frozenset({
    "30 caps", "120 caps", "180 caps", "360 caps",
    "30 g", "120 g",
    "30ml", "15ml", "100ml",
    "dropper 5 ml", "dropper 30 ml", "dropper 50 ml",
    "30roll", "one-step",
})

# Real device/tool bottle_types (see dashboard/shipping.py PROD_BOTTLE_NAMES /
# _STANDARD_BOTTLES for the full prod vocabulary this is drawn from). This is
# the COMPLEMENT of _CONSUMABLE_BOTTLE_TYPES, NOT its inverse: "not a
# consumable" also includes services, consults, info_only, digital ebooks,
# and print books, none of which are "tools you own." This is an explicit
# ALLOWLIST (fail-closed) of the actual physical device/tool bottle_types --
# a bottle_type not listed here is treated as NOT a device, so a
# newly-added service/book/supplement type never silently passes through as
# "tools you own with us" copy. `book` is deliberately EXCLUDED -- a book is
# reading material, not a tool. Normalized lower().strip() before matching.
# Seeded 2026-07-22 from the real data/products.json bottle_type census:
#   own-box (drop-ship devices/tools like Kloud PEMF mats, NES miHealth,
#     water ionizers, plus wearables/tools boxed the same way),
#   harmony-laser (Harmony Laser + cold-laser variants), denas (DENAS PCM Pro
#   family), handcradle (ZYTO Hand Cradle / NES Scanner), nightlight
#   (therapeutic/biocompatible nightlights), dowsing-rods, nasal-clip (NIR
#   intranasal clip), toothbrush (wicking toothbrush).
# Glen: extend this set when a new DEVICE/TOOL bottle_type is added to the
# catalog (cross-ref: adding a bottle_type touches several places --
# shipping.py PROD_BOTTLE_NAMES, _STANDARD_BOTTLES, and here).
_DEVICE_BOTTLE_TYPES = frozenset({
    "own-box", "harmony-laser", "denas", "handcradle",
    "nightlight", "dowsing-rods", "nasal-clip", "toothbrush",
})


def _is_consumable(product) -> bool:
    """True only when the catalog entry is BOTH shippable AND a dosed
    supplement bottle_type (or has no bottle_type at all -- a plain
    supplement).

    The products.json catalog mixes dict entries with occasional plain-string
    entries (data-quality artifact) -- a non-dict is never a consumable,
    treated as excluded rather than crashing. Mirrors the existing
    services/info-only/digital/vendor_shipped exclusion (`shipping.is_shippable`,
    used the same way by `orders.physical_units` / `orders.pack_breakdown`).
    Fails CLOSED (excluded) if `is_shippable` itself raises -- this module's
    philosophy is to degrade by skipping an item, never to guess a device or
    book is safe to tell a client to "restock"."""
    if not isinstance(product, dict):
        return False
    try:
        if not is_shippable(product):
            return False
    except Exception:
        return False
    bottle_type = (product.get("bottle_type") or "").strip().lower()
    if not bottle_type:
        return True  # unset/None bottle_type = a plain dosed supplement
    return bottle_type in _CONSUMABLE_BOTTLE_TYPES


def is_device(product) -> bool:
    """True only when the catalog entry is BOTH shippable AND a real
    device/tool bottle_type (see _DEVICE_BOTTLE_TYPES).

    This is the "tools you own with us" predicate -- deliberately NOT the
    complement of `_is_consumable`, because "not a consumable" also includes
    services/consults (info_only), digital ebooks, and print books, none of
    which are a "tool you own." An unset/blank bottle_type (a plain
    supplement, or a service/consult with no bottle_type at all) is never a
    device. Same non-dict / is_shippable-raises fail-CLOSED philosophy as
    `_is_consumable`: a malformed or unrecognized entry is excluded rather
    than guessed to be a device."""
    if not isinstance(product, dict):
        return False
    try:
        if not is_shippable(product):
            return False
    except Exception:
        return False
    bottle_type = (product.get("bottle_type") or "").strip().lower()
    if not bottle_type:
        return False  # unset/blank bottle_type is never a device
    return bottle_type in _DEVICE_BOTTLE_TYPES


def _as_date(value):
    """Best-effort ISO date/datetime string -> datetime.date. None on failure."""
    try:
        return datetime.date.fromisoformat(str(value or "")[:10])
    except (ValueError, TypeError):
        return None


def _days_since(last_ordered, today):
    last = _as_date(last_ordered)
    if last is None:
        return None
    if today is None:
        ref = datetime.date.today()
    elif isinstance(today, datetime.datetime):
        ref = today.date()
    elif isinstance(today, datetime.date):
        ref = today
    else:
        ref = _as_date(today)
    if ref is None:
        return None
    return (ref - last).days


def _fetch_orders_raw(cx, email, limit=200):
    """Read a client's orders directly, most recent first -- same table,
    same filter/order/limit as `dashboard.orders.list_orders_by_email`, but
    NOT calling that function.

    `list_orders_by_email` builds its result with a single list comprehension
    over `orders._row_to_dict(r)` for every row. `_row_to_dict` calls
    `json.loads` on `items_json`; if ONE order in the client's history has
    malformed `items_json`, that raises INSIDE the comprehension and aborts
    construction of the whole list -- every good order for that client is
    lost, not just the bad one. Since we must not modify `dashboard/orders.py`,
    this reads the same rows with the same query and defers JSON parsing to
    the per-order loop below, where a bad row is caught and skipped instead
    of blanking the whole projection. `cx.row_factory` is assumed to be
    `sqlite3.Row` (same precondition as `list_orders_by_email`)."""
    try:
        cur = cx.execute(
            "SELECT created_at, status, items_json FROM orders "
            "WHERE lower(email)=? ORDER BY created_at DESC, id DESC LIMIT ?",
            ((email or "").strip().lower(), limit))
        return cur.fetchall()
    except Exception:
        return []


def replenish_items(cx, email, *, catalog=None, today=None) -> list:
    """Project the client's owned CONSUMABLE products from their order
    history: for every consumable slug ever ordered, `{slug, name, url,
    last_ordered, times_ordered, running_low}`. Devices, tools, services, and
    info-only lines are excluded. Sorted running-low first, then most
    recently ordered. Never raises -- a malformed order row or line is
    skipped, not fatal, and one bad order never blanks its siblings (see
    `_fetch_orders_raw`)."""
    if catalog is None:
        from dashboard import products as _products
        catalog = _products.load_products()
    catalog = catalog or {}

    client_orders = _fetch_orders_raw(cx, email, limit=200)

    agg = {}
    for row in client_orders:
        try:
            order = dict(row)
            if (order.get("status") or "") == "cancelled":
                continue
            created_at = order.get("created_at") or ""
            items = json.loads(order.get("items_json") or "[]") or []
        except Exception:
            continue
        for line in items:
            try:
                slug = (line.get("slug") or "").strip()
            except Exception:
                continue
            if not slug:
                continue
            product = catalog.get(slug)
            if not _is_consumable(product):
                continue
            entry = agg.get(slug)
            if entry is None:
                entry = {
                    "slug": slug,
                    "name": product.get("name") or slug,
                    "url": product.get("url") or f"/begin/product/{slug}",
                    "last_ordered": created_at,
                    "times_ordered": 0,
                }
                agg[slug] = entry
            entry["times_ordered"] += 1
            if created_at > (entry["last_ordered"] or ""):
                entry["last_ordered"] = created_at

    items_out = []
    for entry in agg.values():
        days = _days_since(entry["last_ordered"], today)
        entry["running_low"] = bool(days is not None and days >= _TYPICAL_BOTTLE_DAYS)
        items_out.append(entry)

    # Stable two-pass sort: most-recently-ordered first, then running-low
    # bubbled to the front (ties keep their recency order from pass 1).
    items_out.sort(key=lambda i: i["last_ordered"] or "", reverse=True)
    items_out.sort(key=lambda i: i["running_low"], reverse=True)
    return items_out
