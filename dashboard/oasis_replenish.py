"""Owned-consumables projection for the "My Healing Oasis" client-portal tile.

`replenish_items(cx, email)` walks a client's order history (via
`dashboard.orders.list_orders_by_email`) and projects it down to the
CONSUMABLE products they have ever ordered -- one row per slug with the most
recent order date, how many times it was ordered, and a lightweight
`running_low` hint (days since last order >= `_TYPICAL_BOTTLE_DAYS`).
Devices/tools/services/info-only lines are excluded via `_is_consumable`.

Pure-ish: the only I/O is the `cx` read the caller supplies (LOG_DB in prod);
`catalog` and `today` are injectable so this unit-tests without app.py or a
real clock. Never raises on a malformed order row or a malformed (non-dict)
catalog entry -- both degrade by skipping that item rather than crashing the
portal tile."""
import datetime

from dashboard import orders as _orders
from dashboard.shipping import is_shippable

_TYPICAL_BOTTLE_DAYS = 30

# Bottle_type used app-wide (see dashboard/shipping.py / data/products.json)
# for one-time durable goods -- devices, wearables, tools, gear -- packed in
# their own box rather than a dosed bottle. Never a consumable.
_DEVICE_BOTTLE_TYPE = "own-box"


def _is_consumable(product) -> bool:
    """True unless the catalog entry is a device/tool/service/info-only item.

    The products.json catalog mixes dict entries with occasional plain-string
    entries (data-quality artifact) -- a non-dict is never a consumable,
    treated as excluded rather than crashing. Mirrors the existing
    services/info-only/digital/vendor_shipped exclusion (`shipping.is_shippable`,
    used the same way by `orders.physical_units` / `orders.pack_breakdown`),
    plus the catalog's own `own-box` bottle_type convention for devices/tools/
    wearables (as opposed to a dosed/bottled consumable)."""
    if not isinstance(product, dict):
        return False
    try:
        if not is_shippable(product):
            return False
    except Exception:
        pass
    if (product.get("bottle_type") or "") == _DEVICE_BOTTLE_TYPE:
        return False
    return True


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


def replenish_items(cx, email, *, catalog=None, today=None) -> list:
    """Project the client's owned CONSUMABLE products from their order
    history: for every consumable slug ever ordered, `{slug, name, url,
    last_ordered, times_ordered, running_low}`. Devices, tools, services, and
    info-only lines are excluded. Sorted running-low first, then most
    recently ordered. Never raises -- a malformed order row or line is
    skipped, not fatal."""
    if catalog is None:
        from dashboard import products as _products
        catalog = _products.load_products()
    catalog = catalog or {}

    try:
        client_orders = _orders.list_orders_by_email(cx, email, limit=200)
    except Exception:
        client_orders = []

    agg = {}
    for order in (client_orders or []):
        try:
            if (order.get("status") or "") == "cancelled":
                continue
            created_at = order.get("created_at") or ""
            items = order.get("items") or []
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
