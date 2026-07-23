"""dashboard/oasis_block.py — the "My Healing Oasis" client-portal tile payload.

Composes the three sibling modules into one block:
  - replenish       -> oasis_replenish.replenish_items: owned CONSUMABLES from
                       the client's order history (restock hints).
  - build_out.owned_from_us    -> DEVICE/TOOL products ordered from us (the
                       non-consumables in the same order history -- ordered
                       products where oasis_replenish._is_consumable is False).
  - build_out.owned_external   -> owned_tools.list_for: tools the client has
                       self-reported, whether or not they map to a catalog slug.
  - build_out.roadmap -> oasis_roadmap.build_roadmap: the personalized gap list,
                       built from the union of device slugs (ours + external),
                       normalized onto the roadmap's simplified hero-family
                       slugs (see _normalize_owned_for_roadmap below).

Returns {"enabled": False} outright when the flag is off. Each sub-field is
independently try/except guarded so one failing source degrades to an empty
list rather than breaking the whole block -- or the rest of the portal
payload, which composes many other unrelated blocks alongside this one."""
import json

from dashboard import oasis_replenish as _rep
from dashboard import oasis_roadmap as _roadmap
from dashboard import owned_tools as _ot

# Real catalog device slugs come in families/variants that differ from the
# roadmap's simplified hero slugs:
#   roadmap "harmony"       <- real "harmony-laser" (+ variants)
#   roadmap "water-ionizer" <- real "water-ionizer-5plate" / "-9plate" / "-15plate"
#   roadmap "kloud"         <- real "kloud-pemf-mini" / "kloud-pemf-maxi"
# Prefix/startswith match so owning ANY real variant of a hero device excludes
# that hero from the roadmap. Order matters only in that the first matching
# prefix wins; families do not currently overlap.
# Glen extends this family map as new device variants are added to the catalog.
_DEVICE_FAMILY_PREFIXES = (
    ("water-ionizer", "water-ionizer"),
    ("harmony", "harmony"),   # covers "harmony-laser*" and the bare "harmony" slug
    ("kloud", "kloud"),
)


def _normalize_owned_for_roadmap(slugs):
    """Map each real owned slug onto its roadmap hero-family slug when it
    matches a known device family prefix (see _DEVICE_FAMILY_PREFIXES);
    otherwise pass the slug through unchanged. Pure, no DB."""
    out = set()
    for s in (slugs or ()):
        s = (s or "").strip()
        if not s:
            continue
        mapped = s
        for prefix, family in _DEVICE_FAMILY_PREFIXES:
            if s.startswith(prefix):
                mapped = family
                break
        out.add(mapped)
    return out


def _device_orders(cx, email):
    """Device/tool products ordered from us: the same order-read + catalog
    approach as oasis_replenish.replenish_items (same table read via
    oasis_replenish._fetch_orders_raw, same catalog lookup), but keeping the
    NON-consumable lines (oasis_replenish._is_consumable False) instead of the
    consumable ones. Never raises -- a malformed order/line/catalog entry is
    skipped, same philosophy as replenish_items."""
    from dashboard import products as _products
    catalog = _products.load_products() or {}

    rows = _rep._fetch_orders_raw(cx, email, limit=200)
    agg = {}
    for row in rows:
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
            if not isinstance(product, dict) or _rep._is_consumable(product):
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

    items_out = list(agg.values())
    items_out.sort(key=lambda i: i["last_ordered"] or "", reverse=True)
    return items_out


def build_block(cx, email, enabled, terrain_phase=None) -> dict:
    """{"enabled": bool, "replenish": [...], "build_out": {"owned_from_us": [...],
    "owned_external": [...], "roadmap": [...]}}. {"enabled": False} when the
    flag is off. Every sub-field is independently guarded so one failing
    source degrades to [] rather than raising into the portal payload."""
    if not enabled:
        return {"enabled": False}

    try:
        replenish = _rep.replenish_items(cx, email)
    except Exception:
        replenish = []

    try:
        owned_from_us = _device_orders(cx, email)
    except Exception:
        owned_from_us = []

    try:
        owned_external = _ot.list_for(cx, email)
    except Exception:
        owned_external = []

    try:
        us_slugs = {d["slug"] for d in owned_from_us if d.get("slug")}
    except Exception:
        us_slugs = set()
    try:
        external_slugs = _ot.owned_slugs(cx, email)
    except Exception:
        external_slugs = set()

    try:
        owned_for_roadmap = _normalize_owned_for_roadmap(us_slugs | external_slugs)
        roadmap = _roadmap.build_roadmap(owned_for_roadmap, terrain_phase)
    except Exception:
        roadmap = []

    return {
        "enabled": True,
        "replenish": replenish,
        "build_out": {
            "owned_from_us": owned_from_us,
            "owned_external": owned_external,
            "roadmap": roadmap,
        },
    }
