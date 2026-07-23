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
  - build_out.wanted  -> _wanted_items: the client's wishlist (shared with My
                       Remedies' "Add to my Oasis" and this tile's own
                       "Add to wishlist" roadmap action), resolved to
                       {slug, name, url} so the client can see what they've
                       flagged, not just an ephemeral "Added" toast.

Returns {"enabled": False} outright when the flag is off. Each sub-field is
independently try/except guarded so one failing source degrades to an empty
list rather than breaking the whole block -- or the rest of the portal
payload, which composes many other unrelated blocks alongside this one."""
import json

from dashboard import oasis_replenish as _rep
from dashboard import oasis_roadmap as _roadmap
from dashboard import owned_tools as _ot
from dashboard import wishlist as _wl

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
    # "harmony-laser*" only, NOT a bare "harmony" prefix -- the catalog has
    # non-device "harmony-*" consumables (e.g. harmony-flower-essence) that a
    # bare prefix would wrongly map onto the Harmony hero. A slug that is exactly
    # "harmony" still passes through unchanged and equals the hero slug anyway.
    ("harmony-laser", "harmony"),
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


def _wanted_items(cx, email):
    """Task 7: the client's wishlist (Build Out's "roadmap-want" AND My
    Remedies' "Add to my Oasis" both land on this SAME shared wishlist store,
    see dashboard/wishlist.py), resolved to {slug, name, url} for display.
    Mirrors oasis_replenish.replenish_items' catalog-lookup shape: a slug
    that no longer resolves to a dict catalog entry (removed/renamed product,
    or a non-catalog roadmap "hero" slug like "harmony") is silently skipped
    rather than shown with placeholder text. Never raises -- any failure
    degrades to an empty list, same philosophy as every other sub-field
    here."""
    from dashboard import products as _products
    catalog = _products.load_products() or {}
    _wl.init_wishlist_table(cx)
    owner = _wl.resolve_owner(email, None)
    slugs = _wl.list_for(cx, owner)
    out = []
    for slug in slugs:
        product = catalog.get(slug)
        if not isinstance(product, dict):
            continue
        out.append({
            "slug": slug,
            "name": product.get("name") or slug,
            "url": product.get("url") or f"/begin/product/{slug}",
        })
    return out


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

    try:
        wanted = _wanted_items(cx, email)
    except Exception:
        wanted = []

    return {
        "enabled": True,
        "replenish": replenish,
        "build_out": {
            "owned_from_us": owned_from_us,
            "owned_external": owned_external,
            "roadmap": roadmap,
            "wanted": wanted,
        },
    }
