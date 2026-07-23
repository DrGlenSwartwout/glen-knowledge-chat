"""dashboard/oasis_block.py — the "My Healing Oasis" client-portal tile payload.

Composes the three sibling modules into one block:
  - replenish       -> oasis_replenish.replenish_items: owned CONSUMABLES from
                       the client's order history (restock hints).
  - build_out.owned_from_us    -> DEVICE/TOOL products ordered from us (an
                       explicit device allowlist, oasis_replenish.is_device --
                       NOT simply the non-consumables, since that complement
                       also includes services/consults, info_only, digital
                       ebooks, and print books, none of which are "tools you
                       own").
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

# Real catalog device slugs come in families/variants that must all collapse
# onto the ONE real SKU the roadmap's HERO_TOOLS uses:
#   water-ionizer* -> water-ionizer-15plate (owning any plate-count excludes it)
#   kloud*         -> kloud-pemf-maxi (owning Mini or Maxi excludes it)
#   harmony-laser* -> harmony-laser (the device itself; already the hero slug)
# Prefix/startswith match so owning ANY real variant of a hero device excludes
# that hero from the roadmap. Order matters only in that the first matching
# prefix wins; families do not currently overlap.
# Glen extends this family map as new device variants are added to the catalog.
_DEVICE_FAMILY_PREFIXES = (
    ("water-ionizer", "water-ionizer-15plate"),
    # "harmony-laser*" only, NOT a bare "harmony" prefix -- the catalog has
    # non-device "harmony-*" consumables (e.g. harmony-flower-essence) that a
    # bare prefix would wrongly map onto the Harmony hero.
    ("harmony-laser", "harmony-laser"),
    ("kloud", "kloud-pemf-maxi"),
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
    oasis_replenish._fetch_orders_raw, same catalog lookup), but keeping only
    lines that pass oasis_replenish.is_device -- an explicit device/tool
    bottle_type allowlist, NOT simply "not a consumable" (that complement
    also includes services/consults, info_only, digital ebooks, and print
    books -- a client who bought a Biofield Analysis or a book must not see
    it under "Tools you own with us"). Never raises -- a malformed
    order/line/catalog entry is skipped, same philosophy as replenish_items."""
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
            if not _rep.is_device(product):
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


def _roadmap_name_lookup():
    """Build a one-shot {slug: name} lookup from every roadmap table
    (HERO_TOOLS + SECONDARY_TOOLS). Pure, no DB -- used only as a
    name-resolution fallback for wishlist slugs that don't resolve in the
    purchasable catalog. Every current roadmap slug IS a real catalog slug
    (see dashboard/oasis_roadmap.py), so this fallback mainly guards a
    roadmap table entry whose product is later removed/renamed from the
    catalog. Never raises -- a malformed table entry is skipped rather than
    crashing the lookup."""
    out = {}
    try:
        tables = list(_roadmap.HERO_TOOLS) + list(_roadmap.SECONDARY_TOOLS)
        for tool in tables:
            try:
                slug = (tool.get("slug") or "").strip()
                if slug and slug not in out:
                    out[slug] = tool.get("name") or slug
            except Exception:
                continue
    except Exception:
        return {}
    return out


def _wanted_items(cx, email):
    """Task 7: the client's wishlist (Build Out's "roadmap-want" AND My
    Remedies' "Add to my Oasis" both land on this SAME shared wishlist store,
    see dashboard/wishlist.py), resolved to {slug, name, url} for display.
    Mirrors oasis_replenish.replenish_items' catalog-lookup shape: a slug is
    first looked up in the purchasable catalog; when it does NOT resolve
    there (removed/renamed product, or a roadmap "hero"/terrain/general
    slug like "harmony" that is never itself a catalog dict entry), it
    falls back to the roadmap tables (see _roadmap_name_lookup) for a
    display name -- this is what makes clicking "Add to wishlist" on a
    roadmap hero/terrain/general item actually surface in Wanted instead of
    silently no-oping. Only a slug that resolves in NEITHER the catalog NOR
    the roadmap tables is skipped. Never raises -- any failure degrades to
    an empty list, same philosophy as every other sub-field here."""
    from dashboard import products as _products
    catalog = _products.load_products() or {}
    roadmap_names = _roadmap_name_lookup()
    _wl.init_wishlist_table(cx)
    owner = _wl.resolve_owner(email, None)
    slugs = _wl.list_for(cx, owner)
    out = []
    for slug in slugs:
        product = catalog.get(slug)
        if isinstance(product, dict):
            out.append({
                "slug": slug,
                "name": product.get("name") or slug,
                "url": product.get("url") or f"/begin/product/{slug}",
            })
            continue
        roadmap_name = roadmap_names.get(slug)
        if roadmap_name is None:
            continue  # unknown in both catalog and roadmap -- truly skip
        out.append({
            "slug": slug,
            "name": roadmap_name,
            "url": f"/begin/product/{slug}",
        })
    return out


def build_block(cx, email, enabled, terrain_phase=None) -> dict:
    """{"enabled": bool, "replenish": [...], "build_out": {"owned_from_us": [...],
    "owned_external": [...], "roadmap": [...], "wanted": [...]}}. {"enabled": False}
    when the flag is off. Every sub-field is independently guarded so one failing
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
