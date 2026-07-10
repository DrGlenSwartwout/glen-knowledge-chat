"""Business-OS Products module over the enriched products.json. Surfaces the
catalog + ingredients and the stale-GrooveKart-page work queue, and persists the
'fixed' set on the /data disk (products.json itself is a read-only repo file)."""
import json
import os
from dashboard.signals import signal as _signal, AMBER, GREEN, GRAY
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA
from dashboard.shipping import is_shippable

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_FIXED_CACHE = None


def _products_path():
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    p = os.path.join(_REPO_DATA, "products.json")
    return p if os.path.exists(p) else None


def _fixed_path():
    d = os.environ.get("DATA_DIR") or _REPO_DATA
    return os.path.join(d, "products-page-fixed.json")


def load_products():
    p = _products_path()
    if not p:
        return {}
    try:
        return (json.load(open(p)) or {}).get("products", {})
    except Exception:
        return {}


_CATALOG_CACHE = {"key": None, "products": {}}


def _cached_products():
    """`load_products()` re-reads a ~900KB file on every call. `superseded_slug` runs on
    the checkout path (once per repertoire read), so key a cache on (path, mtime, size):
    the catalog is a read-only repo file in prod, and a redeploy changes the mtime."""
    p = _products_path()
    if not p:
        return {}
    try:
        st = os.stat(p)
        key = (p, st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    if _CATALOG_CACHE["key"] != key:
        _CATALOG_CACHE["products"] = load_products()
        _CATALOG_CACHE["key"] = key
    return _CATALOG_CACHE["products"]


def superseded_slug(slug, products=None):
    """Follow a retired product's `superseded_by` pointer to its live twin.

    Duplicate records are retired with `inactive: true` rather than deleted (order history
    references their slugs), so a stored slug can name a record that is no longer sellable.
    Returns `slug` unchanged when it is live, unknown, or has no successor. Loop-safe.

    THE one implementation: `app._superseded` delegates here (passing its own in-memory
    catalog), and the purchase_history / repertoire boundaries call it with the default.
    A second copy of this walk would drift."""
    if products is None:
        products = _cached_products()
    seen = set()
    while slug and slug not in seen:
        seen.add(slug)
        p = products.get(slug)
        if not p or not p.get("inactive"):
            return slug
        nxt = (p.get("superseded_by") or "").strip()
        if not nxt:
            return slug
        slug = nxt
    return slug


def _fixed_set():
    try:
        return set(json.load(open(_fixed_path())))
    except Exception:
        return set()


def stale_pages(products=None, fixed=None):
    products = load_products() if products is None else products
    fixed = _fixed_set() if fixed is None else fixed
    return [{"slug": s, "name": p.get("name"), "reason": p.get("gk_stale_reason", "")}
            for s, p in products.items()
            if p.get("gk_stale") and not p.get("inactive") and s not in fixed]


def catalog(with_ingredients_only=True, include_inactive=False):
    out = []
    for s, p in load_products().items():
        if p.get("inactive") and not include_inactive:
            continue
        if with_ingredients_only and not p.get("ingredients"):
            continue
        out.append({"slug": s, "name": p.get("name"), "price_cents": p.get("price_cents"),
                    "ingredients": p.get("ingredients", []), "description": p.get("description", ""),
                    "ingredients_source": p.get("ingredients_source"), "gk_stale": bool(p.get("gk_stale")),
                    # Service-fee lines (e.g. Biofield Analysis): the invoice sorts these to the
                    # top and shows the Value/Regular anchors above the per-client Special price.
                    "service": bool(p.get("service")),
                    # The browser needs the same shippability answer the pricer uses,
                    # so the order builder can disable Pickup when nothing ships.
                    "shippable": is_shippable(p),
                    "service_value_cents": p.get("service_value_cents"),
                    "service_regular_cents": p.get("service_regular_cents")})
    out.sort(key=lambda x: (x["name"] or "").lower())
    return out


def products_signal(cx=None, actor=None):
    try:
        products = load_products()
        total = len(products)
        with_ing = sum(1 for p in products.values() if p.get("ingredients"))
        stale = len(stale_pages(products))
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if total == 0:
        return {"level": GRAY, "summary": "No catalog", "top_actions": [], "count": 0}
    # Backorders take priority — they're an action (reorder) Rae needs to take.
    if cx is not None:
        try:
            from dashboard.orders import backorder_rollup
            bo = backorder_rollup(cx)
            units = sum(b["units_backordered"] for b in bo)
            if units > 0:
                return {"level": AMBER,
                        "summary": f"{units} unit{'s' if units != 1 else ''} on backorder "
                                   f"({len(bo)} product{'s' if len(bo) != 1 else ''})",
                        "top_actions": [{"label": "Reorder list", "href": "/console/products#backorders"}],
                        "count": units}
        except Exception:
            pass
    if stale:
        return {"level": AMBER, "summary": f"{stale} sales page{'s' if stale != 1 else ''} to update",
                "top_actions": [{"label": "Open products", "href": "/console/products"}], "count": stale}
    return {"level": GREEN, "summary": f"{with_ing}/{total} products enriched",
            "top_actions": [{"label": "Open products", "href": "/console/products"}], "count": 0}


products_signal = _signal("products")(products_signal)


def _mark_page_fixed_exec(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    fixed = _fixed_set()
    fixed.add(slug)
    try:
        json.dump(sorted(fixed), open(_fixed_path(), "w"))
    except Exception as e:
        raise RuntimeError(f"could not persist fixed set: {e}")
    return {"slug": slug, "remaining": len(stale_pages()),
            "message": f"Marked {slug}'s GrooveKart page as updated."}


action(key="products.mark_page_fixed", module="products", title="Mark GK page updated",
       description="Record that a product's GrooveKart sales page now matches the current formula.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_mark_page_fixed_exec)
