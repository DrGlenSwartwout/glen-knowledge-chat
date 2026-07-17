"""Rule pricing for bundle SKUs: one-time price = 10% off the summed retail
(price_cents) of the bundle's components, resolved by slug.

Pure module — no Flask, no I/O. The caller passes the products dict."""

DEFAULT_PRICE_CENTS = 6997


def resolve_component(slug: str, products: dict):
    """The component product for a slug, following superseded_by, dropping
    inactive. Returns the product dict (without mutation) or None."""
    seen = set()
    cur = slug
    while cur and cur not in seen:
        seen.add(cur)
        p = products.get(cur)
        if p is None:
            return None
        nxt = p.get("superseded_by")
        if nxt and nxt != cur:
            cur = nxt
            continue
        if p.get("inactive"):
            return None
        return p
    return None


def compute_bundle_price_cents(product: dict, products: dict) -> int:
    """round(0.9 * sum(component price_cents * qty)) in integer cents.
    Raises KeyError if any component slug does not resolve to a sellable product."""
    total = 0
    for comp in product.get("bundle_component_slugs") or []:
        slug = comp["slug"]
        qty = int(comp.get("qty", 1))
        p = resolve_component(slug, products)
        if p is None:
            raise KeyError(f"unresolvable bundle component slug: {slug!r}")
        total += int(p.get("price_cents", DEFAULT_PRICE_CENTS)) * qty
    return int(round(total * 0.9))
