"""Where an order button points.

ONE rule: the new-style in-funnel page, `/begin/product/<slug>`, for every product.

Never `products.json`'s `url` — that is the OLD GrooveKart storefront page, absent on
669 of 966 sellable products, and it drops the client out of the funnel, out of their
courtesy pricing, and onto a page that does not stock two-thirds of the catalog.

A named seam rather than an inline f-string: it is unit-testable, the rule has one
enforcement point, and a future route change touches one line.
"""


def destination_for(slug):
    slug = (slug or "").strip()
    return f"/begin/product/{slug}" if slug else ""
