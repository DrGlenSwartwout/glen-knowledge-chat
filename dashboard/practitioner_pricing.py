"""Pure pricing for practitioner drop-ship: blended base, flat 33% fee, $/% selling-price
with MAP, practitioner margin. Wraps dashboard.wholesale_pricing for the blended base."""
from dashboard import wholesale_pricing as _wp

DEFAULTS = {
    "fee_pct": 0.33,            # service fee on the practitioner's markup (drop-ship only)
    "map_default_cents": 6700,  # $67 minimum advertised price (per-SKU override in console)
}

def load_settings(overrides):
    s = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None:
            s[k] = v
    return s

def drop_ship_base_cents(qty, modules_completed):
    """Per-bottle blended wholesale base for a drop-ship of `qty` bottles at the
    practitioner's certification level (same curve as wholesale stocking)."""
    return _wp.blended_unit_price_cents(int(qty), int(modules_completed), _wp.DEFAULT_B)

def service_fee_cents(selling_cents, base_cents, settings):
    """Flat fee = fee_pct of the markup (selling - base), never negative. Drop-ship only."""
    markup = max(0, int(selling_cents) - int(base_cents))
    return int(round(settings["fee_pct"] * markup))

class MapViolation(ValueError):
    """Selling price resolves below the Minimum Advertised Price."""

def price_for_markup(markup_pct, retail_cents):
    return int(round(int(retail_cents) * (1 + float(markup_pct) / 100.0)))

def markup_pct_for(price_cents, retail_cents):
    if not retail_cents:
        return 0.0
    return round((int(price_cents) - int(retail_cents)) / int(retail_cents) * 100.0, 1)

def resolve_selling_cents(price_input, *, retail_cents, map_cents):
    """price_input: {"price_cents": int} OR {"markup_pct": number} OR {} (default retail).
    Returns the selling price in cents; raises MapViolation if it is below MAP (advertised)."""
    if price_input.get("price_cents") is not None:
        s = int(price_input["price_cents"])
    elif price_input.get("markup_pct") is not None:
        s = price_for_markup(price_input["markup_pct"], retail_cents)
    else:
        s = int(retail_cents)
    if s < int(map_cents):
        raise MapViolation(f"{s} below MAP {map_cents}")
    return s
