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
