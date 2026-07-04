"""Per-practitioner product-discount controls. Pure builder + SQLite config.

Config shape (both schedules optional; program has an extra 'enabled' master flag):
{
  "standard": {"same_sku": {"enabled": bool, "dial": 0..1}, "program_total": {...}, "open_total": {...}},
  "program":  {"enabled": bool, "same_sku": {...}, "program_total": {...}, "open_total": {...}}
}

Also includes pure pricing for practitioner drop-ship: blended base, flat 33% fee, $/% selling-price
with MAP, practitioner margin. Wraps dashboard.wholesale_pricing for the blended base."""
import json
import sqlite3
from datetime import datetime, timezone

from dashboard import wholesale_pricing as _wp
from dashboard import pricing as _pricing

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

def quote_line(*, selling_cents, qty, modules, settings):
    """Per-bottle economics for a drop-ship line. base = blended at this qty+cert;
    fee = 33% of markup; margin = selling - base - fee (>=0); dropship_wholesale = base+fee
    (what the practitioner pays in practitioner-paid mode).

    NOTE: `selling_cents` must already have passed MAP validation via
    `resolve_selling_cents` — quote_line does NOT re-check MAP (the advertised-price floor
    is enforced at price resolution, before a quote is ever built)."""
    base = drop_ship_base_cents(qty, modules)
    fee = service_fee_cents(selling_cents, base, settings)
    margin = max(0, int(selling_cents) - base - fee)
    return {
        "line_selling_cents": int(selling_cents),
        "base_cents": base,
        "fee_cents": fee,
        "margin_cents": margin,
        "dropship_wholesale_cents": base + fee,
    }


# Config persistence layer

_TYPES = ("same_sku", "program_total", "open_total")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS practitioner_pricing ("
        "practitioner_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    cx.commit()


def get_config(cx, pid):
    init_table(cx)
    row = cx.execute(
        "SELECT config_json FROM practitioner_pricing WHERE practitioner_id=?", (str(pid),)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def set_config(cx, pid, config):
    init_table(cx)
    cx.execute(
        "INSERT INTO practitioner_pricing (practitioner_id, config_json, updated_at) "
        "VALUES (?,?,?) ON CONFLICT(practitioner_id) DO UPDATE SET "
        "config_json=excluded.config_json, updated_at=excluded.updated_at",
        (str(pid), json.dumps(config), _now()),
    )
    cx.commit()


# Pure ceilings + effective_settings builder

def _ceiling_anchors(gcfg, ptype):
    # open_total's ceiling is the program_total curve (private-channel decision).
    key = "program_total" if ptype == "open_total" else ptype
    return [list(a) for a in gcfg[key]["anchors"]]


def ceilings(settings):
    gcfg = _pricing._discount_cfg(_pricing.load_settings(settings))
    return {t: float(_ceiling_anchors(gcfg, t)[-1][1]) for t in _TYPES}


def _scaled(anchors, dial):
    d = max(0.0, min(1.0, float(dial)))
    return [[a[0], round(a[1] * d, 4)] for a in anchors]


def effective_settings(config, *, program_member, settings):
    base = _pricing.load_settings(settings)
    gcfg = _pricing._discount_cfg(base)
    cfg = config or {}
    use_program = bool(program_member) and bool((cfg.get("program") or {}).get("enabled"))
    sched = cfg.get("program" if use_program else "standard") or {}
    disc = {}
    for t in _TYPES:
        ent = sched.get(t) or {}
        ceil = _ceiling_anchors(gcfg, t)
        if bool(ent.get("enabled")):
            disc[t] = {"enabled": True, "anchors": _scaled(ceil, ent.get("dial", 0.0))}
        else:
            disc[t] = {"enabled": False, "anchors": ceil}
    out = dict(base)
    out["discounts"] = disc
    return out
