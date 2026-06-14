"""Local sqlite store for per-practitioner pricing + white-label branding.

Schema (in chat_log.db):
    practitioner_settings(
        practitioner_id TEXT PRIMARY KEY,
        branding_json   TEXT DEFAULT '{}',
        pricing_json    TEXT DEFAULT '{}',
        updated_at      TEXT
    )

Public API
----------
init_settings_table(cx)
get_settings(cx, pid)          -> {"branding": {...}, "pricing": {...}}
set_branding(cx, pid, dict)
set_pricing(cx, pid, dict)
price_cents_for(cx, pid, slug, *, retail_cents, map_cents) -> int
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from dashboard import practitioner_pricing as _pp

_PRICING_DEFAULTS = {"default_markup_pct": 0, "overrides": {}}


def init_settings_table(cx) -> None:
    """Create the practitioner_settings table if it does not exist."""
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS practitioner_settings (
            practitioner_id TEXT PRIMARY KEY,
            branding_json   TEXT NOT NULL DEFAULT '{}',
            pricing_json    TEXT NOT NULL DEFAULT '{}',
            updated_at      TEXT
        )
        """
    )
    cx.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_settings(cx, pid: str) -> dict:
    """Return stored settings for practitioner *pid*.

    Falls back to empty branding and default pricing when no row exists or
    when the stored JSON is missing expected keys.
    """
    row = cx.execute(
        "SELECT branding_json, pricing_json FROM practitioner_settings WHERE practitioner_id = ?",
        (pid,),
    ).fetchone()

    if row is None:
        branding = {}
        pricing = dict(_PRICING_DEFAULTS)
        chat_enabled = False
    else:
        branding = json.loads(row["branding_json"] or "{}")
        raw_pricing = json.loads(row["pricing_json"] or "{}")
        pricing = {
            "default_markup_pct": raw_pricing.get("default_markup_pct", 0),
            "overrides": raw_pricing.get("overrides", {}),
        }
        # chat_enabled lives in branding_json to avoid a schema migration
        chat_enabled = bool(branding.pop("chat_enabled", False))

    return {"branding": branding, "pricing": pricing, "chat_enabled": chat_enabled}


def set_branding(cx, pid: str, branding: dict, *, chat_enabled: bool | None = None) -> None:
    """Upsert the branding JSON for *pid*, leaving pricing unchanged.

    If *chat_enabled* is provided it is stored alongside the branding JSON so
    that no schema migration is needed.
    """
    # Read the current branding_json so we can preserve existing chat_enabled
    # when the caller doesn't pass it explicitly.
    row = cx.execute(
        "SELECT branding_json FROM practitioner_settings WHERE practitioner_id = ?",
        (pid,),
    ).fetchone()
    existing = json.loads((row["branding_json"] if row else None) or "{}")
    existing_chat_enabled = existing.get("chat_enabled", False)

    stored = dict(branding)
    # Preserve or update chat_enabled in the blob
    stored["chat_enabled"] = chat_enabled if chat_enabled is not None else existing_chat_enabled

    cx.execute(
        """
        INSERT INTO practitioner_settings (practitioner_id, branding_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(practitioner_id) DO UPDATE SET
            branding_json = excluded.branding_json,
            updated_at    = excluded.updated_at
        """,
        (pid, json.dumps(stored), _now()),
    )
    cx.commit()


def set_pricing(cx, pid: str, pricing: dict) -> None:
    """Upsert the pricing JSON for *pid*, leaving branding unchanged."""
    cx.execute(
        """
        INSERT INTO practitioner_settings (practitioner_id, pricing_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(practitioner_id) DO UPDATE SET
            pricing_json = excluded.pricing_json,
            updated_at   = excluded.updated_at
        """,
        (pid, json.dumps(pricing), _now()),
    )
    cx.commit()


def price_cents_for(cx, pid: str, slug: str, *, retail_cents: int, map_cents: int) -> int:
    """Return the practitioner's selling price for *slug* in cents.

    Resolution order:
    1. Per-SKU dollar override  (overrides[slug])
    2. Default markup %         (price_for_markup(default_markup_pct, retail_cents))
    3. Retail price             (default_markup_pct == 0)

    Always clamped up to *map_cents* (Minimum Advertised Price).
    """
    settings = get_settings(cx, pid)
    pricing = settings["pricing"]

    overrides = pricing.get("overrides", {})
    if slug in overrides:
        price = int(overrides[slug])
    else:
        markup_pct = pricing.get("default_markup_pct", 0)
        price = _pp.price_for_markup(markup_pct, retail_cents)

    # MAP clamp — never sell below the minimum advertised price.
    return max(price, int(map_cents))
