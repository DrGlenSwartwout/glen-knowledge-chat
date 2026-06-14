"""
dashboard/rewards.py — Tier resolution, affiliate cash-earnings ledger, settings.

Pure module: accepts a sqlite connection; no Flask imports, no side effects at import.
All money values are integer cents.
"""

import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULTS = {
    "referral_reward_pct": 0.05,
    "cash_out_threshold_cents": 10000,
    "cash_out_face_pct": 0.70,
}


def load_settings(overrides: dict) -> dict:
    """Merge overrides into DEFAULTS, skipping None values."""
    settings = dict(DEFAULTS)
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


# ---------------------------------------------------------------------------
# People / tags helpers
# ---------------------------------------------------------------------------

def tags_for_email(cx, email: str) -> set:
    """Return the set of tags for a person (case-insensitive email lookup)."""
    row = cx.execute(
        "SELECT tags FROM people WHERE lower(email) = lower(?)", (email,)
    ).fetchone()
    if row is None:
        return set()
    return set(json.loads(row["tags"] or "[]"))


# ---------------------------------------------------------------------------
# Affiliate slug helpers
# ---------------------------------------------------------------------------

def referrer_email_for_slug(cx, slug: str):
    """Return the referrer's email for an approved slug, or None."""
    row = cx.execute(
        "SELECT email FROM affiliate_signups WHERE slug = ? AND status = 'approved'",
        (slug,),
    ).fetchone()
    return row["email"] if row else None


def reward_mode_for_slug(cx, slug: str) -> str:
    """Return 'cash' if the referrer has the tier:pro-influencer tag, else 'points'."""
    email = referrer_email_for_slug(cx, slug)
    if email is None:
        return "points"
    tags = tags_for_email(cx, email)
    return "cash" if "tier:pro-influencer" in tags else "points"


# ---------------------------------------------------------------------------
# Affiliate earnings ledger
# ---------------------------------------------------------------------------

def init_affiliate_earnings_table(cx) -> None:
    """Create the affiliate_earnings table if it doesn't already exist."""
    cx.executescript(
        """
        CREATE TABLE IF NOT EXISTS affiliate_earnings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT    NOT NULL,
            email       TEXT    NOT NULL,
            order_ref   TEXT    NOT NULL,
            amount_cents INTEGER NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'pending',
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            paid_at     TEXT,
            UNIQUE (slug, order_ref)
        );
        CREATE INDEX IF NOT EXISTS idx_affiliate_earnings_slug
            ON affiliate_earnings (slug);
        """
    )
    cx.commit()


def accrue_cash(cx, *, slug: str, email: str, order_ref: str, amount_cents: int) -> None:
    """Record cash earnings for a referrer. Idempotent per (slug, order_ref)."""
    cx.execute(
        """
        INSERT OR IGNORE INTO affiliate_earnings (slug, email, order_ref, amount_cents)
        VALUES (?, ?, ?, ?)
        """,
        (slug, email, order_ref, amount_cents),
    )
    cx.commit()


def pending_cash_total(cx, slug: str) -> int:
    """Return total pending cash earnings (in cents) for the given slug."""
    row = cx.execute(
        "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM affiliate_earnings "
        "WHERE slug = ? AND status = 'pending'",
        (slug,),
    ).fetchone()
    return row["total"] if row else 0


def mark_paid(cx, slug: str) -> None:
    """Mark all pending earnings for the slug as paid."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cx.execute(
        "UPDATE affiliate_earnings SET status = 'paid', paid_at = ? "
        "WHERE slug = ? AND status = 'pending'",
        (now, slug),
    )
    cx.commit()
