"""Settings — generic key/value store + named accessors.

Backs the /console/settings page. Active-Mac lives here.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

LOG_DB = str(Path(__file__).parent.parent / "chat_log.db")

KNOWN_HOSTNAMES = {
    "Drs-MacBook-Pro.local":  "MacBook Pro",
    "Mac-Studio.local":       "Mac Studio (upper)",
    "Drs-Mac-Studio.local":   "Mac Studio (lower-right)",
}


def _ensure_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS settings_kv (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def get(key, default=None):
    _ensure_table()
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT value FROM settings_kv WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set(key, value):
    _ensure_table()
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO settings_kv (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, str(value), ts))


def get_active_mac():
    """Returns the hostname designated as the active write Mac.

    Default: MacBook Pro (the one Glen primarily works on).
    """
    return get("active_mac", "Drs-MacBook-Pro.local")


def set_active_mac(hostname):
    if hostname not in KNOWN_HOSTNAMES:
        raise ValueError(f"Unknown hostname: {hostname}. Must be one of {list(KNOWN_HOSTNAMES)}")
    set("active_mac", hostname)


def active_mac_payload():
    return {
        "active_mac": get_active_mac(),
        "known_macs": [
            {"hostname": h, "label": label} for h, label in KNOWN_HOSTNAMES.items()
        ],
    }
