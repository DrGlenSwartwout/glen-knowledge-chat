"""Business-OS lightweight Home-board signals for the modules whose full domain
logic is a future phase. Each reads real local data (SQLite or a DATA_DIR JSON
file) and is defensive (gray on any error). Registers on import."""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dashboard.signals import signal, RED, AMBER, GREEN, GRAY  # noqa: F401 (RED reserved)

_REPO = Path(__file__).resolve().parent.parent


def _data_file(name):
    """Find a DATA_DIR JSON file across the env-var path and the repo data dir."""
    for base in (os.environ.get("DATA_DIR"), str(_REPO / "data"), str(_REPO)):
        if not base:
            continue
        p = Path(base) / name
        if p.exists():
            return p
    return None


def _plural(n):
    return "s" if n != 1 else ""


@signal("marketing")
def marketing_signal(cx, actor=None):
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM inbound_leads "
            "WHERE source='scoreapp' AND (status IS NULL OR status='pending') "
            "  AND (last_outbound_at IS NULL OR last_outbound_at='')").fetchone()[0]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No new quiz leads", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} new quiz lead{_plural(n)} to reach",
            "top_actions": [{"label": "Open people", "href": "/console"}], "count": n}


@signal("content")
def content_signal(cx, actor=None):
    p = _data_file("atlas-pending.json")
    if not p:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    try:
        n = len(json.loads(p.read_text()).get("concepts") or [])
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No concepts to review", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} atlas concept{_plural(n)} to approve",
            "top_actions": [{"label": "Review atlas", "href": "/admin/atlas"}], "count": n}


@signal("comms")
def comms_signal(cx, actor=None):
    try:
        rows = cx.execute(
            "SELECT start FROM calendar_events WHERE status='visible'").fetchall()
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    now = datetime.now(timezone.utc)
    soon = now + timedelta(hours=48)
    n = 0
    for r in rows:
        s = r[0]
        if not s:
            continue
        try:
            if "T" in s or " " in s:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s[:10])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if now <= dt <= soon:
                n += 1
        except Exception:
            continue
    if n == 0:
        return {"level": GREEN, "summary": "Nothing in the next 48h", "top_actions": [], "count": 0}
    return {"level": AMBER, "summary": f"{n} event{_plural(n)} in the next 48h",
            "top_actions": [{"label": "Open console", "href": "/console"}], "count": n}


@signal("b2b")
def b2b_signal(cx, actor=None):
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM orders "
            "WHERE source IN ('wholesale','dispensary') AND status IN ('new','packed')").fetchone()[0]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if n == 0:
        return {"level": GREEN, "summary": "No active B2B orders", "top_actions": [], "count": 0}
    return {"level": GREEN, "summary": f"{n} active practitioner order{_plural(n)}",
            "top_actions": [{"label": "Open orders", "href": "/console/orders"}], "count": n}
