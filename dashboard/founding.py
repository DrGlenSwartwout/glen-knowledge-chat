"""Founding-launch config + counter. Pure module: the only I/O is reading the
JSON config at import and counting rows via a caller-supplied sqlite connection."""

import json
import logging
import os

from dashboard import subscriptions as _subs

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "founding_launches.json")


def _load():
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("founding: could not load %s: %s", _CONFIG_PATH, e)
        return {}


_CONFIG = _load()


def get_launch(slug: str) -> dict | None:
    return _CONFIG.get(slug)


def count_reserved(cx, slug: str) -> int:
    return _subs.count_founding(cx, slug)


def remaining(cx, slug: str) -> int:
    launch = get_launch(slug)
    if not launch:
        return 0
    return max(0, int(launch.get("cap", 0)) - count_reserved(cx, slug))


def is_open(cx, slug: str, *, now_iso: str | None = None) -> bool:
    launch = get_launch(slug)
    if not launch:
        return False
    if remaining(cx, slug) <= 0:
        return False
    closes_at = (launch.get("closes_at") or "").strip()
    if closes_at and now_iso and now_iso >= closes_at:
        return False
    return True
