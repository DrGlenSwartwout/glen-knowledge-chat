"""5-minute TTL cache for expensive external API calls."""

import time
from threading import Lock

_store = {}
_lock = Lock()
DEFAULT_TTL = 300  # 5 minutes


def cached(key, ttl=DEFAULT_TTL):
    """Decorator factory: cache function result under `key` for `ttl` seconds."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            now = time.time()
            with _lock:
                entry = _store.get(key)
                if entry and now - entry["t"] < ttl:
                    return entry["v"]
            try:
                v = fn(*args, **kwargs)
                with _lock:
                    _store[key] = {"v": v, "t": now, "ok": True}
                return v
            except Exception as e:
                with _lock:
                    stale = _store.get(key)
                    if stale:
                        stale["stale_error"] = str(e)
                        return stale["v"]
                raise
        return wrapper
    return deco


def last_success(key):
    """Return ISO timestamp of last successful cache write for `key`, or None."""
    with _lock:
        entry = _store.get(key)
        if not entry:
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry["t"]))


def clear(key=None):
    """Clear one key or entire cache."""
    with _lock:
        if key is None:
            _store.clear()
        else:
            _store.pop(key, None)
