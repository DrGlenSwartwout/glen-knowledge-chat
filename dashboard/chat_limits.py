"""Pure, Flask-free rate-limit + tier policy for the public chat.
Keep this importable with no app/network deps so it unit-tests in isolation."""
import ipaddress
import threading
import time
from datetime import datetime, timedelta

# Tunable in ONE place. per_min/per_day are per-IP velocity; monthly_full_words
# is the per-email/per-member full-answer ceiling (None = no hard wall).
LIMITS = {
    "anonymous":  {"per_min": 10, "per_day": 40,  "monthly_full_words": None,    "flag_full_words": None},
    "registered": {"per_min": 15, "per_day": 60,  "monthly_full_words": 10_000,  "flag_full_words": None},
    "member":     {"per_min": 30, "per_day": 150, "monthly_full_words": None,    "flag_full_words": 100_000},
}

def client_ip(xff: str, remote_addr: str) -> str:
    """First X-Forwarded-For hop, else remote_addr. IPv6 collapsed to /64."""
    raw = (xff or "").split(",")[0].strip() or (remote_addr or "").strip()
    if not raw:
        return "anon"
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return raw
    if ip.version == 6:
        net = ipaddress.ip_network(f"{raw}/64", strict=False)
        return f"{net.network_address}/64"
    return raw

class VelocityLimiter:
    """In-memory per-IP sliding-window counter. Two windows: 60s and 86400s."""
    _MIN_WINDOW = 60
    _DAY_WINDOW = 86_400

    def __init__(self, clock=time.time):
        self._clock = clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str, per_min: int, per_day: int) -> tuple[bool, int]:
        now = self._clock()
        with self._lock:
            hits = [t for t in self._hits.get(ip, []) if now - t < self._DAY_WINDOW]
            if not hits:
                self._hits.pop(ip, None)
            minute = [t for t in hits if now - t < self._MIN_WINDOW]
            if len(minute) >= per_min:
                self._hits[ip] = hits
                return (False, self._MIN_WINDOW - int(now - minute[0]))
            if len(hits) >= per_day:
                self._hits[ip] = hits
                return (False, self._DAY_WINDOW - int(now - hits[0]))
            hits.append(now)
            self._hits[ip] = hits
            return (True, 0)

def tier_for(has_auth: bool, has_membership: bool, has_email: bool) -> str:
    if has_membership:
        return "member"
    if has_auth or has_email:
        return "registered"
    return "anonymous"

def is_flagged(cx, session_id: str, ip: str, now_iso: str, within_hours: int = 24) -> bool:
    """True if a recent abuse_flags row matches this session_id OR ip."""
    try:
        cutoff = (datetime.fromisoformat(now_iso) - timedelta(hours=within_hours)).isoformat()
        row = cx.execute(
            "SELECT 1 FROM abuse_flags WHERE ts >= ? AND (session_id = ? OR ip = ?) LIMIT 1",
            (cutoff, session_id or "", ip or ""),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def monthly_full_words(cx, email: str, now_iso: str) -> int:
    if not email:
        return 0
    try:
        cutoff = (datetime.fromisoformat(now_iso) - timedelta(days=30)).isoformat()
    except Exception:
        return 0
    row = cx.execute(
        "SELECT COALESCE(SUM(word_count),0) FROM query_log "
        "WHERE email=? AND mode='full' AND ts >= ?",
        (email, cutoff),
    ).fetchone()
    return int(row[0] or 0)
