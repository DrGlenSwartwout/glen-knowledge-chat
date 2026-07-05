"""Zoom Server-to-Server OAuth + scheduled-meeting creation. Stdlib-only."""
import json, base64, urllib.request, urllib.parse

_TOKEN_CACHE = {}  # client_id -> (token, expiry_epoch)


def get_token(account_id, client_id, client_secret, *, _now=None):
    import time
    now = _now if _now is not None else time.time()
    cached = _TOKEN_CACHE.get(client_id)
    if cached and cached[1] > now:
        return cached[0]
    url = "https://zoom.us/oauth/token?" + urllib.parse.urlencode(
        {"grant_type": "account_credentials", "account_id": account_id})
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Authorization": f"Basic {basic}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    tok = d["access_token"]
    _TOKEN_CACHE[client_id] = (tok, now + int(d.get("expires_in", 3600)) - 300)
    return tok


def create_meeting(token, *, host, topic, start_iso, duration_min,
                   timezone="Pacific/Honolulu", opener=None):
    opener = opener or urllib.request.urlopen
    body = {"topic": topic, "type": 2, "start_time": start_iso,
            "duration": int(duration_min), "timezone": timezone,
            "settings": {"waiting_room": True, "join_before_host": False}}
    req = urllib.request.Request(
        f"https://api.zoom.us/v2/users/{host}/meetings",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with opener(req, timeout=30) as r:
        d = json.load(r)
    return {"join_url": d.get("join_url"), "meeting_id": str(d.get("id") or ""),
            "start_url": d.get("start_url")}
