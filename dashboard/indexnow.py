"""IndexNow instant-notify (Bing / Yandex / Seznam / Naver) — best-effort, key-gated.

The *presence* of INDEXNOW_KEY in the environment IS the enablement (no separate
feature flag): when it is set, the approve action pings IndexNow and the app
serves the key-ownership file at /<key>.txt. When it is unset, every entry point
no-ops silently.

Note: Google does NOT consume IndexNow. Google is served by the dynamic
/learn/sitemap.xml (now carrying <lastmod>), which it re-crawls on its own
schedule. This module covers the engines that DO support instant push.
"""
import os
from urllib.parse import urlsplit

ENDPOINT = "https://api.indexnow.org/indexnow"


def key():
    """The configured IndexNow key, or '' when the feature is off."""
    return (os.environ.get("INDEXNOW_KEY") or "").strip()


def submit(url, *, base_url="", k=None, http=None):
    """Best-effort IndexNow ping for a single URL.

    Returns True only if a request was actually issued, False when disabled,
    misconfigured, or the request errored. NEVER raises — callers (e.g. the
    approve action) must not be broken by a notify failure.
    """
    k = (k if k is not None else key()).strip()
    if not k or not url:
        return False
    try:
        host = urlsplit(url).netloc
        if not host:
            return False
        key_location = (f"{base_url.rstrip('/')}/{k}.txt"
                        if base_url else f"https://{host}/{k}.txt")
        payload = {"host": host, "key": k, "keyLocation": key_location,
                   "urlList": [url]}
        if http is None:
            import requests as http
        http.post(ENDPOINT, json=payload,
                  headers={"Content-Type": "application/json; charset=utf-8"},
                  timeout=5)
        return True
    except Exception as exc:  # noqa: BLE001 - notify must never fail the caller
        print(f"[indexnow] ping skipped for {url}: {exc}", flush=True)
        return False
