#!/usr/bin/env python3
"""Run from LOCAL Mac (not Render) to drain the GHL write-queue.
Render's AWS IP is blocked by GHL's Cloudflare WAF; the Mac's residential IP is
not. Uses curl to also bypass JA3 TLS fingerprint blocking. Mirrors
sync-ghl-leads.py.

Usage:
  doppler run --project remedy-match --config prd -- python3 sync-ghl-writes.py [--dry-run]
"""
import json
import os
import subprocess
import sys
import urllib.parse

RENDER_URL     = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GHL_API_KEY    = os.environ.get("GHL_API_KEY", "")
GHL_BASE       = "https://rest.gohighlevel.com/v1"
GHL_PIPELINE_ID = "A6LWJMBoIsOFBMeCa6NY"
GHL_STAGE_NEW   = "397c5fb2-1612-4b7a-aa14-f0dac42a7fda"
GHL_WORKFLOW_ID = "0b02dd3e-b82a-4032-a575-f9269afbd3ac"
DRY_RUN = "--dry-run" in sys.argv


def _curl(method, url, headers=None, payload=None):
    # -w appends the HTTP status on its own trailing line so a GHL 4xx is treated
    # as an error (marked failed + visible) rather than silently "done".
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method, url]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    out = r.stdout or ""
    nl = out.rfind("\n")
    code = out[nl + 1:].strip() if nl >= 0 else ""
    body = out[:nl] if nl >= 0 else out
    try:
        data = json.loads(body) if body.strip() else {}
    except Exception:
        data = None
    if code and not code.startswith("2"):
        return data, f"HTTP {code}: {body[:150]}"
    if data is None:
        return None, body[:200]
    return data, None


def _ghl_headers():
    return {"Authorization": f"Bearer {GHL_API_KEY}"}


def _find_contact(email):
    data, _ = _curl("GET", f"{GHL_BASE}/contacts/lookup?email={urllib.parse.quote(email)}",
                    _ghl_headers())
    contacts = (data or {}).get("contacts") or []
    return contacts[0] if contacts else None


def _do_op(item):
    """Execute one queued GHL write. Returns (ok, message)."""
    op = item["op"]
    email = item["email"]
    payload = json.loads(item.get("payload_json") or "{}")
    contact = _find_contact(email)
    if not contact and op != "tag_add":
        return False, "contact not found"
    cid = contact["id"] if contact else None

    if op in ("tag_add", "tag_remove"):
        tag = payload.get("tag") or payload.get("tags")
        tags = set((contact or {}).get("tags") or [])
        for t in ([tag] if isinstance(tag, str) else (tag or [])):
            if op == "tag_add":
                tags.add(t)
            else:
                tags.discard(t)
        if not cid:  # create the contact with the tag
            data, err = _curl("POST", f"{GHL_BASE}/contacts/",
                              _ghl_headers(), {"email": email, "tags": sorted(tags)})
            return (err is None), (err or "created")
        _, err = _curl("PUT", f"{GHL_BASE}/contacts/{cid}", _ghl_headers(),
                       {"tags": sorted(tags)})
        return (err is None), (err or "tagged")

    if op == "note":
        _, err = _curl("POST", f"{GHL_BASE}/contacts/{cid}/notes", _ghl_headers(),
                       {"body": payload.get("note", "")})
        return (err is None), (err or "noted")

    if op == "opportunity":
        _, err = _curl("POST", f"{GHL_BASE}/pipelines/{GHL_PIPELINE_ID}/opportunities",
                       _ghl_headers(), {"stageId": GHL_STAGE_NEW, "contactId": cid,
                                        "title": payload.get("title", email), "status": "open"})
        return (err is None), (err or "opportunity created")

    if op == "workflow":
        wf = payload.get("workflow_id") or GHL_WORKFLOW_ID
        _, err = _curl("POST", f"{GHL_BASE}/contacts/{cid}/workflow/{wf}",
                       _ghl_headers(), {})
        return (err is None), (err or "enrolled")

    return False, f"unknown op {op}"


def _render(method, path, payload=None):
    return _curl(method, f"{RENDER_URL}{path}",
                 {"X-Webhook-Secret": WEBHOOK_SECRET}, payload)


def main():
    if not (GHL_API_KEY and WEBHOOK_SECRET):
        print("Set GHL_API_KEY + WEBHOOK_SECRET (use doppler run ...)")
        sys.exit(1)
    data, err = _render("GET", "/api/ghl/queue/pending")
    items = (data or {}).get("queue") or []
    print(f"Found {len(items)} queued GHL writes")
    done = failed = 0
    for it in items:
        print(f"  [{it['id']}] {it['op']} {it['email']}")
        if DRY_RUN:
            continue
        ok, msg = _do_op(it)
        _render("POST", "/api/ghl/queue/result",
                {"id": it["id"], "status": "done" if ok else "failed", "result": msg})
        print(f"    -> {'done' if ok else 'FAILED'}: {msg}")
        done += ok
        failed += (not ok)
    if not DRY_RUN:
        print(f"\nDone: {done} synced, {failed} failed")


if __name__ == "__main__":
    main()
