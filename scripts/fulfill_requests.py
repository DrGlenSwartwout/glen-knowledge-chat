#!/usr/bin/env python3
"""Local fulfillment worker for portal analysis-requests.

FMP snapshot is Mac-only, so fulfillment must run here: for each pending request
it resolves the client's latest FMP causal-chain test, builds the portal report,
generates the AI narrative (by request), publishes to prod UN-BLURRED, pins it as
current, emails the client their link, and marks the request done.

Run via: bash ~/deploy-chat/fulfill_requests.sh   (sets CONSOLE_SECRET + OPENAI + paths)
Env: DRY=1 to preview without writing.
"""
import os, sqlite3, json, urllib.request, urllib.error, re, time
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_report import causal_chain_report
from dashboard import fmp_biofield as fb
bpp.authored_report = causal_chain_report

DB = os.path.join(os.environ.get("DATA_DIR", os.path.expanduser("~/deploy-chat")), "chat_log.db")
KEY = os.environ.get("CONSOLE_SECRET", "")
BASE = "https://illtowell.com"
DRY = os.environ.get("DRY", "0") == "1"


def api_get(url, key=None):
    req = urllib.request.Request(url, headers=({"X-Console-Key": key} if key else {}))
    return json.load(urllib.request.urlopen(req, timeout=30))


def api_post(url, body, key):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
        headers={"X-Console-Key": key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def to_iso(s):
    s = (s or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        mo, d, y = m.groups(); y = int(y); y += 2000 if y < 100 else 0
        return f"{y:04d}-{int(mo):02d}-{int(d):02d}"
    return s


def latest_test(cx, email):
    r = cx.execute(
        "SELECT cc.id_fk_test FROM fmp_snap_client_causal_chain cc "
        "JOIN fmp_clients c ON c.id_pk=cc.id_fk_client "
        "WHERE lower(c.email)=? ORDER BY CAST(cc.id_fk_test AS INTEGER) DESC LIMIT 1",
        (email,)).fetchone()
    return r[0] if r else None


def client_name(cx, email):
    r = cx.execute("SELECT name_first, name_last FROM fmp_clients WHERE lower(email)=? LIMIT 1",
                   (email,)).fetchone()
    return (" ".join([x for x in (r or ("", "")) if x]).strip()) if r else ""


def fulfill_one(cx, req):
    email = (req["email"] or "").strip().lower()
    name = client_name(cx, email)
    tid = latest_test(cx, email)
    if tid is None:
        return "no-fmp-test", None
    payload = bpp.build_portal_content(cx, str(tid), special_price_cents=0)
    content = payload.get("content") or {}
    layers = content.get("layers") or []
    if not layers:
        return "empty-build", None
    # AI narrative by request: fill blank meanings via the app LLM
    if not any((l.get("meaning") or "").strip() for l in layers):
        prose = fb.draft_prose(layers, name) or {}
        pl = prose.get("layers") or []
        if prose.get("greeting"):
            content["greeting"] = prose["greeting"]
        for i, l in enumerate(layers):
            if i < len(pl):
                l["title"] = pl[i].get("title") or l.get("title")
                l["meaning"] = pl[i].get("meaning") or l.get("meaning") or ""
    iso = to_iso(payload.get("scan_date"))
    payload["scan_date"] = iso
    payload["name"] = name or payload.get("name")
    if DRY:
        return f"WOULD-fulfill (test {tid}, {len(layers)}L, {iso})", iso
    res = bpp.publish_to_portal(payload, base_url=BASE, console_key=KEY, send=True)
    if not res.get("ok"):
        return f"publish-error:{str(res)[:60]}", iso
    api_post(f"{BASE}/api/console/portal/set-current", {"email": email, "scan_date": iso}, KEY)
    return ("published+notified" if res.get("emailed") else "published+pinned(email-unconfirmed)"), iso


def main():
    q = api_get(f"{BASE}/api/console/analysis-requests?limit=50", KEY)
    reqs = q.get("requests", [])
    print(f"MODE={'DRY' if DRY else 'LIVE'}  pending={len(reqs)}")
    if not reqs:
        print("Nothing to fulfill.")
        return
    cx = sqlite3.connect(DB)
    for req in reqs:
        try:
            action, iso = fulfill_one(cx, req)
        except Exception as e:
            action = f"ERROR:{str(e)[:80]}"
        print(f"  [{req.get('id')}] {req.get('email'):34} req={req.get('scan_date')}  -> {action}")
        if not DRY and not str(action).startswith(("ERROR", "publish-error", "no-fmp", "empty")):
            try:
                api_post(f"{BASE}/api/console/analysis-requests/{req.get('id')}/complete",
                         {"status": "done"}, KEY)
            except Exception as e:
                print(f"        (mark-done failed: {str(e)[:50]})")
        time.sleep(0.4)
    cx.close()


if __name__ == "__main__":
    main()
