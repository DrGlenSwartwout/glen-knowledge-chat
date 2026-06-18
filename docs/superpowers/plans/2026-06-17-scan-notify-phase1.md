# Scan Notification — Phase 1 (notify + preferences + taper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The moment a scan arrives, text (Twilio) + email the client a stable link to their Healing Oasis, with a 3-message taper and unified opt-in/out across SMS, email, and the portal.

**Architecture:** A server-side per-client `portal_notify_state` record (opt status, notify count, engaged, phone, a stored stable portal token) drives eligibility; the local scan trigger reads it, sends SMS+email, and reports the send. Phases 2–3 (processing + unfold) build on `engaged` + the stable link.

**Tech Stack:** Flask + sqlite (server), Python (local trigger: Twilio REST via urllib + smtplib). Spec: `docs/superpowers/specs/2026-06-17-scan-notify-ondemand-unfold-design.md` (Phase 1 = its components 1–3 + 6).

---

## Design decisions filling spec gaps (FLAG for Glen)
- **Stable notification link via a stored token.** Portal tokens are stored hash-only (unrecoverable), but notifications must re-send the *same* link every time. Decision: `portal_notify_state` stores the raw portal token so the link is stable and retrievable without rotating it each scan. The token is already shared with the client, so storing it for our own re-send is a modest, intentional tradeoff (slightly weaker than hash-only-at-rest). If Glen prefers zero raw-token storage, the alternative is a stable HMAC-signed `/p/<sig>` link (bigger change). **Going with stored-token for Phase 1.**
- **Notify-all** (every scanner), per the approved spec.

## Two repos
- **DEPLOY-CHAT** worktree `/tmp/wt-deploy-chat-5326cc61` branch `sess/5326cc61-notify` (Tasks 1–4, PR). Suite: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`
- **VAULT** `~/AI-Training/02 Skills/` (Task 5, auto-snapshot).

**Prerequisite (Glen):** Twilio creds in Doppler — `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` (a sending number). Until set, the sender skips SMS and sends email only (no crash). `SMTP_USER`/`SMTP_PASS` already exist.

---

## Task 1: `portal_notify_state` table + accessors + eligibility

**Files:** Create `dashboard/notify_state.py`; Test `tests/test_notify_state.py`

- [ ] **Step 1: Failing test**

```python
import sqlite3
from dashboard import notify_state as N


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); N.init_table(cx); return cx


def test_defaults_and_decide_taper(tmp_path):
    cx = _cx(tmp_path)
    # unknown client: eligible, variant 0
    d = N.decide(N.get_state(cx, "a@x.com"))
    assert d["eligible"] is True and d["variant"] == 0
    N.incr_notify(cx, "a@x.com"); N.incr_notify(cx, "a@x.com")     # count=2
    d = N.decide(N.get_state(cx, "a@x.com"))
    assert d["eligible"] is True and d["variant"] == 2             # 3rd (last-call)
    N.incr_notify(cx, "a@x.com")                                    # count=3
    assert N.decide(N.get_state(cx, "a@x.com"))["eligible"] is False  # quiet after 3


def test_opt_out_suppresses_and_in_overrides(tmp_path):
    cx = _cx(tmp_path)
    for _ in range(5): N.incr_notify(cx, "b@x.com")
    N.set_opt(cx, "b@x.com", "in")
    assert N.decide(N.get_state(cx, "b@x.com"))["eligible"] is True   # opt-in overrides cap
    N.set_opt(cx, "b@x.com", "out")
    assert N.decide(N.get_state(cx, "b@x.com"))["eligible"] is False  # opt-out hard stop


def test_engaged_keeps_eligible_past_cap(tmp_path):
    cx = _cx(tmp_path)
    for _ in range(4): N.incr_notify(cx, "c@x.com")
    N.mark_engaged(cx, "c@x.com")
    assert N.decide(N.get_state(cx, "c@x.com"))["eligible"] is True
```

- [ ] **Step 2: Run → FAIL** (`-m pytest tests/test_notify_state.py -q`)

- [ ] **Step 3: Implement** — `dashboard/notify_state.py`:

```python
"""Per-client notification preference + engagement state (Phase 1 of scan-notify).
One row per email: opt status (default/in/out), notify_count (taper), engaged,
phone, and the stored stable portal token for the notification link."""
import datetime
import sqlite3

MAX_TAPER = 3  # default (non-engaged, non-opted) clients get this many, then quiet


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_notify_state (
        email TEXT PRIMARY KEY, phone TEXT, opt_status TEXT DEFAULT 'default',
        notify_count INTEGER DEFAULT 0, engaged INTEGER DEFAULT 0,
        portal_token TEXT, updated_at TEXT)""")
    cx.commit()


def _norm(email):
    return (email or "").strip().lower()


def get_state(cx, email):
    init_table(cx)
    row = cx.execute("SELECT email, phone, opt_status, notify_count, engaged, portal_token "
                     "FROM portal_notify_state WHERE email=?", (_norm(email),)).fetchone()
    if not row:
        return {"email": _norm(email), "phone": "", "opt_status": "default",
                "notify_count": 0, "engaged": False, "portal_token": ""}
    return {"email": row[0], "phone": row[1] or "", "opt_status": row[2] or "default",
            "notify_count": row[3] or 0, "engaged": bool(row[4]), "portal_token": row[5] or ""}


def _upsert(cx, email, **fields):
    init_table(cx)
    email = _norm(email)
    if not cx.execute("SELECT 1 FROM portal_notify_state WHERE email=?", (email,)).fetchone():
        cx.execute("INSERT INTO portal_notify_state (email, updated_at) VALUES (?,?)", (email, _now()))
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
    cx.execute(f"UPDATE portal_notify_state SET {sets} WHERE email=?",
               (*fields.values(), _now(), email))
    cx.commit()


def set_opt(cx, email, status):           # status in {'in','out','default'}
    _upsert(cx, email, opt_status=status)


def set_phone(cx, email, phone):
    _upsert(cx, email, phone=(phone or "").strip())


def set_token(cx, email, token):
    _upsert(cx, email, portal_token=token or "")


def mark_engaged(cx, email):
    _upsert(cx, email, engaged=1)


def incr_notify(cx, email):
    s = get_state(cx, email)
    _upsert(cx, email, notify_count=s["notify_count"] + 1)


def decide(state):
    """Eligibility + which of the 3 message variants to send (0,1,2) or None."""
    if state["opt_status"] == "out":
        return {"eligible": False, "variant": None}
    if state["opt_status"] == "in" or state["engaged"]:
        return {"eligible": True, "variant": min(state["notify_count"], MAX_TAPER - 1)}
    if state["notify_count"] < MAX_TAPER:
        return {"eligible": True, "variant": state["notify_count"]}
    return {"eligible": False, "variant": None}


def email_by_phone(cx, phone):
    """Reverse lookup for the Twilio inbound (STOP/START) webhook."""
    init_table(cx)
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())[-10:]
    if not digits:
        return None
    for row in cx.execute("SELECT email, phone FROM portal_notify_state").fetchall():
        if "".join(ch for ch in (row[1] or "") if ch.isdigit()).endswith(digits):
            return row[0]
    return None
```

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** (`-m "notify: portal_notify_state model + taper/opt eligibility"`)

---

## Task 2: opt channels + mark-engaged — portal pref, unsubscribe, Twilio inbound

**Files:** Modify `app.py`; Test `tests/test_notify_routes.py` (new)

- [ ] **Step 1: Failing tests**

```python
def test_portal_notify_pref_sets_opt(client):
    c, appmod = client
    tok = _seed_portal(appmod, "p@y.com", "P", {"layers": []})
    assert c.post(f"/api/portal/{tok}/notify-pref", json={"pref": "out"}).status_code == 200
    import sqlite3; from dashboard import notify_state as N
    cx = sqlite3.connect(appmod.LOG_DB)
    assert N.get_state(cx, "p@y.com")["opt_status"] == "out"


def test_unsubscribe_link_opts_out(client):
    c, appmod = client
    tok = _seed_portal(appmod, "u@y.com", "U", {"layers": []})
    assert c.get(f"/unsubscribe?token={tok}").status_code == 200
    import sqlite3; from dashboard import notify_state as N
    cx = sqlite3.connect(appmod.LOG_DB)
    assert N.get_state(cx, "u@y.com")["opt_status"] == "out"


def test_twilio_inbound_stop_start(client):
    c, appmod = client
    import sqlite3; from dashboard import notify_state as N
    cx = sqlite3.connect(appmod.LOG_DB); N.init_table(cx); N.set_phone(cx, "t@y.com", "+15551230000"); cx.commit()
    c.post("/sms/inbound", data={"From": "+15551230000", "Body": "STOP"})
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "t@y.com")["opt_status"] == "out"
    c.post("/sms/inbound", data={"From": "+15551230000", "Body": "START"})
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "t@y.com")["opt_status"] == "in"


def test_open_marks_engaged(client):
    c, appmod = client
    tok = _seed_portal(appmod, "e@y.com", "E", {"layers": [{"n": 1, "title": "t"}]})
    c.get(f"/api/portal/{tok}")            # opening the content endpoint
    import sqlite3; from dashboard import notify_state as N
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "e@y.com")["engaged"] is True
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add to `app.py` (near the other portal routes):

```python
@app.route("/api/portal/<token>/notify-pref", methods=["POST"])
def api_portal_notify_pref(token):
    from dashboard import client_portal as _cp, notify_state as _ns
    pref = ((request.get_json(silent=True) or {}).get("pref") or "").strip().lower()
    if pref not in ("in", "out"):
        return jsonify({"error": "pref must be in|out"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        portal = _cp.get_portal_by_token(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        _ns.set_opt(cx, portal["email"], pref)
    return jsonify({"ok": True, "pref": pref})


@app.route("/unsubscribe", methods=["GET"])
def portal_unsubscribe():
    from dashboard import client_portal as _cp, notify_state as _ns
    token = (request.args.get("token") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        portal = _cp.get_portal_by_token(cx, token)
        if portal:
            _ns.set_opt(cx, portal["email"], "out")
    return ("You're unsubscribed from Healing Oasis notifications. "
            "You can re-enable them anytime from your portal."), 200


@app.route("/sms/inbound", methods=["POST"])
def sms_inbound():
    from dashboard import notify_state as _ns
    frm = (request.form.get("From") or request.values.get("From") or "").strip()
    body = (request.form.get("Body") or request.values.get("Body") or "").strip().upper()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        email = _ns.email_by_phone(cx, frm)
        if email:
            if body in ("STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "QUIT"):
                _ns.set_opt(cx, email, "out")
            elif body in ("START", "YES", "UNSTOP"):
                _ns.set_opt(cx, email, "in")
    return ("", 204)
```

  And in `api_client_portal` (the content endpoint, ~line 7238), right after `portal = _portal_record_for(cx, token)` succeeds (inside the `with` that has `cx`), mark engaged:

```python
        try:
            from dashboard import notify_state as _ns
            _ns.mark_engaged(cx, (portal.get("email") or ""))
        except Exception as e:
            print(f"[engaged] {e!r}", flush=True)
```
(Place it where `portal` is known and truthy. Best-effort; never break the page.)

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite.** **Step 6: Commit** (`-m "notify: opt-in/out (portal/unsubscribe/SMS) + mark-engaged on open"`)

---

## Task 3: admin bridge — ensure stable link + notify-state read + notify-sent

**Files:** Modify `app.py` + `dashboard/client_portal.py`; Test `tests/test_notify_routes.py`

- [ ] **Step 1: Failing test**

```python
def test_admin_notify_state_returns_decision_and_link(client):
    c, appmod = client
    j = c.post("/api/admin/notify-state?key=test-secret",
               json={"email": "ns@y.com", "name": "NS"}).get_json()
    assert j["eligible"] is True and j["variant"] == 0
    assert j["url"].startswith("https://") and "/portal/" in j["url"]   # stable link minted
    url1 = j["url"]
    # idempotent: same link next call (stored token reused, not rotated)
    j2 = c.post("/api/admin/notify-state?key=test-secret", json={"email": "ns@y.com"}).get_json()
    assert j2["url"] == url1
    # record the send -> count increments
    c.post("/api/admin/notify-sent?key=test-secret", json={"email": "ns@y.com"})
    j3 = c.post("/api/admin/notify-state?key=test-secret", json={"email": "ns@y.com"}).get_json()
    assert j3["variant"] == 1


def test_admin_notify_state_requires_key(client):
    c, _ = client
    assert c.post("/api/admin/notify-state", json={"email": "x@y.com"}).status_code == 401
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.**
  (a) Add `client_portal.ensure_token(cx, email, name="")` — returns a STABLE raw token, creating an empty pending portal if none exists, reusing the stored one otherwise:

```python
def ensure_token(cx, email, name=""):
    """Stable raw token for notification links. Creates a pending portal if the
    client has none. The raw token is held in portal_notify_state so the link is
    re-sendable without rotating it. Returns the raw token."""
    from dashboard import notify_state as _ns
    email = (email or "").strip().lower()
    st = _ns.get_state(cx, email)
    if st.get("portal_token"):
        return st["portal_token"]
    if not cx.execute("SELECT 1 FROM client_portals WHERE email=?", (email,)).fetchone():
        upsert_portal(cx, email, name, {"biofield_status": "pending"})
    token = secrets.token_urlsafe(32)
    cx.execute("UPDATE client_portals SET token_hash=?, updated_at=? WHERE email=?",
               (_hash(token), _now_iso(), email))
    _ns.set_token(cx, email, token)
    cx.commit()
    return token
```
  (Note: for an existing portal with no stored token, this rotates once to a known token, then stores it — stable thereafter. Document this in the function as done above.)

  (b) Add the admin endpoints to `app.py`:

```python
@app.route("/api/admin/notify-state", methods=["POST"])
def api_admin_notify_state():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_portal as _cp, notify_state as _ns
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        if body.get("phone"):
            _ns.set_phone(cx, email, body["phone"])
        token = _cp.ensure_token(cx, email, body.get("name") or "")
        d = _ns.decide(_ns.get_state(cx, email))
    return jsonify({**d, "url": f"{PUBLIC_BASE_URL}/portal/{token}",
                    "unsubscribe": f"{PUBLIC_BASE_URL}/unsubscribe?token={token}"})


@app.route("/api/admin/notify-sent", methods=["POST"])
def api_admin_notify_sent():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import notify_state as _ns
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _ns.incr_notify(cx, email)
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite.** **Step 6: Commit** (`-m "notify: admin bridge — stable link + notify-state/notify-sent"`)

---

## Task 4: PR the server side
- [ ] Run the FULL suite → green. Push `sess/5326cc61-notify`; open a PR (base main) titled "Scan notification Phase 1 (server): preferences, taper, opt-in/out, stable link". Body: lists the model + 5 endpoints; notes the local sender (Task 5) + Twilio creds are the remaining go-live pieces. Do NOT include the Phase 2/3 work.

---

## Task 5: local sender + trigger wiring (VAULT)

**Files:** Create `02 Skills/scan-notify.py`; Modify `02 Skills/e4l-email-trigger.sh`; vault auto-snapshots

- [ ] **Step 1:** Create `02 Skills/scan-notify.py` — given `--email --name --phone`, it: (a) POSTs `/api/admin/notify-state` (with phone) to get `{eligible, variant, url, unsubscribe}`; (b) if not eligible, exits 0 quietly; (c) sends **SMS via Twilio** (REST API over urllib basic-auth to `https://api.twilio.com/2010-04-01/Accounts/<SID>/Messages.json`, body = the `variant`-selected copy + `url`) — **skip if `TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM` unset**; (d) sends **email via smtplib** (STARTTLS 587, `SMTP_USER/SMTP_PASS`, From Dr. Glen, body = copy + `url` + the `unsubscribe` link); (e) POSTs `/api/admin/notify-sent`. The 3 copy variants:
  - v0: "Aloha {name}! Your E4L scan is in — Dr. Glen's system is preparing your personalized analysis in your Healing Oasis: {url}"
  - v1: "Your Healing Oasis analysis is ready to explore, {name} — your scan mapped into a personalized healing path: {url}"
  - v2 (last-call): "Last reminder, {name} — we won't keep nudging. Tap to see your analysis (and stay in the loop): {url}"
  Email always includes the unsubscribe link; SMS relies on Twilio STOP.
- [ ] **Step 2:** Syntax check: `python3 -m py_compile "02 Skills/scan-notify.py"` → `py OK`. (Do NOT live-send.)
- [ ] **Step 3:** Wire into `e4l-email-trigger.sh`: for each newly-detected scan client, resolve email (from `e4l_clients`, as the existing autodraft block does) + phone (one `/api/people?q=<email>` call), then run `scan-notify.py --email --name --phone` (via doppler so Twilio/SMTP/CONSOLE creds are present). Gate behind a new `SCAN_NOTIFY_ENABLED` env (dark by default, like the autodraft flag), so it ships off until Glen sets Twilio creds + flips it. `zsh -n` the script.
- [ ] **Step 4:** Vault auto-snapshots (no git commit).

---

## Self-Review notes
- **Spec coverage (Phase 1):** notification sender (T5), unified preference across SMS/email/portal (T2), eligibility + 3-variant taper (T1), engaged-on-open (T2), admin bridge + stable link (T3). Phases 2–3 (processing, unfold) correctly excluded.
- **Type consistency:** `decide(state)->{eligible,variant}`; state keys `email,phone,opt_status,notify_count,engaged,portal_token`; `opt_status ∈ {default,in,out}`; `ensure_token(cx,email,name)->raw token`; admin notify-state returns `{eligible,variant,url,unsubscribe}`.
- **Verify during impl:** the content endpoint's `portal` var name + where it's truthy (T2 mark-engaged); `_hash`/`_now_iso`/`secrets` already imported in client_portal.py (T3a); the `client` fixture + `_seed_portal` helper in the test files; Twilio/SMTP env names (`TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM`, `SMTP_USER/SMTP_PASS`).
- **FLAG:** the stored-raw-token decision (top of plan) — confirm acceptable, else switch to HMAC-signed links.
