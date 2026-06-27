# Universal Portal — Implementation Plan (sub-project A)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.
> **Spec:** `docs/superpowers/specs/2026-06-27-universal-portal-design.md`.
> **Base:** branch off current `main` (`universal-portal`). Includes the merged portal concierge (#375).

**Goal:** Every remedy buyer gets a portal — minted + emailed on purchase (any channel), a useful no-scan home, and a first-entry TOS gate.

**Architecture:** One fail-open hook in `_ingest_order` mints a portal (idempotent, email-keyed) and emails the `/portal/<token>` link once. The portal payload gains `tos_agreed`; the page gains a no-scan (`biofield_status:"none"`) home and a TOS gate; a new `POST /api/portal/<token>/agree-tos` records consent.

**Tech Stack:** Flask, raw sqlite3, vanilla JS in `client-portal.html`. No new dependencies.

## Global Constraints

- No new dependencies. Reuse existing helpers (exact anchors below).
- **FAIL OPEN in the order path:** portal-mint/email must NEVER raise into `_ingest_order` (it already wraps everything in try/except and "never raises into a checkout path"). Keep that guarantee.
- **One welcome email per portal:** suppression check + `portal_welcome.mark_welcome_sent` once-guard before sending; background thread.
- **Idempotent mint:** `upsert_portal` mints a token only on first-create; a repeat order returns no token → no email.
- TOS agreement must PERSIST (a DB error on `agree-tos` returns a retryable error, never silently drops it).
- Straight ASCII quotes in JS/Python; no smart quotes as delimiters.
- Skip orders with empty email.

## Verified anchors

- `_ingest_order` (app.py:24916) — the single hook; ALL 16 checkout callers route through it; `email`/`name` are kwargs. `_bos_orders.upsert_order(...)` is called at ~:24927 inside a try/except.
- `client_portal.upsert_portal(cx, email, name, content) -> (raw_token_or_None, id)` (mints token first-create only, email-keyed).
- `_send_full_report_email(to, name, subject, body)` (app.py:7162; Gmail→SMTP→log; suppression-aware). `PUBLIC_BASE_URL` (app.py:182). `portal_welcome.mark_welcome_sent(cx, email) -> bool` (dashboard/portal_welcome.py:13). `email_suppression.is_suppressed(cx, email)` (dashboard/email_suppression.py:16). Existing once-guard send pattern: app.py:395-413.
- `record_unlock(cx, *, session_id, trigger, email, tos, tos_version, ...)` (begin_funnel.py:218) — **session_id REQUIRED**; use `_entry_session_id(email)` (app.py:570, deterministic `"entry:"+sha1(email)[:16]`). `BEGIN_TOS_VERSION` constant exists. `is_member(session_id="", email="")` (app.py:550) checks TOS by email.
- `/api/portal/<token>` GET (app.py:11576) returns the payload dict (~:11639); `biofield_status` from `content.get("biofield_status") or "confirmed"` (:11608). `_portal_record_for(cx, token) -> {email,name,content}` (app.py:11263).
- `client-portal.html`: `render(d, v)` (:237); token `seg` from `location.pathname` (:173); `load()` fetches `/api/portal/<seg>` + `/view` (:206); `isPending` spinner/poll branch (:306); concierge "Ask Dr. Glen" section always renders (:443); account (:256), orders (:472).

---

### Task 1: `dashboard/portal_provision.py` — `ensure_portal_for_buyer`

**Files:** Create `dashboard/portal_provision.py`; Test `tests/test_portal_provision.py`.

**Interfaces — Produces:** `ensure_portal_for_buyer(cx, email, name) -> str | None` — mints a portal for `email` if none exists (content `{"biofield_status": "none"}`), returns the raw token **only if newly minted** (else None). Empty email → None.

- [ ] **Step 1: Failing tests**

```python
# tests/test_portal_provision.py
import sqlite3
from dashboard import client_portal as cp
from dashboard.portal_provision import ensure_portal_for_buyer

def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); cp.init_client_portal_table(cx); return cx

def test_mints_token_first_time(tmp_path):
    cx = _cx(tmp_path)
    tok = ensure_portal_for_buyer(cx, "Buyer@X.com ", "Buyer")
    assert tok and isinstance(tok, str)

def test_idempotent_repeat_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert ensure_portal_for_buyer(cx, "b@x.com", "B")          # first mints
    assert ensure_portal_for_buyer(cx, "b@x.com", "B") is None  # repeat: no new token

def test_empty_email_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert ensure_portal_for_buyer(cx, "", "B") is None
    assert ensure_portal_for_buyer(cx, None, "B") is None

def test_mints_none_status(tmp_path):
    cx = _cx(tmp_path)
    ensure_portal_for_buyer(cx, "c@x.com", "C")
    rec = cp.get_portal_content_by_email(cx, "c@x.com")
    assert (rec.get("content") or {}).get("biofield_status") == "none"
```

- [ ] **Step 2: Run → FAIL** — `python3 -m pytest tests/test_portal_provision.py -q`
- [ ] **Step 3: Implement** `dashboard/portal_provision.py`:

```python
"""Mint a client portal for a buyer at order time (idempotent, email-keyed).
Returns the raw token only when a NEW portal is minted, so the caller emails once."""
from dashboard import client_portal as _cp

def ensure_portal_for_buyer(cx, email, name):
    em = (email or "").strip().lower()
    if not em:
        return None
    token, _id = _cp.upsert_portal(cx, em, (name or "").strip(), {"biofield_status": "none"})
    return token   # non-None only on first create
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(portal): ensure_portal_for_buyer (idempotent mint on order)`

---

### Task 2: Wire mint + welcome email into `_ingest_order`

**Files:** Modify `app.py` (`_ingest_order` ~:24916 + a new `_send_portal_welcome` helper). Test `tests/test_order_mints_portal.py`.

**Interfaces — Consumes:** `ensure_portal_for_buyer` (Task 1). **Produces:** an order with a new email mints a portal + sends one welcome email (background, once-guarded, suppression-aware).

- [ ] **Step 1: `_send_portal_welcome(email, name, token)` helper** (mirror app.py:395-413 — suppression + once-guard, then background send):

```python
def _send_portal_welcome(email, name, token):
    em = (email or "").strip().lower()
    if not em or not token:
        return
    try:
        from dashboard import email_suppression as _es, portal_welcome as _pw
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _es.init_table(cx)
            if _es.is_suppressed(cx, em):
                return
            if not _pw.mark_welcome_sent(cx, em):   # already sent
                return
        url = f"{PUBLIC_BASE_URL}/portal/{token}"
        body = (f"Aloha {name or ''},\n\nYour personal healing home is ready:\n\n{url}\n\n"
                f"It is where your remedies, protocol, and your concierge live. "
                f"Reply anytime.\n\nWith aloha,\nDr. Glen & Rae")
        import threading
        threading.Thread(target=_send_full_report_email,
                         args=(em, name, "Your healing home is ready 🌺", body),
                         daemon=True).start()
    except Exception as e:
        print(f"[portal-welcome] {em}: {e!r}", flush=True)
```

- [ ] **Step 2: Hook into `_ingest_order`** — right after the `_bos_orders.upsert_order(...)` call, still inside the try (so the existing outer try/except keeps it fail-open):

```python
            _bos_orders.upsert_order(cx, source=source, external_ref=external_ref, ...)
            # NEW: every buyer gets a portal home (idempotent, fail-open)
            try:
                from dashboard import portal_provision as _pp
                _tok = _pp.ensure_portal_for_buyer(cx, email, name)
                if _tok:
                    _send_portal_welcome(email, name, _tok)
            except Exception as _pe:
                print(f"[orders] portal-provision {source}/{external_ref}: {_pe!r}", flush=True)
```

- [ ] **Step 3: Test** `tests/test_order_mints_portal.py` (reload-app convention; under Doppler). Monkeypatch `app._send_full_report_email` to a recorder (avoid real email). Call `app._ingest_order(source="test", external_ref="o1", email="new@x.com", name="N", items=[{"name":"X","qty":1}], total_cents=1000)`; assert a `client_portals` row exists for `new@x.com` AND `_send_full_report_email` was invoked once with a body containing `/portal/`. Call `_ingest_order` again (`external_ref="o2"`, same email) → assert NO second send (once-guard) and no new portal token. An order with `email=""` → no portal, no send.

Run: `S=/tmp/uos; mkdir -p $S; doppler run -p remedy-match -c prd -- env DATA_DIR=$S python3 -m pytest tests/test_order_mints_portal.py -q -p no:cacheprovider` → PASS.

- [ ] **Step 4: Commit** — `feat(portal): mint portal + welcome email on every order (fail-open)`

---

### Task 3: Portal payload `tos_agreed` + `biofield_status:"none"` + `agree-tos` route

**Files:** Modify `app.py` (`/api/portal/<token>` payload ~:11639 + a new `/api/portal/<token>/agree-tos` route). Test `tests/test_portal_tos.py`.

- [ ] **Step 1: Add `tos_agreed` to the payload.** In the `/api/portal/<token>` GET handler, compute the portal email (the handler already resolves it — reuse that var, e.g. `email_for_reports`) and add to the returned dict:

```python
        "tos_agreed": is_member(email=_portal_email) if _portal_email else True,
```
(Default True when no email so we never gate a portal we can't key — fail-open toward access. Confirm the exact email variable name in the handler.) `biofield_status:"none"` already flows through `content.get("biofield_status")` — no server change needed beyond confirming it isn't coerced.

- [ ] **Step 2: Add `POST /api/portal/<token>/agree-tos`:**

```python
@app.route("/api/portal/<token>/agree-tos", methods=["POST", "OPTIONS"])
def api_portal_agree_tos(token):
    if request.method == "OPTIONS":
        return "", 200
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        from dashboard import client_portal as _cp
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        if not email:
            return jsonify({"ok": True}), 200   # nothing to key TOS to
        try:
            begin_funnel.record_unlock(cx, session_id=_entry_session_id(email),
                                       trigger="tos", email=email, tos=True,
                                       tos_version=BEGIN_TOS_VERSION)
        except Exception as e:
            print(f"[portal-tos] {email}: {e!r}", flush=True)
            return jsonify({"error": "could not record"}), 500
    return jsonify({"ok": True})
```

- [ ] **Step 3: Test** `tests/test_portal_tos.py` (reload-app; Doppler). Seed a portal (`client_portal.upsert_portal(cx, "t@x.com", "T", {"biofield_status":"none"})` → token). Assert the portal payload `tos_agreed` is False initially; `POST /api/portal/<token>/agree-tos` → 200 and `app.is_member(email="t@x.com")` is now True; re-fetch payload → `tos_agreed` True. Bad token → 404.

Run under Doppler → PASS.

- [ ] **Step 4: Commit** — `feat(portal): tos_agreed in payload + /agree-tos route`

---

### Task 4: `client-portal.html` — TOS gate + no-scan home (+ render-verify)

**Files:** Modify `static/client-portal.html` (`render(d, v)` ~:237). Controller render-verify.

- [ ] **Step 1: TOS gate** — at the TOP of `render(d, v)`, before building the home sections:

```javascript
  if (d && d.tos_agreed === false) {
    var g = '<div class="card"><h2>Welcome to your healing home</h2>'
      + '<p>Before we continue, please review and agree to our Terms of Service.</p>'
      + '<p><a href="' + esc((window.TOS_URL || "https://illtowell.com/terms")) + '" target="_blank" rel="noopener">Read the Terms</a></p>'
      + '<button class="btn" id="tosAgreeBtn">I agree to the Terms</button>'
      + '<div class="err" id="tosErr" hidden></div></div>';
    document.getElementById("app").innerHTML = g;
    var b = document.getElementById("tosAgreeBtn");
    if (b) b.addEventListener("click", async function () {
      b.disabled = true;
      try {
        var r = await fetch("/api/portal/" + encodeURIComponent(seg) + "/agree-tos", {method:"POST", credentials:"same-origin"});
        if (r.ok) { load(); return; }
        throw new Error("retry");
      } catch (e) {
        b.disabled = false;
        var er = document.getElementById("tosErr"); if (er) { er.textContent = "Please try again."; er.hidden = false; }
      }
    });
    return;   // suppress the home until agreed
  }
```
(Only gate on explicit `false` — undefined/true renders the home. `seg` and `load` are already in scope.)

- [ ] **Step 2: No-scan home branch** — after the `isPending` block (~:306), add an `else if`:

```javascript
  } else if (d.biofield_status === "none") {
    html += '<div class="card"><h2>Curious what your body is asking for?</h2>'
      + '<p>A quick biofield voice scan reads your stress and terrain patterns and tailors your next steps. It takes about 30 seconds.</p>'
      + '<a class="btn" href="https://Truly.VIP/E4L" target="_blank" rel="noopener">Do your free voice scan</a></div>';
  }
```
This sits alongside the always-rendered account / order-history / "Ask Dr. Glen" concierge sections — so a no-scan buyer sees: account + orders + concierge + this scan CTA, and **no** "Preparing your Biofield Analysis…" spinner, no `process-request`, no polling. Confirm the account/orders/concierge sections render independent of `biofield_status` (they do — they're keyed on `v.account`/`v.orders` and the concierge card is unconditional).

- [ ] **Step 3: Straight ASCII quotes only** — after editing, grep the changed lines for smart quotes (U+2018/2019/201C/201D); must be 0.
- [ ] **Step 4: RENDER-VERIFY (controller, headless, gevent server)** per `feedback_render_verify_not_just_inject` (boot gunicorn `--worker-class gevent`): seed two portals — one `biofield_status:"none"` with NO TOS, one with TOS recorded.
  - No-TOS portal → the TOS gate renders; clicking "I agree" (stub the agree-tos fetch ok) → `load()` re-renders the home.
  - TOS portal (status "none") → home shows: account/orders/concierge + the scan CTA, and assert NO element with the "Preparing your Biofield Analysis" text and NO spinner; `renderSuggestion`/`sendChatMessage` (concierge) present.
  - Zero console errors.
- [ ] **Step 5: Commit** — `feat(portal): first-entry TOS gate + no-scan home`

---

## Verification (end-to-end)

- **Unit (plain):** `tests/test_portal_provision.py`.
- **Under Doppler:** order→mint (`test_order_mints_portal.py`), TOS (`test_portal_tos.py`).
- **Render-verify (headless, gevent):** TOS gate → agree → home; no-scan home shows account+orders+concierge+scan-CTA with NO spinner; zero console errors.
- **Manual smoke:** place a test order with a fresh email → a `/portal/<token>` welcome email is sent once; open it → TOS gate → agree → no-scan home with the concierge; a repeat order sends nothing new.
- **Pre-PR hygiene:** `git diff --name-only origin/main..HEAD | grep -i superpowers` and `git rm --cached` any `.superpowers/sdd` scratch leak.

## Out of scope

Retiring the funnel concierge; backfilling portals for historical buyers (a later one-off `ensure_portal_for_buyer` sweep over existing `orders` emails); enforcing TOS at non-funnel checkouts (captured at portal entry instead). No new commerce.
