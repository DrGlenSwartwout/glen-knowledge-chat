# Studio.com Bridge — Flow A (clinical-wedge welcome) Plan — Mechanic 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A welcome landing page for Studio.com users that layers Dr. Glen's clinical wedge on top of their daily app — a free Biofield voice scan + remedy match, the deeper AI Q&A, free membership, and the free first month of live group — and tags them Studio-sourced for attribution. Reuses existing destinations; ships dark behind `STUDIO_BRIDGE_ENABLED`.

**Architecture:** A single static page + a `/studio` route that serves it (no-store) and stamps a `rm_ref=studio` last-touch attribution cookie (the same mechanism `/begin` uses) plus an `amg_session` cookie, so any later membership/order is attributed `source=studio`. The page links to the EXISTING capabilities — no new backend. Flow B's claim page (`/studio/claim`, #115) is one of the cards.

**Tech Stack:** Python 3.11, Flask, static HTML, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-upgrade-incentive-ladder-design.md` (Mechanic 2, Flow A — "positioned right after the free offers; mostly reuse of the existing free funnel"). Studio.com pays us rev-share on participants, so the wedge converts their users into our Biofield/remedy/group revenue.

**Reuse (link targets, all existing):** `Truly.VIP/E4L` (free voice scan), the concierge chat (`/begin`), `/studio/claim` (free group month, #115), the free-member funnel. The `rm_ref` cookie + `amg_session` pattern from `/begin` (app.py ~1213-1255).

**Test invocation:** App route test → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_studio_welcome.py -q` (worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `/studio` welcome route + page + source tag

**Files:** Modify `app.py`; Create `static/studio-welcome.html`; Test `tests/test_studio_welcome.py`

- [ ] **Step 1: Failing test**

```python
import importlib

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app

def test_studio_welcome_served_no_store_and_sets_source(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/studio")
    assert r.status_code == 200
    assert b"Studio" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
    # stamps a Studio attribution cookie + a session
    cookies = r.headers.getlist("Set-Cookie")
    joined = " ".join(cookies)
    assert "rm_ref=studio" in joined
    assert "amg_session=" in joined

def test_studio_welcome_does_not_clobber_existing_ref(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    c.set_cookie("rm_ref", "someaffiliate")   # adapt to the test client's set_cookie signature
    r = c.get("/studio")
    assert r.status_code == 200
    # an existing affiliate ref is last-touch preserved (do NOT overwrite a real referral)
    assert "rm_ref=studio" not in " ".join(r.headers.getlist("Set-Cookie"))
```

> If the installed Werkzeug's `set_cookie` signature differs, adapt the helper; keep the two assertions' intent (sets `rm_ref=studio` + `amg_session` on a fresh visit; does NOT overwrite an existing `rm_ref`).

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** the route in `app.py` (mirror `/begin`'s cookie handling):

```python
@app.route("/studio")
def studio_welcome():
    resp = send_from_directory(STATIC, "studio-welcome.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    # last-touch attribution: only stamp source=studio if no real referral is already set
    if not (request.cookies.get("rm_ref") or "").strip():
        resp.set_cookie("rm_ref", "studio", max_age=60 * 60 * 24 * 90,
                        httponly=False, samesite="Lax", secure=request.is_secure)
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```
(Match the exact cookie attrs `/begin` uses for `rm_ref`/`amg_session` — copy them so behavior is identical.)

- [ ] **Step 4: Create `static/studio-welcome.html`.** A clean landing page in Dr. Glen's voice (no em dashes, no ALL CAPS; ™ where natural). Headline: "You've got the daily Studio app. Now add Dr. Glen's clinical layer, free." Then cards/CTAs wiring the existing destinations:
  1. **Free Biofield voice scan + remedy match** (the wedge — what a phone app can't do) → `https://Truly.VIP/E4L`.
  2. **Ask Dr. Glen's AI** (deeper, remedy-aware Q&A beyond the app) → `/begin`.
  3. **Your first month of live group coaching, free** → `/studio/claim`.
  4. A line on **free membership + courses** → `/begin`.
  Short, warm, n=1 framing where proof is mentioned; the page is positioning, not a checkout. Style to match the other `begin-*`/`studio-claim` pages. The word "Studio" must appear (the test checks it).

- [ ] **Step 5: Run → pass** + page parses (`html.parser`). 

- [ ] **Step 6: Commit** — `feat(studio-flowA): /studio welcome page (clinical wedge + source tag)`

---

### Task 2: doc

**Files:** Modify `docs/studio-bridge.md` (append a Flow A section)

- [ ] **Step 1:** Append to `docs/studio-bridge.md`: Flow A — `/studio` welcome page layering the clinical wedge (free voice scan + remedy match, the AI Q&A, free membership, the Flow B free group month) on Studio.com users; stamps `rm_ref=studio` (last-touch, doesn't overwrite a real affiliate ref) for attribution; reuses existing destinations, no new backend; same `STUDIO_BRIDGE_ENABLED` feature. Note the flywheel (Studio pays us rev-share; the wedge converts their users into Biofield/remedy/group revenue).
- [ ] **Step 2:** Run `tests/test_studio_welcome.py` once more — green.
- [ ] **Step 3:** Commit — `docs(studio-flowA): Flow A welcome page`

---

## Self-review
- **Spec coverage:** clinical-wedge welcome (voice scan + remedy match + deeper Q&A + free membership) positioned for Studio users + the Flow B group-month card (Task 1); Studio-source attribution via `rm_ref=studio` last-touch (Task 1); reuse-only, no new backend.
- **Type consistency:** route `/studio` → `static/studio-welcome.html`; cookie `rm_ref=studio` (only if unset) + `amg_session`.
- **Deferred:** a dedicated Studio-user onboarding sequence / a Studio-specific concierge mode (the generic funnel + chat suffice); auto-detecting Studio membership (no email export).
- **Risk:** minimal — a static positioning page + a cookie; no money, no PHI; doesn't overwrite an existing affiliate referral (last-touch guard).

## Done
A `/studio` welcome page layers Dr. Glen's clinical wedge on Studio.com users and tags them Studio-sourced, wiring the existing free funnel, voice scan, chat, and free-group-month claim.
