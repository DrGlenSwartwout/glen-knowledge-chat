# Begin Page #1 — Hero + Conversational Identity Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the top of `static/begin.html` into a clean hero (headline + sub + intro video + a simplified self-healing chat) where the AI asks the visitor's name in its first message, captures it conversationally to the one identity record, asks for email at the activation moment to grant Tier-1 membership, and moves the "Explore everything" link to the bottom gated on membership.

**Architecture:** Front-end-only rework of `static/begin.html` plus one backend regression test. A new ungated hero `<section>` is inserted at the top of `<body>` containing the headline, sub, the (relocated) intro video, and a new stripped-down chat panel that streams from the existing `POST /begin/match/chat`. All identity writes reuse the existing `/begin/unlock` backbone: name capture posts `trigger="name"`, activation posts `trigger="tos"` with `email`+`tos=true`. No new server triggers, no new identity store, no feature flag.

**Tech Stack:** Flask (Python 3.11), vanilla JS in `static/begin.html`, SSE streaming from `/begin/match/chat`, SQLite `journey_state` via `begin_funnel`, pytest + Flask test client.

## Global Constraints

- **No emoji. No em dashes.** Client-facing copy uses plain hyphens and text/SVG only (Glen's rule).
- **Live page, no feature flag.** `/begin` is the funnel front door; a manual visual pass is REQUIRED before deploy. Do not gate behind a flag.
- **One human, one record (hard requirement).** Every identity write routes through `/begin/unlock` -> `begin_funnel.record_unlock` -> `journey_state` (unioned by `amg_session` cookie + email). Never create a parallel identity store.
- **Reuse EXISTING triggers only.** `name`, `email`, `tos` are already in `begin_funnel.VALID_TRIGGERS`. Name capture = `trigger="name"` with `name`/`first_name`. Activation = `trigger="tos"` with `email` + `tos=true`; `compute_rung` already promotes email+tos_agreed to `free_tier` -> `is_member` true and fires the existing GHL onboarding hook. DO NOT add new triggers; DO NOT add an `activate` trigger.
- **Hero chat reuses `POST /begin/match/chat`** (the self-healing remedy conversation), streamed. Default `for_whom: "self"` so the visitor can type immediately (no "who is this for" gate). Render stripped-down: message bubbles + input only, NONE of the Rate / Leave-feedback / View-feedback controls.
- **Do not regress existing below-hero content.** The existing layered cards/reveal sections stay intact below the hero (sub-project #2 replaces them later). Only the redundant old layer1 hero-greeting + ask-form is hidden, since the hero supersedes it.
- **Identity backbone is already proven.** `tests/test_begin_routes.py::test_begin_unlock_name_then_email_tos_reaches_free_tier` already pins `name` -> personalize and `tos`+email -> free_tier. Build on it; do not duplicate it.
- **Test harness (deploy-chat):** load app via `importlib.import_module("app")` with `repo_root` on `sys.path` (skip if not importable); `monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))`; create tables with `begin_funnel.init_journey_tables(cx)`; use `app_module.app.test_client()`; set the session cookie with `client.set_cookie("amg_session", "s1")`; on any free-tier transition mock `ghl_onboard_contact` and `_capture_concierge_referral`. NO live Supabase.
- **Run tests:** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_hero_identity.py -v` then `-k "begin"` for no regression. Two pre-existing unrelated failures (`test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`) are expected and ignored.

---

### Task 1: Lock the membership contract the hero depends on (backend test)

The hero relies on two server behaviors: (a) activation via `trigger="tos"` + email + tos sets `is_member` true for BOTH the session and the email, and `/begin/state` reports `tos_agreed_at`; (b) a name captured by session and an email captured at activation resolve to ONE state (union). These behaviors already exist; this task pins them with a focused regression test so a later change cannot silently break the hero. (Characterization tests: they pass on first run.)

**Files:**
- Test: `tests/test_begin_hero_identity.py` (create)

**Interfaces:**
- Consumes (already in codebase): `app.app` (Flask), `app.LOG_DB`, `app.is_member(session_id="", email="")`, `app.ghl_onboard_contact`, `app._capture_concierge_referral`, `begin_funnel.init_journey_tables(cx)`. Route `POST /begin/unlock` accepts `{trigger, name, first_name, email, tos, session_id}`; `GET /begin/state` returns `{current_rung, first_name, email, tos_agreed_at, reveal, ...}`.
- Produces: nothing consumed by later tasks (pure safety net). Confirms the contract the front end in Tasks 3-4 calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_hero_identity.py
"""Begin #1 hero — locks the identity/membership contract the hero front end
relies on: name capture by session, email+tos activation -> Tier-1 member,
session+email union to ONE record. All writes go through /begin/unlock."""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def _fresh_db(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    # Free-tier transition onboards to GHL + concierge referral; neutralize both.
    monkeypatch.setattr(app_module, "ghl_onboard_contact",
                        lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral",
                        lambda *a, **k: None)
    return db


def test_activation_makes_member_by_session_and_email(monkeypatch, tmp_path):
    app_module = _load_app()
    _fresh_db(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "hero1")

    # Name captured conversationally (trigger="name").
    client.post("/begin/unlock", json={"trigger": "name", "name": "Ada"})
    assert app_module.is_member(session_id="hero1") is False

    # Activation: email + ToS via the existing "tos" trigger.
    r = client.post("/begin/unlock", json={
        "trigger": "tos", "email": "ada@example.com", "tos": True})
    assert r.status_code == 200
    assert r.get_json()["current_rung"] == "free_tier"

    # Member now true by BOTH session and email.
    assert app_module.is_member(session_id="hero1") is True
    assert app_module.is_member(email="ada@example.com") is True

    # /begin/state reports the ToS stamp and the captured first name.
    st = client.get("/begin/state").get_json()
    assert st["tos_agreed_at"]
    assert st["first_name"] == "Ada"
    assert st["email"] == "ada@example.com"


def test_name_then_email_resolve_to_one_record(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh_db(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "hero2")
    client.post("/begin/unlock", json={"trigger": "name", "name": "Lee"})
    client.post("/begin/unlock", json={
        "trigger": "tos", "email": "lee@example.com", "tos": True})

    # Exactly one journey_state row carries this session; the union exposes
    # name + email + tos together.
    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM journey_state WHERE session_id=?",
            ("hero2",)).fetchone()[0]
    assert n == 1
    import begin_funnel
    with sqlite3.connect(db) as cx:
        state = begin_funnel.get_state(cx, session_id="hero2",
                                       email="lee@example.com")
    assert state["first_name"] == "Lee"
    assert state["email"] == "lee@example.com"
    assert state["tos_agreed_at"]
```

- [ ] **Step 2: Run the test**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_hero_identity.py -v`
Expected: PASS (characterization — the contract already exists; this test now locks it). If it FAILS, the backbone differs from the plan's assumptions — STOP and report, do not "fix" production code to match.

- [ ] **Step 3: Commit**

```bash
git add tests/test_begin_hero_identity.py
git commit -m "test: lock begin hero identity/membership contract"
```

---

### Task 2: Hero scaffold + responsive layout

Insert a new ungated hero `<section class="hero">` at the very top of `<body>` in `static/begin.html` (immediately after the `layer0` brandbar block that ends at line 589, before the `layer1` video section at line 591). The hero holds: a personalized headline (`data-name`), a one-line sub, the intro video (relocate the existing `.video-wrap`/`.video` markup from the old `layer1` section), and an empty chat panel container the JS in Task 3 fills. Then hide the now-redundant old `layer1` hero-greeting + ask-form so the page does not show two welcomes/two videos.

**Files:**
- Modify: `static/begin.html` (insert hero after line 589; relocate `.video-wrap` out of the `layer1` section at 591-610; hide old hero-greeting section 615 onward; add hero CSS in the `<style>` block before line 572)
- Test: `tests/test_begin_hero_identity.py` (add a serve assertion)

**Interfaces:**
- Consumes: existing `.video` markup (begin.html 593-608), existing `data-name` personalization hook, existing `--gold`/`--muted`/`--radius` CSS vars.
- Produces (for Task 3): DOM ids `#hero-chat` (panel), `#hero-messages` (transcript), `#hero-input` (textarea), `#hero-send` (button). For Task 4: `#hero-activate` (activation prompt container, empty here).

- [ ] **Step 1: Add the hero CSS**

In the `<style>` block (before the closing `</style>` at line 572), add:

```css
  /* Begin #1 hero: video | chat, above the fold. */
  .hero { max-width: 1080px; margin: 18px auto 0; padding: 0 20px; }
  .hero-head { text-align: center; margin-bottom: 18px; }
  .hero-head h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px; }
  .hero-head p { font-size: 15px; color: var(--muted); margin: 0; }
  .hero-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
  .hero-chat { display: flex; flex-direction: column; min-height: 360px;
    border: 1px solid rgba(212,168,67,0.25); border-radius: var(--radius); overflow: hidden; }
  .hero-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .hero-msg { max-width: 88%; padding: 10px 13px; border-radius: 12px; font-size: 15px; line-height: 1.45; }
  .hero-msg.user { align-self: flex-end; background: rgba(212,168,67,0.14); }
  .hero-msg.assistant { align-self: flex-start; background: rgba(255,255,255,0.05); }
  .hero-inputbar { display: flex; gap: 8px; padding: 10px; border-top: 1px solid rgba(255,255,255,0.08); }
  .hero-inputbar textarea { flex: 1; resize: none; min-height: 44px; max-height: 120px; padding: 11px 13px; }
  .hero-send { min-width: 64px; }
  @media (max-width: 760px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-head h1 { font-size: 24px; }
  }
```

- [ ] **Step 2: Insert the hero section**

Immediately after the `layer0` block (after line 589 `</div>`), insert:

```html
  <!-- BEGIN #1 HERO: headline + sub + video + simplified self-healing chat.
       Ungated (always visible, above the fold). Deeper layered content below
       stays intact for sub-project #2. -->
  <section class="hero" data-layer="hero">
    <div class="hero-head">
      <h1>Welcome<span data-name-prefix>, </span><span data-name></span> - let&rsquo;s talk about your health goals.</h1>
      <p>A ten-second conversation to point you toward what your body needs next.</p>
    </div>
    <div class="hero-grid">
      <div class="video-wrap reveal">
        <div class="video" role="button" aria-label="Play introduction video from Dr. Glen Swartwout">
          <div class="play" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          </div>
          <div class="lower-third">
            <div class="lt-rule" aria-hidden="true"></div>
            <p class="lt-text">
              Guided by the lifework of <strong>Dr. Glen Swartwout, O.D., N.D.</strong>, featured in USA&nbsp;Today:
              <em>&ldquo;Revolutionizing the World of Medicine with Natural Therapies.&rdquo;</em>
            </p>
          </div>
          <div class="scrub" aria-hidden="true"><i></i></div>
        </div>
      </div>
      <div class="hero-chat" id="hero-chat">
        <div class="hero-messages" id="hero-messages"></div>
        <div id="hero-activate"></div>
        <div class="hero-inputbar">
          <textarea id="hero-input" rows="1" placeholder="Type your reply&hellip;"></textarea>
          <button class="send-btn hero-send" id="hero-send" type="button">Send</button>
        </div>
      </div>
    </div>
  </section>
```

- [ ] **Step 3: Remove the relocated video from the old layer1 section**

The intro video now lives in the hero. Delete the old `.video-wrap` block at begin.html 593-609 (inside `<section class="shell locked" data-layer="layer1">` at 591). Leave the empty `<section ... data-layer="layer1">...</section>` wrapper in place if other layer1 reveal logic references it; if the section becomes empty, keep the opening/closing tags so `data-layer="layer1"` still exists for the reveal engine.

- [ ] **Step 4: Hide the redundant old hero-greeting + ask form**

The hero supersedes the old welcome. On the old hero-greeting `<section class="shell pad-y locked" data-layer="layer1">` (begin.html 615), add an inline `style="display:none;"` to that section's opening tag so its eyebrow/greeting/ask-form no longer render. Do NOT delete it (the reveal engine and #2 may reference its layers); only hide it.

- [ ] **Step 5: Add a serve assertion**

Append to `tests/test_begin_hero_identity.py`:

```python
def test_begin_serves_hero(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert 'class="hero"' in html
    assert 'id="hero-chat"' in html
    assert 'id="hero-messages"' in html
    assert "health goals" in html
    # The hero video is present (relocated, single instance kept in the hero).
    assert 'class="video"' in html
```

- [ ] **Step 6: Run the serve test**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_hero_identity.py -v`
Expected: PASS (all tests including `test_begin_serves_hero`).

- [ ] **Step 7: Commit**

```bash
git add static/begin.html tests/test_begin_hero_identity.py
git commit -m "feat: begin #1 hero scaffold (headline + video + chat panel, responsive)"
```

Note for the human reviewer: this task ships visual structure; a manual visual pass (desktop two-column above the fold; mobile stacked) is required before deploy and is tracked separately.

---

### Task 3: Hero chat widget + conversational name capture

Wire the hero chat panel. On load it shows ONE scripted assistant message that asks the name (instant, not a model call). The visitor's first reply is captured as their name: cleaned client-side, POSTed via `unlock("name", {...})`, and also sent to `/begin/match/chat` as their first real message so the conversation flows. Subsequent messages stream from `/begin/match/chat` as normal. No Rate / Leave / View-feedback controls render. `personalize()` greets them by name once captured.

**Files:**
- Modify: `static/begin.html` (add hero-chat JS inside the existing `<script>` block, after the `unlock()` helper at line 791 and before / alongside the `arrival()` IIFE; reuse the existing `unlock`, `personalize`, `getRef` helpers)
- Test: `tests/test_begin_hero_identity.py` (serve assertion: scripted greeting present, no feedback controls)

**Interfaces:**
- Consumes: `unlock(trigger, extra)` (begin.html 783) -> POSTs `/begin/unlock`, returns the new state; `personalize(name)` (779); `getRef()` (used by `unlock`); DOM ids from Task 2 (`#hero-messages`, `#hero-input`, `#hero-send`). Endpoint `POST /begin/match/chat` accepts `{query, history, for_whom, session_id, name, email}` and streams SSE `data: {...}` lines (see `static/begin-match.html` sendQuery, lines 313-430, for the exact stream/event shape).
- Produces (for Task 4): JS globals `heroHistory` (array of `{role, content}`), `heroExchanges` (int counter), and function `heroAppend(role, text)`.

- [ ] **Step 1: Add the hero chat JS**

Inside the `<script>` block, after the `unlock()` function (line 791), add:

```javascript
  // ---- Begin #1 hero chat: scripted name-ask, then stream /begin/match/chat ----
  (function heroChat(){
    var msgsEl = document.getElementById('hero-messages');
    var inputEl = document.getElementById('hero-input');
    var sendEl = document.getElementById('hero-send');
    if (!msgsEl || !inputEl || !sendEl) return;

    window.heroHistory = [];
    window.heroExchanges = 0;
    var nameCaptured = false;
    var loading = false;

    window.heroAppend = function(role, text){
      var d = document.createElement('div');
      d.className = 'hero-msg ' + (role === 'user' ? 'user' : 'assistant');
      d.textContent = text;
      msgsEl.appendChild(d);
      msgsEl.scrollTop = msgsEl.scrollHeight;
      return d;
    };

    // Scripted first assistant message asks the name (instant, no model call).
    var GREETING = "Aloha. I am here to help with your health goals. So I can "
      + "tailor this and remember you - what should I call you?";
    heroAppend('assistant', GREETING);
    heroHistory.push({ role: 'assistant', content: GREETING });

    // Clean a free-text reply into a first name: drop a leading "I'm/my name
    // is/it's/this is/call me", strip punctuation, take the first 1-2 tokens.
    function cleanName(raw){
      var s = (raw || '').trim()
        .replace(/^(i\s*am|i'm|my name is|it'?s|this is|call me)\s+/i, '')
        .replace(/[^A-Za-z' -]/g, ' ')
        .trim();
      if (!s) return '';
      var parts = s.split(/\s+/).slice(0, 2);
      return parts.join(' ').slice(0, 40);
    }

    function streamMatch(query){
      var box = heroAppend('assistant', '');
      var acc = '';
      return fetch('/begin/match/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          query: query, history: heroHistory.slice(),
          for_whom: 'self', session_id: '', name: '', email: ''
        })
      }).then(function(r){
        if (!r.ok || !r.body) { box.textContent = 'Something went sideways. Please try again.'; return; }
        var reader = r.body.getReader();
        var decoder = new TextDecoder();
        var buf = '';
        function pump(){
          return reader.read().then(function(chunk){
            if (chunk.done) { heroHistory.push({ role: 'assistant', content: acc }); return; }
            buf += decoder.decode(chunk.value, { stream: true });
            var lines = buf.split('\n'); buf = lines.pop();
            lines.forEach(function(line){
              if (line.indexOf('data: ') !== 0) return;
              var raw = line.slice(6).trim(); if (!raw) return;
              var evt; try { evt = JSON.parse(raw); } catch (_) { return; }
              if (evt.delta) { acc += evt.delta; box.textContent = acc; msgsEl.scrollTop = msgsEl.scrollHeight; }
              else if (evt.text) { acc = evt.text; box.textContent = acc; }
            });
            return pump();
          });
        }
        return pump();
      }).catch(function(){ box.textContent = 'Something went sideways. Please try again.'; });
    }

    function send(){
      if (loading) return;
      var q = (inputEl.value || '').trim();
      if (!q) return;
      inputEl.value = ''; inputEl.style.height = 'auto';
      heroAppend('user', q);
      heroHistory.push({ role: 'user', content: q });
      heroExchanges += 1;
      loading = true; sendEl.disabled = true;

      var done = Promise.resolve();
      // First reply after the scripted name-ask = the name. Capture it (best
      // effort) to the one record, then continue the conversation.
      if (!nameCaptured) {
        nameCaptured = true;
        var nm = cleanName(q);
        if (nm) {
          done = unlock('name', { name: nm, first_name: nm }).then(function(){
            personalize(nm);
          }).catch(function(){});
        }
      }
      done.then(function(){ return streamMatch(q); })
        .then(function(){ loading = false; sendEl.disabled = false;
          if (typeof window.heroMaybeActivate === 'function') window.heroMaybeActivate(); });
    }

    sendEl.addEventListener('click', send);
    inputEl.addEventListener('keydown', function(e){
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
  })();
```

Note: the evd `evt.delta`/`evt.text` field names MUST match what `/begin/match/chat` emits. Before writing, open `static/begin-match.html` lines 373-430 and confirm the exact SSE event field names the endpoint sends; use those names here (adjust `evt.delta`/`evt.text` if they differ). This is the one place to verify against the live stream shape.

- [ ] **Step 2: Add the serve assertion**

Append to `tests/test_begin_hero_identity.py`:

```python
def test_hero_chat_scripted_greeting_and_no_feedback_controls(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "what should I call you" in html
    assert "id=\"hero-send\"" in html
    # Hero chat surface must not render Rate / feedback controls.
    hero_start = html.index('id="hero-chat"')
    hero_end = html.index('</section>', hero_start)
    hero_block = html[hero_start:hero_end]
    assert "Rate" not in hero_block
    assert "feedback" not in hero_block.lower()
```

- [ ] **Step 3: Run the tests**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_hero_identity.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add static/begin.html tests/test_begin_hero_identity.py
git commit -m "feat: begin #1 hero chat - scripted name-ask + conversational name capture"
```

Note for the human reviewer: manual visual pass required - confirm the scripted greeting renders, typing a name greets the visitor and the conversation streams, and no feedback controls appear.

---

### Task 4: Email-to-activate + Explore link moved to bottom (membership-gated)

After a couple of exchanges, surface an inline email + ToS opt-in inside the chat panel. On submit, POST `unlock("tos", {email, tos:true, first_name})` which sets `tos_agreed_at` -> member. Relocate the primary "Explore everything" link to the bottom of the page, shown/enabled only for members (gate on `STATE.tos_agreed_at`), with a soft "activate to explore" nudge for non-members.

**Files:**
- Modify: `static/begin.html` (add the activation prompt JS + a bottom Explore block + gating JS; soften the existing top Explore link at line 587)
- Test: `tests/test_begin_hero_identity.py` (serve assertion: email field + ToS copy + bottom explore present)

**Interfaces:**
- Consumes: `unlock(trigger, extra)` (begin.html 783); `STATE` global (669) and `STATE.tos_agreed_at` from `/begin/state`; `heroExchanges` + `heroAppend` + `#hero-activate` from Tasks 2-3.
- Produces: nothing for later tasks (this is the last task of #1).

- [ ] **Step 1: Add the bottom Explore block + soften the top link**

At the END of `<body>` (just before the closing `</body>`), add a bottom Explore block:

```html
  <!-- Begin #1: Explore entry relocated to the bottom, member-gated. -->
  <section class="shell pad-y" id="explore-bottom" style="text-align:center;">
    <a id="explore-link" href="/begin/explore"
       style="color:var(--gold);border-bottom:1px solid rgba(212,168,67,0.35);padding-bottom:1px;display:none;">
       Explore everything</a>
    <span id="explore-nudge" style="color:var(--muted);font-size:13px;display:none;">
      Activate your free membership above to explore everything.</span>
  </section>
```

Then soften the existing top Explore link at line 587: remove that `<p>` (lines 586-588) so the only Explore entry is the bottom one. (The line 637 "See everything" link lives inside the now-hidden old hero-greeting section from Task 2 Step 4, so it no longer renders - no change needed there.)

- [ ] **Step 2: Add the activation + gating JS**

Inside the `<script>` block, after the hero chat IIFE from Task 3, add:

```javascript
  // ---- Begin #1: membership gate for the bottom Explore entry ----
  function applyExploreGate(){
    var link = document.getElementById('explore-link');
    var nudge = document.getElementById('explore-nudge');
    if (!link || !nudge) return;
    var member = !!(STATE && STATE.tos_agreed_at);
    link.style.display = member ? 'inline' : 'none';
    nudge.style.display = member ? 'none' : 'inline';
  }

  // ---- Begin #1: email-to-activate prompt, surfaced after a couple exchanges ----
  (function heroActivate(){
    var slot = document.getElementById('hero-activate');
    if (!slot) return;
    var shown = false;

    window.heroMaybeActivate = function(){
      if (shown) return;
      if (STATE && STATE.tos_agreed_at) return;       // already a member
      if ((window.heroExchanges || 0) < 2) return;    // wait for value first
      shown = true;
      slot.innerHTML =
        '<div style="padding:12px 14px;border-top:1px solid rgba(255,255,255,0.08);">'
        + '<p style="margin:0 0 8px;font-size:14px;color:var(--muted);">'
        + 'Want me to save your progress and unlock your free membership? '
        + 'Add your email and we are set.</p>'
        + '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        + '<input id="hero-email" type="email" placeholder="you@example.com" '
        + 'style="flex:1;min-width:180px;padding:10px 12px;" />'
        + '<button class="send-btn" id="hero-activate-btn" type="button" '
        + 'style="min-width:96px;">Activate</button>'
        + '</div>'
        + '<label style="display:flex;gap:7px;align-items:flex-start;margin-top:8px;font-size:12px;color:var(--muted);">'
        + '<input id="hero-tos" type="checkbox" /> '
        + '<span>I agree to the <a href="https://remedymatch.com/info/terms-and-conditions" '
        + 'target="_blank" rel="noopener" style="color:var(--gold);">Terms</a>, and to receive '
        + 'helpful guidance by email.</span></label>'
        + '<p id="hero-activate-err" style="display:none;color:#d98;font-size:12px;margin:6px 0 0;"></p>'
        + '</div>';

      document.getElementById('hero-activate-btn').addEventListener('click', function(){
        var email = (document.getElementById('hero-email').value || '').trim();
        var tos = document.getElementById('hero-tos').checked;
        var err = document.getElementById('hero-activate-err');
        if (!email || email.indexOf('@') < 1) {
          err.textContent = 'Please enter a valid email.'; err.style.display = 'block'; return;
        }
        if (!tos) {
          err.textContent = 'Please agree to the Terms to activate.'; err.style.display = 'block'; return;
        }
        err.style.display = 'none';
        var first = (STATE && STATE.first_name) || '';
        unlock('tos', { email: email, tos: true, first_name: first }).then(function(st){
          slot.innerHTML = '<p style="padding:12px 14px;font-size:14px;color:var(--gold);margin:0;">'
            + 'You are in. Your free membership is active.</p>';
          applyExploreGate();
        }).catch(function(){
          err.textContent = 'Could not activate just now. Please try again.';
          err.style.display = 'block';
        });
      });
    };
  })();
```

- [ ] **Step 3: Call the gate on state load**

The `arrival()` IIFE (begin.html 794) and the `unlock()` helper both set `STATE` then call `applyReveal`/`personalize`/`renderCards`. Add `applyExploreGate();` immediately after each of those three `personalize(...)` / `renderCards(...)` updates so the bottom Explore reflects membership on first load AND after activation. Concretely, append `applyExploreGate();` after `renderCards(st.surfaced_cards);` in: the `unlock()` `.then` (line 789-790), the `arrival()` state-load `.then` (807-808), and the `postMessage` handler (829). Guard with `if (typeof applyExploreGate === 'function')` since `unlock` is defined before `applyExploreGate`.

- [ ] **Step 4: Add the serve assertion**

Append to `tests/test_begin_hero_identity.py`:

```python
def test_hero_has_activation_and_bottom_explore(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    # Bottom explore block + gated link + non-member nudge.
    assert 'id="explore-bottom"' in html
    assert 'id="explore-link"' in html
    assert 'id="explore-nudge"' in html
    # Activation wiring present (the email field/markup is injected by JS, so
    # assert the JS that mints it ships in the page).
    assert "hero-activate-btn" in html
    assert "unlock('tos'" in html
    # The old top explore <p> was removed (only the bottom entry remains).
    assert html.count('href="/begin/explore"') == 1
```

- [ ] **Step 5: Run the tests**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_hero_identity.py -v`
Expected: PASS (all five+ tests).

- [ ] **Step 6: No-regression sweep**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "begin" -v`
Expected: existing `test_begin_routes.py` / `test_begin_funnel.py` still PASS; no new failures.

- [ ] **Step 7: Commit**

```bash
git add static/begin.html tests/test_begin_hero_identity.py
git commit -m "feat: begin #1 email-to-activate + bottom member-gated Explore link"
```

Note for the human reviewer: manual visual pass required - after a couple of replies the email+ToS prompt appears, activating sets membership and reveals the bottom Explore link; before activation the soft nudge shows.

---

## Self-Review

**1. Spec coverage** (against `docs/superpowers/specs/2026-06-20-begin-hero-identity-design.md`):
- Hero layout (headline + sub + video + chat, responsive) -> Task 2.
- Reuse `/begin/match/chat` stripped-down, no feedback controls -> Task 3 (+ serve assertion).
- Scripted first AI message asks the name -> Task 3.
- Conversational name capture to one record via `trigger="name"` -> Task 3 (+ Task 1 contract lock).
- Email -> activate free membership via email+tos -> Task 4 (+ Task 1 contract lock). NOTE: spec said "add `name`/`activate` triggers"; reality is `name`/`email`/`tos` already exist and email+tos already grants membership, so the plan reuses them and adds NO triggers. This is a strict simplification preserving every approved behavior; flag it to the human at handoff.
- Explore link -> bottom, membership-gated -> Task 4.
- One record (session+email union) -> enforced by routing every write through `/begin/unlock`; pinned in Task 1.
- Existing below-hero content left intact -> Task 2 hides only the redundant old hero-greeting + relocates the video; deeper layers untouched.
- Live page, no flag, visual pass required -> stated in Global Constraints + per-task reviewer notes.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". The one verification step (Task 3 Step 1 note: confirm `/begin/match/chat` SSE field names against `begin-match.html`) is a concrete, located instruction, not a placeholder.

**3. Type consistency:** `unlock(trigger, extra)` returns the state promise (used in Tasks 3-4). `heroExchanges`/`heroHistory`/`heroAppend`/`heroMaybeActivate` defined in Task 3, consumed in Task 4. DOM ids `#hero-chat`/`#hero-messages`/`#hero-input`/`#hero-send`/`#hero-activate` defined in Task 2, consumed in Tasks 3-4. `STATE.tos_agreed_at` matches the `/begin/state` payload key (confirmed in `begin_funnel.get_state`). Activation uses `trigger="tos"` consistently. Consistent.
