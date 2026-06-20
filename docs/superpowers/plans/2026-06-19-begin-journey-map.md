# Begin Page #2 — 4-card Unfolding Journey Map — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed 4-card journey strip below the `/begin` hero that unfolds left-to-right (triggered by the first AI answer or the video being played), weaves a framing line into the chat when the chat triggers it, and colors each card by the visitor's progress through the journey gates.

**Architecture:** A new pure `begin_funnel.journey_map(state, ref)` returns the 4 ordered cards with computed status; `/begin/state` injects it (like the existing `surfaced_cards`); `static/begin.html` renders the strip and re-renders on the same state-refresh sites #1 wired, with an idempotent unfold animation triggered by the first of {chat cue, video cue}. No new journey gates, no schema change; existing `surface()`/`/begin/explore` untouched.

**Tech Stack:** Flask (Python 3.11), vanilla JS in `static/begin.html`, SQLite `journey_state` via `begin_funnel`, pytest + Flask test client.

## Global Constraints

- **No emoji. No em dashes.** Plain hyphens and text/SVG glyphs only.
- **Live page, no feature flag.** `main` auto-deploys to prod; the merge ships it. A manual visual pass is REQUIRED before it is considered launched.
- **No new gates, no schema change.** Reuse existing `VALID_TRIGGERS`: the 4 cards key off `scan`, `question`, `paid_fork`, `share_video` (all already present). Do NOT add to `VALID_TRIGGERS` or alter `journey_state` columns.
- **`card_href` behavior must stay byte-identical after refactor.** Task 1 factors a shared `_thread_href` helper; the existing `card_href` output (internal pass-through; external utm `begin-card-<key>`) must not change. The existing `test_begin_funnel.py` card/surface tests must still pass.
- **XSS-safe rendering.** Card label/paren/href come from the server-side static `JOURNEY_STEPS` (never user input). Build card DOM with `textContent` for text nodes (consistent with #1's reviewed pattern); do not interpolate any value into `innerHTML`.
- **Do not touch the contextual surfacing system.** `surface()` / `surfaced_cards` / `renderCards` / `/begin/explore` are a separate, data-driven system and stay as-is. This strip is a distinct fixed 4-step map.
- **Reuse #1 plumbing.** `STATE`, `unlock(trigger, extra)`, `heroAppend(role, text)`, and the state-refresh sites (each currently calling `applyExploreGate()`) already exist in `begin.html` from #1.
- **Test harness (deploy-chat):** load app via `importlib.import_module("app")` with `repo_root` on `sys.path` (skip if not importable); `monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))`; create tables with `begin_funnel.init_journey_tables(cx)`; `app_module.app.test_client()`; set `amg_session` cookie; mock `ghl_onboard_contact` + `_capture_concierge_referral` on any free-tier transition.
- **Run tests:** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <targets> -v`.

---

### Task 1: `journey_map` + shared href helper (backend)

Add the journey definition and the pure status function to `begin_funnel.py`, factoring the internal/external href threading out of `card_href` into a shared helper so the new function reuses it without changing `card_href`'s output.

**Files:**
- Modify: `begin_funnel.py` (add `_thread_href`, refactor `card_href` at 418-425 to use it, add `JOURNEY_STEPS` + `journey_map` near the card section)
- Test: `tests/test_begin_journey_map.py` (create)

**Interfaces:**
- Consumes: `state` dict from `get_state` (has `state["unlocked_gates"]`, a sorted list).
- Produces: `begin_funnel.JOURNEY_STEPS` (ordered list), `begin_funnel.journey_map(state, ref="")` -> list of 4 dicts `{"key","label","paren","href","status"}` with `status` in `{"done","next","available"}`. `begin_funnel._thread_href(base_url, ref, campaign)` -> str.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_journey_map.py
"""Begin #2 - journey_map status logic + href threading."""

import importlib
import sys
from pathlib import Path

import pytest


def _bf():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("begin_funnel")
    except Exception as e:
        pytest.skip(f"begin_funnel not importable: {e}")


def _state(gates):
    return {"unlocked_gates": list(gates)}


def test_no_gates_scan_is_next():
    bf = _bf()
    m = bf.journey_map(_state([]), "")
    assert [c["key"] for c in m] == ["scan", "find", "heal", "earn"]
    assert [c["status"] for c in m] == ["next", "available", "available", "available"]


def test_scan_done_find_is_next():
    bf = _bf()
    m = bf.journey_map(_state(["scan"]), "")
    assert [c["status"] for c in m] == ["done", "next", "available", "available"]


def test_scan_and_question_done_heal_is_next():
    bf = _bf()
    m = bf.journey_map(_state(["scan", "question"]), "")
    assert [c["status"] for c in m] == ["done", "done", "next", "available"]


def test_all_gates_done_none_next():
    bf = _bf()
    m = bf.journey_map(_state(["scan", "question", "paid_fork", "share_video"]), "")
    assert [c["status"] for c in m] == ["done", "done", "done", "done"]
    assert all(c["status"] != "next" for c in m)


def test_labels_and_internal_hrefs():
    bf = _bf()
    m = bf.journey_map(_state([]), "someslug")
    by = {c["key"]: c for c in m}
    assert by["scan"]["label"] == "Scan" and by["scan"]["paren"] == "Your Biofield"
    assert by["find"]["label"] == "Find" and by["find"]["paren"] == "Your Remedy Match"
    assert by["heal"]["label"] == "Heal" and by["heal"]["paren"] == "the root causes"
    assert by["earn"]["label"] == "Earn" and by["earn"]["paren"] == "Ambassador"
    # All four destinations are internal -> hrefs are the bare path (no utm).
    assert by["scan"]["href"] == "/begin/voice"
    assert by["find"]["href"] == "/begin/match"
    assert by["heal"]["href"] == "/begin/ascend"
    assert by["earn"]["href"] == "/begin/path"


def test_thread_href_external_threads_utm():
    bf = _bf()
    h = bf._thread_href("https://x.example.com", "slug", "begin-journey-scan")
    assert h.startswith("https://x.example.com?utm_source=slug")
    assert "utm_campaign=begin-journey-scan" in h
    # internal pass-through unchanged
    assert bf._thread_href("/begin/voice", "slug", "begin-journey-scan") == "/begin/voice"


def test_card_href_unchanged_after_refactor():
    bf = _bf()
    # internal CARD_CATALOG entry returns bare base_url
    assert bf.card_href("voice_distinctions", "slug") == "/begin/voice"
    # external entry threads the original begin-card-<key> campaign
    h = bf.card_href("quiz", "slug")
    assert h.startswith("https://healing.scoreapp.com?utm_source=slug")
    assert "utm_campaign=begin-card-quiz" in h
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: FAIL (`journey_map` / `_thread_href` / `JOURNEY_STEPS` not defined).

- [ ] **Step 3: Add `_thread_href`, refactor `card_href`, add `JOURNEY_STEPS` + `journey_map`**

In `begin_funnel.py`, replace the existing `card_href` (lines 418-425):

```python
def _thread_href(base_url, ref, campaign):
    """Internal (/...) base returned as-is; external base threaded with the
    ref-based utm. Shared by card_href and journey_map so threading stays in sync."""
    if base_url.startswith("/"):
        return base_url
    slug = (ref or "remedy-match").strip() or "remedy-match"
    sep = "&" if "?" in base_url else "?"
    return (f"{base_url}{sep}utm_source={urllib.parse.quote(slug)}"
            f"&utm_medium=affiliate&utm_campaign={campaign}")


def card_href(key, ref=""):
    c = CARD_CATALOG[key]
    return _thread_href(c["base_url"], ref, f"begin-card-{key}")
```

Then add, just below `card_href`/`_card` (after line ~430):

```python
# ---------------------------------------------------------------------------
# Begin #2 - fixed 4-step journey map (distinct from surface()/CARD_CATALOG).
# All copy provisional (BNSN site pass later). done_gate/click_trigger are all
# existing VALID_TRIGGERS - no new gates.
# ---------------------------------------------------------------------------
JOURNEY_STEPS = [
    {"key": "scan", "label": "Scan", "paren": "Your Biofield",
     "base_url": "/begin/voice",  "done_gate": "scan",        "click_trigger": "scan"},
    {"key": "find", "label": "Find", "paren": "Your Remedy Match",
     "base_url": "/begin/match",  "done_gate": "question",     "click_trigger": "question"},
    {"key": "heal", "label": "Heal", "paren": "the root causes",
     "base_url": "/begin/ascend", "done_gate": "paid_fork",    "click_trigger": "paid_fork"},
    {"key": "earn", "label": "Earn", "paren": "Ambassador",
     "base_url": "/begin/path",   "done_gate": "share_video",  "click_trigger": "share_video"},
]


def journey_map(state, ref=""):
    """Ordered 4 cards with progress status. done = its done_gate is set;
    next = the first not-done step; rest = available. Pure; never mutates."""
    gates = set((state or {}).get("unlocked_gates") or ())
    out = []
    next_assigned = False
    for step in JOURNEY_STEPS:
        if step["done_gate"] in gates:
            status = "done"
        elif not next_assigned:
            status = "next"
            next_assigned = True
        else:
            status = "available"
        out.append({
            "key": step["key"], "label": step["label"], "paren": step["paren"],
            "href": _thread_href(step["base_url"], ref, f"begin-journey-{step['key']}"),
            "status": status,
        })
    return out
```

- [ ] **Step 4: Run the new test + the regression suite**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py tests/test_begin_funnel.py -v`
Expected: PASS (new journey tests AND existing begin_funnel card/surface tests; `card_href` unchanged).

- [ ] **Step 5: Commit**

```bash
git add begin_funnel.py tests/test_begin_journey_map.py
git commit -m "feat: begin #2 journey_map + shared _thread_href helper"
```

---

### Task 2: Inject `journey_map` into `/begin/state` (backend wiring)

**Files:**
- Modify: `app.py` (`begin_state`, ~1355-1366)
- Test: `tests/test_begin_journey_map.py` (add a Flask-test-client payload test)

**Interfaces:**
- Consumes: `begin_funnel.journey_map(state, ref)` (Task 1); existing `begin_state` locals `state` and `ref_slug`.
- Produces: `/begin/state` JSON now contains `journey_map` (list of 4).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_begin_journey_map.py`:

```python
import sqlite3


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_state_payload_includes_journey_map(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "j1")
    body = client.get("/begin/state").get_json()
    jm = body["journey_map"]
    assert [c["key"] for c in jm] == ["scan", "find", "heal", "earn"]
    assert jm[0]["status"] == "next"            # nothing done yet
    assert all(set(c) >= {"key", "label", "paren", "href", "status"} for c in jm)
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py::test_state_payload_includes_journey_map -v`
Expected: FAIL (KeyError `journey_map`).

- [ ] **Step 3: Inject the key**

In `app.py` `begin_state`, after the line `payload["surfaced_cards"] = begin_funnel.surface(state, query_texts, ref_slug)`, add:

```python
    payload["journey_map"] = begin_funnel.journey_map(state, ref_slug)
```

- [ ] **Step 4: Run to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_journey_map.py
git commit -m "feat: begin #2 inject journey_map into /begin/state"
```

---

### Task 3: Journey strip markup, CSS, and progress rendering (front-end)

Add the hidden strip below the hero, its CSS, the client fallback/trigger constants, and `renderJourney()` + `refreshJourney()`, and wire `refreshJourney()` beside every existing `applyExploreGate()` call so the strip colors by progress on each state update.

**Files:**
- Modify: `static/begin.html` (markup after the hero `</section>` at ~642; CSS before `</style>`; JS in the `<script>` block; wiring at the `applyExploreGate()` sites ~827, ~997, ~1022, ~1044)
- Test: `tests/test_begin_journey_map.py` (serve assertion)

**Interfaces:**
- Consumes: `STATE.journey_map` (Task 2); existing `STATE`, `unlock` from #1.
- Produces (for Task 4): JS `renderJourney()`, `refreshJourney()`, globals `JOURNEY_FALLBACK`, `JOURNEY_TRIGGER`; DOM `#journey-strip`, `#journey-cards`; CSS classes `.journey-card`, `.journey-card.in`, `.status-done|next|available`, `#journey-strip.unfolded`.

- [ ] **Step 1: Add the strip markup**

Immediately after the hero section's closing `</section>` (the hero block that opens at `<section class="hero" data-layer="hero">`), insert:

```html
  <!-- BEGIN #2: 4-card journey map. Hidden until unfold (Task 4). Renders from
       STATE.journey_map; colors by progress. Distinct from the layer-5 surfaced
       cards and /begin/explore. -->
  <section id="journey-strip" aria-label="Your healing journey">
    <p class="journey-caption">Your healing journey</p>
    <div id="journey-cards"></div>
  </section>
```

- [ ] **Step 2: Add the CSS**

Before `</style>`, add:

```css
  /* Begin #2 journey strip */
  #journey-strip { max-width: 1080px; margin: 18px auto 0; padding: 0 20px; display: none; }
  #journey-strip.unfolded { display: block; }
  .journey-caption { text-align: center; font-size: 13px; letter-spacing: 0.04em; color: var(--muted); margin: 0 0 12px; }
  #journey-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .journey-card { display: block; padding: 16px 14px; border: 1px solid rgba(212,168,67,0.22);
    border-radius: var(--radius); text-decoration: none; color: inherit;
    opacity: 0; transform: translateX(-12px); transition: opacity .45s ease, transform .45s ease; }
  .journey-card.in { opacity: 1; transform: translateX(0); }
  .journey-card .jc-label { font-size: 16px; font-weight: 600; }
  .journey-card .jc-paren { display: block; font-size: 13px; color: var(--muted); margin-top: 3px; }
  .journey-card .jc-tag { display: block; margin-top: 8px; font-size: 11px; letter-spacing: 0.04em; }
  .journey-card.status-done { border-color: rgba(255,255,255,0.12); }
  .journey-card.status-done .jc-label { color: var(--muted); }
  .journey-card.status-next { border-color: var(--gold); box-shadow: 0 0 0 1px var(--gold) inset; }
  .journey-card.status-next .jc-tag { color: var(--gold); }
  .jc-check { color: var(--gold); }
  @media (max-width: 760px) { #journey-cards { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 420px) { #journey-cards { grid-template-columns: 1fr; } }
```

- [ ] **Step 3: Add the client constants + render functions**

In the `<script>` block (place near the other Begin #2 / #1 helpers, e.g. right before the `applyExploreGate` definition), add:

```javascript
  // ---- Begin #2: journey strip ----
  var JOURNEY_FALLBACK = [
    { key:'scan', label:'Scan', paren:'Your Biofield',     href:'/begin/voice',  status:'available' },
    { key:'find', label:'Find', paren:'Your Remedy Match', href:'/begin/match',  status:'available' },
    { key:'heal', label:'Heal', paren:'the root causes',   href:'/begin/ascend', status:'available' },
    { key:'earn', label:'Earn', paren:'Ambassador',        href:'/begin/path',   status:'available' }
  ];
  var JOURNEY_TRIGGER = { scan:'scan', find:'question', heal:'paid_fork', earn:'share_video' };
  var journeyUnfolded = false;

  function paintJourneyIn(staggered){
    var strip = document.getElementById('journey-strip');
    if (!strip) return;
    var cards = strip.querySelectorAll('.journey-card');
    Array.prototype.forEach.call(cards, function(c, i){
      c.style.transitionDelay = staggered ? (i * 120) + 'ms' : '0ms';
      requestAnimationFrame(function(){ requestAnimationFrame(function(){ c.classList.add('in'); }); });
    });
  }

  function renderJourney(){
    var wrap = document.getElementById('journey-cards');
    if (!wrap) return;
    var steps = (STATE && STATE.journey_map) || JOURNEY_FALLBACK;
    wrap.innerHTML = '';
    steps.forEach(function(s){
      var a = document.createElement('a');
      a.className = 'journey-card status-' + (s.status || 'available');
      a.href = s.href || '#';
      var lab = document.createElement('span');
      lab.className = 'jc-label';
      lab.textContent = s.label;                       // static server label - textContent (XSS-safe)
      if (s.status === 'done') {
        var ck = document.createElement('span');
        ck.className = 'jc-check'; ck.setAttribute('aria-hidden', 'true');
        ck.textContent = ' [v]';
        lab.appendChild(ck);
      }
      var par = document.createElement('span');
      par.className = 'jc-paren'; par.textContent = s.paren;
      a.appendChild(lab); a.appendChild(par);
      if (s.status === 'next') {
        var tg = document.createElement('span');
        tg.className = 'jc-tag'; tg.textContent = 'your next step';
        a.appendChild(tg);
      }
      a.addEventListener('click', function(){
        var t = JOURNEY_TRIGGER[s.key];
        if (t && typeof unlock === 'function') { try { unlock(t); } catch(_){} }
        // navigation proceeds regardless (fire-and-forget gate advance)
      });
      wrap.appendChild(a);
    });
  }

  function refreshJourney(){
    renderJourney();
    if (journeyUnfolded) paintJourneyIn(false);   // re-render while open: show immediately
  }
```

- [ ] **Step 4: Wire `refreshJourney()` into the state-refresh sites**

After EACH existing `applyExploreGate();` call (there are calls in: the `arrival()` state-load `.then` ~828; the activation `unlock('tos')` success `.then` ~997; the `unlock()` helper `.then` ~1022; the `postMessage` handler ~1044), add on the next line:

```javascript
        if (typeof refreshJourney === 'function') refreshJourney();
```

(Match the indentation of the adjacent `applyExploreGate();`. Use grep for `applyExploreGate();` to find every call site and add the line after each.)

- [ ] **Step 5: Add the serve assertion**

Append to `tests/test_begin_journey_map.py`:

```python
def test_begin_serves_journey_strip(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="journey-strip"' in html
    assert 'id="journey-cards"' in html
    for label in ("Scan", "Find", "Heal", "Earn"):
        assert label in html
    assert "function renderJourney" in html
```

- [ ] **Step 6: Run the tests**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add static/begin.html tests/test_begin_journey_map.py
git commit -m "feat: begin #2 journey strip markup + progress rendering"
```

Note for the human reviewer: the strip stays hidden (`display:none`) until Task 4 wires the unfold; manual visual pass happens after Task 4.

---

### Task 4: Unfold animation + triggers + chat framing line (front-end)

Reveal the strip with a left-to-right stagger on the first of {chat cue, video cue}; weave a framing line into the chat when the chat triggers it; show it already-open on a return visit that is underway.

**Files:**
- Modify: `static/begin.html` (add `unfoldJourney()` + the framing constant; hook the chat-cue site in the hero `send()` flow ~927; hook the video-cue site at the `.video` click handler; hook the reload case in `arrival()`)
- Test: `tests/test_begin_journey_map.py` (serve assertion for the unfold JS + framing copy)

**Interfaces:**
- Consumes: `renderJourney()`, `paintJourneyIn()`, `journeyUnfolded`, `JOURNEY...` (Task 3); `window.heroAppend(role, text)` and the hero `send()` `streamMatch(...)` resolution + the `.video` click handler (all from #1).
- Produces: `unfoldJourney(source)`.

- [ ] **Step 1: Add `unfoldJourney()` + framing constant**

In the `<script>` block, right after `refreshJourney()` from Task 3, add:

```javascript
  var JOURNEY_FRAMING = "Here's the path we'll walk together - so I can get to "
    + "know you and help you find the best solutions for your unique needs. "
    + "Start anywhere.";

  function unfoldJourney(source){
    if (journeyUnfolded) return;
    journeyUnfolded = true;
    var strip = document.getElementById('journey-strip');
    if (!strip) return;
    if (source === 'chat' && typeof window.heroAppend === 'function') {
      window.heroAppend('assistant', JOURNEY_FRAMING);   // weave meaning into the chat
    }
    strip.classList.add('unfolded');
    renderJourney();
    paintJourneyIn(source !== 'reload');   // animate fresh triggers; immediate on reload
  }
```

- [ ] **Step 2: Hook the chat cue (first AI answer completes)**

In the hero `send()` function, the chain that ends the turn is:

```javascript
      done.then(function(){ return streamMatch(q); })
        .then(function(){ loading = false; sendEl.disabled = false;
          if (typeof window.heroMaybeActivate === 'function') window.heroMaybeActivate(); });
```

Add `unfoldJourney('chat');` inside that final `.then` (it is idempotent, so firing on every turn is fine; only the first matters):

```javascript
      done.then(function(){ return streamMatch(q); })
        .then(function(){ loading = false; sendEl.disabled = false;
          if (typeof unfoldJourney === 'function') unfoldJourney('chat');
          if (typeof window.heroMaybeActivate === 'function') window.heroMaybeActivate(); });
```

- [ ] **Step 3: Hook the video cue (video played)**

Find the hero video click handler (grep `querySelector('.video')`; from #1 it reads roughly `var vid = document.querySelector('.video'); if(vid){ vid.addEventListener('click', function(){ unlock('video'); }); }`). Add the unfold call inside the handler:

```javascript
  var vid = document.querySelector('.video');
  if (vid) { vid.addEventListener('click', function(){
    unlock('video');
    if (typeof unfoldJourney === 'function') unfoldJourney('video');
  }); }
```

- [ ] **Step 4: Hook the reload-when-underway case**

In the `arrival()` IIFE's state-load `.then` (the one that sets `STATE = st;` then calls `applyReveal`/`personalize`/`renderCards`/`applyExploreGate`/`refreshJourney`), after `refreshJourney()`, add:

```javascript
        var anyDone = ((st.journey_map) || []).some(function(s){ return s.status === 'done'; });
        if (anyDone && typeof unfoldJourney === 'function') unfoldJourney('reload');
```

- [ ] **Step 5: Add the serve assertion**

Append to `tests/test_begin_journey_map.py`:

```python
def test_begin_serves_unfold_and_framing(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "function unfoldJourney" in html
    assert "the path we'll walk together" in html
    assert "unfoldJourney('chat')" in html
    assert "unfoldJourney('video')" in html
```

- [ ] **Step 6: Run the tests + the begin sweep**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Then: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "begin" -v`
Expected: new tests PASS; existing begin tests (`test_begin_routes`, `test_begin_funnel`, `test_begin_hero_identity`) still PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
git add static/begin.html tests/test_begin_journey_map.py
git commit -m "feat: begin #2 unfold animation + chat/video triggers + framing line"
```

Note for the human reviewer: manual visual pass required - confirm the strip is hidden on load, unfolds left-to-right when the video is played OR the first AI answer lands (chat cue also drops the framing line into the chat), progress coloring tracks gates, and a return visit shows it already open.

---

## Self-Review

**1. Spec coverage** (against `docs/superpowers/specs/2026-06-19-begin-journey-map-design.md`):
- 4 cards in order with labels/parentheticals/destinations -> Task 1 `JOURNEY_STEPS` + Task 3 render.
- Strip below hero, responsive -> Task 3 markup + CSS.
- Progress status (done/next/available) from gates -> Task 1 `journey_map` + Task 3 coloring; payload -> Task 2.
- Click navigates AND advances the map via existing triggers -> Task 3 click handler (`JOURNEY_TRIGGER`).
- Unfold left-to-right, once -> Task 4 `unfoldJourney` + `paintJourneyIn` stagger + `journeyUnfolded` flag.
- Trigger = first of {chat cue after first AI answer, video play} -> Task 4 Steps 2-3.
- Chat cue weaves a framing line -> Task 4 `JOURNEY_FRAMING` via `heroAppend`.
- Reload-when-underway shows already-open -> Task 4 Step 4.
- No new gates / schema; existing triggers only -> Global Constraints; `done_gate`/`click_trigger` all in `VALID_TRIGGERS`.
- `card_href` unchanged after refactor -> Task 1 `_thread_href` + regression test `test_card_href_unchanged_after_refactor`.
- `surface()`/`/begin/explore` untouched -> not modified by any task.
- XSS-safe -> Task 3 builds DOM via `textContent`.
- Live page, no flag; visual pass -> Global Constraints + per-task reviewer notes.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step shows complete code. The one grep-to-locate instruction (Task 3 Step 4, Task 4 Step 3) targets a concrete, named anchor.

**3. Type consistency:** `journey_map` returns `{key,label,paren,href,status}` (Task 1) consumed identically by the payload test (Task 2) and `renderJourney` (Task 3). `journeyUnfolded`/`paintJourneyIn`/`renderJourney`/`refreshJourney` defined in Task 3, consumed in Task 4. `JOURNEY_TRIGGER` values (`scan`/`question`/`paid_fork`/`share_video`) match `JOURNEY_STEPS[].click_trigger` and `done_gate`. `unfoldJourney(source)` source values `'chat'|'video'|'reload'` consistent across Task 4. Consistent.
