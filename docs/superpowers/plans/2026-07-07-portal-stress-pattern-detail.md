# Portal Stress-Pattern Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a client tap a stress-pattern chip in the portal to reveal that pattern's description inline, for the chips that have one, and make the chips persist on every render.

**Architecture:** Frontend-only change in the single inlined file `static/client-portal.html`. The E4L `description` already flows to the browser in `d.findings[i].description`; we surface it. Chips with a description become `<button>`s that toggle a single inline `#patDetail` panel; chips without one stay plain `<span>`s. A verification harness (`tests/manual/portal_pattern_harness.py`, pure stdlib) serves the real page against a controlled payload for headless assertions.

**Tech Stack:** Vanilla JS + CSS inside `static/client-portal.html`; Python stdlib `http.server` for the test harness; headless Chrome (claude-in-chrome MCP) for verification.

## Global Constraints

- **One file only** for the feature: `static/client-portal.html`. No backend, no new endpoints, no schema change.
- **Copy rules (Glen):** no em dashes, no ALL CAPS words, no "Hook:" label in any client-facing string.
- **Escaping:** use the file's existing `esc()` helper (escapes `& < > "`) for every dynamic value, including `data-*` attribute values.
- **Only described chips are interactive.** A finding with an empty/blank `description` renders exactly as today: a plain, non-clickable `<span class="pat-chip">`.
- **Chips persist on every render** (return visits + confirmed reports), not just the first-view unfold. The `reveal` stagger animation stays first-view-only (gated on the existing `firstTime`).
- **Reduced motion:** any new animation must be disabled under `@media (prefers-reduced-motion: reduce)`, mirroring the existing `.reveal` rule.
- **Reuse brand tokens:** `--brand`, `--brand-soft`, `--line`, `--card2`, `--btn-fg`, `--ink`. No hard-coded colors.
- **One panel open at a time**, handled by a single delegated listener bound once to the persistent `#app` container (survives poll re-renders).

---

### Task 1: Verification harness + persistent, described-aware chip rendering

Rebuild the `patternsBlock` chip builder so chips render on every view and described findings become buttons carrying their description. Stand up the headless harness first so the change is observable.

**Files:**
- Create: `tests/manual/portal_pattern_harness.py`
- Modify: `static/client-portal.html` (the `patternsBlock` builder, currently lines 708-718)

**Interfaces:**
- Consumes: `d.findings` = array of `{code, name, description, rank}` (already produced by `api_client_portal`); the existing `esc()` helper and `firstTime`/`rvlCls`/`findings`/`findingsCount` locals in `render()`.
- Produces: chip markup where a described finding is `<button class="pat-chip pat-chip--detail" data-pname data-pdesc aria-expanded aria-controls="patDetail">` and a blank one is `<span class="pat-chip">`; plus an empty `<div class="pat-detail" id="patDetail" hidden></div>` after the `.pat-wrap`. Task 2 relies on these exact class names, `data-pname`/`data-pdesc`, and the `#patDetail` id.

- [ ] **Step 1: Create the headless verification harness**

Create `tests/manual/portal_pattern_harness.py`:

```python
"""Manual headless-verification harness for the portal stress-pattern detail feature.

Serves the REAL static/client-portal.html at /portal/<anything> and returns a
controlled /api/portal payload: two findings WITH a description + one WITHOUT,
status=confirmed, so the pattern chips render as a mix of clickable + plain.
Pure stdlib: no Flask, no Doppler, no Pinecone, no full-app boot.

Run:   python3 tests/manual/portal_pattern_harness.py [PORT]   (default 8799)
Open:  http://localhost:PORT/portal/testtoken
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC = Path(__file__).resolve().parents[2] / "static" / "client-portal.html"

FINDINGS = [
    {"code": "ED1", "name": "Source Driver", "rank": 1,
     "description": "The Source Driver bioenergetically supports the strength of the body's fields."},
    {"code": "ET1", "name": "Heart Driver", "rank": 2,
     "description": "The Heart Driver bioenergetically supports the heartbeat and the midbrain."},
    {"code": "ER9", "name": "Environmental Load", "rank": 3, "description": ""},
]
D = {"name": "Test Client", "biofield_status": "confirmed", "blurred": False,
     "actionable": False, "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
     "greeting": "Aloha Test,", "layers": [{"n": 1, "title": "Surface", "meaning": "x"}],
     "findings": FINDINGS, "reorder_items": [], "messages": [],
     "membership_category": "none", "notify_on": True, "tos_agreed": True,
     "element_state": None, "element_backdrop_enabled": False}
V = {"biofield": {"visible": True, "status": "confirmed", "blurred": False,
     "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
     "layers": [{"n": 1, "title": "Surface", "meaning": "x"}]}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p.startswith("/portal/"):
            self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
        elif p.endswith("/view"):
            self._send(200, json.dumps(V).encode(), "application/json")
        elif p.startswith("/api/portal/"):
            self._send(200, json.dumps(D).encode(), "application/json")
        else:
            self._send(200, b"null", "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    print(f"serving http://localhost:{port}/portal/testtoken")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
```

- [ ] **Step 2: Run the harness and observe the RED baseline**

Run (background): `python3 tests/manual/portal_pattern_harness.py 8799`
Then headless-load `http://localhost:8799/portal/testtoken` (claude-in-chrome: `navigate`, then `read_console_messages` and `javascript_tool`).
Evaluate: `document.querySelectorAll('.pat-chip').length`
Expected NOW (before the change): **0** — confirmed reports render no chips today. Also confirm `read_console_messages` shows no uncaught error (if `render()` throws on a missing field, add that field to `D` in the harness and re-run — the biofield card must reach the chip block).

- [ ] **Step 3: Rewrite the `patternsBlock` builder**

In `static/client-portal.html`, replace the current block (lines 708-718):

```javascript
    // Helpers: a reveal class (only on first view) and a per-element delay style.
    const rvlCls = firstTime ? " reveal" : "";
    const layerDelay = (i)=> firstTime
      ? ` style="animation-delay:${(findingsCount*120 + 300 + i*400)}ms"` : "";
    // Patterns block (chips) — staged before the layers on first view.
    const patternsBlock = (firstTime && findingsCount)
      ? `<p class="muted small${rvlCls}" style="margin:.2rem 0 .2rem">Your scan identified these stress patterns</p>
         <div class="pat-wrap">${findings.map((f,i)=>
           `<span class="pat-chip reveal" style="animation-delay:${i*120}ms">${esc(f.name||f.code||"")}</span>`
         ).join("")}</div>`
      : "";
```

with:

```javascript
    // Helpers: a reveal class (only on first view) and a per-element delay style.
    const rvlCls = firstTime ? " reveal" : "";
    const layerDelay = (i)=> firstTime
      ? ` style="animation-delay:${(findingsCount*120 + 300 + i*400)}ms"` : "";
    const patDelay = (i)=> firstTime ? ` style="animation-delay:${i*120}ms"` : "";
    // Patterns block (chips). Rendered on EVERY view so a returning client (and
    // confirmed reports) can revisit their patterns — not just the first-view
    // unfold. The reveal stagger stays a first-view-only flourish (rvlCls/patDelay
    // are empty when !firstTime). A finding WITH a description becomes a <button>
    // that toggles the inline #patDetail panel (wired in wirePatternDetails());
    // a finding WITHOUT one stays a plain, non-interactive <span> — about half the
    // E4L catalog (ER stresses, Nutrition, Environmental) has no description.
    const anyDetail = findings.some(f => (f.description||"").trim());
    const patternsBlock = findingsCount
      ? `<p class="muted small${rvlCls}" style="margin:.2rem 0 .2rem">Your scan identified these stress patterns${anyDetail ? " &middot; tap one to learn more" : ""}</p>
         <div class="pat-wrap">${findings.map((f,i)=>{
           const nm = esc(f.name||f.code||"");
           const ds = (f.description||"").trim();
           return ds
             ? `<button type="button" class="pat-chip pat-chip--detail${rvlCls}"${patDelay(i)} data-pname="${esc(f.name||f.code||"")}" data-pdesc="${esc(ds)}" aria-expanded="false" aria-controls="patDetail">${nm}</button>`
             : `<span class="pat-chip${rvlCls}"${patDelay(i)}>${nm}</span>`;
         }).join("")}</div>
         <div class="pat-detail" id="patDetail" hidden></div>`
      : "";
```

- [ ] **Step 4: Add the chip-affordance CSS**

In the `<style>` block, immediately after the `.pat-chip{...}` rule (currently lines 229-230), add:

```css
  .pat-chip--detail{-webkit-appearance:none;appearance:none;color:inherit;font:inherit;
    line-height:inherit;cursor:pointer;transition:background .15s ease,color .15s ease,border-color .15s ease}
  .pat-chip--detail::after{content:"";display:inline-block;width:5px;height:5px;margin-left:6px;
    border-right:1.5px solid var(--brand);border-bottom:1.5px solid var(--brand);
    transform:rotate(45deg) translateY(-1px);opacity:.7;transition:transform .15s ease}
  .pat-chip--detail:hover,.pat-chip--detail:focus-visible,
  .pat-chip--detail[aria-expanded="true"]{background:var(--brand);color:var(--btn-fg)}
  .pat-chip--detail:hover::after,.pat-chip--detail:focus-visible::after,
  .pat-chip--detail[aria-expanded="true"]::after{border-color:var(--btn-fg)}
  .pat-chip--detail[aria-expanded="true"]::after{transform:rotate(-135deg) translateY(1px)}
  .pat-chip--detail:focus-visible{outline:2px solid var(--brand);outline-offset:2px}
```

- [ ] **Step 5: Run the harness and verify rendering (GREEN for Task 1)**

Reload `http://localhost:8799/portal/testtoken` headless. Evaluate via `javascript_tool`:

```javascript
JSON.stringify({
  totalChips: document.querySelectorAll('.pat-chip').length,
  detailButtons: [...document.querySelectorAll('button.pat-chip--detail')].map(b=>b.dataset.pname),
  plainSpans: [...document.querySelectorAll('span.pat-chip')].map(s=>s.textContent),
  panelExists: !!document.getElementById('patDetail'),
  panelHidden: document.getElementById('patDetail')?.hidden,
  firstDesc: document.querySelector('button.pat-chip--detail')?.dataset.pdesc
})
```

Expected:
- `totalChips` = 3
- `detailButtons` = `["Source Driver","Heart Driver"]` (the two with descriptions, as buttons)
- `plainSpans` = `["Environmental Load"]` (the blank one, still a span)
- `panelExists` = true, `panelHidden` = true (nothing open yet)
- `firstDesc` starts with "The Source Driver bioenergetically supports"

Also confirm `read_console_messages` shows no uncaught errors.

- [ ] **Step 6: Commit**

```bash
git add tests/manual/portal_pattern_harness.py static/client-portal.html
git commit -m "feat(portal): persist stress-pattern chips + mark described ones as buttons"
```

---

### Task 2: Inline detail panel + click interaction

Add the single delegated handler that opens/swaps/closes the `#patDetail` panel, and the panel's CSS.

**Files:**
- Modify: `static/client-portal.html` (add `.pat-detail` CSS in the `<style>` block; add `wirePatternDetails()` and call it in `render()`)

**Interfaces:**
- Consumes: the `button.pat-chip--detail` markup + `#patDetail` panel from Task 1; the existing `esc()` helper; the persistent `#app` container (`<div id="app">`, line 321).
- Produces: `wirePatternDetails()` (idempotent; safe to call on every render).

- [ ] **Step 1: Add an interaction assertion to observe the RED state**

With Task 1 merged, reload the harness page and simulate a click on the first detail chip, then read the panel:

```javascript
(function(){
  document.querySelector('button.pat-chip--detail').click();
  const panel = document.getElementById('patDetail');
  return JSON.stringify({hidden: panel.hidden, text: panel.textContent});
})()
```

Expected NOW (no handler yet): `hidden` = true, `text` = "" — clicking does nothing.

- [ ] **Step 2: Add the panel CSS**

In the `<style>` block, right after the chip-affordance rules from Task 1 Step 4, add:

```css
  .pat-detail{margin:.15rem 0 .7rem;padding:.6rem .8rem;border-radius:10px;
    background:var(--card2);border:1px solid var(--line);
    box-shadow:inset 3px 0 0 var(--brand);font-size:.9rem;line-height:1.5}
  .pat-detail .pat-detail-name{font-weight:700;font-size:.82rem;letter-spacing:.02em;
    margin-bottom:.25rem;color:var(--brand)}
  .pat-detail:not([hidden]){animation:patExpand .28s ease}
  @keyframes patExpand{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
  @media (prefers-reduced-motion: reduce){.pat-detail:not([hidden]){animation:none}}
```

- [ ] **Step 3: Add the delegated handler**

In the `<script>`, add a module-level flag + function. Place it just above the `function render(d, v){` definition (line 470):

```javascript
// Stress-pattern detail: one delegated click listener bound ONCE to the
// persistent #app container, so it survives render()'s poll re-renders. A
// described chip (button.pat-chip--detail) toggles the single inline #patDetail
// panel: click to open, click again (or another chip) to swap/close. Only one
// panel is ever open. data-pname/data-pdesc are read back decoded, then re-esc()'d
// on inject (never trust them as raw HTML).
let _patWired = false;
function wirePatternDetails(){
  if(_patWired) return;
  const app = document.getElementById("app");
  if(!app) return;
  _patWired = true;
  app.addEventListener("click", (e)=>{
    const chip = e.target.closest(".pat-chip--detail");
    if(!chip || !app.contains(chip)) return;
    const panel = document.getElementById("patDetail");
    if(!panel) return;
    const wasOpen = chip.getAttribute("aria-expanded") === "true";
    app.querySelectorAll('.pat-chip--detail[aria-expanded="true"]')
       .forEach(c => c.setAttribute("aria-expanded", "false"));
    if(wasOpen){
      panel.hidden = true;
      panel.innerHTML = "";
      return;
    }
    chip.setAttribute("aria-expanded", "true");
    const nm = chip.getAttribute("data-pname") || "";
    const ds = chip.getAttribute("data-pdesc") || "";
    panel.innerHTML = `<div class="pat-detail-name">${esc(nm)}</div><div>${esc(ds)}</div>`;
    panel.hidden = false;
  });
}
```

- [ ] **Step 4: Call the handler from `render()`**

In `render()`, next to the other init calls (after `initCoachesCard();` / `initPeerCard();`, around line 1113-1114), add:

```javascript
  wirePatternDetails();
```

- [ ] **Step 5: Verify the full interaction (GREEN for Task 2)**

Reload the harness page headless. Run this scripted sequence via `javascript_tool`:

```javascript
(function(){
  const btns = [...document.querySelectorAll('button.pat-chip--detail')];
  const panel = document.getElementById('patDetail');
  const out = {};
  // open first
  btns[0].click();
  out.afterOpen = {hidden: panel.hidden, hasName: !!panel.querySelector('.pat-detail-name'),
                   text: panel.textContent.slice(0,40), expanded0: btns[0].getAttribute('aria-expanded')};
  // swap to second
  btns[1].click();
  out.afterSwap = {text: panel.textContent.slice(0,40),
                   expanded0: btns[0].getAttribute('aria-expanded'),
                   expanded1: btns[1].getAttribute('aria-expanded'),
                   openCount: document.querySelectorAll('.pat-chip--detail[aria-expanded="true"]').length};
  // close second
  btns[1].click();
  out.afterClose = {hidden: panel.hidden, expanded1: btns[1].getAttribute('aria-expanded')};
  return JSON.stringify(out, null, 2);
})()
```

Expected:
- `afterOpen`: `hidden` false, `hasName` true, `text` starts "The Source Driver", `expanded0` = "true"
- `afterSwap`: `text` starts "The Heart Driver", `expanded0` = "false", `expanded1` = "true", `openCount` = 1 (one open at a time)
- `afterClose`: `hidden` true, `expanded1` = "false"

Also confirm the blank chip has no handler effect: `document.querySelector('span.pat-chip').closest('button')` is `null`.

- [ ] **Step 6: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(portal): inline detail panel for described stress-pattern chips"
```

---

### Task 3: Stop the harness + final live-data sanity check

**Files:** none (verification only)

- [ ] **Step 1: Stop the harness server**

Kill the background `portal_pattern_harness.py` process.

- [ ] **Step 2: Confirm no other chip render path was missed**

Run: `grep -n "pat-chip" static/client-portal.html`
Expected: matches only in the CSS rules and the single `patternsBlock` builder from Task 1 — no second, un-updated chip render path.

- [ ] **Step 3: Commit any stray formatting (if needed)**

If `git status` is clean, skip. Otherwise:

```bash
git add static/client-portal.html
git commit -m "chore(portal): tidy stress-pattern chip render"
```

---

## Notes for the reviewer / executor

- The harness (`tests/manual/portal_pattern_harness.py`) is a manual verification tool, not a pytest test — the repo has no JS test runner and frontend changes here are verified by headless render (house practice). It is intentionally kept under `tests/manual/`.
- No backend change is required: `api_client_portal` already emits `findings` with `code/name/description/rank` (app.py:15069-15071). If a future reviewer wants a live-data check, load a real `/portal/<token>` whose latest scan has driver-category findings (ET/ED/EI/ES/MB) and confirm described chips expand.
- Descriptions are absent for ER / Nutrition / Environmental items (136 of 223). Those chips staying plain is intended, not a bug.
