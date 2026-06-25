# Navigation Shell 1a — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, injected navigation shell across illtowell.com's standalone static pages — a journey-map ribbon (rendering the existing `begin_funnel` 4-land journey), a "My Path" this-visit trail, persistent Back/Home, and safe external-link handling — fixing the "everyone gets lost" feedback.

**Architecture:** A single self-contained client bundle (`static/shell.js` + `static/shell.css`) is injected into every public HTML response by a Flask `after_request` hook (replacing `</head>`), behind the `JOURNEY_SHELL_ENABLED` flag. The shell fetches the existing `GET /begin/state` and renders its `journey_map` (4 lands: Scan→Find→Heal→Give); mode (funnel vs member) is decided server-side and passed in as `window.__SHELL__`. No new journey data model, no new endpoint, no points.

**Tech Stack:** Python 3 / Flask 3.1, vanilla JS (no framework), pytest. Existing modules reused: `begin_funnel.py` (`journey_map`, `JOURNEY_STEPS`), `app.py` (`get_authenticated_user`, `GET /begin/state`, `amg_session` cookie).

## Global Constraints

- Flag **`JOURNEY_SHELL_ENABLED`** (env, Doppler `remedy-match/prd`), **dark by default**. Flag off → `after_request` injects nothing; responses are byte-identical to today.
- **Do NOT modify** `begin_funnel.py` or the `begin_funnel.JOURNEY_STEPS` engine. 1a renders existing state only.
- **No points / rewards / celebrations in 1a** (would collide with `dashboard/points.py`). Deferred to 1b.
- **Injection exclusions** (never inject the shell): paths under `/console/`, `/admin/`, `/api/`, `/static/`; `/begin/state`; any non-`text/html` content-type; any non-200 status; the email `.j2` templates (never served as web pages).
- **Idempotent injection** — never inject twice into one response.
- **`send_from_directory` gotcha:** file responses have `direct_passthrough=True`; set `response.direct_passthrough = False` before `response.get_data()` or the rewrite silently no-ops.
- All shell CSS is scoped under `#journey-shell` / `.js-shell-*` classes so it never clobbers page styles.
- Asset URLs are `/static/shell.js`, `/static/shell.css`, `/static/shell-map.json` (served by the existing `/static/<path:filename>` route).

---

### Task 1: Server-side injection (flag + pure helpers + `after_request`)

**Files:**
- Create: `shell_nav.py` (repo root, beside `begin_funnel.py`)
- Modify: `app.py` (add flag near the other `_ENABLED` flags ~line 3245; add `after_request` hook near the bottom of the route definitions)
- Test: `tests/test_journey_shell_inject.py`

**Interfaces:**
- Produces:
  - `shell_nav.should_inject(path: str, content_type: str, status: int) -> bool`
  - `shell_nav.resolve_mode(path: str, authenticated: bool) -> str` (`"member"` | `"funnel"`)
  - `shell_nav.inject_shell_html(html: str, mode: str) -> str` (returns html unchanged if no `</head>` or already injected)
- Consumes: nothing (Task 1 is the foundation).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_journey_shell_inject.py
"""Journey shell after_request injection — pure helpers + wired behavior.
Mirrors the LOG_DB/CONSOLE_SECRET monkeypatch fixture pattern (see test_calendar.py)."""
import pytest
import shell_nav


# ---- pure helpers ----
def test_should_inject_public_html_200():
    assert shell_nav.should_inject("/begin", "text/html; charset=utf-8", 200) is True

@pytest.mark.parametrize("path", ["/console/orders", "/admin/x", "/api/journey",
                                  "/static/shell.js", "/begin/state"])
def test_should_not_inject_excluded_paths(path):
    assert shell_nav.should_inject(path, "text/html", 200) is False

def test_should_not_inject_non_html():
    assert shell_nav.should_inject("/begin/state", "application/json", 200) is False

def test_should_not_inject_non_200():
    assert shell_nav.should_inject("/begin", "text/html", 302) is False

def test_resolve_mode_member_when_authenticated():
    assert shell_nav.resolve_mode("/begin", True) == "member"

def test_resolve_mode_member_for_member_paths():
    assert shell_nav.resolve_mode("/client-portal", False) == "member"
    assert shell_nav.resolve_mode("/coaching", False) == "member"

def test_resolve_mode_funnel_default():
    assert shell_nav.resolve_mode("/begin/match", False) == "funnel"

def test_inject_adds_assets_before_head_close():
    out = shell_nav.inject_shell_html("<head><title>x</title></head><body></body>", "funnel")
    assert "/static/shell.js" in out and "/static/shell.css" in out
    assert '"mode":"funnel"' in out or "'mode':'funnel'" in out
    assert out.index("shell.js") < out.index("</head>")

def test_inject_is_idempotent():
    once = shell_nav.inject_shell_html("<head></head>", "funnel")
    twice = shell_nav.inject_shell_html(once, "funnel")
    assert twice.count("/static/shell.js") == 1

def test_inject_noop_without_head():
    assert shell_nav.inject_shell_html("<body>no head</body>", "funnel") == "<body>no head</body>"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_journey_shell_inject.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shell_nav'`.

- [ ] **Step 3: Write `shell_nav.py`**

```python
# shell_nav.py — pure helpers for the injected navigation shell (1a).
# No Flask import: everything here is unit-testable in isolation.

_EXCLUDE_PREFIXES = ("/console/", "/admin/", "/api/", "/static/")
_EXCLUDE_EXACT = ("/begin/state",)
_MEMBER_PREFIXES = ("/client-portal", "/coaching", "/affiliate-hub",
                    "/cert-portal", "/practitioner", "/dashboard", "/workspace")

_MARKER = 'id="journey-shell-assets"'


def should_inject(path: str, content_type: str, status: int) -> bool:
    """True only for public HTML 200 pages the shell should attach to."""
    if status != 200:
        return False
    if "text/html" not in (content_type or "").lower():
        return False
    p = (path or "").rstrip("/") or "/"
    if p in _EXCLUDE_EXACT:
        return False
    if any(p == pre.rstrip("/") or p.startswith(pre) for pre in _EXCLUDE_PREFIXES):
        return False
    return True


def resolve_mode(path: str, authenticated: bool) -> str:
    """Member when logged in OR on a member surface; funnel otherwise."""
    if authenticated:
        return "member"
    p = (path or "")
    if any(p.startswith(pre) for pre in _MEMBER_PREFIXES):
        return "member"
    return "funnel"


def inject_shell_html(html: str, mode: str) -> str:
    """Insert the shell <link>+<script> tags before </head>. Idempotent;
    no-op when there is no </head>."""
    if _MARKER in (html or ""):
        return html
    if "</head>" not in html:
        return html
    mode = "member" if mode == "member" else "funnel"
    tags = (
        f'<link {_MARKER} rel="stylesheet" href="/static/shell.css">'
        f'<script>window.__SHELL__={{"mode":"{mode}"}};</script>'
        f'<script defer src="/static/shell.js"></script>'
    )
    return html.replace("</head>", tags + "\n</head>", 1)
```

- [ ] **Step 4: Run pure-helper tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_journey_shell_inject.py -v`
Expected: PASS (all the helper tests above).

- [ ] **Step 5: Add the flag + `after_request` wiring in `app.py`**

Add the flag alongside the other `_ENABLED` flags (near line 3245):

```python
JOURNEY_SHELL_ENABLED = os.environ.get("JOURNEY_SHELL_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

Add `import shell_nav` with the other top-level imports (near `import begin_funnel`).

Add this `after_request` hook (place it after the last route, before `if __name__ == "__main__"`):

```python
@app.after_request
def _inject_journey_shell(response):
    if not JOURNEY_SHELL_ENABLED:
        return response
    try:
        if not shell_nav.should_inject(request.path, response.content_type or "", response.status_code):
            return response
        response.direct_passthrough = False  # static file responses default to True
        html = response.get_data(as_text=True)
        if "</head>" not in html:
            return response
        authed = bool(get_authenticated_user(request))
        mode = shell_nav.resolve_mode(request.path, authed)
        response.set_data(shell_nav.inject_shell_html(html, mode))
    except Exception as e:  # never let the shell break a page
        print(f"[journey-shell] inject skipped: {e!r}", flush=True)
    return response
```

- [ ] **Step 6: Add the integration tests**

Append to `tests/test_journey_shell_inject.py`:

```python
@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "JOURNEY_SHELL_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_shell_injected_on_begin_page(client):
    c, _ = client
    body = c.get("/begin").get_data(as_text=True)
    assert "/static/shell.js" in body
    assert 'window.__SHELL__' in body


def test_shell_not_injected_on_begin_state_json(client):
    c, _ = client
    body = c.get("/begin/state").get_data(as_text=True)
    assert "/static/shell.js" not in body


def test_shell_noop_when_flag_off(client):
    c, appmod = client
    appmod.JOURNEY_SHELL_ENABLED = False
    body = c.get("/begin").get_data(as_text=True)
    assert "/static/shell.js" not in body
```

- [ ] **Step 7: Run the full test file to verify it passes**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_journey_shell_inject.py -v`
Expected: PASS (pure helpers + 3 integration tests). If `test_shell_not_injected_on_begin_state_json` fails, confirm `/begin/state` returns `application/json` (it uses `jsonify`).

- [ ] **Step 8: Commit**

```bash
git add shell_nav.py tests/test_journey_shell_inject.py app.py
git commit -m "feat(nav): inject journey shell assets via after_request (flag-gated, dark)"
```

---

### Task 2: Presentational map config + validator

**Files:**
- Create: `static/shell-map.json`
- Modify: `shell_nav.py` (add `validate_shell_map`)
- Test: `tests/test_shell_map_config.py`

**Interfaces:**
- Consumes: `begin_funnel.JOURNEY_STEPS` (the canonical land keys `scan/find/heal/give`).
- Produces: `shell_nav.validate_shell_map(cfg: dict, land_keys: list[str]) -> list[str]` (list of error strings; empty = valid).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shell_map_config.py
import json
from pathlib import Path
import begin_funnel
import shell_nav

CFG = Path(__file__).resolve().parent.parent / "static" / "shell-map.json"


def _land_keys():
    return [s["key"] for s in begin_funnel.JOURNEY_STEPS]


def test_shipped_config_is_valid():
    cfg = json.loads(CFG.read_text())
    assert shell_nav.validate_shell_map(cfg, _land_keys()) == []


def test_validator_flags_unknown_land():
    bad = {"lands": {"scan": {"name": "x", "category": "scan", "intrigue": "y"},
                     "BOGUS": {"name": "z", "category": "scan", "intrigue": "y"}},
           "categories": {"scan": {"icon": "🌀"}}}
    errs = shell_nav.validate_shell_map(bad, ["scan", "find", "heal", "give"])
    assert any("BOGUS" in e for e in errs)


def test_validator_flags_missing_category_style():
    bad = {"lands": {"scan": {"name": "x", "category": "missing", "intrigue": "y"}},
           "categories": {}}
    errs = shell_nav.validate_shell_map(bad, ["scan", "find", "heal", "give"])
    assert any("missing" in e for e in errs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_shell_map_config.py -v`
Expected: FAIL — `static/shell-map.json` missing and `validate_shell_map` undefined.

- [ ] **Step 3: Create `static/shell-map.json`**

```json
{
  "lands": {
    "scan": {"name": "The Listening Pool", "category": "scan", "intrigue": "Your body is already speaking. Step in and listen."},
    "find": {"name": "The Hall of Mirrors", "category": "find", "intrigue": "See the one remedy your body is asking for."},
    "heal": {"name": "The Sanctuary", "category": "heal", "intrigue": "Where the root causes finally settle."},
    "give": {"name": "The Beacon", "category": "give", "intrigue": "Light the way for someone still in the fog."}
  },
  "categories": {
    "scan": {"icon": "🌀", "hue": "#4aa3a2"},
    "find": {"icon": "🔮", "hue": "#7a6cc4"},
    "heal": {"icon": "🌿", "hue": "#5aa36a"},
    "give": {"icon": "✨", "hue": "#caa64a"}
  }
}
```

- [ ] **Step 4: Add `validate_shell_map` to `shell_nav.py`**

```python
def validate_shell_map(cfg: dict, land_keys) -> list:
    """Return a list of human-readable errors. Empty list == valid.
    Every land must map to a real engine land key; every land's category
    must have a style in `categories`."""
    errors = []
    lands = (cfg or {}).get("lands") or {}
    cats = (cfg or {}).get("categories") or {}
    valid = set(land_keys or ())
    for key, land in lands.items():
        if key not in valid:
            errors.append(f"unknown land '{key}' (not a JOURNEY_STEPS key)")
        cat = (land or {}).get("category")
        if cat not in cats:
            errors.append(f"land '{key}' references missing category style '{cat}'")
    return errors
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_shell_map_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add static/shell-map.json shell_nav.py tests/test_shell_map_config.py
git commit -m "feat(nav): presentational shell-map config + validator (lands map to engine)"
```

---

### Task 3: Client shell — ribbon, journey render, Back/Home, My Path, external links

**Files:**
- Create: `static/shell.css`
- Create: `static/shell.js`
- Test: manual/visual QA (no JS test runner in this repo — the Python surface is covered in Tasks 1–2). Verification steps below.

**Interfaces:**
- Consumes: `window.__SHELL__.mode` (injected by Task 1); `GET /begin/state` → `{journey_map: [{key,label,paren,href,status,fill,steps:[{key,label,done}]}], ...}`; `GET /static/shell-map.json` (Task 2).
- Produces: a global `#journey-shell` DOM frame; `localStorage["jshell.trail"]` (this-visit page list).

- [ ] **Step 1: Create `static/shell.css`**

```css
/* Journey shell — all scoped under #journey-shell to never clobber page CSS. */
:root { --jshell-h: 52px; }
body.js-shell-on { padding-top: var(--jshell-h); }
#journey-shell { position: fixed; top: 0; left: 0; right: 0; height: var(--jshell-h);
  display: flex; align-items: center; gap: 12px; padding: 0 12px; z-index: 9999;
  background: #faf8f4; border-bottom: 1px solid #e6e0d6; font: 14px/1.2 system-ui, sans-serif; }
#journey-shell .js-home, #journey-shell .js-back { background: none; border: none;
  cursor: pointer; font-size: 18px; color: #555; padding: 4px 6px; border-radius: 6px; }
#journey-shell .js-home:hover, #journey-shell .js-back:hover { background: #efe9df; }
#journey-shell .js-path { flex: 1; display: flex; align-items: center; gap: 6px;
  overflow-x: auto; cursor: pointer; }
.js-land { display: flex; align-items: center; gap: 6px; white-space: nowrap;
  padding: 4px 8px; border-radius: 14px; color: #6b6256; opacity: .65; }
.js-land .js-icon { font-size: 16px; }
.js-land.done { opacity: 1; color: #3c6b4a; }
.js-land.next { opacity: 1; color: #8a6d1f; font-weight: 600; }
.js-land.next::after { content: "💎"; margin-left: 2px; }   /* gold gem on current */
.js-land.fog { opacity: .35; filter: blur(.3px); }
.js-trail-link { color: #b8ad99; }                          /* drawn path between lands */
.js-trail-link.done { color: #6aa37a; }
#journey-shell .js-mypath-btn { background: none; border: 1px solid #e6e0d6;
  border-radius: 14px; padding: 4px 10px; cursor: pointer; color: #555; }
.js-mypath { position: fixed; top: var(--jshell-h); right: 8px; width: 280px; max-height: 60vh;
  overflow-y: auto; background: #fff; border: 1px solid #e6e0d6; border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,.12); padding: 8px; z-index: 9999; display: none; }
.js-mypath.open { display: block; }
.js-mypath h4 { margin: 4px 8px; font-size: 12px; text-transform: uppercase; color: #999; }
.js-mypath a { display: block; padding: 6px 8px; color: #444; text-decoration: none;
  border-radius: 6px; }
.js-mypath a:hover { background: #f4f0e8; }
.js-ext-mark { font-size: .8em; opacity: .6; margin-left: 2px; }
/* member efficient nav */
#journey-shell .js-mnav { display: flex; gap: 10px; }
#journey-shell .js-mnav a { color: #555; text-decoration: none; padding: 4px 6px; }
#journey-shell .js-maptoggle { font-size: 18px; }
```

- [ ] **Step 2: Create `static/shell.js`**

```javascript
/* Journey navigation shell (1a). Vanilla, self-contained, idempotent. */
(function () {
  if (window.__jshellBooted) return;
  window.__jshellBooted = true;
  var MODE = (window.__SHELL__ && window.__SHELL__.mode) || "funnel";
  var TRAIL_KEY = "jshell.trail";

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function isExternal(href) {
    if (!href) return false;
    if (/^(mailto:|tel:|#|javascript:)/i.test(href)) return false;
    try { return new URL(href, location.href).origin !== location.origin; }
    catch (e) { return false; }
  }

  // --- My Path (this-visit trail) ---
  function recordVisit() {
    var trail = [];
    try { trail = JSON.parse(localStorage.getItem(TRAIL_KEY) || "[]"); } catch (e) {}
    var here = { t: document.title || location.pathname, p: location.pathname + location.search };
    if (!trail.length || trail[trail.length - 1].p !== here.p) trail.push(here);
    if (trail.length > 50) trail = trail.slice(-50);
    try { localStorage.setItem(TRAIL_KEY, JSON.stringify(trail)); } catch (e) {}
    return trail;
  }

  // --- external links open in a new tab + get a marker ---
  function tagExternalLinks() {
    document.querySelectorAll("a[href]").forEach(function (a) {
      if (a.dataset.jshellExt) return;
      if (isExternal(a.getAttribute("href"))) {
        a.target = "_blank"; a.rel = "noopener noreferrer";
        a.dataset.jshellExt = "1";
        a.appendChild(el("span", "js-ext-mark", "↗"));
      }
    });
  }

  // --- ribbon scaffold ---
  function buildRibbon(trail) {
    var bar = el("div"); bar.id = "journey-shell";
    var home = el("button", "js-home", "🏠"); home.title = "Home";
    home.onclick = function () { location.href = "/"; };
    var back = el("button", "js-back", "←"); back.title = "Back";
    back.onclick = function () {
      if (document.referrer && new URL(document.referrer).origin === location.origin) history.back();
      else location.href = "/";
    };
    var path = el("div", "js-path"); path.id = "js-path";
    path.title = "Open your journey map";
    var mypathBtn = el("button", "js-mypath-btn", "My Path");
    bar.appendChild(home); bar.appendChild(back); bar.appendChild(path);

    if (MODE === "member") {
      var mnav = el("div", "js-mnav",
        '<a href="/client-portal">Journal</a><a href="/coaching">Coaching</a>' +
        '<a href="/client-portal">Account</a>');
      var toggle = el("button", "js-maptoggle", "🗺️"); toggle.title = "Map / nav";
      bar.insertBefore(toggle, home);  // unobtrusive upper-left toggle
      bar.appendChild(mnav);
      toggle.onclick = function () { path.classList.toggle("js-hide"); mnav.classList.toggle("js-hide"); };
    }

    bar.appendChild(mypathBtn);
    document.body.appendChild(bar);
    document.body.classList.add("js-shell-on");

    var drawer = buildMyPath(trail);
    mypathBtn.onclick = function () { drawer.classList.toggle("open"); };
    return path;
  }

  function buildMyPath(trail) {
    var d = el("div", "js-mypath");
    d.appendChild(el("h4", null, "My Path — this visit"));
    trail.slice().reverse().forEach(function (v) {
      var a = el("a", null, v.t); a.href = v.p; d.appendChild(a);
    });
    document.body.appendChild(d);
    return d;
  }

  // --- render the 4 lands from /begin/state journey_map ---
  function renderLands(pathEl, journey, mapCfg) {
    pathEl.innerHTML = "";
    var lands = (mapCfg && mapCfg.lands) || {};
    var cats = (mapCfg && mapCfg.categories) || {};
    var seenNext = false;
    journey.forEach(function (card, i) {
      if (i > 0) {
        var link = el("span", "js-trail-link" + (journey[i - 1].status === "done" ? " done" : ""), "—");
        pathEl.appendChild(link);
      }
      var meta = lands[card.key] || {};
      var icon = (cats[meta.category] || {}).icon || "•";
      var cls = "js-land";
      if (card.status === "done") cls += " done";
      else if (card.status === "next") { cls += " next"; seenNext = true; }
      else if (seenNext) cls += " fog";  // fog upcoming lands beyond the current next
      var land = el("div", cls,
        '<span class="js-icon">' + icon + '</span>' +
        '<span>' + (meta.name || card.label) + '</span>');
      land.title = (meta.intrigue || card.paren || "");
      land.onclick = function (e) { e.stopPropagation(); if (card.href) location.href = card.href; };
      pathEl.appendChild(land);
    });
  }

  function boot() {
    var trail = recordVisit();
    tagExternalLinks();
    var pathEl = buildRibbon(trail);
    Promise.all([
      fetch("/begin/state", { credentials: "same-origin" }).then(function (r) { return r.json(); }).catch(function () { return {}; }),
      fetch("/static/shell-map.json").then(function (r) { return r.json(); }).catch(function () { return {}; })
    ]).then(function (res) {
      var journey = (res[0] && res[0].journey_map) || [];
      if (journey.length) renderLands(pathEl, journey, res[1]);
      else pathEl.appendChild(el("span", "js-land", "illtowell.com"));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
```

- [ ] **Step 3: Verify locally with the flag on**

Run (background): `cd ~/deploy-chat && JOURNEY_SHELL_ENABLED=1 doppler run -p remedy-match -c prd -- python3 app.py` then open `http://localhost:<port>/begin`.
Expected: a fixed top ribbon with 🏠 / ← / the 4 lands (gem 💎 on the current stage, completed stages connected by a green trail) / "My Path". Clicking "My Path" opens the this-visit drawer. Any off-site link shows "↗" and opens a new tab.

- [ ] **Step 4: Re-run the Python suite (no regressions)**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_journey_shell_inject.py tests/test_shell_map_config.py tests/test_begin_journey_map.py -v`
Expected: PASS (the existing journey-map engine test still passes — we did not touch `begin_funnel`).

- [ ] **Step 5: Commit**

```bash
git add static/shell.js static/shell.css
git commit -m "feat(nav): client shell — ribbon, journey render, Back/Home, My Path, external links"
```

---

### Task 4: Expand-to-map overlay + member toggle polish

**Files:**
- Modify: `static/shell.js` (add overlay open/close + render pavilions)
- Modify: `static/shell.css` (overlay styles)
- Test: manual/visual QA.

**Interfaces:**
- Consumes: the same `journey` array + `mapCfg` from Task 3 (each land's `steps:[{label,done}]` are the pavilions inside).
- Produces: a `.js-overlay` full-screen map; reuses `renderLands`'s data.

- [ ] **Step 1: Add overlay CSS to `static/shell.css`**

```css
.js-overlay { position: fixed; inset: 0; background: rgba(28,24,18,.78); z-index: 10000;
  display: none; align-items: center; justify-content: center; }
.js-overlay.open { display: flex; }
.js-overlay-inner { background: #faf8f4; border-radius: 16px; padding: 20px;
  max-width: 860px; width: 92vw; max-height: 86vh; overflow-y: auto;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
.js-pav-land { border: 1px solid #e6e0d6; border-radius: 12px; padding: 14px; }
.js-pav-land.fog { opacity: .45; }
.js-pav-land h3 { margin: 0 0 4px; font-size: 16px; }
.js-pav-land .js-intrigue { font-size: 12px; color: #8a8073; margin-bottom: 8px; }
.js-pav { display: block; padding: 5px 0; color: #555; text-decoration: none; font-size: 13px; }
.js-pav.done::before { content: "✓ "; color: #5aa36a; }
.js-overlay-close { position: fixed; top: 14px; right: 18px; font-size: 26px; color: #fff;
  background: none; border: none; cursor: pointer; z-index: 10001; }
```

- [ ] **Step 2: Add the overlay logic to `static/shell.js`**

Inside the IIFE, add this function and wire it into `renderLands`' caller. Add after `renderLands`:

```javascript
  function buildOverlay(journey, mapCfg) {
    var lands = (mapCfg && mapCfg.lands) || {};
    var cats = (mapCfg && mapCfg.categories) || {};
    var ov = el("div", "js-overlay");
    var close = el("button", "js-overlay-close", "×");
    close.onclick = function () { ov.classList.remove("open"); };
    var inner = el("div", "js-overlay-inner");
    var seenNext = false;
    journey.forEach(function (card) {
      var meta = lands[card.key] || {};
      var icon = (cats[meta.category] || {}).icon || "•";
      var fog = (card.status !== "done" && card.status !== "next" && seenNext);
      if (card.status === "next") seenNext = true;
      var box = el("div", "js-pav-land" + (fog ? " fog" : ""));
      box.appendChild(el("h3", null, icon + " " + (meta.name || card.label)));
      box.appendChild(el("div", "js-intrigue", meta.intrigue || card.paren || ""));
      (card.steps || []).forEach(function (s) {
        var a = el("a", "js-pav" + (s.done ? " done" : ""), s.label);
        a.href = card.href || "#";
        box.appendChild(a);
      });
      inner.appendChild(box);
    });
    ov.appendChild(close); ov.appendChild(inner);
    ov.onclick = function (e) { if (e.target === ov) ov.classList.remove("open"); };
    document.body.appendChild(ov);
    return ov;
  }
```

Then change the `.then` in `boot()` so the ribbon click opens the overlay:

```javascript
      if (journey.length) {
        renderLands(pathEl, journey, res[1]);
        var overlay = buildOverlay(journey, res[1]);
        pathEl.addEventListener("click", function () { overlay.classList.add("open"); });
      } else pathEl.appendChild(el("span", "js-land", "illtowell.com"));
```

- [ ] **Step 3: Verify the overlay locally**

With the flag-on server from Task 3, click the ribbon path on `/begin`.
Expected: a full-screen park overlay — 4 land cards, each with its flavor name, intrigue line, and its pavilions (sub-steps; completed ones show ✓). Lands beyond the current stage are fogged. `×` or backdrop closes it.

- [ ] **Step 4: Commit**

```bash
git add static/shell.js static/shell.css
git commit -m "feat(nav): expand-to-map overlay — lands open to pavilions with fog"
```

---

### Task 5: Integration smoke + rollout wiring

**Files:**
- Modify: `docs/superpowers/specs/2026-06-24-navigation-shell-journey-map-design.md` (mark 1a built)
- Test: manual smoke + full suite.

- [ ] **Step 1: Full suite green**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_journey_shell_inject.py tests/test_shell_map_config.py -v && python3 -m pytest tests/ -k "begin" -q`
Expected: PASS, no regressions in the `begin*` tests.

- [ ] **Step 2: Two-surface manual smoke (flag on)**

With `JOURNEY_SHELL_ENABLED=1`: load a **funnel** page (`/begin/match`) → ribbon in funnel mode (no member nav). Load a **member** surface (`/client-portal`) → ribbon shows the efficient nav + the 🗺️ toggle. Load `/console/orders` (if accessible) → **no** shell injected. View-source any page → exactly one `/static/shell.js`.

- [ ] **Step 3: Confirm dark-by-default**

Restart the server **without** the flag. Load `/begin` → no `#journey-shell`, page byte-identical to today.

- [ ] **Step 4: Note the go-live step in the spec**

Add to the spec's Rollout section: "**1a go-live:** set `JOURNEY_SHELL_ENABLED=1` in Doppler `remedy-match/prd` after pilot review." Commit:

```bash
git add docs/superpowers/specs/2026-06-24-navigation-shell-journey-map-design.md
git commit -m "docs(nav): mark 1a built; record JOURNEY_SHELL_ENABLED go-live step"
```

- [ ] **Step 5: Open the PR** (do not flip the flag; ships dark)

```bash
git push -u origin sess/6a686b75
gh pr create --title "Navigation Shell 1a — journey ribbon + map + My Path (dark)" \
  --body "Persistent injected nav shell across standalone static pages. Renders the existing begin_funnel 4-land journey, adds My Path this-visit trail, Back/Home, and safe external-link handling. Flag-gated (JOURNEY_SHELL_ENABLED), dark by default — no behavior change until flipped. Reuses GET /begin/state; begin_funnel engine untouched; no points (1b).

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review

**Spec coverage:**
- Shell injection across standalone pages → Task 1. ✓
- Funnel vs member modes → Task 1 (`resolve_mode`, server-decided) + Task 3 (member nav) + Task 4 (toggle). ✓
- Ribbon renders existing `journey_map`, gem on `status:"next"`, drawn trail on done → Task 3. ✓
- Fog on upcoming lands → Task 3 (ribbon) + Task 4 (overlay). ✓
- Expand-to-map (lands → pavilions) → Task 4. ✓
- My Path this-visit trail (localStorage, anonymous) → Task 3. ✓
- Back/Home + external-link new-tab + marker → Task 3. ✓
- Categorical style library → `static/shell-map.json` categories (Task 2), consumed Tasks 3–4. ✓
- No new journey table/endpoint; reuse `/begin/state` + `amg_session` → Tasks 1, 3. ✓
- No points in 1a → enforced (none added); Global Constraints. ✓
- Flag dark-by-default, exclusions, idempotent, `direct_passthrough` → Task 1 + Global Constraints. ✓

**Placeholder scan:** none — every step has concrete code or exact commands.

**Type consistency:** `should_inject` / `resolve_mode` / `inject_shell_html` / `validate_shell_map` signatures match between Tasks 1–2 and their tests; JS consumes the `journey_map` shape (`key/label/paren/href/status/fill/steps[]`) exactly as produced by `begin_funnel.journey_map` (verified in source); `window.__SHELL__.mode` produced in Task 1, consumed in Task 3.
