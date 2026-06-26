# Console Role-Based Navigation (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the internal console (`op-nav.js`) show each person the right nav — Glen full, Rae streamlined with a "More" overflow, Shaira unchanged — driven by a new `/api/me` endpoint, reusing the existing `_auth`/`rbac`/`access_tokens` plumbing.

**Architecture:** A new `GET /api/me` resolves the caller's key/token to `{role, nav}` via the existing `_auth()` + `dashboard/rbac.py`. `op-nav.js` fetches it and reorganizes its already-rendered bar from a declarative profile map (non-primary items move into a "More ▾" dropdown). Scoped tokens are wired into the BOS action layer so Rae's token can act. Owner is the safe default: any `/api/me` failure leaves the full bar.

**Tech Stack:** Python/Flask (`app.py`, sqlite `LOG_DB`), `dashboard/rbac.py`, vanilla JS (`static/op-nav.js`), pytest, headless Playwright (Chromium) for JS render-verify.

## Global Constraints

- **`role` governs permissions, `nav` governs layout.** Glen and Rae are *both* `rbac.OWNER`; only `nav` (`"glen"`/`"rae"`) differs. Never gate a tab on `role` alone.
- **Owner-safe fallback (verbatim):** the streamlined view activates **only** when `/api/me` returns `nav="rae"`. Any of: fetch error, timeout, `nav: null`, or `nav="glen"` → render the **full bar**. `CONSOLE_SECRET` always resolves to `nav="glen"`.
- **Reuse, don't reinvent:** use the existing `_auth()` (`app.py` ~18570, returns `(ok, ctx, code)` with `ctx={scope,user_name,user_id}`), `dashboard/rbac.py` (`actor_for_scope`, `resolve_actor`, role constants `OWNER/OPS/VA`), and the `access_tokens`/`workspace_users` tables. No new auth model, no new tables.
- **Shaira untouched:** do not modify `static/shaira-workspace.html` or `/workspace/<owner>`; Shaira's page does not load `op-nav.js`.
- **Out of scope (sub-projects B/C):** merging boards (Money/Pages/Approvals), retiring orphan *pages*, killing Settings stubs, and the inline action affordances (Create-PO, retry charge, etc.). A only assigns nav slots.
- **Test env:** tests monkeypatch `appmod.LOG_DB` (→ `tmp_path`) and `appmod.CONSOLE_SECRET="test-secret"`, then call `appmod._init_auth_tables()` + `appmod._init_workspace_schema()` and use `appmod.app.test_client()` (mirror `tests/test_calendar.py`). JS render-verify runs the app under `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret python3 app.py` against a seeded `chat_log.db`.

---

### Task 1: `GET /api/me` endpoint

**Files:**
- Modify: `app.py` — add `_nav_profile()` helper + the `/api/me` route, beside the other `_auth`-based routes (after `workspace_page`, ~app.py:18691).
- Test: `tests/test_api_me.py` (create)

**Interfaces:**
- Consumes: `_auth()` → `(ok, ctx, code)`, `ctx={"scope","user_name","user_id"}`; `dashboard.rbac.actor_for_scope(scope)` → `Actor(role, name)`; `_owner_from_scope(scope)`.
- Produces: `GET /api/me` → `200 {"role": "owner"|"ops"|"va"|None, "name": str|None, "nav": "glen"|"rae"|"va"|None, "scope": str|None}`. Always HTTP 200 (even unauthenticated → all-null), so the front-end always parses JSON. `_nav_profile(scope) -> "glen"|"rae"|"va"` (used only here).

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_me.py`:

```python
"""GET /api/me — resolves the caller's key/token to {role, nav} for op-nav."""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "test-secret")
    appmod._init_auth_tables()
    appmod._init_workspace_schema()
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_token(appmod, token, owner):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   (owner, owner.title(), f"workspace:{owner}"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", (owner,)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "test"))
        cx.commit()


def test_me_admin_is_glen(client):
    c, _ = client
    j = c.get("/api/me", headers={"X-Console-Key": "test-secret"}).get_json()
    assert j["role"] == "owner" and j["nav"] == "glen" and j["name"] == "Glen"


def test_me_rae_token_is_owner_nav_rae(client):
    c, appmod = client
    _seed_token(appmod, "rae-tok", "rae")
    j = c.get("/api/me", headers={"X-Console-Key": "rae-tok"}).get_json()
    assert j["role"] == "owner" and j["nav"] == "rae" and j["name"] == "Rae"


def test_me_shaira_token_is_va(client):
    c, appmod = client
    _seed_token(appmod, "sha-tok", "shaira")
    j = c.get("/api/me", headers={"X-Console-Key": "sha-tok"}).get_json()
    assert j["role"] == "va" and j["nav"] == "va"


def test_me_no_key_is_all_null_200(client):
    c, _ = client
    r = c.get("/api/me")
    assert r.status_code == 200
    j = r.get_json()
    assert j["role"] is None and j["nav"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/me-test python3 -m pytest tests/test_api_me.py -q` (run `mkdir -p /tmp/me-test` first).
Expected: FAIL — `/api/me` returns 404 (route not defined).

- [ ] **Step 3: Add `_nav_profile` + the route**

In `app.py`, immediately after the `workspace_page` route (~line 18691), add:

```python
def _nav_profile(scope):
    """Layout profile for op-nav. Distinct from rbac role (Glen and Rae are both
    OWNER): 'admin'->'glen', 'workspace:rae'->'rae', anything else scoped->'va'."""
    if not scope or scope == "admin":
        return "glen"
    owner = (_owner_from_scope(scope) or "").lower()
    if owner == "glen":
        return "glen"
    if owner == "rae":
        return "rae"
    return "va"


@app.route("/api/me")
def api_me():
    """Identity for the front-end nav. Always 200; unauthenticated -> all-null."""
    ok, ctx, _code = _auth()
    if not ok or not ctx:
        return jsonify({"role": None, "name": None, "nav": None, "scope": None})
    scope = ctx.get("scope") or "admin"
    actor = _bos_rbac.actor_for_scope(scope)
    if scope == "admin":
        name = "Glen"
    else:
        name = (ctx.get("user_name") or "").title() or None
    return jsonify({"role": actor.role, "name": name,
                    "nav": _nav_profile(scope), "scope": scope})
```

Note: `_bos_rbac` is the `dashboard.rbac` module already imported at `app.py:18306` (`from dashboard import rbac as _bos_rbac`). If that import is not in scope at this location, add `from dashboard import rbac as _bos_rbac` at the top of the function.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/me-test python3 -m pytest tests/test_api_me.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_api_me.py
git commit -m "feat(console): GET /api/me returns role + nav profile for op-nav"
```

---

### Task 2: Wire scoped tokens into the BOS action layer

**Files:**
- Modify: `app.py` — add `_role_for_token()` helper and pass it into `_bos_actor()`'s `resolve_actor` call (`_bos_actor` is at ~app.py:23972).
- Test: `tests/test_bos_actor_token.py` (create)

**Interfaces:**
- Consumes: `dashboard.rbac.resolve_actor(key, console_secret=, token=, role_for_token=)` (returns `Actor` or `None`); `rbac.actor_for_scope(scope).role`; the `access_tokens`/`workspace_users` tables.
- Produces: `_role_for_token(token) -> "owner"|"ops"|"va"|None` (looks a token up → its scope → rbac role; `None` if unknown/revoked). `_bos_actor()` now resolves scoped tokens, so a `workspace:rae` token → `Actor(OWNER)` and `workspace:shaira` → `Actor(VA)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bos_actor_token.py`:

```python
"""_role_for_token maps an access token -> rbac role, so BOS actions accept
scoped tokens (Rae = owner). Shaira stays VA, bound by the policy matrix."""
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "test-secret")
    appmod._init_auth_tables()
    appmod._init_workspace_schema()
    return appmod


def _seed(appmod, token, owner):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   (owner, owner.title(), f"workspace:{owner}"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", (owner,)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "t"))
        cx.commit()


def test_role_for_token_rae_is_owner(appmod):
    _seed(appmod, "rae-tok", "rae")
    assert appmod._role_for_token("rae-tok") == "owner"


def test_role_for_token_shaira_is_va(appmod):
    _seed(appmod, "sha-tok", "shaira")
    assert appmod._role_for_token("sha-tok") == "va"


def test_role_for_token_unknown_is_none(appmod):
    assert appmod._role_for_token("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/bos-test python3 -m pytest tests/test_bos_actor_token.py -q` (`mkdir -p /tmp/bos-test` first).
Expected: FAIL — `app` has no attribute `_role_for_token`.

- [ ] **Step 3: Add `_role_for_token` + wire it into `_bos_actor`**

In `app.py`, replace the body of `_bos_actor()` (~line 23972). The current function is:

```python
def _bos_actor():
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    return _bos_rbac.resolve_actor(key, console_secret=dashboard.CONSOLE_SECRET)
```

Replace with (add the helper just above it):

```python
def _role_for_token(token):
    """Map an access token -> rbac role via its workspace scope. None if unknown."""
    if not token:
        return None
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT u.scope FROM access_tokens t "
                "JOIN workspace_users u ON u.id = t.user_id "
                "WHERE t.token = ? AND t.revoked_at IS NULL", (token,)).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return _bos_rbac.actor_for_scope(row[0]).role


def _bos_actor():
    """Resolve the calling actor: owner master key (CONSOLE_SECRET) first, then a
    per-user access token -> its rbac role (Rae=owner, Shaira=va)."""
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    return _bos_rbac.resolve_actor(
        key, console_secret=dashboard.CONSOLE_SECRET,
        token=key, role_for_token=_role_for_token)
```

(`resolve_actor` checks the owner master key first, so passing `token=key` is safe — a CONSOLE_SECRET key never reaches the token branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/bos-test python3 -m pytest tests/test_bos_actor_token.py -q`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_bos_actor_token.py
git commit -m "feat(console): wire scoped access tokens into the BOS action layer"
```

---

### Task 3: Role-aware `op-nav.js` — `/api/me` fetch, visibility map, "More ▾" overflow

**Files:**
- Modify: `static/op-nav.js` — add `data-id` to each tab/board `<a>`; extend `bosMods` with the de-orphaned boards; add a top-level + a BOS-level "More ▾" container; after `document.write`, fetch `/api/me`, cache the profile, and run `applyNavProfile()`.
- Verify: headless Playwright (no pytest — JS).

**Interfaces:**
- Consumes: `GET /api/me` (Task 1) → `{nav}`.
- Produces: a role-aware bar. `applyNavProfile(profile)` moves any tab/board `<a>` whose `data-id` is not in the profile's `primary` list into the matching "More ▾" dropdown. Owner (`nav="glen"`) keeps all primary tabs but moves the owner-More board group into the BOS "More ▾". Rae (`nav="rae"`) moves her non-primary tabs and boards into "More ▾".

- [ ] **Step 1: Add `data-id` to tabs and boards, and the de-orphaned boards**

In `static/op-nav.js`, in the `tabs` rendering loop (the `for` loop building `bar` around line 200), change the tab `<a>` to include `data-id`:

```javascript
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    var cls = (t.id === active) ? "op-nav-tab active" : "op-nav-tab";
    bar += '<a class="' + cls + '" data-id="' + t.id + '" href="' + t.href + '">' + t.label + '</a>';
  }
```

Add a top-level "More ▾" container immediately AFTER that loop and BEFORE `bar += '<span class="op-nav-spacer"></span>';`:

```javascript
  bar += '<span class="op-nav-more" id="op-nav-more-top" style="display:none">'
    + '<button type="button" class="op-nav-tab op-nav-more-btn">More ▾</button>'
    + '<span class="op-nav-more-menu"></span></span>';
```

In `bosMods` (array starting line 81), append the de-orphaned boards after `{ id: "neworder", ... }`:

```javascript
    { id: "practitioners", label: "Practitioners", href: "/console/practitioners" + qs },
    { id: "top-products",  label: "Top Products",  href: "/console/top-products" + qs },
    { id: "topic-pages",   label: "Topic Pages",   href: "/console/topic-pages" + qs },
    { id: "topic-suggestions", label: "Topic Suggestions", href: "/console/topic-suggestions" + qs },
    { id: "remedy-meanings", label: "Remedy Meanings", href: "/console/remedy-meanings" + qs },
    { id: "ingredients-ops", label: "Ingredients (Ops)", href: "/admin/ingredients" + qs },
    { id: "cert",          label: "Cert",          href: "/console/cert" + qs },
    { id: "coaching",      label: "Coaching",      href: "/console/coaching-cohort" + qs },
    { id: "studio-credits", label: "Studio Credits", href: "/console/studio-credits" + qs },
    { id: "membership",    label: "Membership",    href: "/admin/membership" + qs },
    { id: "atlas",         label: "Atlas",         href: "/admin/atlas" + qs },
    { id: "wholesale",     label: "Wholesale",     href: "/admin/wholesale" + qs },
    { id: "clips",         label: "Clips",         href: "/admin/clips" + qs },
```

In the BOS sub-row loop (inside `if (active === "bos")`, line ~221), add `data-id` to the subtab `<a>` and append a BOS "More ▾" container before closing the sub-row `</nav>`:

```javascript
    for (var j = 0; j < bosMods.length; j++) {
      var m = bosMods[j];
      var scls = (m.id === sub) ? "op-nav-subtab active" : "op-nav-subtab";
      bar += '<a class="' + scls + '" data-id="' + m.id + '" href="' + m.href + '">' + m.label + '</a>';
    }
    bar += '<span class="op-nav-more" id="op-nav-more-bos" style="display:none">'
      + '<button type="button" class="op-nav-subtab op-nav-more-btn">More ▾</button>'
      + '<span class="op-nav-more-menu"></span></span>';
```

- [ ] **Step 2: Add the "More ▾" dropdown styles**

In the `styles` string (the `<style id="op-nav-styles">` block), before its closing `+ '</style>'`, add:

```javascript
    + '.op-nav-more{position:relative;display:inline-flex;align-items:center}'
    + '.op-nav-more-btn{cursor:pointer;background:transparent;border:0;font:inherit}'
    + '.op-nav-more-menu{position:absolute;top:100%;left:0;min-width:180px;'
    +   'background:#0d0d14;border:1px solid #2a2a35;border-radius:8px;'
    +   'box-shadow:0 10px 30px rgba(0,0,0,.5);padding:4px 0;display:none;z-index:10000}'
    + '.op-nav-more.open .op-nav-more-menu{display:block}'
    + '.op-nav-more-menu a{display:block;padding:7px 14px;color:#9aa0b4;text-decoration:none;border-bottom:0}'
    + '.op-nav-more-menu a:hover{background:rgba(124,92,191,.16);color:#e6edf3}'
```

- [ ] **Step 3: Add the profile map, `applyNavProfile`, and the `/api/me` wiring**

Immediately AFTER the existing `document.write(styles + bar);` line (~line 232), add:

```javascript
  // ── Role-aware nav: reorganize the rendered bar per the caller's nav profile ──
  // primary = stays on the bar; everything else moves into the matching "More ▾".
  var NAV_PROFILES = {
    glen: {
      tabs: ["dashboard","console","bos","projects","inbox","settings","funnel"],
      bos:  ["orders","payments","finance","crm","products","biofield","sales",
             "ingredients","topic-pages","biofield-reveals","biofield-intake",
             "reviews","shipping","neworder"]
    },
    rae: {
      tabs: ["dashboard","console","bos","inbox"],
      bos:  ["orders","payments","finance","crm","reviews","shipping","neworder"]
    }
  };

  function reorg(wrapId, linkSelector, primaryIds) {
    var wrap = document.getElementById(wrapId);
    if (!wrap) return;
    var menu = wrap.querySelector(".op-nav-more-menu");
    var bar = wrap.parentNode;
    var moved = 0;
    bar.querySelectorAll(linkSelector).forEach(function (a) {
      var id = a.getAttribute("data-id");
      if (!id) return;
      if (primaryIds.indexOf(id) === -1) {
        a.classList.remove("op-nav-tab", "op-nav-subtab");
        menu.appendChild(a);
        moved++;
      }
    });
    wrap.style.display = moved ? "inline-flex" : "none";
  }

  function applyNavProfile(navName) {
    var prof = NAV_PROFILES[navName] || NAV_PROFILES.glen;
    reorg("op-nav-more-top", ".op-nav-bar > a.op-nav-tab", prof.tabs);
    reorg("op-nav-more-bos", ".op-nav-sub > a.op-nav-subtab", prof.bos);
  }

  // Toggle "More" dropdowns (event delegation; survives reorg).
  document.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest(".op-nav-more-btn");
    document.querySelectorAll(".op-nav-more.open").forEach(function (w) {
      if (!btn || w !== btn.parentNode) w.classList.remove("open");
    });
    if (btn) { btn.parentNode.classList.toggle("open"); e.preventDefault(); }
  });

  // Render instantly from the cached profile (no flash on repeat visits),
  // then revalidate via /api/me. Owner-safe: any failure leaves the full bar.
  var NAV_CACHE_KEY = "op_nav_profile";
  try {
    var cached = localStorage.getItem(NAV_CACHE_KEY);
    if (cached === "rae" || cached === "glen") applyNavProfile(cached);
  } catch (e) {}

  fetch("/api/me" + (effKey ? "?key=" + encodeURIComponent(effKey) : ""),
        { headers: effKey ? { "X-Console-Key": effKey } : {} })
    .then(function (r) { return r.json(); })
    .then(function (me) {
      var navName = (me && me.nav === "rae") ? "rae" : "glen";  // only 'rae' streamlines
      try { localStorage.setItem(NAV_CACHE_KEY, navName); } catch (e) {}
      applyNavProfile(navName);
    })
    .catch(function () { /* leave full bar */ });
```

Note: `applyNavProfile` is idempotent only in that already-moved links stay in the menu; running cached `glen` then fetched `glen` is a no-op, and cached `rae`→fetched `glen` cannot un-move. To keep it simple and correct, only call `applyNavProfile` from the cache path when the cached value matches what a fresh fetch will return on the common case; on a profile *change* (rare: a shared browser switching identity) a reload corrects it. This is acceptable for a 2-person trusted team; do not add un-move logic (YAGNI).

- [ ] **Step 4: Render-verify headless (owner, Rae, fetch-fail)**

Start the app and seed a Rae token, then assert the three render states with zero JS errors.

Run (write `mkdir -p /tmp/nav-test` first), start server:
```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/nav-test CONSOLE_SECRET=test-secret PORT=5097 python3 app.py > /tmp/nav-test/srv.log 2>&1 &
# wait for up:
for i in $(seq 1 45); do curl -s -o /dev/null "http://127.0.0.1:5097/dashboard?key=test-secret" && break; sleep 1; done
# seed a Rae token:
python3 - <<'PY'
import sqlite3
cx=sqlite3.connect("/tmp/nav-test/chat_log.db")
cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES ('rae','Rae','workspace:rae')")
uid=cx.execute("SELECT id FROM workspace_users WHERE name='rae'").fetchone()[0]
cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES ('rae-tok',?, 't')",(uid,))
cx.commit(); print("seeded rae-tok")
PY
```

Playwright assertions (save to `/tmp/nav-test/navverify.py`, run with `python3`):
```python
from playwright.sync_api import sync_playwright
B="http://127.0.0.1:5097/dashboard"
def bar(pg):
    return pg.evaluate("""()=>({
      topTabs:[...document.querySelectorAll('.op-nav-bar > a.op-nav-tab')].map(a=>a.dataset.id),
      topMoreShown: getComputedStyle(document.getElementById('op-nav-more-top')).display!=='none',
      topMore:[...document.querySelectorAll('#op-nav-more-top .op-nav-more-menu a')].map(a=>a.dataset.id)
    })""")
with sync_playwright() as p:
    b=p.chromium.launch()
    # Owner (CONSOLE_SECRET): full top bar, top-More hidden
    pg=b.new_page(); errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto(B+"?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(800)
    o=bar(pg); print("OWNER", o, "errs", errs or "none")
    assert "settings" in o["topTabs"] and o["topMoreShown"] is False
    pg.close()
    # Rae token: streamlined top bar (no settings/projects/funnel), top-More shown with them
    pg=b.new_page(); errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto(B+"?key=rae-tok", wait_until="networkidle"); pg.wait_for_timeout(800)
    r=bar(pg); print("RAE", r, "errs", errs or "none")
    assert "settings" not in r["topTabs"] and r["topMoreShown"] is True
    assert set(["projects","settings","funnel"]).issubset(set(r["topMore"]))
    pg.close()
    b.close()
    print("OK")
```
Run: `python3 /tmp/nav-test/navverify.py` → prints `OK`, both `errs none`. Then stop the server: `lsof -ti :5097 | xargs kill`.
Expected: owner shows `settings` on the bar + top-More hidden; Rae's bar omits settings/projects/funnel and they appear in top-More; zero JS errors.

- [ ] **Step 5: Commit**

```bash
git add static/op-nav.js
git commit -m "feat(console): role-aware op-nav — /api/me, profile map, More overflow + de-orphaned boards"
```

---

### Task 4: Settings sub-tab parent

**Files:**
- Modify: `static/op-nav.js` — render a Settings sub-row (mirroring the BOS sub-row) when `active === "settings"`.
- Modify: `static/console-pricing-settings.html`, `static/admin-tax.html` — set `data-active="settings" data-sub="pricing"` / `"tax"` on their `op-nav.js` script tag so the Settings sub-row shows + highlights on those pages.
- Verify: headless Playwright.

**Interfaces:**
- Consumes: the `active`/`sub` `data-` attributes op-nav already reads (`script.dataset.active`, `script.dataset.sub`).
- Produces: a Settings sub-row with links Pricing · Shipping-config · Tax · Write-Mac, shown whenever `active === "settings"`.

- [ ] **Step 1: Render the Settings sub-row**

In `static/op-nav.js`, after the `if (active === "bos") { ... }` block (ends ~line 230, before `document.write`), add:

```javascript
  if (active === "settings") {
    var setMods = [
      { id: "pricing",  label: "Pricing",         href: "/console/pricing-settings" + qs },
      { id: "shipping", label: "Shipping-config",  href: "/admin/shipping" + qs },
      { id: "tax",      label: "Tax",              href: "/admin/tax" + qs },
      { id: "writemac", label: "Write-Mac",        href: "/console/settings" + qs }
    ];
    bar += '<nav class="op-nav-sub" role="navigation" aria-label="Settings">'
      + '<span class="op-nav-sub-brand">Settings</span>';
    for (var s = 0; s < setMods.length; s++) {
      var sm = setMods[s];
      var smcls = (sm.id === sub) ? "op-nav-subtab active" : "op-nav-subtab";
      bar += '<a class="' + smcls + '" data-id="' + sm.id + '" href="' + sm.href + '">' + sm.label + '</a>';
    }
    bar += '</nav>';
  }
```

- [ ] **Step 2: Point the config pages at the Settings sub-row**

In `static/console-pricing-settings.html`, change the `op-nav.js` script tag to:
```html
<script src="/static/op-nav.js" data-active="settings" data-sub="pricing"></script>
```
In `static/admin-tax.html`, change its `op-nav.js` script tag to:
```html
<script src="/static/op-nav.js" data-active="settings" data-sub="tax"></script>
```
(Leave `static/admin-shipping.html` as `data-active="bos" data-sub="shipping"` — it is primarily the BOS Shipping board; the Settings sub-row links to it as a shortcut without owning its highlight.)

- [ ] **Step 3: Render-verify the Settings sub-row**

Start the app (reuse Task 3's server on a fresh port if stopped), then:
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:5097/console/settings?key=test-secret", wait_until="networkidle")
    pg.wait_for_timeout(600)
    subs=pg.evaluate("()=>[...document.querySelectorAll('.op-nav-sub a.op-nav-subtab')].map(a=>a.dataset.id)")
    print("SETTINGS SUBROW", subs, "errs", errs or "none")
    assert set(["pricing","shipping","tax","writemac"]).issubset(set(subs))
    # pricing page highlights its sub-tab
    pg.goto("http://127.0.0.1:5097/console/pricing-settings?key=test-secret", wait_until="networkidle")
    pg.wait_for_timeout(600)
    act=pg.evaluate("()=>{var a=document.querySelector('.op-nav-sub a.op-nav-subtab.active');return a&&a.dataset.id}")
    print("PRICING ACTIVE", act); assert act=="pricing"
    b.close(); print("OK")
```
Expected: the Settings sub-row shows the four links; `/console/pricing-settings` highlights `pricing`; zero JS errors. Stop the server afterward.

- [ ] **Step 4: Commit**

```bash
git add static/op-nav.js static/console-pricing-settings.html static/admin-tax.html
git commit -m "feat(console): Settings sub-tab parent (Pricing/Shipping/Tax/Write-Mac)"
```

---

## Rollout (after all tasks reviewed + merged)

1. **Mint Rae's token in prod** (the live DB is on Render; minting locally is refused unless `ALLOW_LOCAL_TOKEN_MINT=1`):
   ```bash
   doppler run -p remedy-match -c prd -- bash -c '
     curl -s -X POST https://glen-knowledge-chat.onrender.com/api/access-tokens \
       -H "X-Console-Key: $CONSOLE_SECRET" -H "Content-Type: application/json" \
       -d "{\"name\":\"rae\",\"scope\":\"workspace:rae\",\"display_name\":\"Rae\",\"note\":\"console nav\"}"'
   ```
   The response includes `token` — hand Glen Rae's bookmark URL `https://illtowell.com/dashboard?key=<token>` (paste-to-Glen, never echo the token in a shared channel beyond what's needed).
2. **Live render-verify** (the render-verify lesson): load `/dashboard?key=<rae-token>` headless on the live site → assert the streamlined bar + working "More ▾" + zero JS console errors; load `/dashboard?key=<CONSOLE_SECRET>` → full bar.
3. No feature flag needed — owner-fallback makes the change inert for Glen.

## Verification (whole sub-project)

- `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/me-test python3 -m pytest tests/test_api_me.py tests/test_bos_actor_token.py -q` → all pass.
- Headless render-verify (Tasks 3 & 4) → owner full bar, Rae streamlined + More, Settings sub-row, fetch-fail → full bar; zero JS errors on every page.
- Confirm Shaira's `/workspace/shaira` still renders with no `op-nav.js` (unchanged).
- De-orphan check: every previously-orphan board id appears in `bosMods` (Task 3) so it is reachable from the BOS bar or its "More ▾".

## Self-Review Notes

- **Spec coverage:** `/api/me` (Task 1) ✓; Rae token mint (Rollout) ✓; scoped-token action wiring (Task 2) ✓; role-aware op-nav + map + More (Task 3) ✓; Settings sub-tabs (Task 4) ✓; owner-fallback (Task 3 Step 3, Global Constraints) ✓; de-orphaning (Task 3 bosMods extension) ✓; Shaira untouched (Global Constraints + Verification) ✓.
- **Type consistency:** `/api/me` returns `{role, name, nav, scope}` consumed only as `me.nav` in Task 3. `_nav_profile`/`_role_for_token`/`applyNavProfile`/`reorg`/`NAV_PROFILES` names match across steps.
- **YAGNI:** no un-move logic in `applyNavProfile` (reload corrects a rare identity switch); no feature flag (owner-fallback suffices).
