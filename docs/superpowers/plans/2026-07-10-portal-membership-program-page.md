# Membership Program Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate, personalized-by-tier membership program page inside the client portal that sells Free → Paid → Family, gives ambassadors their affiliate links, and lightly introduces practitioner/coach/certification.

**Architecture:** A pure `dashboard/program_tiers.py` module produces the sellable-tier blocks (Free/Paid/Family) from ownership + enabled booleans, with prices referenced from existing constants. A new `GET /api/portal/<token>/program` endpoint composes those tiers with the existing `_ambassador_block` and a static grow-with-us list. A new `GET /portal/<token>/program` route serves `static/portal-program.html`, which fetches the API and renders. An entry card in `client-portal.html` links to it. Whole feature ships dark behind `PORTAL_PROGRAM_PAGE_ENABLED`.

**Tech Stack:** Python 3 / Flask (`app.py`), sqlite3, vanilla-JS static HTML pages, pytest. Design spec: `docs/superpowers/specs/2026-07-10-portal-membership-program-page-design.md`.

## Global Constraints

- **Prices never hardcoded.** Paid price = `dashboard/portal_offers.MEMBERSHIP_PRICE_CENTS`; Family price/value/label = `dashboard/family_plan.PLAN["amount_cents"|"value_cents"|"label"]`. No dollar literal in Python or HTML.
- **`dashboard/program_tiers.py` is pure** — it never imports `app`; it may import `dashboard.family_plan` and `dashboard.portal_offers` only. It performs no DB or env access (ownership/enabled come in as parameters). This keeps it bare-pytest testable.
- **Flag presence-checked, ships dark.** `PORTAL_PROGRAM_PAGE_ENABLED` defaults off; check presence via a truthy tuple, never print its value.
- **No dead buy buttons.** A tier whose commerce flag is off (`SUBSCRIPTIONS_ENABLED` for Paid, `FAMILY_PLAN_ENABLED` for Family) renders `state == "coming_soon"`, not an `available` buy CTA.
- **A field reaches the page only if it is in BOTH the payload dict AND rendered.**
- **Copy rules (Glen):** no em dashes, no ALL CAPS words, no "Hook:" label. Benefit copy below is first-draft; Glen tunes it.
- **Test commands:** pure `dashboard/*` tests run bare: `python3 -m pytest tests/<file> -q`. Tests that `import app` run under doppler: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -q`.
- **Ambassador uses the affiliate-slug system only** (`_ambassador_block`); do not touch `dashboard/referrals.py`.

---

## File Structure

- **Create** `dashboard/program_tiers.py` — pure tier descriptors + `program_blocks()`, `current_tier_key()`, `GROW_PATHS`.
- **Create** `tests/test_program_tiers.py` — pure unit tests (bare pytest).
- **Modify** `app.py` — add `_portal_program_page_enabled()`; add `GET /api/portal/<token>/program`; add `GET /portal/<token>/program`; add `payload["program_page"]` in `api_client_portal`.
- **Create** `static/portal-program.html` — page shell + render JS.
- **Modify** `static/client-portal.html` — entry card near the top of the stack, gated on `v.program_page.enabled`.
- **Create** `tests/test_program_page_routes.py` — endpoint + route + entry-card payload tests (doppler).

---

### Task 1: Pure tier module `dashboard/program_tiers.py`

**Files:**
- Create: `dashboard/program_tiers.py`
- Test: `tests/test_program_tiers.py`

**Interfaces:**
- Consumes: `dashboard.family_plan.PLAN` (`{"amount_cents":14700,"value_cents":19700,"label":"Family Plan"}`), `dashboard.portal_offers.MEMBERSHIP_PRICE_CENTS` (int, currently 9900).
- Produces:
  - `program_blocks(*, paid_owned: bool, family_owned: bool, paid_enabled: bool, family_enabled: bool) -> list[dict]` — ordered `[free, paid, family]`, each `{"key","name","benefits":list[str],"price_cents":int|None,"value_cents":int|None,"period":str,"cta_label":str|None,"checkout_path":str|None,"state":"owned"|"available"|"coming_soon"}`.
  - `current_tier_key(tiers: list[dict]) -> str` — highest owned key among `family` > `paid` > `free`.
  - `GROW_PATHS: list[dict]` — `[{"key","name","blurb","url"}]` for practitioner/coach/cert.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_program_tiers.py
from dashboard import program_tiers as pt
from dashboard import family_plan as fp
from dashboard import portal_offers as po


def _by_key(tiers):
    return {t["key"]: t for t in tiers}


def test_free_is_always_owned():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["free"]["state"] == "owned"
    assert t["free"]["checkout_path"] is None


def test_paid_available_when_enabled_and_not_owned():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["paid"]["state"] == "available"
    assert t["paid"]["price_cents"] == po.MEMBERSHIP_PRICE_CENTS
    assert t["paid"]["checkout_path"] == "/portal/offer/live-group/checkout"


def test_paid_coming_soon_when_flag_off():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=False, family_enabled=True))
    assert t["paid"]["state"] == "coming_soon"


def test_paid_owned_wins_over_enabled():
    t = _by_key(pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["paid"]["state"] == "owned"


def test_family_prices_are_data_sourced():
    t = _by_key(pt.program_blocks(
        paid_owned=False, family_owned=False,
        paid_enabled=True, family_enabled=True))
    assert t["family"]["price_cents"] == fp.PLAN["amount_cents"]
    assert t["family"]["value_cents"] == fp.PLAN["value_cents"]
    assert t["family"]["name"] == fp.PLAN["label"]


def test_current_tier_key_prefers_family():
    tiers = pt.program_blocks(
        paid_owned=True, family_owned=True,
        paid_enabled=True, family_enabled=True)
    assert pt.current_tier_key(tiers) == "family"
    tiers2 = pt.program_blocks(
        paid_owned=True, family_owned=False,
        paid_enabled=True, family_enabled=True)
    assert pt.current_tier_key(tiers2) == "paid"


def test_grow_paths_shape():
    keys = {g["key"] for g in pt.GROW_PATHS}
    assert keys == {"practitioner", "coach", "cert"}
    assert all(g["url"] for g in pt.GROW_PATHS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_program_tiers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.program_tiers'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/program_tiers.py
"""Sellable membership tiers for the client-portal program page.

Pure + parameter-based; never imports app, so it unit-tests in isolation.
Prices are referenced from the canonical constants (family_plan.PLAN,
portal_offers.MEMBERSHIP_PRICE_CENTS) and never hardcoded here.
"""
from dashboard import family_plan as _fp
from dashboard import portal_offers as _po


def _state(owned, enabled):
    if owned:
        return "owned"
    return "available" if enabled else "coming_soon"


def program_blocks(*, paid_owned, family_owned, paid_enabled, family_enabled):
    """The three sellable tiers, in ladder order, with per-viewer state."""
    free = {
        "key": "free",
        "name": "Free membership",
        "benefits": [
            "Your private portal with your Biofield Analysis and matched remedies",
            "Order your remedies whenever you want",
            "Referral tracking so you can share and be credited",
        ],
        "price_cents": 0,
        "value_cents": None,
        "period": "",
        "cta_label": None,
        "checkout_path": None,
        "state": "owned",
    }
    paid = {
        "key": "paid",
        "name": "Guided membership",
        "benefits": [
            "Live group coaching with Dr. Glen",
            "Your protocol re-matched as you progress",
            "Your AI ally and Terrain Restore support",
        ],
        "price_cents": _po.MEMBERSHIP_PRICE_CENTS,
        "value_cents": None,
        "period": "/mo",
        "cta_label": "Join",
        "checkout_path": "/portal/offer/live-group/checkout",
        "state": _state(paid_owned, paid_enabled),
    }
    family = {
        "key": "family",
        "name": _fp.PLAN["label"],
        "benefits": [
            "Everything in guided membership for your whole household",
            "Cover the people you care for under one plan",
            "One simple monthly price for the family",
        ],
        "price_cents": _fp.PLAN["amount_cents"],
        "value_cents": _fp.PLAN["value_cents"],
        "period": "/mo",
        "cta_label": "Add your family",
        "checkout_path": "/portal/offer/family-plan/checkout",
        "state": _state(family_owned, family_enabled),
    }
    return [free, paid, family]


def current_tier_key(tiers):
    """Highest owned tier: family > paid > free."""
    owned = {t["key"] for t in tiers if t.get("state") == "owned"}
    for key in ("family", "paid", "free"):
        if key in owned:
            return key
    return "free"


GROW_PATHS = [
    {"key": "practitioner", "name": "Become a Practitioner",
     "blurb": "Offer Biofield Analysis to your own clients at wholesale.",
     "url": "/practitioner/register"},
    {"key": "coach", "name": "Coach Training",
     "blurb": "Train as a coach and grow into the certification path.",
     "url": "/practitioner/register"},
    {"key": "cert", "name": "Certification",
     "blurb": "Earn your certification with Dr. Glen.",
     "url": "/cert"},
]
```

Note: `checkout_path` for family is `/portal/offer/family-plan/checkout`. Verify this route exists during Task 2 wiring; if the live family checkout path differs, update this constant (the design mandates reusing the existing family-plan checkout, not inventing one). Grep: `grep -n "family-plan/checkout\|family_plan.*checkout" app.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_program_tiers.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/program_tiers.py tests/test_program_tiers.py
git commit -m "feat(program): pure tier-blocks module for portal program page"
```

---

### Task 2: Program flag + `GET /api/portal/<token>/program` endpoint

**Files:**
- Modify: `app.py` (add `_portal_program_page_enabled` near `_portal_offers_enabled` at `app.py:15139`; add the endpoint near the other `/api/portal/*` routes, e.g. after `api_client_portal`)
- Test: `tests/test_program_page_routes.py`

**Interfaces:**
- Consumes: `dashboard.program_tiers.program_blocks/current_tier_key/GROW_PATHS` (Task 1); existing `_active_membership_for_email(email)->dict|None`, `_family_plan_enabled()`, `_subscriptions_enabled()`, `_client_login_enabled()`, `dashboard.portal_identity.resolve_identity`, `dashboard.family_plan.covers`/`init_family_plan_table`, `dashboard.portal_view._ambassador_block`, module globals `QUIZ_URL`, `PUBLIC_BASE_URL`, `LOG_DB`.
- Produces: JSON `{"email":str,"current_tier":str,"tiers":list,"ambassador":dict,"grow":list}`; helper `_portal_program_page_enabled()->bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_program_page_routes.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email, token):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_portal_table(cx) if hasattr(cp, "init_portal_table") else None
        cp.upsert_portal(cx, email=email, name="Test Client",
                         token=token, content={})


def test_api_program_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PORTAL_PROGRAM_PAGE_ENABLED", raising=False)
    r = c.get("/api/portal/anytoken/program")
    assert r.status_code == 404


def test_api_program_returns_tiers_for_free_client(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "1")
    _seed_portal(appmod, "free@x.com", "tok-free")
    r = c.get("/api/portal/tok-free/program")
    assert r.status_code == 200
    body = r.get_json()
    keys = {t["key"] for t in body["tiers"]}
    assert keys == {"free", "paid", "family"}
    tiers = {t["key"]: t for t in body["tiers"]}
    assert tiers["free"]["state"] == "owned"
    assert tiers["paid"]["state"] == "available"
    assert body["current_tier"] == "free"
    assert body["ambassador"]["status"] == "none"
    assert {g["key"] for g in body["grow"]} == {"practitioner", "coach", "cert"}
```

Note: match `_seed_portal` to the real `client_portal` API surfaced in `tests/test_client_portal_routes.py` (it uses a `_seed_portal(appmod, ...)` helper + `cp.upsert_portal`). Copy that file's exact seeding helper rather than guessing signatures.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py -q`
Expected: FAIL — 404 test passes trivially only if route absent returns 404; the tiers test fails (route not defined → 404 not 200).

- [ ] **Step 3: Add the flag helper**

In `app.py`, immediately after `_portal_offers_enabled()` (ends ~`app.py:15143`):

```python
def _portal_program_page_enabled() -> bool:
    """Master flag for the client-portal membership program page. Dark by default."""
    return os.environ.get("PORTAL_PROGRAM_PAGE_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")
```

- [ ] **Step 4: Add the endpoint**

In `app.py`, after the `api_client_portal` function (near `app.py:15702`):

```python
@app.route("/api/portal/<token>/program", methods=["GET"])
def api_portal_program(token):
    """Personalized membership program blocks for the program page."""
    if not _portal_program_page_enabled():
        return jsonify({"error": "not found"}), 404
    from dashboard import portal_identity as _pi
    from dashboard import family_plan as _fp
    from dashboard import portal_view as _pv
    from dashboard import program_tiers as _pt
    sess_cookie = request.cookies.get("rm_portal_session", "")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ident = _pi.resolve_identity(
            cx, token=token, session_token=sess_cookie,
            client_login_enabled=_client_login_enabled())
        if not ident:
            return jsonify({"error": "not found"}), 404
        email = ident.email
        family_owned = False
        if _family_plan_enabled():
            try:
                _fp.init_family_plan_table(cx)
                family_owned = bool(_fp.covers(cx, email))
            except Exception:
                family_owned = False
        try:
            amb = _pv._ambassador_block(cx, email, QUIZ_URL, PUBLIC_BASE_URL)
        except Exception:
            amb = {"status": "none",
                   "signup_url": f"{PUBLIC_BASE_URL.rstrip('/')}/affiliate/apply-form"}
    paid_owned = bool(_active_membership_for_email(email))
    tiers = _pt.program_blocks(
        paid_owned=paid_owned,
        family_owned=family_owned,
        paid_enabled=_subscriptions_enabled(),
        family_enabled=_family_plan_enabled(),
    )
    return jsonify({
        "email": email,
        "current_tier": _pt.current_tier_key(tiers),
        "tiers": tiers,
        "ambassador": amb,
        "grow": _pt.GROW_PATHS,
    })
```

- [ ] **Step 5: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_program_page_routes.py
git commit -m "feat(program): flag + /api/portal/<token>/program endpoint"
```

---

### Task 3: `GET /portal/<token>/program` page route + `static/portal-program.html`

**Files:**
- Modify: `app.py` (add route near `portal_analyze_page` at `app.py:16211`)
- Create: `static/portal-program.html`
- Test: `tests/test_program_page_routes.py` (add cases)

**Interfaces:**
- Consumes: `_portal_program_page_enabled()` (Task 2), `send_from_directory`, `STATIC`, `/api/portal/<token>/program` (Task 2).
- Produces: an HTML page served at `/portal/<token>/program` containing marker `id="program-root"`.

- [ ] **Step 1: Write the failing test (append to `tests/test_program_page_routes.py`)**

```python
def test_program_page_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PORTAL_PROGRAM_PAGE_ENABLED", raising=False)
    r = c.get("/portal/tok/program")
    assert r.status_code == 404


def test_program_page_served_when_flag_on(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    r = c.get("/portal/tok/program")
    assert r.status_code == 200
    assert b'id="program-root"' in r.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py -q`
Expected: FAIL — `/portal/tok/program` returns 404 even with flag on (route + file absent).

- [ ] **Step 3: Add the page route**

In `app.py`, after `portal_analyze_page` (~`app.py:16217`):

```python
@app.route("/portal/<token>/program")
def portal_program_page(token):
    """Membership program page. Dark until PORTAL_PROGRAM_PAGE_ENABLED."""
    if not _portal_program_page_enabled():
        return ("Not found", 404)
    resp = send_from_directory(STATIC, "portal-program.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Create the static page**

```html
<!-- static/portal-program.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Your membership program</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
           color: #1F5A4D; background: #f6f7f5; }
    .wrap { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
    .card { background: #fff; border-radius: 14px; padding: 20px;
            margin: 14px 0; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .badge { display: inline-block; background: #2f6f5e; color: #fff;
             border-radius: 999px; padding: 4px 12px; font-size: 13px; }
    .owned { border-left: 4px solid #2f6f5e; }
    .cta { display: inline-block; background: #d4a843; color: #1F5A4D;
           text-decoration: none; border-radius: 10px; padding: 10px 18px;
           font-weight: 600; margin-top: 10px; }
    .soon { color: #6b7a72; font-size: 13px; }
    h1 { font-size: 22px; } h2 { font-size: 18px; margin: 0 0 6px; }
    ul { padding-left: 18px; } li { margin: 4px 0; }
    a.link { color: #2f6f5e; }
  </style>
</head>
<body>
  <div class="wrap" id="program-root">
    <h1 id="hero">Your membership program</h1>
    <div id="tiers"></div>
    <div id="ambassador" class="card"></div>
    <div id="grow" class="card"></div>
    <div class="card">
      <a class="link" id="back" href="#">Back to your portal</a>
    </div>
  </div>
  <script>
    const token = location.pathname.split("/")[2];
    document.getElementById("back").href = "/portal/" + token;

    function money(cents) {
      if (cents === 0) return "Free";
      if (cents == null) return "";
      return "$" + (cents / 100).toFixed(cents % 100 ? 2 : 0);
    }

    function tierCard(t) {
      const owned = t.state === "owned";
      const bullets = (t.benefits || []).map(b => "<li>" + b + "</li>").join("");
      let action = "";
      if (owned) {
        action = '<div class="badge">You have this</div>';
      } else if (t.state === "available" && t.checkout_path) {
        const price = money(t.price_cents) + (t.period || "");
        action = '<a class="cta" href="' + t.checkout_path + '">' +
                 (t.cta_label || "Choose") + " " + price + "</a>";
      } else {
        action = '<div class="soon">Coming soon</div>';
      }
      return '<div class="card ' + (owned ? "owned" : "") + '">' +
             "<h2>" + t.name + "</h2><ul>" + bullets + "</ul>" + action + "</div>";
    }

    function ambassadorCard(a) {
      if (a.status === "enrolled") {
        return "<h2>Your ambassador links</h2>" +
          '<p>Share your link and be credited.</p>' +
          '<p><b>Your link:</b> <a class="link" href="' + a.referral_url +
          '">' + a.referral_url + "</a></p>" +
          '<p><b>Invite ambassadors:</b> <a class="link" href="' + a.recruit_url +
          '">' + a.recruit_url + "</a></p>";
      }
      if (a.status === "pending") {
        return "<h2>Ambassador</h2><p>Your application is under review.</p>";
      }
      return "<h2>Become an ambassador</h2>" +
        '<p>Share what helped you and earn as others heal.</p>' +
        '<a class="cta" href="' + a.signup_url + '">Apply</a>';
    }

    function growCard(paths) {
      const items = (paths || []).map(g =>
        '<li><a class="link" href="' + g.url + '">' + g.name + "</a> — " + g.blurb + "</li>"
      ).join("");
      return "<h2>Grow with us</h2><ul>" + items + "</ul>";
    }

    fetch("/api/portal/" + token + "/program")
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(d => {
        document.getElementById("hero").textContent =
          "Your membership program";
        document.getElementById("tiers").innerHTML =
          (d.tiers || []).map(tierCard).join("");
        document.getElementById("ambassador").innerHTML =
          ambassadorCard(d.ambassador || { status: "none", signup_url: "#" });
        document.getElementById("grow").innerHTML = growCard(d.grow);
      })
      .catch(() => {
        document.getElementById("tiers").innerHTML =
          '<div class="card">We could not load your program right now.</div>';
      });
  </script>
</body>
</html>
```

Copy note: the em-dash-looking separator in `growCard` uses ` — ` for display only; if Glen's no-em-dash rule extends to rendered UI, swap it for `: `. Confirm with Glen at review; default to `: ` if unsure.

- [ ] **Step 5: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Render-verify manually (per [[feedback_render_the_page_not_the_payload]])**

With the flag on locally, open `/portal/<a real token>/program` in headless Chrome and confirm the tier cards, ambassador state, and grow band render — not just that the JSON is correct.

- [ ] **Step 7: Commit**

```bash
git add app.py static/portal-program.html tests/test_program_page_routes.py
git commit -m "feat(program): program page route + static render"
```

---

### Task 4: Entry card in the main portal

**Files:**
- Modify: `app.py` (`api_client_portal`, add `payload["program_page"]` near the payload assembly ~`app.py:15194`)
- Modify: `static/client-portal.html` (render an entry card near the top of `load()`, ~`:648-712`)
- Test: `tests/test_program_page_routes.py` (add a payload case)

**Interfaces:**
- Consumes: `_portal_program_page_enabled()` (Task 2), the existing `api_client_portal` payload dict and `token`.
- Produces: `payload["program_page"] = {"enabled": bool, "url": str}` and a rendered entry card linking to it.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_client_portal_payload_exposes_program_page(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    _seed_portal(appmod, "pp@x.com", "tok-pp")
    r = c.get("/api/portal/tok-pp")
    assert r.status_code == 200
    body = r.get_json()
    assert body["program_page"]["enabled"] is True
    assert body["program_page"]["url"] == "/portal/tok-pp/program"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py::test_client_portal_payload_exposes_program_page -q`
Expected: FAIL — `KeyError: 'program_page'`.

- [ ] **Step 3: Add the payload block**

In `app.py`, inside `api_client_portal`, alongside the other best-effort `payload[...]` blocks (~`app.py:15194`):

```python
        try:
            payload["program_page"] = {
                "enabled": _portal_program_page_enabled(),
                "url": f"/portal/{token}/program",
            }
        except Exception:
            pass
```

- [ ] **Step 4: Render the entry card**

In `static/client-portal.html`, inside `load()` near the top of the card stack (after the hero, ~`:655`), append when enabled:

```javascript
      if (v.program_page && v.program_page.enabled) {
        html += '<div class="card program-entry">' +
          '<h2>See everything your membership unlocks</h2>' +
          '<p>Your program, family options, and ways to share and grow.</p>' +
          '<a class="cta" href="' + v.program_page.url + '">Explore your program</a>' +
          '</div>';
      }
```

Match `html +=` / card-append idiom to the surrounding code in `load()` (it builds cards into a string or appends nodes — mirror whichever the neighboring cards use; the variable may be named differently). Confirm the exact accumulation pattern before editing.

- [ ] **Step 5: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_program_page_routes.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Render-verify** the entry card appears near the top of the main portal with the flag on, and is absent with it off.

- [ ] **Step 7: Commit**

```bash
git add app.py static/client-portal.html tests/test_program_page_routes.py
git commit -m "feat(program): entry card in client portal linking to program page"
```

---

## Self-Review

**Spec coverage:**
- Route `/portal/<token>/program` + API + entry card → Tasks 2,3,4. ✓
- Personalized by tier (free/paid/family states from predicates) → Tasks 1,2. ✓
- Ambassador hub v1 = links + tools → Task 2 (compose `_ambassador_block`) + Task 3 (render). ✓
- Grow-with-us practitioner/coach/cert intro → Task 1 `GROW_PATHS` + Task 3 render. ✓
- Single source of truth `program_tiers.py`, prices from constants → Task 1 (+ Global Constraints). ✓
- Flag `PORTAL_PROGRAM_PAGE_ENABLED`, ships dark, no dead buy buttons → Tasks 2,3 + `_state` coming_soon. ✓
- Coexists with Options & Pricing card; referral-code system untouched → nothing modifies them. ✓
- Field in payload AND rendered → Task 4 does both. ✓

**Placeholder scan:** No TBD/TODO left; benefit copy is first-draft-real (Glen-tunable), not a placeholder. Two explicit "confirm at review" flags (family checkout path; em-dash in grow separator) are verification steps with a stated default, not open holes.

**Type consistency:** `program_blocks` / `current_tier_key` / `GROW_PATHS` names and the tier dict keys (`key,name,benefits,price_cents,value_cents,period,cta_label,checkout_path,state`) are used identically across Tasks 1→3. Ambassador dict keys (`status,slug,referral_url,recruit_url,signup_url`) match `_ambassador_block`'s real returns. Endpoint JSON keys (`email,current_tier,tiers,ambassador,grow`) match what `portal-program.html` reads.
