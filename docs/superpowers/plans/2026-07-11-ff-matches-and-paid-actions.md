# FF Matches + Paid Actions (Slice 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show every client an AI-generated set of Functional Formulation (FF) product matches for their scan, and let paid/family members turn the reviewed set into an unpaid, unpublished invoice with one click — without ever charging a card or stranding an invoice.

**Architecture:** Three independently shippable, flag-gated sub-slices. **3a** a pure, dependency-injected FF-match generator (scan items → ranked FF products via Pinecone `specific-formulations`, resolved to product slugs, no dosing). **3b** a `ff_match_drafts` cache table + a generate-once portal endpoint + a member-aware portal payload + the FF card in the portal UI. **3c** the paid `add-to-invoice` action (creates one unpaid/unpublished `orders` row, idempotent, gated on `family_plan.covers()`), enabled only after Glen publishes a draft through a small console review surface.

**Tech Stack:** Python 3 / Flask (`app.py`), sqlite (`chat_log.db` via `LOG_DB`), Pinecone (`specific-formulations` namespace) + the existing embedding helper, vanilla-JS portal (`static/client-portal.html`), pytest. Prod deploy is Render auto-deploy on merge to main; env flags live in **Doppler** (project `remedy-match`, config `prd`), never the Render API.

## Global Constraints

- **All new behavior is dark by default** behind `FF_MATCHES_ENABLED` (3a/3b) and stays dark until Glen flips it in Doppler. Flag-off: the portal payload is byte-identical to today and the endpoints 404. Mirror the exact reader shape of `_scan_recommendations_enabled()` (`app.py:14973`).
- **Never charge a card.** `add-to-invoice` creates an `orders` row with `pay_status='unpaid'`, `portal_published=0`, `status='proposed'`. It is NOT a Stripe Checkout session.
- **Free-tier FF matches carry NO dosing.** Names + meanings only. The generated draft never contains dosing; dosing is added by Glen at review and shown only to covered members after publish.
- **Free-tier disclose copy must not promise a review that is never coming.** Free copy: `"matched automatically from your scan"`. Paid copy: `"AI-generated, pending Dr. Glen's review"`. These two strings are load-bearing.
- **Member-aware always.** Every per-client lookup keys off `email_for_reports` (the `?member=`-repointed email), never `primary_email`. Mirror `_request_analysis_core` (`app.py:16155`, member re-point at `16173-16184`).
- **Order links are new-style only.** Every product link goes through `order_destination.destination_for(slug)` → `/begin/product/<slug>`. Never `remedymatch.com`.
- **The paid gate is the report-unlock predicate, NOT `_is_paid_member`.** `_is_paid_member` (`app.py:5288`) excludes trial members and is a *pricing* predicate. Use the same combination `_portal_biofield_unlocked` uses (`app.py:10661`): `_has_paid_biofield(email) or _active_membership_for_email(email) or (_family_plan_enabled() and family_plan.covers(cx, email))`.
- **The FF request button appears for animals too** (design decision 6). For a client whose `client_species.is_animal` is true, the endpoint returns the scan's infoceuticals (from `scan_recommendations`) as the recommendation instead of an FF set; the button is still present.
- **Generate-once per scan.** A second generation for the same `(email, scan_date)` returns the cached draft verbatim — never regenerates. This cache is also the rate-limit: a free member can only ever trigger generation once per distinct scan.

---

## File Structure

- Create `dashboard/ff_matcher.py` — pure generator (no Flask, no DB, no network of its own; deps injected).
- Create `dashboard/ff_match_drafts.py` — sqlite store for the draft cache + status.
- Modify `app.py` — flag reader, generator wiring, portal endpoint, payload key, console review endpoints, add-to-invoice endpoint.
- Modify `static/client-portal.html` — FF request button + FF matches card + (3c) add-to-invoice button.
- Create `static/console-ff-drafts.html` — minimal console review/publish surface (3c).
- Tests: `tests/test_ff_matcher.py`, `tests/test_ff_match_drafts.py`, `tests/test_ff_matches_api.py`, `tests/test_ff_matches_payload.py`, `tests/test_ff_add_to_invoice.py`.

**Test harness note (applies to every API/payload test):** `app` imports at module load and opens `LOG_DB = DATA_DIR/chat_log.db`; `DATA_DIR` is `/data` in prod config and does not exist locally. Run pytest as: `doppler run -p remedy-match -c prd -- env DATA_DIR="$SOME_WRITABLE_DIR" python3 -m pytest <file> -q`. Strip ANSI before grepping pytest output (`sed 's/\x1b\[[0-9;]*m//g'`); a skipped suite reads as green, so confirm "N passed".

---

# Slice 3a — the FF-match generator

**Files:**
- Create: `dashboard/ff_matcher.py`
- Test: `tests/test_ff_matcher.py`

**Interfaces:**
- Consumes: nothing from this repo at runtime — Pinecone query + slug resolver are injected as callables so the module is unit-testable offline.
- Produces:
  ```python
  def generate_ff_matches(
      scan_items,          # list[dict]: the scan's recommendations, each with "label" and "category"
      *,
      query_matches,       # callable(query_text:str, top_k:int) -> list[dict]; each {"id","score","metadata":{"name":...}}
      resolve_slug,        # callable(name:str) -> str|None
      destination,         # callable(slug:str) -> str   (order_destination.destination_for)
      top_k=5,
  ) -> list[dict]:
      """Each result: {"name","slug","url","meaning","score"}. No dosing key, ever.
      Drops any candidate whose name won't resolve to a slug. Dedupes by slug.
      Deterministic order: by descending score, then name. Never raises on an
      empty candidate list — returns []."""
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ff_matcher.py
from dashboard.ff_matcher import generate_ff_matches

SCAN = [
    {"label": "Adrenal / stress axis", "category": "ES"},
    {"label": "Liver detox pathway", "category": "ED"},
]

def _query(text, top_k):
    # verifies the generator built a query from the scan labels
    assert "Adrenal" in text and "Liver" in text
    return [
        {"id": "a", "score": 0.91, "metadata": {"name": "Adrenal Restore"}},
        {"id": "b", "score": 0.88, "metadata": {"name": "Liver Support"}},
        {"id": "c", "score": 0.80, "metadata": {"name": "Adrenal Restore"}},  # dup name
        {"id": "d", "score": 0.70, "metadata": {"name": "Unresolvable Thing"}},
    ][:top_k]

def _resolve(name):
    return {"Adrenal Restore": "adrenal-restore", "Liver Support": "liver-support"}.get(name)

def _dest(slug):
    return f"/begin/product/{slug}"

def test_generate_ranks_dedupes_resolves_and_carries_no_dosing():
    out = generate_ff_matches(SCAN, query_matches=_query, resolve_slug=_resolve,
                              destination=_dest, top_k=4)
    # unresolvable dropped, dup slug collapsed, ordered by score
    assert [m["slug"] for m in out] == ["adrenal-restore", "liver-support"]
    assert out[0]["url"] == "/begin/product/adrenal-restore"
    assert all("dosing" not in m for m in out)
    assert out[0]["score"] >= out[1]["score"]

def test_empty_candidates_returns_empty_not_error():
    assert generate_ff_matches(SCAN, query_matches=lambda t, k: [],
                               resolve_slug=_resolve, destination=_dest) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ff_matcher.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.ff_matcher'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/ff_matcher.py
"""Scan -> ranked Functional Formulation product matches. Pure + injected deps
(Pinecone query, slug resolver, destination) so it is unit-testable offline.
Names + meanings only — NEVER dosing; dosing is a clinical instruction added at
Glen's review."""


def _query_text(scan_items):
    parts = []
    for it in scan_items or []:
        label = (it.get("label") or "").strip()
        if label:
            parts.append(label)
    return "; ".join(parts)


def generate_ff_matches(scan_items, *, query_matches, resolve_slug, destination, top_k=5):
    text = _query_text(scan_items)
    if not text:
        return []
    try:
        candidates = query_matches(text, top_k) or []
    except Exception:
        return []
    out, seen = [], set()
    for c in candidates:
        name = ((c.get("metadata") or {}).get("name") or "").strip()
        if not name:
            continue
        slug = resolve_slug(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append({
            "name": name,
            "slug": slug,
            "url": destination(slug),
            "meaning": ((c.get("metadata") or {}).get("meaning") or "").strip(),
            "score": float(c.get("score") or 0.0),
        })
    out.sort(key=lambda m: (-m["score"], m["name"]))
    return out[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ff_matcher.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ff_matcher.py tests/test_ff_matcher.py
git commit -m "feat(ff): pure scan->FF-match generator (Slice 3a)"
```

---

# Slice 3b — drafts store, endpoint, payload, card

## Task 3b.1: `ff_match_drafts` store

**Files:**
- Create: `dashboard/ff_match_drafts.py`
- Test: `tests/test_ff_match_drafts.py`

**Interfaces:**
- Produces:
  ```python
  def init_table(cx): ...
  def get(cx, email, scan_date) -> dict | None
      # {"email","scan_date","items":[...],"status","created_at","updated_at","published_at"}
  def get_or_create(cx, email, scan_date, make_items) -> dict
      # generate-once: if a row exists, return it verbatim (make_items NOT called);
      # else call make_items() -> list, insert status='draft', return it.
  def set_items(cx, email, scan_date, items) -> None      # console edit; bumps updated_at
  def publish(cx, email, scan_date) -> bool               # status->'published', stamps published_at; False if no row
  def list_by_status(cx, status=None, limit=200) -> list  # console review surface
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ff_match_drafts.py
import sqlite3
from dashboard import ff_match_drafts as d

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    d.init_table(cx)
    return cx

def test_get_or_create_is_generate_once():
    cx = _cx()
    calls = []
    def make():
        calls.append(1)
        return [{"slug": "x", "name": "X"}]
    r1 = d.get_or_create(cx, "a@b.com", "2026-07-01", make)
    r2 = d.get_or_create(cx, "a@b.com", "2026-07-01", make)  # must NOT regenerate
    assert calls == [1]
    assert r1["items"] == r2["items"] == [{"slug": "x", "name": "X"}]
    assert r1["status"] == "draft"

def test_publish_and_status_filter():
    cx = _cx()
    d.get_or_create(cx, "a@b.com", "2026-07-01", lambda: [{"slug": "x"}])
    assert d.publish(cx, "a@b.com", "2026-07-01") is True
    assert d.get(cx, "a@b.com", "2026-07-01")["status"] == "published"
    assert d.get(cx, "a@b.com", "2026-07-01")["published_at"]
    assert [r["email"] for r in d.list_by_status(cx, "published")] == ["a@b.com"]
    assert d.list_by_status(cx, "draft") == []

def test_publish_missing_row_returns_false():
    assert d.publish(_cx(), "no@b.com", "2026-07-01") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ff_match_drafts.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/ff_match_drafts.py
"""Per-(email, scan_date) cache of generated FF matches + review status.
Generate-once: get_or_create never regenerates an existing row."""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ff_match_drafts (
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            items_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            published_at TEXT,
            PRIMARY KEY (email, scan_date)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ffmd_status ON ff_match_drafts(status)")


def _row(r):
    if r is None:
        return None
    return {"email": r["email"], "scan_date": r["scan_date"],
            "items": json.loads(r["items_json"] or "[]"), "status": r["status"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "published_at": r["published_at"]}


def get(cx, email, scan_date):
    r = cx.execute("SELECT * FROM ff_match_drafts WHERE email=? AND scan_date=?",
                   (email.lower(), scan_date)).fetchone()
    return _row(r)


def get_or_create(cx, email, scan_date, make_items):
    email = email.lower()
    existing = get(cx, email, scan_date)
    if existing is not None:
        return existing
    items = make_items() or []
    now = _now()
    cx.execute("INSERT INTO ff_match_drafts "
               "(email, scan_date, items_json, status, created_at, updated_at) "
               "VALUES (?,?,?,?,?,?)",
               (email, scan_date, json.dumps(items), "draft", now, now))
    cx.commit()
    return get(cx, email, scan_date)


def set_items(cx, email, scan_date, items):
    cx.execute("UPDATE ff_match_drafts SET items_json=?, updated_at=? WHERE email=? AND scan_date=?",
               (json.dumps(items or []), _now(), email.lower(), scan_date))
    cx.commit()


def publish(cx, email, scan_date):
    now = _now()
    cur = cx.execute("UPDATE ff_match_drafts SET status='published', published_at=?, updated_at=? "
                     "WHERE email=? AND scan_date=?", (now, now, email.lower(), scan_date))
    cx.commit()
    return cur.rowcount == 1


def list_by_status(cx, status=None, limit=200):
    if status:
        rows = cx.execute("SELECT * FROM ff_match_drafts WHERE status=? "
                          "ORDER BY updated_at DESC LIMIT ?", (status, limit)).fetchall()
    else:
        rows = cx.execute("SELECT * FROM ff_match_drafts ORDER BY updated_at DESC LIMIT ?",
                          (limit,)).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ff_match_drafts.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ff_match_drafts.py tests/test_ff_match_drafts.py
git commit -m "feat(ff): ff_match_drafts store, generate-once + publish (Slice 3b)"
```

## Task 3b.2: flag reader + generator wiring in app.py

**Files:**
- Modify: `app.py` (add near `_scan_recommendations_enabled` at `app.py:14973`)
- Test: covered by 3b.3's API test (wiring has no standalone behavior)

**Interfaces:**
- Produces:
  - `_ff_matches_enabled() -> bool` — mirrors `_scan_recommendations_enabled()`.
  - `_ff_query_specific_formulations(text, top_k) -> list[dict]` — thin adapter over the existing Pinecone `specific-formulations` query used at `app.py:12862` / `journal_blueprint.py:293`. Returns `[{"id","score","metadata":{...}}]`. On any Pinecone/embedding error, returns `[]` (never raises).
  - `_make_ff_items_for(email, scan_date) -> list[dict]` — reads that scan's `scan_recommendations` rows (via the existing `_scan_recommendations_for(email, scan_date)` at `app.py:15050`), maps them to `scan_items` (label/category), and calls `ff_matcher.generate_ff_matches(...)` injecting `_ff_query_specific_formulations`, `_resolve_buy_slug` (`app.py:5907`), and `order_destination.destination_for`.

- [ ] **Step 1: Add the flag reader** (mirror `app.py:14973` exactly, new name/env)

```python
def _ff_matches_enabled():
    return os.environ.get("FF_MATCHES_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 2: Add the Pinecone adapter.** Locate the existing `specific-formulations` query (around `app.py:12862`) and factor its embed+query into `_ff_query_specific_formulations(text, top_k)`. It must catch all exceptions and return `[]`. Do NOT duplicate the embedding client setup — reuse whatever `journal_blueprint.match_remedies` / the `12862` path already constructs.

- [ ] **Step 3: Add `_make_ff_items_for(email, scan_date)`** that composes `_scan_recommendations_for` → `generate_ff_matches`. Import `from dashboard import ff_matcher, ff_match_drafts, order_destination` at the top of app.py beside the other `dashboard` imports.

- [ ] **Step 4: Manual smoke** (no test yet — exercised in 3b.3). Confirm `import app` still succeeds:
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$D" python3 -c "import app; print(app._ff_matches_enabled())"`
Expected: prints `False` (flag unset), no import error.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(ff): flag reader + inline generator wiring (Slice 3b)"
```

## Task 3b.3: portal endpoint `POST /api/portal/<token>/ff-matches`

**Files:**
- Modify: `app.py` (add route beside `api_portal_request_analysis` at `app.py:16212`)
- Test: `tests/test_ff_matches_api.py`

**Interfaces:**
- Consumes: `_portal_record_for` (`app.py:15409`), the member re-point block (`app.py:16173-16184`), `_client_species_for` (`app.py:16008`), `ff_match_drafts.get_or_create`, `_make_ff_items_for`, `_scan_recommendations_for`.
- Produces: `POST /api/portal/<token>/ff-matches` → JSON `{"ff_matches": {...}}` (shape defined in Step 3). 404 when `_ff_matches_enabled()` is false.

- [ ] **Step 1: Write the failing test** (mirror the fixtures in `tests/test_client_species_payload.py`; monkeypatch `LOG_DB`, seed a portal token + a `scan_recommendations` row, monkeypatch `app._make_ff_items_for` to a stub so no Pinecone call happens)

```python
# tests/test_ff_matches_api.py  (abridged — full fixtures per test_client_species_payload.py)
def test_ff_matches_flag_off_404(app_env):
    app, client, token = app_env  # flag unset
    assert client.post(f"/api/portal/{token}/ff-matches").status_code == 404

def test_ff_matches_generate_once_and_cached(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    calls = []
    monkeypatch.setattr(app, "_make_ff_items_for",
                        lambda e, d: (calls.append(1) or [{"name": "X", "slug": "x", "url": "/begin/product/x", "meaning": "m", "score": 0.9}]))
    a = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    b = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    assert calls == [1]                       # generate-once
    assert a["items"] == b["items"]
    assert all("dosing" not in it for it in a["items"])
    assert a["reviewed"] is False

def test_animal_returns_infoceuticals_not_ff(app_env, monkeypatch):
    app, client, token = app_env  # token's client seeded is_animal=True in fixture
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    out = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    assert out["kind"] == "infoceutical"      # animals get the scan's infoceuticals
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$D" python3 -m pytest tests/test_ff_matches_api.py -q`
Expected: FAIL — route returns 404/405 for the enabled cases (route not defined).

- [ ] **Step 3: Write the route**

```python
@app.route("/api/portal/<token>/ff-matches", methods=["POST"])
def api_portal_ff_matches(token):
    if not _ff_matches_enabled():
        return ("", 404)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "unknown token"}), 404
        email = (portal.get("email") or "").lower()
        # member re-point (mirror app.py:16173-16184)
        member = (request.args.get("member") or "").lower()
        if member and _household_view_enabled() and _hh.can_view(email, member):
            email = member
        scan_date = _current_scan_date_for(email)   # the selected/most-recent scan
        covered = _ff_covered(cx, email)            # defined in 3c; in 3b return False
        species = _client_species_for(email)
        if species.get("is_animal"):
            items = _scan_recommendations_for(email, scan_date) or []
            return jsonify({"ff_matches": {"kind": "infoceutical", "items": items,
                                           "reviewed": False, "covered": covered,
                                           "scan_date": scan_date}})
        draft = ff_match_drafts.get_or_create(cx, email, scan_date,
                                              lambda: _make_ff_items_for(email, scan_date))
        reviewed = draft["status"] == "published"
        items = draft["items"]
        if not covered and reviewed:
            items = [{k: v for k, v in it.items() if k != "dosing"} for it in items]
        return jsonify({"ff_matches": {"kind": "ff", "items": items, "reviewed": reviewed,
                                       "covered": covered, "scan_date": scan_date}})
```

Notes for the implementer:
- `_current_scan_date_for(email)` — reuse the same selected-scan-date logic `_scan_recommendations_for` / the payload uses (see `app.py:16000`). If a dedicated helper does not exist, thread the same `req_date or None` fallback the payload uses; do not invent a new selection rule.
- In 3b, define `_ff_covered(cx, email)` as a stub returning `False` (no paid affordance yet); 3c replaces its body. This keeps 3b shippable and dosing-stripped for everyone.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$D" python3 -m pytest tests/test_ff_matches_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_ff_matches_api.py
git commit -m "feat(ff): portal ff-matches endpoint, generate-once, animal branch (Slice 3b)"
```

## Task 3b.4: portal payload key + FF card

**Files:**
- Modify: `app.py` (`api_client_portal`, beside `payload["scan_recommendations"]` at `app.py:16000-16005`)
- Modify: `static/client-portal.html` (beside the `.scanrec-card` at `1039-1060`)
- Test: `tests/test_ff_matches_payload.py`

**Interfaces:**
- Consumes: `ff_match_drafts.get`, `_ff_covered`, `_client_species_for`, `email_for_reports`.
- Produces: `payload["ff_matches"]` — present only when `_ff_matches_enabled()` AND a draft already exists for the selected scan (the GET path never generates; generation is the POST button). Flag-off or no-draft → key absent (byte-identical payload).

- [ ] **Step 1: Write the failing payload test** — assert: flag off → no `ff_matches` key; flag on + existing published draft for a covered member → `ff_matches.items` present with dosing; flag on + free member → items present, no `dosing`; `?member=` returns the member's draft not the caregiver's.

- [ ] **Step 2: Run it — FAIL** (`ff_matches` key absent).

- [ ] **Step 3: Add the payload block** (GET-side, never generates):

```python
# in api_client_portal, after the scan_recommendations block (~app.py:16005)
if _ff_matches_enabled():
    try:
        _ffd = ff_match_drafts.get(cx, email_for_reports, req_date or _latest_scan_date(email_for_reports))
        if _ffd:
            _cov = _ff_covered(cx, email_for_reports)
            _items = _ffd["items"]
            _reviewed = _ffd["status"] == "published"
            if not _cov:
                _items = [{k: v for k, v in it.items() if k != "dosing"} for it in _items]
            payload["ff_matches"] = {"items": _items, "reviewed": _reviewed, "covered": _cov}
    except Exception as e:
        app.logger.warning("ff_matches payload skipped: %r", e)   # never break the load
```

- [ ] **Step 4: Add the card + request button** in `static/client-portal.html` after the `.scanrec-card` block. The card:
  - Always renders a **"See my formulation matches"** button (present for animals too) that `POST`s `/api/portal/<token>/ff-matches` (carry `?member=` if set), then renders the returned items.
  - If `d.ff_matches` is already present on load, render the card immediately.
  - Each item: name (linked to `it.url`), meaning. **No dosing element unless `d.ff_matches.covered && it.dosing`.**
  - Disclose copy: `d.ff_matches.covered ? "AI-generated, pending Dr. Glen's review" : "matched automatically from your scan"`. Use `esc()` for all interpolated text (match the existing card's escaping).
  - No add-to-invoice button in 3b (added in 3c).

- [ ] **Step 5: Run the payload test — PASS.** Then render-verify in a headless browser against a seeded local draft (per the "render the page, not the payload" rule): confirm the free view shows names with no dosing and the free disclose copy, and a covered view shows dosing + the review copy.

- [ ] **Step 6: Commit**

```bash
git add app.py static/client-portal.html tests/test_ff_matches_payload.py
git commit -m "feat(ff): ff_matches payload + portal card, tier-aware, member-aware (Slice 3b)"
```

---

# Slice 3c — the paid add-to-invoice action + review surface

## Task 3c.1: the coverage predicate + publish/review console endpoints

**Files:**
- Modify: `app.py` (replace the `_ff_covered` stub; add console endpoints)
- Test: `tests/test_ff_add_to_invoice.py` (coverage + publish portions)

**Interfaces:**
- Produces:
  - `_ff_covered(cx, email) -> bool` — `_has_paid_biofield(email) or _active_membership_for_email(email) or (_family_plan_enabled() and family_plan.covers(cx, email))`. Fail-closed. (Mirrors `_portal_biofield_unlocked`'s inner test at `app.py:10661`, minus the paid-gate flag — coverage is about entitlement, not the blur flag.)
  - `GET /api/console/ff-match-drafts` (console-key gated, mirror `api_console_analysis_requests` at `app.py:10972`) → `{"drafts": ff_match_drafts.list_by_status(cx, request.args.get("status"))}`.
  - `POST /api/console/ff-match-drafts/publish` (console-key gated) — body `{email, scan_date, items?}`; if `items` present call `set_items` first (Glen's edits, incl. dosing), then `publish`. Returns `{"published": bool}`.

- [ ] **Step 1: Write the failing test** — `_ff_covered` true via `family_plan.covers` (seed a caregiver plan for a member) and false for a bare free email; console publish flips a draft to `published` and stores edited items incl. a `dosing` field.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the predicate (replace 3b stub) and the two console endpoints. Reuse the exact console-key check the neighbouring `/api/console/*` routes use (`X-Console-Key`).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** (`feat(ff): coverage predicate + console publish/review endpoints (Slice 3c)`).

## Task 3c.2: `add-to-invoice` endpoint (idempotent, unpaid, unpublished)

**Files:**
- Modify: `app.py` (route beside the ff-matches endpoint)
- Test: `tests/test_ff_add_to_invoice.py` (invoice portions)

**Interfaces:**
- Consumes: `_ff_covered`, `ff_match_drafts.get`, `_bos_orders.upsert_order` (the composer insert at `app.py:33614`), `order_destination`.
- Produces: `POST /api/portal/<token>/ff-matches/add-to-invoice` → `{"ok": true, "order_ref": "FFINV-..."}` | 403 (not covered) | 409 (draft not published).

- [ ] **Step 1: Write the failing test:**
  - free member → **403**, and **no** `orders` row created.
  - covered member, draft **not** published → **409**.
  - covered member, published draft → **201/200**, exactly one `orders` row with `pay_status='unpaid'`, `portal_published=0`, `status='proposed'`, `items_json` = the published items.
  - **double POST → still exactly one row** (idempotent via deterministic `external_ref`).
  - covered **via caregiver's** `family_plan.covers`, not own payment → allowed.

- [ ] **Step 2: Run — FAIL** (route missing).

- [ ] **Step 3: Implement:**

```python
@app.route("/api/portal/<token>/ff-matches/add-to-invoice", methods=["POST"])
def api_portal_ff_add_to_invoice(token):
    if not _ff_matches_enabled():
        return ("", 404)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "unknown token"}), 404
        email = (portal.get("email") or "").lower()
        member = (request.args.get("member") or "").lower()
        if member and _household_view_enabled() and _hh.can_view(email, member):
            email = member
        if not _ff_covered(cx, email):
            return jsonify({"error": "not covered"}), 403
        scan_date = _current_scan_date_for(email)
        draft = ff_match_drafts.get(cx, email, scan_date)
        if not draft or draft["status"] != "published":
            return jsonify({"error": "not published"}), 409
        ext = f"FFINV-{email}-{scan_date}"           # deterministic => idempotent via UNIQUE(source, external_ref)
        items = [{"slug": it["slug"], "name": it["name"], "qty": 1,
                  "unit_cents": 0, "line_cents": 0} for it in draft["items"]]
        _bos_orders.upsert_order(cx, source="in-house", external_ref=ext, status="proposed",
                                 email=email, name=(portal.get("name") or ""),
                                 items=items, total_cents=0, channel="ff-invoice",
                                 invoice_note="FF matches — pending Rae invoicing")
        cx.commit()
        return jsonify({"ok": True, "order_ref": ext})
```

Notes:
- Confirm the exact `upsert_order` kwarg names against `dashboard/orders.py:101` and the composer call at `app.py:33614`; the explore found `pay_status`/`portal_published` are NOT kwargs — they fall to the column defaults (`'unpaid'` / `0`), which is exactly what we want. Do not pass them.
- `unit_cents=0`: pricing is set by Rae in the composer at review time. The row exists to enter the review/invoice queue, not to price. If the composer requires non-zero prices to list the order, price each line via the existing FF client-price lookup (`client_prices.__all_ff__`) instead — implementer confirms which the composer needs and matches it. Prefer the real FF price if trivially available.

- [ ] **Step 4: Run — PASS** (all five cases).

- [ ] **Step 5: Add the add-to-invoice button** in `static/client-portal.html`: visible only when `d.ff_matches.covered && d.ff_matches.reviewed`. Posts the endpoint; on success shows "Added to your next invoice — Rae will finalize it." Disable-after-click to avoid a double-submit (belt-and-suspenders on top of the idempotent server).

- [ ] **Step 6: Commit** (`feat(ff): paid add-to-invoice, idempotent unpaid order (Slice 3c)`).

## Task 3c.3: minimal console review surface

**Files:**
- Create: `static/console-ff-drafts.html`
- Modify: `app.py` (a `GET /console/ff-drafts` static-serve route, mirror an existing console page route)

- [ ] **Step 1:** Serve `static/console-ff-drafts.html` at `/console/ff-drafts` (mirror an existing `console-*.html` serve route). Page lists drafts from `GET /api/console/ff-match-drafts?status=draft`, lets Glen edit item meanings/dosing inline, and `POST`s `/api/console/ff-match-drafts/publish`. Console-key entry mirrors the other console pages.
- [ ] **Step 2:** Render-verify the page loads and lists a seeded draft; publish flips it and it drops off the `draft` filter.
- [ ] **Step 3: Commit** (`feat(ff): console review-and-publish surface for FF drafts (Slice 3c)`).

---

## Rollout (operator, not code)

1. Merge 3a → 3b → 3c PRs in order.
2. Flip `FF_MATCHES_ENABLED=1` in **Doppler** (project `remedy-match`, config `prd`) — never the Render API (deploy resync prunes Render-only vars). Verify the portal card renders for a test scan and that a free view shows no dosing.
3. `add-to-invoice` stays inert until a draft is published through `/console/ff-drafts`, so 3c is safe to ship dark-then-lit independently.

## Non-goals (this plan)

- Upgrading the generator to the evidence-ranked `e4l-scan-remedy-matcher` agent (option B) — deferred; the injected `query_matches` seam makes it a drop-in later.
- A human "cannot take herbal remedies" flag — only `is_animal` drives the infoceutical branch today. Deferred to a tiny follow-up.
- The monthly family shipment, member discounts, group coaching — the Family Plan's other benefits, off the same `covers()` predicate, out of scope here.
- A separate monthly quota table — the generate-once cache is the rate-limit (one generation per distinct scan). Revisit only if generation cost proves material.

## Self-review checklist (done while writing)

- Spec coverage: FF matches visible to all ✓ (3b); no dosing free ✓ (global constraint + 3b.3/3b.4); disclose copy both tiers ✓ (3b.4); add-to-invoice paid-only, after publish, idempotent, unpaid ✓ (3c.2); animals get infoceuticals ✓ (3b.3); flag-gated dark ✓ (global); member-aware ✓ (global).
- Corrected spec assumptions: no `analysis_requests` rail (own `ff_match_drafts` + console surface); paid gate is the unlock predicate not `_is_paid_member`; `upsert_order` uses column defaults for `pay_status`/`portal_published`; generator must be built (3a).
- Type consistency: `generate_ff_matches` returns `{name,slug,url,meaning,score}`; drafts store `items` round-trips that shape; endpoint/payload strip only `dosing`; `_ff_covered(cx, email)` signature identical in 3b stub and 3c real.
