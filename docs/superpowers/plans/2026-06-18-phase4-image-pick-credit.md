# Phase 4 — Pairwise Image Pick + Order-Redeemable Credit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let viewers pick a favorite of each image pair (botanical + mechanism) on the sales page, tally the votes, and grant 1 order-redeemable point when they pick both pairs and later order that product — behind a new `SALES_PAGES_IMAGE_PICK` flag.

**Architecture:** A `sales_page_votes` SQLite table stores one upsertable pick per (session, product, kind). A pick route records the choice (keyed on the `amg_session` cookie + member email when known, with email backfilled onto the session's prior anonymous votes). Page-data surfaces the pairs to pick (or the chosen hero); the credit is granted in `_settle_order_points` (email-based, idempotent) when an ordered product has both-pair votes for the buyer.

**Tech Stack:** Python 3.11 / Flask, SQLite (`chat_log.db` via `LOG_DB`), the existing points engine (`dashboard/points.py`), vanilla static JS.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-18-phase4-image-pick-credit-design.md` (authoritative).
- **Flag:** `SALES_PAGES_IMAGE_PICK`, default OFF, `.strip().lower() in ("1","true","yes")`. Requires `SALES_PAGES_AI_IMAGES` on. Flag off → images render exactly as Phase 3 (no `pick` field, pick route 404, no credit).
- **Kinds** exactly `botanical`, `mechanism` (from `dashboard.sales_image_prompts.IMAGE_KINDS`). Variants are ints ≥1; **"neither" = variant 0** (excluded from credit + tally).
- **Credit:** 1 point = `IMAGE_PICK_REWARD_CENTS` env (default `100` = $1), via `points.credit(...,reason="image_pick", order_ref=f"imgpick_{slug}")`; granted at most once per (email, product), only at order settlement, only when both pairs picked. Email-based (order has no session).
- **Anti-gaming:** credit never granted at pick time; only at order, only via `has_entry` guard.
- **DB:** `chat_log.db` via `LOG_DB`; data layer takes an open `cx`. **No emoji** in client-facing copy.
- **Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$(mktemp -d)" ~/.venvs/deploy-chat311/bin/python -m pytest <file> -v`. Mock Supabase; importorskip playwright.

---

## File Structure

- **Create** `dashboard/sales_votes.py` — votes data layer.
- **Modify** `app.py` — flag `_SALES_IMAGE_PICK_ENABLED`; reward constant; `POST /begin/product-image-pick/<slug>`; pick block in `begin_product_page_data`; credit block in `_settle_order_points`.
- **Modify** `static/begin-product.html` — `renderImagesBody` pick UI.
- **Test** `tests/test_sales_pages_phase4.py`.

---

## Task 1: `sales_page_votes` data layer

**Files:** Create `dashboard/sales_votes.py`; Test `tests/test_sales_pages_phase4.py`

**Interfaces (produces):** `init_table(cx)`; `record_pick(cx, slug, kind, variant, session_id, email="")` (upsert on `(session_id, slug, kind)`; backfills email onto the session's `email=''` rows when email given); `get_picks(cx, slug, *, session_id="", email="")` → `{"botanical": variant|None, "mechanism": variant|None}` (match rows by session_id OR non-empty email; a real pick ≥1 or 0 for neither); `picked_both(cx, slug, *, session_id="", email="")` → bool (both kinds have a pick ≥1); `tally(cx, slug)` → `{kind: {variant: count}}` (variant 0 excluded).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase4.py
import sqlite3
from dashboard import sales_votes as sv

def _cx(): return sqlite3.connect(":memory:")

def test_record_pick_upsert_one_row_per_session_kind():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    sv.record_pick(cx, "longevity", "botanical", 2, "sessA")   # re-pick updates
    assert sv.get_picks(cx, "longevity", session_id="sessA")["botanical"] == 2
    assert sv.tally(cx, "longevity") == {"botanical": {2: 1}}   # one row, last choice

def test_picked_both_requires_real_pick_in_both():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 0, "sessA")    # neither
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is True

def test_email_backfill_enables_match_by_email():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")            # anon
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA", "a@b.co")  # identified -> backfills
    assert sv.picked_both(cx, "longevity", email="a@b.co") is True      # both now carry the email

def test_tally_excludes_neither():
    cx = _cx()
    sv.record_pick(cx, "x", "botanical", 1, "s1")
    sv.record_pick(cx, "x", "botanical", 1, "s2")
    sv.record_pick(cx, "x", "botanical", 0, "s3")   # neither
    assert sv.tally(cx, "x") == {"botanical": {1: 2}}
```

- [ ] **Step 2: Run to verify it fails** — `... -m pytest tests/test_sales_pages_phase4.py -v` → FAIL (no module).

- [ ] **Step 3: Implement**

```python
# dashboard/sales_votes.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_page_votes ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, kind TEXT, "
               "chosen_variant INTEGER, session_id TEXT, email TEXT DEFAULT '', "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', "
               "UNIQUE(session_id, product_slug, kind))")
    cx.commit()

def record_pick(cx, slug, kind, variant, session_id, email=""):
    init_table(cx); now = _now(); email = (email or "").strip().lower()
    cx.execute("INSERT INTO sales_page_votes (product_slug, kind, chosen_variant, session_id, email, created_at, updated_at) "
               "VALUES (?,?,?,?,?,?,?) ON CONFLICT(session_id, product_slug, kind) DO UPDATE SET "
               "chosen_variant=excluded.chosen_variant, "
               "email=CASE WHEN excluded.email!='' THEN excluded.email ELSE sales_page_votes.email END, "
               "updated_at=excluded.updated_at",
               (slug, kind, int(variant), session_id, email, now, now))
    if email and session_id:
        cx.execute("UPDATE sales_page_votes SET email=? WHERE session_id=? AND email=''", (email, session_id))
    cx.commit()

def _match_rows(cx, slug, session_id, email):
    init_table(cx)
    return cx.execute(
        "SELECT kind, chosen_variant FROM sales_page_votes WHERE product_slug=? AND "
        "(session_id=? OR (email!='' AND email=?))",
        (slug, session_id or "\x00", (email or "").strip().lower() or "\x00")).fetchall()

def get_picks(cx, slug, *, session_id="", email=""):
    out = {"botanical": None, "mechanism": None}
    for kind, var in _match_rows(cx, slug, session_id, email):
        if kind in out:
            out[kind] = var
    return out

def picked_both(cx, slug, *, session_id="", email=""):
    p = get_picks(cx, slug, session_id=session_id, email=email)
    return (p.get("botanical") or 0) >= 1 and (p.get("mechanism") or 0) >= 1

def tally(cx, slug):
    init_table(cx)
    rows = cx.execute("SELECT kind, chosen_variant, COUNT(*) FROM sales_page_votes "
                      "WHERE product_slug=? AND chosen_variant>=1 GROUP BY kind, chosen_variant", (slug,)).fetchall()
    out = {}
    for kind, var, n in rows:
        out.setdefault(kind, {})[var] = n
    return out
```

- [ ] **Step 4: Run** → PASS (4).
- [ ] **Step 5: Commit** — `git add dashboard/sales_votes.py tests/test_sales_pages_phase4.py && git commit -m "feat: sales_page_votes data layer (pick upsert, email backfill, tally)"`

---

## Task 2: flag + pick route

**Files:** Modify `app.py`; Test append.

**Interfaces:** `_SALES_IMAGE_PICK_ENABLED`; `_IMAGE_PICK_REWARD_CENTS`; `POST /begin/product-image-pick/<slug>`.

**Consumes:** `dashboard.sales_votes`, `dashboard.sales_image_prompts.IMAGE_KINDS`, `_get_product`, `get_authenticated_user`, `LOG_DB`.

- [ ] **Step 1: Write the failing test**

```python
import importlib

def _reload(monkeypatch, tmp_path, pick="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES", "true"); monkeypatch.setenv("SALES_PAGES_IMAGE_PICK", pick)
    import app as appmod; importlib.reload(appmod); return appmod

def test_pick_route_records_and_both(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sX")  # localhost default
    r1 = c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": 1})
    assert r1.status_code == 200 and r1.get_json()["both_picked"] is False
    r2 = c.post(f"/begin/product-image-pick/{slug}", json={"kind": "mechanism", "variant": 2})
    assert r2.get_json()["both_picked"] is True

def test_pick_route_neither_and_bad_input(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sY")
    assert c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": "neither"}).status_code == 200
    assert c.post(f"/begin/product-image-pick/{slug}", json={"kind": "bogus", "variant": 1}).status_code == 400

def test_pick_route_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, pick="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().post(f"/begin/product-image-pick/{slug}", json={"kind":"botanical","variant":1}).status_code == 404
```

- [ ] **Step 2: Run** → FAIL (route undefined).

- [ ] **Step 3: Implement**

```python
# app.py — near _SALES_AI_IMAGES_ENABLED (~L2302)
_SALES_IMAGE_PICK_ENABLED = os.environ.get("SALES_PAGES_IMAGE_PICK", "").strip().lower() in ("1", "true", "yes")
_IMAGE_PICK_REWARD_CENTS = int(os.environ.get("IMAGE_PICK_REWARD_CENTS", "100"))
```

```python
# app.py — after begin_product_image_gen
@app.route("/begin/product-image-pick/<slug>", methods=["POST"])
def begin_product_image_pick(slug):
    from dashboard import sales_image_prompts as _sip
    if not _SALES_IMAGE_PICK_ENABLED or not _get_product(slug):
        return ("", 404)
    data = request.get_json(silent=True) or {}
    kind = (data.get("kind") or "").strip()
    if kind not in _sip.IMAGE_KINDS:
        return jsonify({"ok": False}), 400
    raw = data.get("variant")
    if raw == "neither":
        variant = 0
    else:
        try:
            variant = int(raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False}), 400
        if variant < 1:
            return jsonify({"ok": False}), 400
    session_id = request.cookies.get("amg_session", "")
    au = get_authenticated_user(request)
    email = ((au or {}).get("email") or "").strip().lower() if au else ""
    from dashboard import sales_votes as _sv
    with sqlite3.connect(LOG_DB) as cx:
        _sv.record_pick(cx, slug, kind, variant, session_id, email)
        picks = _sv.get_picks(cx, slug, session_id=session_id, email=email)
        both = _sv.picked_both(cx, slug, session_id=session_id, email=email)
    return jsonify({"ok": True, "picks": picks, "both_picked": both})
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4.py && git commit -m "feat: image-pick flag + reward + pick route"`

---

## Task 3: page-data pick state

**Files:** Modify `app.py` (`begin_product_page_data`); Test append.

**Interfaces (produces):** when `_SALES_IMAGE_PICK_ENABLED` AND a kind has ≥2 ready variants, the images body gains `pick`: `{botanical: {chosen, options:[{variant,url}...]}, mechanism: {...}, both_picked}`. `chosen` is the viewer's pick (variant ≥1) or None; `options` are the variants to choose from. Flag off → no `pick` field (Phase-3 images body unchanged).

- [ ] **Step 1: Write the failing test**

```python
import sqlite3

def test_page_data_pick_state(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx:
        si.record_image(cx, slug, "botanical", 1, "botanical-1.png")
        si.record_image(cx, slug, "botanical", 2, "botanical-2.png")
        si.record_image(cx, slug, "mechanism", 1, "mechanism-1.png")
        si.record_image(cx, slug, "mechanism", 2, "mechanism-2.png")
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sP")
    body = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert "pick" in body
    assert len(body["pick"]["botanical"]["options"]) == 2
    assert body["pick"]["botanical"]["chosen"] is None
    # after a pick, chosen reflects it
    c.post(f"/begin/product-image-pick/{slug}", json={"kind": "botanical", "variant": 2})
    body2 = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert body2["pick"]["botanical"]["chosen"] == 2

def test_page_data_no_pick_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, pick="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    body = next(s for s in appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert "pick" not in body
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — in `begin_product_page_data`, AFTER the `_SALES_AI_IMAGES_ENABLED` images block, add:

```python
    if _SALES_IMAGE_PICK_ENABLED:
        import sqlite3 as _sq3
        from dashboard import sales_images as _si3, sales_votes as _sv3, sales_image_prompts as _sip3
        try:
            _sess = request.cookies.get("amg_session", "")
            _au = get_authenticated_user(request)
            _em = ((_au or {}).get("email") or "").strip().lower() if _au else ""
            with _sq3.connect(LOG_DB) as _cx3:
                _all = _si3.get_images(_cx3, slug)
                _picks = _sv3.get_picks(_cx3, slug, session_id=_sess, email=_em)
                _both = _sv3.picked_both(_cx3, slug, session_id=_sess, email=_em)
            _by_kind = {}
            for im in _all:
                _by_kind.setdefault(im["kind"], []).append(im)
            _pick = {}
            for _k in _sip3.IMAGE_KINDS:
                _opts = [{"variant": im["variant"], "url": f"/begin/product-image/{slug}/{im['filename']}"}
                         for im in sorted(_by_kind.get(_k, []), key=lambda x: x["variant"])]
                if len(_opts) >= 2:
                    _pick[_k] = {"chosen": _picks.get(_k) if (_picks.get(_k) or 0) >= 1 else None, "options": _opts}
            if _pick:
                _img_sec3 = next((s for s in sections if s["id"] == "images"), None)
                if _img_sec3 is not None and isinstance(_img_sec3["body"], dict):
                    _pick["both_picked"] = _both
                    _img_sec3["body"]["pick"] = _pick
        except Exception as _e:
            print(f"[img-pick] page-data skipped: {_e}", flush=True)
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4.py && git commit -m "feat: page-data image-pick state (pairs + chosen)"`

---

## Task 4: credit at order settlement

**Files:** Modify `app.py` (`_settle_order_points`); Test append.

**Interfaces:** in `_settle_order_points`, after the earn/redeem logic and still inside the `with sqlite3.connect(LOG_DB) as cx:` block, grant the image-pick credit for each ordered product the buyer picked both pairs for.

**Consumes:** `dashboard.sales_votes.picked_both`, `dashboard.points.credit/has_entry`, `_IMAGE_PICK_REWARD_CENTS`, `order["items"]`, `email`.

- [ ] **Step 1: Write the failing test**

```python
def test_credit_granted_once_on_order_when_both_picked(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_votes as sv, points as pts
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sv.record_pick(cx, slug, "botanical", 1, "sQ", "buyer@x.co")
        sv.record_pick(cx, slug, "mechanism", 1, "sQ", "buyer@x.co")
    order = {"email": "buyer@x.co", "items": [{"slug": slug, "name": "X"}],
             "total_cents": 0, "shipping_cents": 0, "get_cents": 0,
             "points_redeemed_cents": 0, "discount_cents": 0}
    appmod._settle_order_points(order, order_ref="INV-1")
    appmod._settle_order_points(order, order_ref="INV-1")  # idempotent re-settle
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pts.has_entry(cx, order_ref=f"imgpick_{slug}", reason="image_pick") is True

def test_no_credit_when_only_one_pair_or_other_product(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_votes as sv, points as pts
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sv.record_pick(cx, slug, "botanical", 1, "sR", "b2@x.co")  # only one pair
    appmod._settle_order_points({"email": "b2@x.co", "items": [{"slug": slug}],
        "total_cents":0,"shipping_cents":0,"get_cents":0,"points_redeemed_cents":0,"discount_cents":0},
        order_ref="INV-2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pts.has_entry(cx, order_ref=f"imgpick_{slug}", reason="image_pick") is False
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — in `_settle_order_points`, inside the existing `with sqlite3.connect(LOG_DB) as cx:` block, after the earn block, add:

```python
        # ── Phase 4: image-pick reward (1 point per product the buyer picked both pairs for) ──
        try:
            from dashboard import sales_votes as _sv4
            for _it in (order.get("items") or []):
                _islug = (_it.get("slug") or "").strip()
                if not _islug:
                    continue
                if _sv4.picked_both(cx, _islug, email=email) and not _points.has_entry(
                        cx, order_ref=f"imgpick_{_islug}", reason="image_pick"):
                    _points.credit(cx, email, value_cents=_IMAGE_PICK_REWARD_CENTS,
                                   reason="image_pick", order_ref=f"imgpick_{_islug}")
        except Exception as _ipe:
            print(f"[img-pick] credit skipped: {_ipe!r}", flush=True)
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4.py && git commit -m "feat: grant image-pick point at order settlement (email-based, idempotent)"`

---

## Task 5: frontend pick UI

**Files:** Modify `static/begin-product.html` (`renderImagesBody`); Manual verification.

**Interfaces:** consumes images body `pick` (Task 3) + the existing `state`/`images`. Uses `BASE`, `slug`.

- [ ] **Step 1: Implement** — extend `renderImagesBody(body)`:
  - If `body.pick` present: for each kind in `pick`:
    - if `pick[kind].chosen` is set → render that variant's image (find it in `options` by `variant`) as the hero (`<img class="sp-product-img">`).
    - else → render the two `options` side-by-side, each an `<img>` with a "Pick this" button below, plus a single "Neither — show me something else" text link beneath the pair. On "Pick this" → `POST BASE+'/begin/product-image-pick/'+slug` `{kind, variant}`; on "neither" → same with `variant:"neither"`. On success, replace that pair with the chosen image (or, for neither, a muted "we'll refresh these" line).
  - When `body.pick.both_picked` (or both kinds chosen) → append a muted line "Thanks — you've helped shape this, and earned a credit toward this product." (no emoji).
  - If `body.pick` absent → existing Phase-3 behavior (ready images / generating / none).
  - Add CSS: a `.sp-pick-row{display:flex;gap:10px}` + `.sp-pick-opt{flex:1}` + a `.sp-pick-btn` styled like the existing small buttons; reuse `.sp-product-img`. NO emoji.

- [ ] **Step 2: Manual verification** — locally with all four flags on + a product with 2 variants/kind: open images → see two pairs → pick one each → each pair collapses to the chosen image → "credit earned" line shows; reload → choices persist (page-data `chosen`); "neither" records and refreshes the prompt. Flag off → Phase-3 display. Full visual is a manual product-owner pass.

- [ ] **Step 3: Commit** — `git add static/begin-product.html && git commit -m "feat: image-pick UI (pairs, pick/neither, credit-earned note)"`

---

## Task 6: integration + flag default

**Files:** Modify `tests/test_sales_pages_phase4.py`

- [ ] **Step 1: Write the test**

```python
def test_flag_defaults_off(monkeypatch, tmp_path):
    monkeypatch.delenv("SALES_PAGES_IMAGE_PICK", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES", "true")
    import importlib, app as appmod; importlib.reload(appmod)
    assert appmod._SALES_IMAGE_PICK_ENABLED is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().post(f"/begin/product-image-pick/{slug}", json={"kind":"botanical","variant":1}).status_code == 404
```

- [ ] **Step 2: Run the full Phase-4 file + 1/2/3** — `... -m pytest tests/test_sales_pages_phase4.py tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass.
- [ ] **Step 3: Confirm flag default OFF in Render** — do NOT set `SALES_PAGES_IMAGE_PICK`. Ship dark.
- [ ] **Step 4: Commit** — `git add tests/test_sales_pages_phase4.py && git commit -m "test: phase-4 integration + flag-default-off"`

---

## Verification (end to end)

1. `... -m pytest tests/test_sales_pages_phase4.py tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass; full suite no new failures.
2. **Flag off (default):** images render exactly as Phase 3 (no `pick` field, pick route 404, no credit).
3. **Flag on locally:** images section shows both pairs → pick one each → pair collapses to the chosen image → "credit earned" note; choices persist on reload; "neither" records variant 0 (no credit, tally-excluded). Ordering the product after picking both pairs grants exactly 1 `image_pick` point (idempotent); only one pair / neither / different product → no credit.
