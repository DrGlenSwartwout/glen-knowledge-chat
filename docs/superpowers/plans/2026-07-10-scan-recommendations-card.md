# Scan-recommendations card — Implementation Plan (Slice 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show every client the infoceuticals their own E4L scan matched, free, with a working order link — and show the miHealth cycles alongside them, clearly not purchasable.

**Architecture:** Slice 1 put 5,914 rows into prod's `scan_recommendations`, write-only. This slice adds three read helpers, a `payload["scan_recommendations"]` block honouring `?member=`, a card in `client-portal.html`, and a one-line destination resolver. It also adds the console GET that finally lets us confirm what production stored.

**Tech Stack:** Python 3, Flask, sqlite3, vanilla JS, pytest, headless Chrome for render-verification.

## Global Constraints

- Flag `SCAN_RECOMMENDATIONS_ENABLED`, default OFF. Flag off → `payload` is byte-identical to today and no card renders.
- **Every order link is `/begin/product/<slug>`. Never `remedymatch.com`.** The catalog's `url` field is the OLD GrooveKart page; the new-style in-funnel page exists for all 966 sellable products.
- **Only Infoceuticals get an order button.** `ER`/`MR` are miHealth device cycles, not products — zero of them resolve to a slug, and that is correct. Render them as a secondary, informational list with no button.
- `section` values are exactly `"Infoceuticals"` and `"miHealth Functions"`.
- `?member=` re-points the card at the member's scan, exactly as the report does. A member's card must never show the caregiver's scan.
- This slice sells nothing new and sends no email. It renders stored rows and links to existing product pages.
- Best-effort: a failure building this block must never break the portal load (same pattern as `payload["invoices"]`).

## Facts measured against production (2026-07-10)

- `scan_recommendations` holds **5,914 rows / 570 scans / 162 clients**, pushed idempotently.
- **70 distinct infoceutical item codes.** 69 resolve to a live product via `_resolve_remedy_slug({"name": <bare code>})`, because the catalog's storefront twin carries the bare code as its `pinecone_title` (`es1-lymph` → `"ES1"`). Examples: `ED6 → ed6-heart-driver` ($39.97), `ES7 → es7-muscle`, `MB1 → mb1-brain-stem-hologram`, `EI8 → ei8-microbes-liver-integrator`.
- **`BFA` is the one failure**, and it is rank 1 on **161 scans**. Both its records carry long `pinecone_title`s, so the bare code matches neither.
- **Zero `ER`/`MR` codes resolve.** Correct — nothing to buy.
- Of the two BFA records: `bfa-big-field-aligner` has the storefront `url` but **no `bottle_type`** (even after #762); `bfa-big-field-aligner-infoceutical` has `bottle_type: "30ml"`, the dosing `description`, `fmp_id: 198` — and is the slug already stored in a live client's `reorder_items`.
- No product currently uses an `aliases` key (0 of 978).

## A design decision, stated up front

**The card keys off the E4L scan, not the published report.** `payload["scan_date"]` is a *published report* date, and those can be phantom: at least one live report is filed under a date on which that client has no scan at all. Keying the card on the report date would show them nothing. So:

> The card shows the **latest scan_date present in `scan_recommendations`** for `email_for_reports`, unless `?scan_date=` names one we actually hold.

## File Structure

| file | responsibility |
|---|---|
| `dashboard/scan_recommendations.py` (modify) | add `scan_dates_for`, `for_scan_date`, `split_by_section` |
| `dashboard/order_destination.py` (create) | `destination_for(slug) -> str`. One function, one rule. |
| `data/products.json` (modify) | `"aliases": ["BFA"]` on `bfa-big-field-aligner-infoceutical` |
| `app.py` (modify) | index `aliases` into `_TITLE_TO_SLUG`; `_scan_recommendations_for()`; `payload["scan_recommendations"]`; console GET |
| `static/client-portal.html` (modify) | the card |
| `tests/test_order_destination.py` (create) | resolver unit tests |
| `tests/test_scan_recommendations_read.py` (create) | store read helpers + BFA alias |
| `tests/test_scan_recommendations_payload.py` (create) | payload, flag-off, `?member=` |

---

### Task 1: the destination resolver

Small on purpose. It exists so the "new-style page, always" rule has exactly one enforcement point, and so a future route change touches one line.

**Files:** Create `dashboard/order_destination.py`; Test `tests/test_order_destination.py`

**Interfaces:**
- Produces: `destination_for(slug: str) -> str` — `/begin/product/<slug>`; `""` for a blank/None slug.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_order_destination.py
"""Every order link points at the NEW-STYLE in-funnel product page.

`products.json`'s `url` field is the OLD GrooveKart page and is absent on 669 of 966
sellable products. `/begin/product/<slug>` renders from catalog data for all of them,
keeps the client inside the funnel where the upgrade CTA lives, and honours their
courtesy pricing. So the rule is: never link to remedymatch.com.
"""
import json
from pathlib import Path

from dashboard.order_destination import destination_for


def test_a_slug_becomes_a_new_style_product_page():
    assert destination_for("ed6-heart-driver") == "/begin/product/ed6-heart-driver"


def test_a_blank_slug_yields_no_link():
    assert destination_for("") == ""
    assert destination_for(None) == ""


def test_the_destination_is_never_the_old_storefront():
    for slug in ("es1-lymph", "bfa-big-field-aligner-infoceutical", "mb1-brain-stem-hologram"):
        assert "remedymatch.com" not in destination_for(slug)


def test_it_works_for_a_product_that_has_no_storefront_url():
    """bfa-big-field-aligner-infoceutical is `no_groovekart` with no `url` — and still
    has a new-style page. This is exactly why we do not read the `url` field."""
    p = json.loads((Path(__file__).resolve().parent.parent / "data" / "products.json").read_text())["products"]
    rec = p["bfa-big-field-aligner-infoceutical"]
    assert not rec.get("url")
    assert destination_for("bfa-big-field-aligner-infoceutical") == "/begin/product/bfa-big-field-aligner-infoceutical"
```

- [ ] **Step 2: Run and watch them fail**

`doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_order_destination.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name 'order_destination'`

- [ ] **Step 3: Implement**

```python
# dashboard/order_destination.py
"""Where an order button points.

ONE rule: the new-style in-funnel page, `/begin/product/<slug>`, for every product.

Never `products.json`'s `url` — that is the OLD GrooveKart storefront page, absent on
669 of 966 sellable products, and it drops the client out of the funnel, out of their
courtesy pricing, and onto a page that does not stock two-thirds of the catalog.

A named seam rather than an inline f-string: it is unit-testable, the rule has one
enforcement point, and a future route change touches one line.
"""


def destination_for(slug):
    slug = (slug or "").strip()
    return f"/begin/product/{slug}" if slug else ""
```

- [ ] **Step 4: Green.** Expected `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/order_destination.py tests/test_order_destination.py
git commit -m "feat(portal): destination_for() — order links point at the new-style product page"
```

---

### Task 2: `BFA` must resolve to a slug

**Files:** Modify `data/products.json`, `app.py` (`_TITLE_TO_SLUG`); Test `tests/test_scan_recommendations_read.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_resolve_remedy_slug({"name": "BFA"})` → `"bfa-big-field-aligner-infoceutical"`.

**Why this record and not the other.** Two active BFA twins. `bfa-big-field-aligner` carries the storefront `url` but has **no `bottle_type`**, so an order of it resolves the packer's `"default"` bottle — a phantom bottle that poisons the shipping quote. `bfa-big-field-aligner-infoceutical` carries `bottle_type: "30ml"`, the dosing `description`, and is already the slug in a live client's `reorder_items`. Order links target `/begin/product/<slug>`, never the storefront, so the `url` buys us nothing.

**Neither twin is retired.** Retiring the FMP twin would strip `bottle_type` from the survivor. Deduplicating the pair is a separate cleanup — see Non-goals.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_recommendations_read.py  (part 1 of 2 — read helpers land in Task 3)
"""BFA is rank 1 on 161 scans and resolves to nothing.

69 of 70 infoceutical codes resolve because the catalog's storefront twin carries the
bare code as its `pinecone_title` (es1-lymph -> "ES1"). Both BFA records carry long
titles, so the bare code matches neither. A new `aliases` list fixes it without
touching `pinecone_title`, which would orphan the product's Pinecone vector.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BFA_SLUG = "bfa-big-field-aligner-infoceutical"


def _app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def test_the_bfa_record_carries_the_bare_code_as_an_alias():
    assert _products()[BFA_SLUG]["aliases"] == ["BFA"]


def test_the_aliased_record_is_the_one_with_a_bottle_type():
    """The other BFA twin has no bottle_type; ordering it resolves the packer's
    'default' bottle, which poisons the shipping quote."""
    rec = _products()[BFA_SLUG]
    assert rec["bottle_type"] == "30ml"
    assert rec.get("description")


def test_pinecone_title_is_untouched():
    assert _products()[BFA_SLUG]["pinecone_title"] == "BFA Big Field Aligner Infoceutical"


def test_the_bare_code_bfa_now_resolves_to_a_live_product():
    app = _app()
    slug = app._resolve_remedy_slug({"name": "BFA"})
    assert slug == BFA_SLUG
    assert app._get_product(slug)


def test_the_other_infoceutical_codes_still_resolve():
    app = _app()
    for code, expected in (("ED6", "ed6-heart-driver"), ("ES7", "es7-muscle"),
                           ("ES1", "es1-lymph"), ("MB1", "mb1-brain-stem-hologram")):
        assert app._resolve_remedy_slug({"name": code}) == expected


def test_mihealth_codes_still_resolve_to_nothing():
    """ER/MR are device cycles, not products. Resolving them would be the bug."""
    app = _app()
    for code in ("ER2", "ER18", "MR4", "MR6"):
        assert not app._resolve_remedy_slug({"name": code})


def test_an_alias_never_shadows_a_real_product_name():
    """A collision would silently hand one product's code to another."""
    p = _products()
    names = {(r.get("pinecone_title") or r.get("name") or "").strip().lower() for r in p.values()}
    for slug, rec in p.items():
        for a in rec.get("aliases") or []:
            assert a.strip().lower() not in names, f"{slug} alias {a!r} collides with a product title"
```

- [ ] **Step 2: Run and watch them fail**

Expected: `KeyError: 'aliases'`, and `test_the_bare_code_bfa_now_resolves_to_a_live_product` fails with `assert None == 'bfa-big-field-aligner-infoceutical'`.

- [ ] **Step 3: Add the alias to `data/products.json`**

On `bfa-big-field-aligner-infoceutical`, add one key. Change nothing else:

```json
      "bottle_type": "30ml",
      "aliases": ["BFA"]
```

- [ ] **Step 4: Index aliases in `_TITLE_TO_SLUG`**

`app.py`, replacing the dict comprehension near `_TITLE_TO_SLUG`:

```python
# pinecone_title -> catalog slug (deterministic in-catalog resolution; avoids the
# "Stress Release" vs "Emotional Stress Release" false match).
# `aliases` adds the spellings a record answers to but is not named after — an E4L scan
# says "BFA", the catalog says "BFA Big Field Aligner Infoceutical". Aliases are indexed
# LAST with setdefault, so they can never shadow a real product's title or name.
_TITLE_TO_SLUG = {}
for _s, _p in (_PRODUCTS.get("products") or {}).items():
    _TITLE_TO_SLUG.setdefault((_p.get("pinecone_title") or _p.get("name")), _s)
for _s, _p in (_PRODUCTS.get("products") or {}).items():
    for _a in (_p.get("aliases") or []):
        _TITLE_TO_SLUG.setdefault(_a, _s)
```

- [ ] **Step 5: Green + the whole-catalog check**

```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_scan_recommendations_read.py -q -p no:cacheprovider
```
Expected `7 passed`. Then prove no other code's resolution changed:

```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python - <<'PY'
import sqlite3, sys; sys.path.insert(0, ".")
import app
e = sqlite3.connect("file:/Users/remedymatch/AI-Training/e4l.db?mode=ro", uri=True)
codes = [r[0] for r in e.execute("SELECT DISTINCT item_code FROM e4l_scan_results WHERE section_context='Infoceuticals'")]
bad = [c for c in codes if not app._get_product(app._resolve_remedy_slug({"name": c}) or "")]
print("infoceutical codes:", len(codes), "| unresolved:", bad)
PY
```
Expected: `infoceutical codes: 70 | unresolved: []`

- [ ] **Step 6: Commit**

```bash
git add data/products.json app.py tests/test_scan_recommendations_read.py
git commit -m "fix(catalog): BFA answers to its bare code, so its order link resolves"
```

---

### Task 3: store read helpers + the console read path

**Files:** Modify `dashboard/scan_recommendations.py`, `app.py`; Test `tests/test_scan_recommendations_read.py` (append)

**Interfaces:**
- Consumes: `init_table`, `replace_scan` (Slice 1).
- Produces:
  - `scan_dates_for(cx, email) -> list[str]` — descending, newest first.
  - `for_scan_date(cx, email, scan_date) -> list[dict]` — ordered by `priority_rank`.
  - `split_by_section(rows) -> tuple[list, list]` — `(infoceuticals, mihealth)`, order preserved.
  - `GET /api/console/scan-recommendations` → `{"ok", "total_rows", "clients", "scans"}` when no `email` is given.
    With `?email=` (and optional `&scan_date=`) it adds `{"email", "scan_dates", "scan_date", "infoceuticals": [...], "mihealth": [...]}`.
    Console-key gated. **This is the read path that lets us confirm what Slice 1 actually stored** — and the corpus-wide
    check needs no client's email, so verification never has to handle client data.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_scan_recommendations_read.py`)

```python
import sqlite3

from dashboard import scan_recommendations as sr

EMAIL = "caregiver@example.com"
ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
    {"item_code": "ER2", "priority_rank": 3, "protocol_days": 2,
     "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
]


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    sr.init_table(con)
    sr.replace_scan(con, EMAIL, "10", "2026-07-02", ITEMS)
    sr.replace_scan(con, EMAIL, "20", "2026-06-13", ITEMS[:1])
    yield con
    con.close()


def test_scan_dates_are_newest_first(cx):
    assert sr.scan_dates_for(cx, EMAIL) == ["2026-07-02", "2026-06-13"]


def test_for_scan_date_returns_that_scan_in_rank_order(cx):
    got = [r["item_code"] for r in sr.for_scan_date(cx, EMAIL, "2026-07-02")]
    assert got == ["BFA", "ED6", "ER2"]


def test_for_an_unknown_date_returns_nothing(cx):
    assert sr.for_scan_date(cx, EMAIL, "1999-01-01") == []


def test_for_an_unknown_email_returns_nothing(cx):
    assert sr.scan_dates_for(cx, "stranger@example.com") == []


def test_split_by_section_preserves_rank_order(cx):
    info, mih = sr.split_by_section(sr.for_scan_date(cx, EMAIL, "2026-07-02"))
    assert [r["item_code"] for r in info] == ["BFA", "ED6"]
    assert [r["item_code"] for r in mih] == ["ER2"]


def test_split_by_section_on_an_empty_list(cx):
    assert sr.split_by_section([]) == ([], [])
```

Plus endpoint tests (same file, using the `tmp_db` fixture and a `test_client`, `X-Console-Key: testkey`):

```python
def test_console_read_requires_the_key(client):
    assert client.get("/api/console/scan-recommendations").status_code == 401


def test_console_read_without_an_email_returns_corpus_totals(client, tmp_db):
    body = client.get("/api/console/scan-recommendations", headers=HDRS).get_json()
    assert body["ok"] is True
    assert set(body) == {"ok", "total_rows", "clients", "scans"}   # no client data leaked
    assert body["total_rows"] == 4 and body["clients"] == 1 and body["scans"] == 2


def test_console_read_with_an_email_adds_that_clients_scan(client):
    body = client.get(f"/api/console/scan-recommendations?email={EMAIL}", headers=HDRS).get_json()
    assert body["scan_date"] == "2026-07-02"
    assert [i["item_code"] for i in body["infoceuticals"]] == ["BFA", "ED6"]
    assert [m["item_code"] for m in body["mihealth"]] == ["ER2"]
```

- [ ] **Step 2: Watch them fail** (`AttributeError: ... 'scan_dates_for'`)

- [ ] **Step 3: Implement the helpers**

```python
# append to dashboard/scan_recommendations.py

def scan_dates_for(cx, email):
    """This client's E4L scan dates, newest first. NOTE: these are SCAN dates, not
    published-report dates — a report can be filed under a date on which the client has
    no scan, so the card must key off these."""
    rows = cx.execute(
        "SELECT DISTINCT scan_date FROM scan_recommendations WHERE email=? AND scan_date<>'' "
        "ORDER BY scan_date DESC", (_norm(email),)).fetchall()
    return [r[0] for r in rows]


def for_scan_date(cx, email, scan_date):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_date=? ORDER BY priority_rank",
        (_norm(email), (scan_date or "").strip())).fetchall()
    return [dict(r) for r in rows]


def split_by_section(rows):
    """(infoceuticals, mihealth), rank order preserved. ER/MR are miHealth device cycles,
    not products — they are shown but never carry an order button."""
    info = [r for r in rows if r.get("section") == SECTION_INFOCEUTICAL]
    mih = [r for r in rows if r.get("section") != SECTION_INFOCEUTICAL]
    return info, mih
```

- [ ] **Step 4: Add the console GET** (`app.py`, beside `api_console_scan_recommendations_sync`)

```python
@app.route("/api/console/scan-recommendations", methods=["GET"])
def api_console_scan_recommendations_read():
    """Owner: read back what the pusher stored. Slice 1 shipped write-only, so this is
    the first way to confirm production's row count and spot-check a scan. Read-only."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import scan_recommendations as _sr
    email = (request.args.get("email") or "").strip().lower()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _sr.init_table(cx)
        out = {"ok": True,
               "total_rows": cx.execute("SELECT COUNT(*) FROM scan_recommendations").fetchone()[0],
               "clients": cx.execute("SELECT COUNT(DISTINCT email) FROM scan_recommendations").fetchone()[0],
               "scans": cx.execute("SELECT COUNT(DISTINCT email || '|' || scan_id) FROM scan_recommendations").fetchone()[0]}
        # No email -> corpus totals only. The backfill check needs no client data.
        if not email:
            return jsonify(out)
        dates = _sr.scan_dates_for(cx, email)
        picked = (request.args.get("scan_date") or "").strip() or (dates[0] if dates else "")
        info, mih = _sr.split_by_section(_sr.for_scan_date(cx, email, picked) if picked else [])
    out.update({"email": email, "scan_dates": dates, "scan_date": picked,
                "infoceuticals": info, "mihealth": mih})
    return jsonify(out)
```

- [ ] **Step 5: Green.** Expected `16 passed` in that file.

- [ ] **Step 6: Commit**

```bash
git add dashboard/scan_recommendations.py app.py tests/test_scan_recommendations_read.py
git commit -m "feat(e4l): scan_recommendations read helpers + console read path"
```

---

### Task 4: the portal payload

**Files:** Modify `app.py`; Test `tests/test_scan_recommendations_payload.py`

**Interfaces:**
- Consumes: `scan_dates_for`, `for_scan_date`, `split_by_section` (Task 3); `destination_for` (Task 1); `_resolve_remedy_slug` (Task 2).
- Produces: `payload["scan_recommendations"]`:

```json
{"scan_date": "2026-07-02", "scan_dates": ["2026-07-02", "2026-06-13"],
 "infoceuticals": [{"code": "BFA", "label": "Big Field Aligner (BFA)", "rank": 1,
                    "protocol_days": 15, "order_url": "/begin/product/bfa-..."}],
 "mihealth": [{"code": "ER2", "label": "Large Intestine", "rank": 6, "protocol_days": 2}]}
```

`mihealth` entries carry **no** `order_url`. An infoceutical whose code does not resolve carries `order_url: ""` rather than a dead link.

**Label rule.** For `BFA`, render `"Big Field Aligner (BFA)"` — Glen's rule: the label says BFA, and "aligner" both expands the acronym and describes the benefit. Every other code renders `"<CODE> <label>"`, e.g. `"ED6 Heart"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_recommendations_payload.py
"""The card's payload. Flag-gated, member-aware, best-effort.

Keyed on the E4L SCAN date, not the published-report date — a live client's report is
filed under a date on which she has no scan, and keying on it would show her nothing.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import scan_recommendations as sr
from dashboard import household as hh

CARE = "caregiver@example.com"
PET = "pet@example.com"

ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
    {"item_code": "ER2", "priority_rank": 3, "protocol_days": 2,
     "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
]


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        sr.init_table(cx)
        hh.init_household_tables(cx)
        sr.replace_scan(cx, CARE, "10", "2026-07-02", ITEMS)
        sr.replace_scan(cx, CARE, "20", "2026-06-13", ITEMS[:1])
        sr.replace_scan(cx, PET, "30", "2026-07-05", ITEMS[:2])
    return app


def test_flag_off_returns_nothing(app_db, monkeypatch):
    monkeypatch.delenv("SCAN_RECOMMENDATIONS_ENABLED", raising=False)
    assert app_db._scan_recommendations_for(CARE) is None


def test_flag_on_returns_the_latest_scan(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    block = app_db._scan_recommendations_for(CARE)
    assert block["scan_date"] == "2026-07-02"
    assert block["scan_dates"] == ["2026-07-02", "2026-06-13"]


def test_infoceuticals_and_mihealth_are_separated(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE)
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA", "ED6"]
    assert [m["code"] for m in b["mihealth"]] == ["ER2"]


def test_mihealth_rows_carry_no_order_url(app_db, monkeypatch):
    """ER/MR are device cycles. A dead order button is worse than no button."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    for m in app_db._scan_recommendations_for(CARE)["mihealth"]:
        assert "order_url" not in m


def test_every_infoceutical_has_a_working_order_url(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    for i in app_db._scan_recommendations_for(CARE)["infoceuticals"]:
        assert i["order_url"].startswith("/begin/product/")
        assert "remedymatch.com" not in i["order_url"]


def test_bfa_renders_glens_label(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    bfa = app_db._scan_recommendations_for(CARE)["infoceuticals"][0]
    assert bfa["label"] == "Big Field Aligner (BFA)"
    assert bfa["rank"] == 1


def test_other_codes_render_code_then_label(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    ed6 = app_db._scan_recommendations_for(CARE)["infoceuticals"][1]
    assert ed6["label"] == "ED6 Heart"


def test_an_explicit_scan_date_wins(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE, scan_date="2026-06-13")
    assert b["scan_date"] == "2026-06-13"
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA"]


def test_an_unknown_scan_date_falls_back_to_the_latest(app_db, monkeypatch):
    """A published-report date can name a day on which the client has no scan."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE, scan_date="2026-07-07")
    assert b["scan_date"] == "2026-07-02"


def test_a_client_with_no_scans_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    assert app_db._scan_recommendations_for("stranger@example.com") is None


def test_a_member_sees_their_own_scan_not_the_caregivers(app_db, monkeypatch):
    """?member= re-points email_for_reports; the card must follow it."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(PET)
    assert b["scan_date"] == "2026-07-05"
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA", "ED6"]
    assert b["mihealth"] == []


def test_a_broken_lookup_never_breaks_the_portal(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    monkeypatch.setattr(sr, "scan_dates_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    assert app_db._scan_recommendations_for(CARE) is None
```

- [ ] **Step 2: Watch them fail** (`AttributeError: ... '_scan_recommendations_for'`)

- [ ] **Step 3: Implement**

`app.py`, beside `_portal_options_for`:

```python
def _scan_recommendations_enabled():
    """The scan-recommendations card. Default OFF — when off the portal payload never
    gains the key, so responses are byte-identical to pre-card behavior."""
    return (os.environ.get("SCAN_RECOMMENDATIONS_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _scan_rec_label(code, label):
    """Glen's rule for BFA: the label says BFA, and "aligner" both expands the acronym
    and describes the benefit. Every other code reads "<CODE> <label>"."""
    code, label = (code or "").strip(), (label or "").strip()
    if code == "BFA":
        return "Big Field Aligner (BFA)"
    return f"{code} {label}".strip()


def _scan_recommendations_for(email, scan_date=None):
    """The client's own E4L scan recommendations. None when off, unknown, or broken.

    Keyed on the E4L SCAN date, never the published-report date: a report can be filed
    under a date on which the client has no scan, and keying on that shows them nothing.
    An unknown scan_date falls back to the latest.

    miHealth rows (ER/MR) carry NO order_url — they are device cycles your practitioner
    runs, not products. Zero of them resolve to a slug, which is correct.
    """
    if not _scan_recommendations_enabled() or not email:
        return None
    try:
        from dashboard import scan_recommendations as _sr
        from dashboard.order_destination import destination_for
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _sr.init_table(cx)
            dates = _sr.scan_dates_for(cx, email)
            if not dates:
                return None
            picked = scan_date if (scan_date and scan_date in dates) else dates[0]
            info, mih = _sr.split_by_section(_sr.for_scan_date(cx, email, picked))
        out_info = []
        for r in info:
            slug = _resolve_remedy_slug({"name": r["item_code"]}) or ""
            if slug and not _get_product(slug):
                slug = ""
            out_info.append({"code": r["item_code"],
                             "label": _scan_rec_label(r["item_code"], r["label"]),
                             "rank": r["priority_rank"], "protocol_days": r["protocol_days"],
                             "order_url": destination_for(slug)})
        out_mih = [{"code": r["item_code"], "label": (r["label"] or r["item_code"]),
                    "rank": r["priority_rank"], "protocol_days": r["protocol_days"]}
                   for r in mih]
        return {"scan_date": picked, "scan_dates": dates,
                "infoceuticals": out_info, "mihealth": out_mih}
    except Exception as _e:
        print(f"[scan-recs] {_e!r}", flush=True)
        return None
```

Then wire it into `api_client_portal`, immediately before `return jsonify(payload)`, using the **member-aware** `email_for_reports`:

```python
    # Scan-recommendations card (flag-gated, best-effort). email_for_reports is already
    # re-pointed by ?member=, so a member's card shows THEIR scan, not the caregiver's.
    try:
        _sr_block = _scan_recommendations_for(email_for_reports, req_date or None)
        if _sr_block:
            payload["scan_recommendations"] = _sr_block
    except Exception as _e:
        print(f"[scan-recs/payload] {_e!r}", flush=True)
```

- [ ] **Step 4: Green.** Expected `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_scan_recommendations_payload.py
git commit -m "feat(portal): scan_recommendations payload — member-aware, flag-gated"
```

---

### Task 5: the card

**Files:** Modify `static/client-portal.html`

No unit test — this is markup. It is **render-verified** in a headless browser against a local page proxied to prod's API. That technique is what caught the household-switcher bug, where the API payload was correct and the rendered page showed the caregiver's report under the member's name.

- [ ] **Step 1: Add the card**

In `static/client-portal.html`, after the biofield report card and before "History & receipts":

```javascript
  // Your scan's own matches, straight from E4L. Free, and orderable.
  // Infoceuticals get an order link to the new-style product page. miHealth cycles
  // (ER/MR) are what your practitioner runs on the device — shown, never a button.
  if (d.scan_recommendations) {
    const sr = d.scan_recommendations;
    let s = `<div class="card scanrec-card">
      <h2>What your scan matched</h2>
      <p class="muted" style="margin:0 0 .7rem">From your voice scan on ${esc(fmtDate(sr.scan_date))}.</p>`;
    if (sr.infoceuticals.length) {
      s += `<p style="margin:.2rem 0 .4rem"><strong>Your infoceuticals</strong></p>`;
      s += sr.infoceuticals.map(i => {
        const name = esc(i.label);
        const days = i.protocol_days ? ` <span class="muted">· ${i.protocol_days}-day protocol</span>` : "";
        return i.order_url
          ? `<p style="margin:.25rem 0">${name}${days} — <a href="${esc(i.order_url)}">order</a></p>`
          : `<p style="margin:.25rem 0">${name}${days}</p>`;
      }).join("");
    }
    if (sr.mihealth.length) {
      s += `<p class="muted" style="margin:.9rem 0 .3rem">miHealth cycles — these are run on the device by your practitioner, not taken as drops.</p>`;
      s += sr.mihealth.map(m => `<p class="muted" style="margin:.2rem 0;font-size:.9em">${esc(m.code)} ${esc(m.label)}</p>`).join("");
    }
    s += `</div>`;
    html += s;
  }
```

- [ ] **Step 2: Syntax-check the page's inline JS**

```bash
python3 - <<'PY' > /tmp/portal.js
import re
src = open("static/client-portal.html").read()
print('\n;\n'.join(re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', src, re.S)))
PY
node --check /tmp/portal.js && echo "JS PARSES OK"
```

- [ ] **Step 3: Render-verify against a real client's payload**

Serve the LOCAL page, proxy `/api/*` to prod, and set `SCAN_RECOMMENDATIONS_ENABLED=1` only in the proxy's injected response. Load the page in headless Chrome, strip `<script>` and `<style>` before grepping (`--dump-dom` includes the inline script source — grepping it matches your own code and looks like a pass), and assert:
- the heading "What your scan matched" appears;
- `Big Field Aligner (BFA)` appears, followed by an `order` link;
- an `ER` code appears with **no** adjacent `order` link;
- no occurrence of `remedymatch.com`.

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(portal): the scan-recommendations card"
```

---

### Task 6: full-suite regression, PR, then flip the flag (controller)

- [ ] **Step 1: Baseline vs branch, ANSI-stripped, concierge eval deselected**

```bash
cd ~/deploy-chat && git worktree add -f --detach /tmp/wt-s2-base origin/main
cd /tmp/wt-s2-base && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_journey_assets.py --deselect tests/test_portal_concierge_eval.py::test_grounding_and_style_pass_rate 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "^FAILED tests/" | sed 's/ - .*//' | sort -u > /tmp/s2-base.txt
```

Run the identical command on the branch into `/tmp/s2-branch.txt`, then `comm -23 /tmp/s2-branch.txt /tmp/s2-base.txt` must be EMPTY.

`test_journey_assets.py` is excluded (a `PIL` import error aborts collection). `test_portal_concierge_eval` is deselected on BOTH sides — it is an LLM pass-rate eval that fails ~1 run in 3 in isolation and will otherwise appear as a phantom regression. `grep -E "^FAILED"` does not match pytest's colourised output; the `sed` is load-bearing.

- [ ] **Step 2: Remove the baseline worktree, push, open the PR.**

- [ ] **Step 3: After merge + deploy, confirm what Slice 1 actually stored** — the read path exists now:

```bash
KEY=$(doppler secrets get CONSOLE_SECRET -p remedy-match -c prd --plain)
curl -s -H "X-Console-Key: $KEY" "https://illtowell.com/api/console/scan-recommendations" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['total_rows'], 'rows |', d['clients'], 'clients |', d['scans'], 'scans')"
```

**Acceptance: `5914 rows | 162 clients | 570 scans`.** That is the direct confirmation Slice 1 could not make, and
it proves the atomic replace never accumulated. No client's email is needed for it.

- [ ] **Step 4: Flip the flag in Doppler, not the Render API.**

```bash
doppler secrets set SCAN_RECOMMENDATIONS_ENABLED=1 -p remedy-match -c prd
```

deploy-chat syncs env FROM Doppler and prunes anything not there — a Render-API-only var silently disappears on the next resync. That is exactly how `SCAN_REQUEST_ENABLED` drifted off and left live client emails pointing at a disabled endpoint.

- [ ] **Step 5: Render-verify on production** — load a real client's portal in headless Chrome, confirm the card, the BFA order link, and that no `ER` row carries a button.

---

## Non-goals

- **Deduplicating the two BFA records.** `bfa-big-field-aligner` has the storefront `url` but no `bottle_type`; `bfa-big-field-aligner-infoceutical` has `bottle_type`, dosing, and `fmp_id`. Retiring either loses something. A proper merge is its own slice.
- **The FF request button, the AI matches, and the paid actions.** That is Slice 3.
- **Species / "Give our Aloha to <name>".** Slice 4.
- **Making `ER`/`MR` purchasable.** They are device cycles.
- **Touching the paywall.** `_portal_biofield_unlocked` governs the $300 Causal Biofield Analysis and is untouched. E4L scan results are free.
