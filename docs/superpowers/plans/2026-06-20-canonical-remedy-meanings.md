# Canonical Remedy Meanings + Reveal Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A slug-keyed canonical meaning per catalog product that the Biofield-reveal ingest applies to every new reveal, populated by auto-captured reveal edits (default-on "remember" toggle), a dedicated console curation page, and an AI pre-load; plus an ingest guardrail that auto-drops non-catalog remedies and records what it dropped.

**Architecture:** A new `dashboard/biofield_meanings.py` store + AI-propose. The `/api/e4l/reveal-draft` ingest resolves each remedy to a catalog slug (drop + record non-catalog), then overrides the meaning from the store. The console reveal-edit promotes edited meanings to the store by default. A new `/console/remedy-meanings` page (+ dispatch actions) lists and curates all meanings and AI-pre-loads them. Canonical only sets a NEW reveal's initial meanings; the reveal-approval flow stays the member-facing gate.

**Tech Stack:** Python 3.11 / Flask (single `app.py`), SQLite (`LOG_DB`), the dispatch spine (`dashboard/actions.py`, `dashboard/rbac.py`), the Anthropic client `_cl` (model `claude-haiku-4-5-20251001`), pytest. Console front-end is vanilla JS modeled on `static/console-biofield-reveals.html`.

## Global Constraints

- No emoji, no em dashes (code, comments, commit messages).
- Console-only; OWNER/OPS + CONSOLE_SECRET-gated. No new public flag (the reveal-approval flow is the member-facing gate).
- Key the store by slug. Post-guardrail every surviving remedy resolves to a catalog product.
- Canonical applies ONLY at ingest to set a NEW reveal's initial meanings; never retroactive; a per-reveal edit always wins for that reveal.
- Auto-drop non-catalog remedies at ingest, but ALWAYS record the dropped names (no silent drop).
- Everything wrapped/never-raises into callers: store ops, the resolver, the canonical override, the "remember" promotion, and `propose_meaning` (returns "" on failure) must never break ingest, the reveal edit, or the page.
- Test runner: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. tmp `LOG_DB` via `monkeypatch.setattr(app_module, "LOG_DB", db)`; mock the LLM client; CONSOLE_SECRET via `monkeypatch.setattr(app_module, "CONSOLE_SECRET", "")` (disables the gate) or pass `?key=`.

---

## Critical files

- `dashboard/biofield_meanings.py` (new) — the store (`init_table`, `upsert`, `get_map`, `get_all`, `delete`) + `propose_meaning(product, client)`.
- `dashboard/biofield_reveals.py` — add a `dropped` column (idempotent ALTER in `init_table`, parse in `_row`, `set_dropped`).
- `app.py` — `_resolve_remedy_slug(r)`; the guardrail+override block in `/api/e4l/reveal-draft` (~10169); `/console/remedy-meanings` + `/api/console/remedy-meanings`; register the new actions near line 20226.
- `dashboard/biofield_reveal_actions.py` — extend `_exec_edit` with the default-on "remember" promotion.
- `dashboard/remedy_meaning_actions.py` (new) — dispatch actions `remedy_meaning.save/.delete/.propose/.propose_all`.
- `static/console-remedy-meanings.html` (new) — the curation page (modeled on `console-biofield-reveals.html`).
- `tests/test_biofield_meanings.py` (new).

---

## Task 1: Meaning store + AI propose (`dashboard/biofield_meanings.py`)

**Files:**
- Create: `dashboard/biofield_meanings.py`
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Produces:
  - `init_table(cx)`
  - `upsert(cx, slug, meaning, by, source)` — insert/update one slug (single row per slug).
  - `get_map(cx) -> {slug: meaning}` (omits empty meanings).
  - `get_all(cx) -> [{slug, meaning, source, updated_at}]`.
  - `delete(cx, slug)`
  - `propose_meaning(product, client) -> str` — 1-2 sentence function-leading meaning; never raises (returns "" on any failure or when client is None).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_biofield_meanings.py`:

```python
# tests/test_biofield_meanings.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


class _FakeClient:
    """Mimics anthropic client.messages.create(...).content[0].text."""
    def __init__(self, text="Supports the body's natural detox pathways.", raises=False):
        self._t, self._raises = text, raises
        outer = self
        class _M:
            def create(self, **kw):
                if outer._raises:
                    raise RuntimeError("llm down")
                return type("R", (), {"content": [type("C", (), {"text": outer._t})()]})()
        self.messages = _M()


def _db(tmp_path):
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "m.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
    return bm, db


def test_upsert_get_map_and_all(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "nous-energy", "Guides healing and mobilizes metals.", "glen", "glen")
        assert bm.get_map(cx) == {"nous-energy": "Guides healing and mobilizes metals."}
        allrows = bm.get_all(cx)
    assert len(allrows) == 1 and allrows[0]["slug"] == "nous-energy" and allrows[0]["source"] == "glen"


def test_upsert_updates_single_row(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "cistus", "First.", "ai", "ai")
        bm.upsert(cx, "cistus", "Second.", "glen", "glen")
        rows = bm.get_all(cx)
    assert len(rows) == 1 and rows[0]["meaning"] == "Second." and rows[0]["source"] == "glen"


def test_delete(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "x", "m", "glen", "glen")
        bm.delete(cx, "x")
        assert bm.get_map(cx) == {}


def test_get_map_omits_empty(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "x", "", "ai", "ai")
        assert bm.get_map(cx) == {}


def test_propose_meaning_builds_text():
    bm = _load("dashboard.biofield_meanings")
    product = {"name": "Nous Energy", "ingredients": [{"name": "spirit minerals"}],
               "benefits": ["mental energy"], "description": "Guides healing."}
    out = bm.propose_meaning(product, _FakeClient("Guides healing processes and mobilizes heavy metals."))
    assert out == "Guides healing processes and mobilizes heavy metals."


def test_propose_meaning_never_raises():
    bm = _load("dashboard.biofield_meanings")
    assert bm.propose_meaning({"name": "X"}, _FakeClient(raises=True)) == ""
    assert bm.propose_meaning({"name": "X"}, None) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -v`
Expected: FAIL (`dashboard.biofield_meanings` does not exist).

- [ ] **Step 3: Create the store module**

Create `dashboard/biofield_meanings.py`:

```python
"""Canonical per-product remedy meaning store. One curated 1-2 sentence meaning
per catalog slug, applied to new Biofield reveals at ingest. All functions are
wrapped or pure; none raise into callers (propose_meaning returns "" on failure)."""
import json
from datetime import datetime, timezone

_MODEL = "claude-haiku-4-5-20251001"


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS biofield_remedy_meanings "
        "(slug TEXT PRIMARY KEY, meaning TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT '', updated_by TEXT, updated_at TEXT)")
    cx.commit()


def upsert(cx, slug, meaning, by, source):
    slug = (slug or "").strip()
    if not slug:
        return
    cx.execute(
        "INSERT INTO biofield_remedy_meanings (slug, meaning, source, updated_by, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(slug) DO UPDATE SET "
        "meaning=excluded.meaning, source=excluded.source, updated_by=excluded.updated_by, updated_at=excluded.updated_at",
        (slug, (meaning or "").strip(), (source or "").strip(), (by or "").strip(), _now()))
    cx.commit()


def get_map(cx):
    """{slug: meaning} for non-empty meanings."""
    rows = cx.execute("SELECT slug, meaning FROM biofield_remedy_meanings").fetchall()
    return {r[0]: r[1] for r in rows if (r[1] or "").strip()}


def get_all(cx):
    rows = cx.execute(
        "SELECT slug, meaning, source, updated_at FROM biofield_remedy_meanings ORDER BY slug").fetchall()
    return [{"slug": r[0], "meaning": r[1], "source": r[2], "updated_at": r[3]} for r in rows]


def delete(cx, slug):
    cx.execute("DELETE FROM biofield_remedy_meanings WHERE slug=?", ((slug or "").strip(),))
    cx.commit()


def propose_meaning(product, client):
    """1-2 sentence meaning that LEADS with the remedy's major functions, warm lay
    voice, no disease claims. Never raises -> "" on any failure or no client."""
    if client is None:
        return ""
    try:
        name = product.get("name") or product.get("slug") or ""
        ingredients = product.get("ingredients") or []
        if isinstance(ingredients, list):
            ing = ", ".join(
                str(i.get("name") if isinstance(i, dict) else i) for i in ingredients[:20])
        else:
            ing = str(ingredients)
        benefits = product.get("benefits") or []
        ben = "; ".join(str(b) for b in benefits) if isinstance(benefits, list) else str(benefits)
        desc = product.get("description") or ""
        user = (
            f"Remedy: {name}\nKey ingredients: {ing}\nBenefits: {ben}\nDescription: {desc}\n\n"
            "Write a 1 to 2 sentence remedy 'meaning' that LEADS with this remedy's major functions, "
            "in warm, plain, lay language. No disease claims, no diagnosis, no hype. "
            "Return only the sentence or two, with no preamble.")
        msg = client.messages.create(
            model=_MODEL, max_tokens=160, messages=[{"role": "user", "content": user}])
        parts = getattr(msg, "content", None) or []
        return "".join(getattr(p, "text", "") for p in parts).strip()
    except Exception:
        return ""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_meanings.py tests/test_biofield_meanings.py
git commit -m "feat: canonical remedy meanings store + AI propose"
```

---

## Task 2: `dropped` column on the reveal store (`dashboard/biofield_reveals.py`)

**Files:**
- Modify: `dashboard/biofield_reveals.py` (`init_table` ALTER, `_row` parse, new `set_dropped`)
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Produces: `set_dropped(cx, rid, names)`; `_row` now includes `dropped` (a list); `init_table` adds the `dropped` column idempotently.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_biofield_meanings.py`:

```python
def test_reveal_dropped_column(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        br.init_table(cx)  # idempotent (no error on the ALTER second time)
        rid, is_new = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"},
                                [{"name": "Top", "slug": "top"}], "s")
        assert is_new
        br.set_dropped(cx, rid, ["Mineral Binder", "Made Up"])
        row = br.get(cx, rid)
    assert row["dropped"] == ["Mineral Binder", "Made Up"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py::test_reveal_dropped_column -v`
Expected: FAIL (`set_dropped` missing / `dropped` not in row).

- [ ] **Step 3: Add the column + accessor**

In `dashboard/biofield_reveals.py`, change `init_table` to add the column idempotently (append the ALTER before the final `cx.commit()`):

```python
def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            interpretation_json TEXT NOT NULL DEFAULT '{}',
            remedies_json TEXT NOT NULL DEFAULT '[]',
            first_approved INTEGER NOT NULL DEFAULT 0,
            token_hash TEXT,
            approved_at TEXT, approved_by TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(email, scan_date)
        )
    """)
    # Additive column for the ingest guardrail (idempotent).
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN dropped TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
    cx.commit()
```

In `_row`, parse `dropped` (after the `remedies` line):

```python
def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["interpretation"] = json.loads(d.pop("interpretation_json") or "{}")
    d["remedies"] = json.loads(d.pop("remedies_json") or "[]")
    d["dropped"] = json.loads(d.pop("dropped", "[]") or "[]")
    d["first_approved"] = bool(d.get("first_approved"))
    return d
```

Add `set_dropped` (next to `set_remedies`):

```python
def set_dropped(cx, rid, names):
    cx.execute("UPDATE biofield_reveals SET dropped=?, updated_at=? WHERE id=?",
               (json.dumps(names or []), _now(), rid))
    cx.commit()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py::test_reveal_dropped_column -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveals.py tests/test_biofield_meanings.py
git commit -m "feat: dropped column on biofield_reveals (ingest guardrail record)"
```

---

## Task 3: Ingest resolver + guardrail + canonical override (`app.py`)

**Files:**
- Modify: `app.py` (add `_resolve_remedy_slug`; the guardrail/override block in `api_e4l_reveal_draft` ~10169)
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Consumes: `_get_product`, `_TITLE_TO_SLUG`, `dashboard.biofield_meanings`, `dashboard.biofield_reveals` (`upsert`, `set_dropped`).
- Produces: `_resolve_remedy_slug(r) -> str|None`; `/api/e4l/reveal-draft` now drops non-catalog remedies (recording names in `dropped`), assigns a valid slug to survivors, and overrides their meaning from the canonical store.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_meanings.py` (these load `app`):

```python
def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br, biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        bm.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def _push(app_module, remedies, key):
    return app_module.app.test_client().post(
        "/api/e4l/reveal-draft",
        headers={"X-Console-Key": key},
        json={"email": "a@b.com", "scan_date": "2026-06-20",
              "interpretation": {"body": "x"}, "remedies": remedies})


def test_ingest_drops_non_catalog_and_records(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key:
        pytest.skip("no CRON_SECRET/CONSOLE_SECRET in env")
    # 'top' is a placeholder slug that won't resolve; force a known real slug via monkeypatch.
    real = next(iter((app_module._PRODUCTS.get("products") or {}).keys()), None)
    if not real:
        pytest.skip("no catalog products")
    r = _push(app_module, [{"name": app_module._PRODUCTS["products"][real]["name"], "slug": real, "meaning": "pushed"},
                           {"name": "Totally Made Up", "slug": "nope-xyz", "meaning": "ghost"}], key)
    assert r.get_json().get("ok") is True
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rows = br.list_pending(cx)
    row = rows[0]
    slugs = [x.get("slug") for x in row["remedies"]]
    assert real in slugs and "nope-xyz" not in slugs
    assert "Totally Made Up" in row["dropped"]


def test_ingest_applies_canonical_override(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key:
        pytest.skip("no CRON_SECRET/CONSOLE_SECRET in env")
    real = next(iter((app_module._PRODUCTS.get("products") or {}).keys()), None)
    if not real:
        pytest.skip("no catalog products")
    from dashboard import biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, real, "CANONICAL MEANING", "glen", "glen")
    _push(app_module, [{"name": app_module._PRODUCTS["products"][real]["name"], "slug": real, "meaning": "pushed text"}], key)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    assert row["remedies"][0]["meaning"] == "CANONICAL MEANING"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k ingest -v`
Expected: FAIL (non-catalog remedy is NOT dropped; canonical override absent).

- [ ] **Step 3: Add the resolver**

In `app.py`, immediately after `_get_product` (~line 2985-end of that function), add:

```python
def _resolve_remedy_slug(r):
    """Resolve a pushed remedy to a catalog slug: its slug if valid, else its name
    via _TITLE_TO_SLUG (exact then case-insensitive), else None. Never raises."""
    try:
        s = (r.get("slug") or "").strip()
        if s and _get_product(s):
            return s
        name = (r.get("name") or "").strip()
        if not name:
            return None
        if name in _TITLE_TO_SLUG:
            return _TITLE_TO_SLUG[name]
        low = name.lower()
        for title, slug in _TITLE_TO_SLUG.items():
            if (title or "").strip().lower() == low:
                return slug
        return None
    except Exception:
        return None
```

- [ ] **Step 4: Add the guardrail + override in the ingest handler**

In `app.py`, inside `api_e4l_reveal_draft`, replace the line `rid, is_new = _br.upsert(cx, email, scan_date, interp, remedies, (data.get("source") or "").strip())` and its surrounding `with` block so the guardrail runs first and `dropped` is persisted. The new body of the `try:` (from `_br.init_table(cx)` through the token mint) becomes:

```python
    from dashboard import biofield_reveals as _br, biofield_meanings as _bm
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _br.init_table(cx)
            _bm.init_table(cx)
            # Guardrail + canonical override.
            try:
                canon = _bm.get_map(cx)
            except Exception:
                canon = {}
            cleaned, dropped = [], []
            for r in (remedies or []):
                slug = _resolve_remedy_slug(r) if isinstance(r, dict) else None
                if not slug:
                    nm = (r.get("name") if isinstance(r, dict) else "") or "(unnamed)"
                    dropped.append(nm.strip() or "(unnamed)")
                    continue
                rr = dict(r)
                rr["slug"] = slug
                cm = canon.get(slug)
                if cm:
                    rr["meaning"] = cm
                cleaned.append(rr)
            rid, is_new = _br.upsert(cx, email, scan_date, interp, cleaned, (data.get("source") or "").strip())
            try:
                _br.set_dropped(cx, rid, dropped)
            except Exception:
                pass
            if is_new:
                token = secrets.token_urlsafe(32)
                _br.set_token(cx, rid, _hash_token(token))
                now = datetime.now(timezone.utc)
                cx.execute(
                    "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                    (_hash_token(token), email, "biofield_reveal", now.isoformat(),
                     (now + timedelta(days=30)).isoformat()))
                cx.commit()
```

(The `if is_new:` email-send block below it and the `return jsonify({"ok": True, "id": rid})` stay unchanged. Remove the old `from dashboard import biofield_reveals as _br` line that preceded the original `try:` if it now duplicates the import inside the try — keep a single import.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k ingest -v`
Expected: PASS. Then run the existing reveal/trial tests for no regression: `... -m pytest tests/test_biofield_trial.py tests/test_biofield_cart.py -v` (green).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_biofield_meanings.py
git commit -m "feat: reveal ingest guardrail (drop non-catalog) + canonical meaning override"
```

---

## Task 4: Default-on "remember" promotion on the reveal edit (`dashboard/biofield_reveal_actions.py`)

**Files:**
- Modify: `dashboard/biofield_reveal_actions.py` (`_exec_edit`)
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Consumes: `dashboard.biofield_meanings.upsert`, `dashboard.biofield_reveals.set_remedies`.
- Produces: `_exec_edit` promotes each edited remedy's meaning to canonical when its `remember` flag is truthy (default true); the `remember` key is stripped from what is stored on the reveal row.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_meanings.py`:

```python
def _seed_reveal(br, db):
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"},
                           [{"name": "Top", "slug": "top", "meaning": "old"}], "s")
    return rid


def test_edit_remember_default_promotes(tmp_path):
    bra = _load("dashboard.biofield_reveal_actions")
    br = _load("dashboard.biofield_reveals")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
    rid = _seed_reveal(br, db)
    with sqlite3.connect(db) as cx:
        bra._exec_edit({"id": rid, "remedies": [{"name": "Top", "slug": "top", "meaning": "NEW MEANING"}]},
                       {"cx": cx, "actor": None})
        canon = bm.get_map(cx)
        row = br.get(cx, rid)
    assert canon.get("top") == "NEW MEANING"           # promoted (remember defaults on)
    assert row["remedies"][0]["meaning"] == "NEW MEANING"  # reveal row updated
    assert "remember" not in row["remedies"][0]         # remember stripped from stored remedy


def test_edit_remember_false_skips_canonical(tmp_path):
    bra = _load("dashboard.biofield_reveal_actions")
    br = _load("dashboard.biofield_reveals")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
    rid = _seed_reveal(br, db)
    with sqlite3.connect(db) as cx:
        bra._exec_edit({"id": rid, "remedies": [{"name": "Top", "slug": "top", "meaning": "ONE TIME", "remember": False}]},
                       {"cx": cx, "actor": None})
        canon = bm.get_map(cx)
        row = br.get(cx, rid)
    assert "top" not in canon                            # NOT promoted
    assert row["remedies"][0]["meaning"] == "ONE TIME"   # reveal row still updated
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k edit_remember -v`
Expected: FAIL (no promotion; `remember` not stripped).

- [ ] **Step 3: Extend `_exec_edit`**

In `dashboard/biofield_reveal_actions.py`, replace the `if isinstance(params.get("remedies"), list):` block inside `_exec_edit` with:

```python
    if isinstance(params.get("remedies"), list):
        from dashboard import biofield_meanings as _bm
        _bm.init_table(ctx["cx"])
        stored = []
        for rem in params["remedies"]:
            if not isinstance(rem, dict):
                stored.append(rem)
                continue
            remember = rem.get("remember", True)  # default ON
            clean = {k: v for k, v in rem.items() if k != "remember"}
            stored.append(clean)
            slug = (clean.get("slug") or "").strip()
            meaning = (clean.get("meaning") or "").strip()
            if remember and slug and meaning:
                try:
                    _bm.upsert(ctx["cx"], slug, meaning, _actor_name(ctx.get("actor")), "glen")
                except Exception as e:
                    print(f"[remedy-meaning] promote {e!r}", flush=True)
        _br.set_remedies(ctx["cx"], rid, stored)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k edit_remember -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveal_actions.py tests/test_biofield_meanings.py
git commit -m "feat: default-on remember promotes reveal meaning edits to canonical"
```

---

## Task 5: Console actions module (`dashboard/remedy_meaning_actions.py`) + registration

**Files:**
- Create: `dashboard/remedy_meaning_actions.py`
- Modify: `app.py` (register + configure the actions near line 20226)
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Consumes: `dashboard.actions` (`register_action`, `Action`, `LOW_WRITE`, `get_action`), `dashboard.rbac` (`OWNER`, `OPS`), `dashboard.biofield_meanings`.
- Produces: action executors `_exec_save`, `_exec_delete`, `_exec_propose`, `_exec_propose_all`; `configure(**kw)` (injects `client`, `products`); `register()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_meanings.py`:

```python
def test_actions_save_delete_propose(tmp_path):
    rma = _load("dashboard.remedy_meaning_actions")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "a.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
    rma.configure(client=_FakeClient("AI MEANING."),
                  products={"nous-energy": {"name": "Nous Energy", "benefits": ["energy"]}})
    with sqlite3.connect(db) as cx:
        rma._exec_save({"slug": "nous-energy", "meaning": "Glen text."}, {"cx": cx, "actor": None})
        assert bm.get_map(cx)["nous-energy"] == "Glen text."
        rma._exec_propose({"slug": "nous-energy"}, {"cx": cx, "actor": None})
        assert bm.get_map(cx)["nous-energy"] == "AI MEANING."  # propose overwrites with ai text
        rma._exec_delete({"slug": "nous-energy"}, {"cx": cx, "actor": None})
        assert bm.get_map(cx) == {}


def test_action_propose_all_fills_missing(tmp_path):
    rma = _load("dashboard.remedy_meaning_actions")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "a.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
        bm.upsert(cx, "has", "already", "glen", "glen")
    rma.configure(client=_FakeClient("AI."),
                  products={"has": {"name": "Has"}, "missing": {"name": "Missing"}})
    with sqlite3.connect(db) as cx:
        out = rma._exec_propose_all({}, {"cx": cx, "actor": None})
        mp = bm.get_map(cx)
    assert out["proposed"] == 1 and mp["missing"] == "AI." and mp["has"] == "already"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k action -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create the actions module**

Create `dashboard/remedy_meaning_actions.py`:

```python
"""Console actions for the canonical remedy meanings store: save/delete one slug,
AI-propose one slug, and propose-all-missing. Registered on the dispatch spine."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_meanings as _bm

_DEPS = {}


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_save(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _bm.init_table(ctx["cx"])
    _bm.upsert(ctx["cx"], slug, (params.get("meaning") or "").strip(),
               _actor_name(ctx.get("actor")), "glen")
    return {"ok": True}


def _exec_delete(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _bm.init_table(ctx["cx"])
    _bm.delete(ctx["cx"], slug)
    return {"ok": True}


def _exec_propose(params, ctx):
    slug = (params.get("slug") or "").strip()
    prod = (_DEPS.get("products") or {}).get(slug)
    if not prod:
        raise ValueError("unknown product")
    _bm.init_table(ctx["cx"])
    text = _bm.propose_meaning(dict(prod, slug=slug), _DEPS.get("client"))
    if text:
        _bm.upsert(ctx["cx"], slug, text, "ai", "ai")
    return {"ok": bool(text), "meaning": text}


def _exec_propose_all(params, ctx):
    products = _DEPS.get("products") or {}
    _bm.init_table(ctx["cx"])
    existing = _bm.get_map(ctx["cx"])
    cap = int(params.get("cap") or 200)
    proposed, failed = 0, 0
    for slug, prod in list(products.items()):
        if proposed + failed >= cap:
            break
        if existing.get(slug):
            continue
        text = _bm.propose_meaning(dict(prod, slug=slug), _DEPS.get("client"))
        if text:
            _bm.upsert(ctx["cx"], slug, text, "ai", "ai")
            proposed += 1
        else:
            failed += 1
    print(f"[remedy-meaning] propose_all proposed={proposed} failed={failed}", flush=True)
    return {"ok": True, "proposed": proposed, "failed": failed}


def register():
    if get_action("remedy_meaning.save"):
        return
    register_action(Action(key="remedy_meaning.save", module="remedy_meaning", title="Save remedy meaning",
        description="Set the canonical meaning for a product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_save))
    register_action(Action(key="remedy_meaning.delete", module="remedy_meaning", title="Delete remedy meaning",
        description="Remove the canonical meaning for a product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_delete))
    register_action(Action(key="remedy_meaning.propose", module="remedy_meaning", title="Propose remedy meaning (AI)",
        description="AI-propose a function-covering meaning for one product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_propose))
    register_action(Action(key="remedy_meaning.propose_all", module="remedy_meaning", title="Propose all missing (AI)",
        description="AI-propose meanings for every product without one.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_propose_all))
```

- [ ] **Step 4: Register at startup**

In `app.py`, after the `reviews_actions` registration block (~line 20230), add:

```python
# ── Canonical remedy meanings console actions ────────────────────────────────
from dashboard import remedy_meaning_actions as _rma
_rma.configure(client=_cl, products=(_PRODUCTS.get("products") or {}))
_rma.register()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k action -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/remedy_meaning_actions.py app.py tests/test_biofield_meanings.py
git commit -m "feat: remedy meaning console actions (save/delete/propose/propose-all)"
```

---

## Task 6: Console page + data endpoint (`app.py` + `static/console-remedy-meanings.html`)

**Files:**
- Modify: `app.py` (add `/console/remedy-meanings` + `/api/console/remedy-meanings`)
- Create: `static/console-remedy-meanings.html`
- Test: `tests/test_biofield_meanings.py`

**Interfaces:**
- Consumes: `CONSOLE_SECRET`, `_PRODUCTS`, `dashboard.biofield_meanings.get_all`, the dispatch action endpoint `/api/action/...` (existing).
- Produces: `GET /console/remedy-meanings` (page), `GET /api/console/remedy-meanings` (`{rows:[{slug,name,meaning,source,updated_at}]}`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_meanings.py`:

```python
def test_console_meanings_list_auth(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    c = app_module.app.test_client()
    assert c.get("/api/console/remedy-meanings").status_code == 401  # no key
    r = c.get("/api/console/remedy-meanings?key=sek")
    assert r.status_code == 200
    rows = r.get_json()["rows"]
    assert isinstance(rows, list) and all("slug" in x and "name" in x and "meaning" in x for x in rows)


def test_console_meanings_page_serves(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    html = app_module.app.test_client().get("/console/remedy-meanings?key=sek").get_data(as_text=True)
    assert "remedy-meanings" in html and "remedy_meaning" in html  # data endpoint + action markers
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -k console_meanings -v`
Expected: FAIL (routes 404).

- [ ] **Step 3: Add the routes**

In `app.py`, next to the other console-page routes (e.g. after `console_biofield_reveals_page` ~8173), add:

```python
@app.route("/api/console/remedy-meanings", methods=["GET"])
def api_console_remedy_meanings():
    if CONSOLE_SECRET:
        _key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if _key != CONSOLE_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    from dashboard import biofield_meanings as _bm
    with sqlite3.connect(LOG_DB) as cx:
        _bm.init_table(cx)
        by_slug = {m["slug"]: m for m in _bm.get_all(cx)}
    rows = []
    for slug, p in (_PRODUCTS.get("products") or {}).items():
        m = by_slug.get(slug) or {}
        rows.append({"slug": slug, "name": p.get("name", ""),
                     "meaning": m.get("meaning", ""), "source": m.get("source", ""),
                     "updated_at": m.get("updated_at", "")})
    rows.sort(key=lambda r: (r["name"] or "").lower())
    return jsonify({"rows": rows})


@app.route("/console/remedy-meanings", methods=["GET"])
def console_remedy_meanings_page():
    if CONSOLE_SECRET:
        _key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if _key != CONSOLE_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    resp = send_from_directory(STATIC, "console-remedy-meanings.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Create the console page**

Create `static/console-remedy-meanings.html`. Model it on `static/console-biofield-reveals.html` (read that file for the console styling + the `?key=` capture + the `/api/action/...` POST helper). It must: read `key` from the query string; `GET /api/console/remedy-meanings?key=<key>`; render one row per product with the name, an editable `<textarea>` for the meaning, a source badge, and three buttons wired via the dispatch endpoint `POST /api/action/remedy_meaning.save` / `.delete` / `.propose` (body `{params:{slug, meaning}}` plus the console key header `X-Console-Key`), plus a "Propose all missing" button calling `remedy_meaning.propose_all`. All dynamic text via `textContent`/`setAttribute` (no innerHTML of server data). Keep the literal strings `remedy-meanings` and `remedy_meaning` in the page so the serve test passes. No emoji, no em dashes.

(Concrete minimum the serve test needs and the page must contain: a `fetch("/api/console/remedy-meanings?key=" + key)` call and `"/api/action/remedy_meaning."` action calls.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_meanings.py -v`
Expected: PASS (the whole suite).

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-remedy-meanings.html tests/test_biofield_meanings.py
git commit -m "feat: console remedy-meanings curation page + data endpoint"
```

---

## Verification

- Per task: the named `pytest` target passes (doppler + venv).
- Full sweep after Task 6: `tests/test_biofield_meanings.py` all green; plus `tests/test_biofield_trial.py tests/test_biofield_cart.py -v` for no reveal-path regression (the ingest change is additive + wrapped).
- Final Opus whole-branch review (focus: the guardrail never rejects a push and always records `dropped`; canonical override only at ingest, never retroactive; the resolver maps slug-and-name correctly and returns None for junk; the default-on "remember" promotes and `remember` is stripped from stored remedies, while `remember:false` leaves canonical untouched; everything wrapped/never-raises; console endpoints + actions CONSOLE_SECRET + OWNER/OPS gated; `propose_meaning` never raises and `propose_all` logs proposed/failed; XSS-safe page; no emoji/em-dash).
- Manual visual pass (live, console): `/console/remedy-meanings` lists products, edit+save persists, "Propose with AI" fills a row, "Propose all missing" seeds the catalog; a reveal edit with "remember" checked carries forward to the next ingest; a non-catalog remedy shows in the reveal's "dropped" notice.
- Ship via PR + merge to `main` (auto-deploys). Console-only, no flag; gentle probe of `/api/console/remedy-meanings?key=` (200 with rows) per the warm-up rule. Update memory.

## Build order
Task 1 (store + propose) -> Task 2 (dropped column) -> Task 3 (ingest guardrail/override) -> Task 4 (remember promotion) -> Task 5 (console actions) -> Task 6 (console page). Tasks 3-6 depend on Task 1; Task 4 on Tasks 1-2; Task 6 on Task 5. After Task 6, optionally run `remedy_meaning.propose_all` once in prod to pre-load (it is AI-proposed-then-curated, and the reveal-approval flow gates anything member-facing).
