# Portal Findings Population Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `findings` (with `e4l_description`) into published portal content so the stress-pattern chips shipped in PR #659 render on real portals.

**Architecture:** Single-function change in `dashboard/biofield_portal_publish.py` → `build_portal_content`, the one builder both publish routes consume (confirmed: local `biofield_local_app._do_publish` builds the payload via this function). Replace the hardcoded `"findings": []` with findings pulled from the scan via an injectable `scan_context`. Unit-tested with a stubbed `scan_context` so tests never touch the machine's `e4l.db`.

**Tech Stack:** Python; pytest. The test file imports only `dashboard` modules (no `app`), so plain `python3 -m pytest` runs it — no Doppler/Pinecone needed.

## Global Constraints

- **One module changed:** `dashboard/biofield_portal_publish.py` (+ its test file). No `app.py` change, no prod render-path change, no schema change, no new endpoint.
- **None-raising contract:** `build_portal_content` must never raise. Blank `email`/`scan_date`, or a raising/absent `scan_context`, must yield `findings == []`.
- **Injectable `scan_context`:** add keyword param `scan_context=None`; when `None`, lazily import `dashboard.biofield_e4l.scan_context` (lazy import avoids any load-time cycle).
- **Scan alignment:** call `scan_context(email, scan_date)` — pass the published `scan_date` as the `today` arg so findings match the report being published, not "latest."
- **Trim to exactly four fields per finding:** `{"code", "name", "description", "rank"}`. Drop `category`/`group` (the portal ignores them).
- **All findings** (infoceuticals + ER/MR stresses), in the order `scan_context` returns them. **Forward-only** — no backfill of existing portals.

---

### Task 1: Populate `findings` from the scan in `build_portal_content`

**Files:**
- Modify: `dashboard/biofield_portal_publish.py:96-160` (`build_portal_content`)
- Test: `tests/test_biofield_portal_publish_build.py`

**Interfaces:**
- Consumes: `dashboard.biofield_e4l.scan_context(email, today) -> {"findings": [{"code","name","description","rank","category","group"}, ...], ...}` (already exists).
- Produces: `build_portal_content(..., scan_context=None)` now returns `content["findings"]` as a list of `{"code","name","description","rank"}` dicts (empty on any failure). No other return field changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_biofield_portal_publish_build.py` (the `_seed_karin` helper and `CATALOG`/imports already exist at the top of the file):

```python
def test_build_populates_findings_from_scan(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    captured = {}
    def fake_scan_context(email, today):
        captured["args"] = (email, today)
        return {"findings": [
            {"code": "ED3", "name": "Cell Driver", "rank": 1,
             "description": "The Cell Driver supports cellular energy.",
             "category": "ED", "group": "infoceutical"},
            {"code": "ER9", "name": "Environmental Load", "rank": 2,
             "description": "", "category": "ER", "group": "stress"},
        ]}
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, scan_context=fake_scan_context)
    f = out["content"]["findings"]
    assert len(f) == 2
    # trimmed to exactly the four portal-consumed fields; description preserved
    assert f[0] == {"code": "ED3", "name": "Cell Driver",
                    "description": "The Cell Driver supports cellular energy.", "rank": 1}
    # blank-description finding kept, description == ""
    assert f[1] == {"code": "ER9", "name": "Environmental Load",
                    "description": "", "rank": 2}
    # aligned to the published scan_date, not "latest"
    assert captured["args"] == ("permanentlyyours777@hawaiiantel.net", "2026-06-25")


def test_build_findings_empty_when_scan_context_raises(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    def boom(email, today):
        raise RuntimeError("e4l.db unreadable")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, scan_context=boom)
    assert out["content"]["findings"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_biofield_portal_publish_build.py -q`
Expected: FAIL — the current `build_portal_content` has no `scan_context` param (TypeError: unexpected keyword argument 'scan_context'), and `content["findings"]` is `[]`.

- [ ] **Step 3: Implement the change**

In `dashboard/biofield_portal_publish.py`, update the signature (lines 96-97):

```python
def build_portal_content(cx, test_id, *, special_price_cents, catalog=None,
                         audio_url=None, report_pdf_url=None, scan_context=None):
```

Then insert the findings computation between the reorder loop and the `content = {` dict — i.e. after line 138 (`reorder.append(...)`) and before line 140 (`content = {`):

```python
    # Bake the scan's findings (name + e4l_description) into the portal content so
    # the client-portal stress-pattern chips render. scan_context reads the local
    # e4l.db; passing scan_date as `today` aligns findings to THIS report's scan,
    # not "latest". Injectable for tests; never raises (portal must publish even
    # when e4l.db is missing/unreadable). Trimmed to the fields the portal uses.
    email = (client.get("email") or "").strip().lower()
    scan_date = rep.get("date") or ""
    _scan_ctx = scan_context
    if _scan_ctx is None:
        from dashboard.biofield_e4l import scan_context as _scan_ctx
    findings = []
    if email and scan_date:
        try:
            raw = _scan_ctx(email, scan_date).get("findings") or []
            findings = [{"code": f.get("code", ""), "name": f.get("name", ""),
                         "description": f.get("description", ""), "rank": f.get("rank")}
                        for f in raw]
        except Exception:
            findings = []
```

Change the content dict's findings line (was line 146):

```python
        "findings": findings,
```

Reuse the hoisted `email` in the return dict (was line 154 `"email": (client.get("email") or "").strip().lower(),`):

```python
    return {
        "email": email,
        "name": name,
        "scan_date": rep.get("date") or "",
        "scan_id": "",
        "content": content,
        "unresolved": unresolved,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_biofield_portal_publish_build.py -q`
Expected: PASS — all tests in the file green (the two new ones plus the three pre-existing `test_build_*`, which don't assert on findings and stay deterministic: the injected/real `scan_context` returns `[]` when `e4l.db` is absent).

- [ ] **Step 5: Commit**

Stage ONLY these two paths (never `git add -A`/`.`; git-ignored scratch lives under `.superpowers/`):

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_build.py
git commit -m "feat(portal): bake scan findings into published biofield content"
```

---

## Notes for the reviewer / executor

- Guard `if email and scan_date` short-circuits when either is blank (no `scan_context` call, `findings == []`). It is a trivial defensive branch; the raise-test proves the none-raising contract on the populated path.
- The change is forward-only by design: existing published portals keep `findings: []` until re-published. Not a bug — a scoped decision.
- Live confirmation (optional, needs the local intake tool + a real client): re-publish one client from `:8011`, then load their `/portal/<token>` — the chips now appear. Not automated here.
