# Biofield Reveal Console Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/console/biofield-reveals` reviewable: show each draft's id + scan date + client name + tags, show readable stress factors per layer, and add a Delete action - while the member reveal stays unchanged (no patterns shown).

**Architecture:** The bridge (vault) attaches readable `pattern_labels` to each pushed layer. The deploy-chat console endpoint joins `people` for name + tags; a new `biofield_reveal.delete` action removes a draft; the console page renders the new context and preserves `patterns`/`pattern_labels` across edits. No schema change (labels ride inside `layers_json`).

**Tech Stack:** Python 3.11, Flask, SQLite, vanilla JS console page.

## Global Constraints

- Two repos: Task 1 is the AI-Training vault (`02 Skills/`, no PR, auto-snapshot). Tasks 2-4 are deploy-chat (PR + merge).
- No emoji, no em dashes (plain hyphen).
- **Member reveal must not show patterns.** `_biofield_layer_payload` is unchanged; a regression test proves a stored layer's `patterns`/`pattern_labels` never reach the member payload.
- Console changes are CONSOLE_SECRET-gated; no new public flag.
- All enrichment is best-effort: a missing/blank `people` row or malformed `tags` must never break the listing.
- **Edits must preserve `patterns` + `pattern_labels`** - `collectLayers` currently rebuilds layers without them, which would erase the stress factors on Save.
- deploy-chat test harness: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Vault: `cd ~/AI-Training && ~/.venvs/deploy-chat311/bin/python -m pytest "02 Skills/tests/test_e4l_reveal_push.py" -v`.

---

## File Structure

- `~/AI-Training/02 Skills/e4l-reveal-push.py` (vault): `build_payload` gains `label_map`; `run` builds it.
- `dashboard/biofield_reveals.py`: add `delete(cx, rid)`.
- `dashboard/biofield_reveal_actions.py`: add `_exec_delete` + register `biofield_reveal.delete`.
- `app.py`: `api_console_biofield_reveals` enriches drafts with `client_name` + `tags` (new helper `_people_brief`).
- `static/console-biofield-reveals.html`: header (id/scan-date/name/tags), per-layer stress factors, edit-preservation, Delete button + `doDelete`.
- Tests: `02 Skills/tests/test_e4l_reveal_push.py` (vault); `tests/test_biofield_layers.py` (deploy-chat).

---

## Task 1 (vault): bridge attaches readable `pattern_labels` per layer

**Files:**
- Modify: `~/AI-Training/02 Skills/e4l-reveal-push.py`
- Test: `~/AI-Training/02 Skills/tests/test_e4l_reveal_push.py`

**Interfaces:**
- `build_payload(content, email, scan_date, label_map=None) -> dict | None` - each output layer gains `pattern_labels = [label_map.get(c, c) for c in patterns]` (readable name, falling back to the raw code). `patterns` (codes) retained. `label_map` defaults to `{}`.

- [ ] **Step 1: Add the failing test** (append to `test_e4l_reveal_push.py`)

```python
def test_build_payload_attaches_pattern_labels():
    m = _load()
    content = {"greeting": "Aloha.", "layers": [
        {"n": 1, "title": "T", "meaning": "M", "remedy": "Nous Energy",
         "patterns": ["ER26", "ZZ99"]}]}
    label_map = {"ER26": "Adrenal Rejuvenator"}
    p = m.build_payload(content, "a@x.com", "2026-06-01", label_map=label_map)
    L = p["layers"][0]
    assert L["patterns"] == ["ER26", "ZZ99"]
    assert L["pattern_labels"] == ["Adrenal Rejuvenator", "ZZ99"]  # missing code -> code itself
```

- [ ] **Step 2: Run -> FAIL**

Run: `cd ~/AI-Training && ~/.venvs/deploy-chat311/bin/python -m pytest "02 Skills/tests/test_e4l_reveal_push.py::test_build_payload_attaches_pattern_labels" -v`
Expected: FAIL (`pattern_labels` not present / unexpected `label_map` kwarg).

- [ ] **Step 3: Implement** - change `build_payload`'s signature and per-layer dict in `e4l-reveal-push.py`:

```python
def build_payload(content, email, scan_date, label_map=None):
    """Map e4l_synthesis.to_portal_content output to the reveal-draft payload.
    Returns None when there are no layers (a failed/empty synthesis)."""
    label_map = label_map or {}
    layers_in = (content or {}).get("layers") or []
    if not layers_in:
        return None
    out_layers = []
    for L in layers_in:
        remedy_str = (L.get("remedy") or "").strip()
        first = remedy_str.split(" + ")[0].strip() if remedy_str else ""
        patterns = L.get("patterns") or []
        out_layers.append({
            "n": L.get("n"),
            "title": (L.get("title") or "").strip(),
            "summary": (L.get("meaning") or "").strip(),
            "patterns": patterns,
            "pattern_labels": [label_map.get(c, c) for c in patterns],
            "remedy": {"name": first} if first else None,
        })
    return {
        "email": email,
        "scan_date": scan_date,
        "interpretation": {"greeting": (content.get("greeting") or ""), "body": ""},
        "layers": out_layers,
        "source": "e4l-synthesis",
    }
```

- [ ] **Step 4: Build the label map in `run`** - after `patterns = E.pull_patterns(...)` and before/at the `build_payload` call, change the `run` body:

```python
    patterns = E.pull_patterns(cx, scan["scan_id"], limit=top)
    print(f"scan {scan['scan_id']} ({scan['scan_date']}): {len(patterns)} top patterns")
    label_map = {p["item_code"]: (p.get("full_name") or p.get("name") or p["item_code"])
                 for p in patterns if p.get("item_code")}
    ...
    payload = build_payload(content, email, scan["scan_date"], label_map=label_map)
```

(Leave the rest of `run` unchanged.)

- [ ] **Step 5: Run the suite -> PASS**

Run: `cd ~/AI-Training && ~/.venvs/deploy-chat311/bin/python -m pytest "02 Skills/tests/test_e4l_reveal_push.py" -v`
Expected: PASS (7 passed - the 6 existing plus the new one; the optional kwarg keeps the others green).

- [ ] **Step 6: Commit** (vault auto-snapshots; explicit optional)

```bash
cd ~/AI-Training && git add "02 Skills/e4l-reveal-push.py" "02 Skills/tests/test_e4l_reveal_push.py" && git commit -m "feat: bridge attaches readable pattern_labels per layer"
```

---

## Task 2 (deploy-chat): delete action + member anti-leak regression

**Files:**
- Modify: `dashboard/biofield_reveals.py` (add `delete`)
- Modify: `dashboard/biofield_reveal_actions.py` (add `_exec_delete` + register)
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- `biofield_reveals.delete(cx, rid)` - `DELETE FROM biofield_reveals WHERE id=?`, commit. Idempotent.
- Action `biofield_reveal.delete` (executor `_exec_delete(params, ctx)` -> `{"deleted": rid}`), OWNER/OPS, LOW_WRITE.

- [ ] **Step 1: Add the failing tests** (append to `tests/test_biofield_layers.py`; reuse that file's existing `_load_app`/`_fresh` helpers and `_extract_reveal`)

```python
def test_delete_action_removes_draft(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "d@x.com", "2026-06-01", {"greeting": "hi"},
                           [{"name": "Top", "slug": "top", "meaning": "m"}], "s")
    c = app_module.app.test_client()
    r = c.post("/api/action/biofield_reveal.delete",
               json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r.status_code == 200 and r.get_json().get("status") == "done"
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM biofield_reveals WHERE id=?", (rid,)).fetchone()[0] == 0
    # idempotent: deleting again still succeeds
    r2 = c.post("/api/action/biofield_reveal.delete",
                json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r2.status_code == 200


def test_member_payload_never_leaks_patterns(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "is_member", lambda **kw: True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "p@x.com", "2026-06-01", {"greeting": "Hi"},
                           [{"name": "Top", "slug": "top", "meaning": "m"}], "s",
                           layers=[{"n": 1, "title": "Layer", "summary": "S",
                                    "patterns": ["ER26"], "pattern_labels": ["Adrenal Rejuvenator"],
                                    "remedy": {"name": "Top", "slug": "top", "meaning": "m"}}])
        br.set_token(cx, rid, th)
        br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, "p@x.com", "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").data
    assert b"ER26" not in html, "raw pattern code leaked to member"
    assert b"Adrenal Rejuvenator" not in html, "pattern label leaked to member"
    reveal = _extract_reveal(html)
    for L in (reveal.get("layers") or []):
        assert "patterns" not in L and "pattern_labels" not in L
```

- [ ] **Step 2: Run -> FAIL**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_delete_action_removes_draft -v`
Expected: FAIL (action `biofield_reveal.delete` not registered -> non-done status). The anti-leak test may already pass (it is a guard); confirm it does.

- [ ] **Step 3: Add `delete` to the store** (`dashboard/biofield_reveals.py`, after `set_dropped`)

```python
def delete(cx, rid):
    cx.execute("DELETE FROM biofield_reveals WHERE id=?", (int(rid),))
    cx.commit()
```

- [ ] **Step 4: Add the executor + registration** (`dashboard/biofield_reveal_actions.py`)

After `_exec_approve`:

```python
def _exec_delete(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _br.delete(ctx["cx"], rid)
    return {"deleted": rid}
```

In `register()`, after the `biofield_reveal.approve` registration (still inside the function, before it returns), add:

```python
    register_action(Action(
        key="biofield_reveal.delete", module="biofield_reveal", title="Delete Biofield reveal",
        description="Delete a reveal draft (removes it from the queue).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_delete))
```

(The `get_action("biofield_reveal.approve")` guard at the top still works: on a fresh process all three register together; on repeat calls it returns early.)

- [ ] **Step 5: Run -> PASS**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS (all existing layer tests plus the two new ones).

- [ ] **Step 6: Commit**

```bash
git add dashboard/biofield_reveals.py dashboard/biofield_reveal_actions.py tests/test_biofield_layers.py
git commit -m "feat: biofield_reveal.delete action + member anti-leak regression test"
```

---

## Task 3 (deploy-chat): console endpoint enriches drafts with name + tags

**Files:**
- Modify: `app.py` (`api_console_biofield_reveals` + new `_people_brief`)
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- `_people_brief(cx, email) -> {"client_name": str, "tags": [str, ...]}` - best-effort `people` lookup; empty fields on miss/error.
- `api_console_biofield_reveals` returns `{"drafts": [...], "approved": [...]}` where each item now also has `client_name` and `tags`.

- [ ] **Step 1: Add the failing test** (append to `tests/test_biofield_layers.py`)

```python
def test_console_list_enriches_name_and_tags(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, email TEXT, name TEXT, tags TEXT)")
        cx.execute("INSERT INTO people (email, name, tags) VALUES (?,?,?)",
                   ("c@x.com", "Jane Client", '["e4l account", "type:client"]'))
        br.upsert(cx, "c@x.com", "2026-06-01", {"greeting": "hi"},
                  [{"name": "Top", "slug": "top", "meaning": "m"}], "s")
        br.upsert(cx, "nobody@x.com", "2026-06-02", {"greeting": "hi"}, [], "s")
        cx.commit()
    c = app_module.app.test_client()
    body = c.get("/api/console/biofield-reveals", headers={"X-Console-Key": "sek"}).get_json()
    by_email = {d["email"]: d for d in body["drafts"]}
    assert by_email["c@x.com"]["client_name"] == "Jane Client"
    assert by_email["c@x.com"]["tags"] == ["e4l account", "type:client"]
    # no people row -> empty, still listed, no error
    assert by_email["nobody@x.com"]["client_name"] == ""
    assert by_email["nobody@x.com"]["tags"] == []
```

- [ ] **Step 2: Run -> FAIL**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_console_list_enriches_name_and_tags -v`
Expected: FAIL (`client_name` KeyError).

- [ ] **Step 3: Add `_people_brief`** (`app.py`, near `api_console_biofield_reveals`)

```python
def _people_brief(cx, email):
    """Best-effort name + tags for a reveal's owner email. Empty on miss/error."""
    out = {"client_name": "", "tags": []}
    try:
        row = cx.execute("SELECT name, tags FROM people WHERE lower(email)=lower(?)",
                         ((email or "").strip(),)).fetchone()
        if row:
            out["client_name"] = (row[0] or "").strip()
            try:
                t = json.loads(row[1] or "[]")
                out["tags"] = [str(x) for x in t] if isinstance(t, list) else []
            except Exception:
                out["tags"] = []
    except Exception:
        pass
    return out
```

- [ ] **Step 4: Enrich the listing** - in `api_console_biofield_reveals`, replace the body after loading drafts/approved:

```python
    from dashboard import biofield_reveals as _br
    with sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        drafts = _br.list_pending(cx)
        approved = _br.list_approved(cx)
        for d in drafts + approved:
            d.update(_people_brief(cx, d.get("email")))
    return jsonify({"drafts": drafts, "approved": approved})
```

- [ ] **Step 5: Run -> PASS**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_biofield_layers.py
git commit -m "feat: enrich biofield reveal console list with client name + tags"
```

---

## Task 4 (deploy-chat): console page header, stress factors, edit-preservation, delete

**Files:**
- Modify: `static/console-biofield-reveals.html`
- Test: `tests/test_biofield_layers.py` (serve markers)

**Interfaces:** consumes `d.id`, `d.scan_date`, `d.client_name`, `d.tags`, and each `L.pattern_labels` / `L.patterns` from Task 3's payload.

- [ ] **Step 1: Add the failing serve test** (append to `tests/test_biofield_layers.py`)

```python
def test_console_page_has_enrichment_markers(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    html = app_module.app.test_client().get(
        "/console/biofield-reveals", headers={"X-Console-Key": "sek"}).data.decode()
    assert "client_name" in html
    assert "pattern_labels" in html
    assert "stress-factors" in html
    assert "biofield_reveal.delete" in html
```

- [ ] **Step 2: Run -> FAIL**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_console_page_has_enrichment_markers -v`
Expected: FAIL (markers absent).

- [ ] **Step 3: Header - id, scan date, name, tags** - in `buildCard`, replace the `meta` block (the `var meta ... wrap.appendChild(meta);` section) with:

```javascript
  var meta = document.createElement('div');
  meta.className = 'meta';
  var idSpan = document.createElement('span');
  idSpan.textContent = '#' + (d.id != null ? d.id : '?') + '  ';
  var emailSpan = document.createElement('span');
  emailSpan.textContent = (d.email || '') + ' - ' + (d.scan_date || '');
  meta.appendChild(idSpan);
  meta.appendChild(emailSpan);
  if (d.client_name) {
    var nameSpan = document.createElement('span');
    nameSpan.textContent = '  -  ' + d.client_name;
    meta.appendChild(nameSpan);
  }
  wrap.appendChild(meta);
  var tags = Array.isArray(d.tags) ? d.tags : [];
  if (tags.length) {
    var tagRow = document.createElement('div');
    tagRow.className = 'client-tags';
    tagRow.textContent = 'Tags: ' + tags.join(', ');
    wrap.appendChild(tagRow);
  }
```

- [ ] **Step 4: Stress factors per layer + stash for edit-preservation** - inside the `layers.forEach(function(L, idx){ ... })` block, right after the summary textarea is appended (after `row.appendChild(summaryArea);`), insert:

```javascript
      // Read-only stress factors (console only; never shown to the member).
      var factors = Array.isArray(L.pattern_labels) && L.pattern_labels.length
        ? L.pattern_labels
        : (Array.isArray(L.patterns) ? L.patterns : []);
      if (factors.length) {
        var sf = document.createElement('div');
        sf.className = 'stress-factors';
        sf.textContent = 'Stress factors: ' + factors.join(', ');
        row.appendChild(sf);
      }
      // Stash the originals so Save (collectLayers) preserves them.
      row.dataset.patterns = JSON.stringify(L.patterns || []);
      row.dataset.patternLabels = JSON.stringify(L.pattern_labels || []);
```

- [ ] **Step 5: Preserve patterns/labels in `collectLayers`** - in `collectLayers`, add the stashed fields to each pushed object:

```javascript
  remedyRows.forEach(function(row, idx){
    var titleEl = row.querySelector('.layer-title-field');
    var summaryEl = row.querySelector('.layer-summary-field');
    var nameEl = row.querySelector('.remedy-name');
    var slugEl = row.querySelector('.remedy-slug');
    var meaningEl = row.querySelector('.remedy-meaning');
    var rememberEl = row.querySelector('.remedy-remember');
    var patterns = [];
    var patternLabels = [];
    try { patterns = JSON.parse(row.dataset.patterns || '[]'); } catch (e) {}
    try { patternLabels = JSON.parse(row.dataset.patternLabels || '[]'); } catch (e) {}
    out.push({
      n: idx + 1,
      title: titleEl ? titleEl.value : '',
      summary: summaryEl ? summaryEl.value : '',
      patterns: patterns,
      pattern_labels: patternLabels,
      remedy: {
        name: nameEl ? nameEl.value : '',
        slug: slugEl ? slugEl.value : '',
        meaning: meaningEl ? meaningEl.value : '',
        remember: rememberEl ? rememberEl.checked : false
      }
    });
  });
```

- [ ] **Step 6: Delete button + `doDelete`** - in `buildCard`, after the `approveBtn` is appended to its `row`, add a delete button:

```javascript
  var deleteBtn = document.createElement('button');
  deleteBtn.className = 'btn ghost';
  deleteBtn.textContent = 'Delete';
  deleteBtn.onclick = function(){ doDelete(wrap, d.id, deleteBtn); };
  row.appendChild(deleteBtn);
```

And add the `doDelete` function next to `doApprove`:

```javascript
async function doDelete(card, id, btn){
  if(!window.confirm('Delete reveal #' + id + '? This cannot be undone.')) return;
  var statusEl = card.querySelector('.status');
  statusEl.textContent = 'Deleting...';
  btn.disabled = true;
  var r = await api('POST', '/api/action/biofield_reveal.delete', { id: id });
  if(r.ok && r.json.status === 'done'){
    statusEl.textContent = 'Deleted.';
    setTimeout(loadList, 600);
  } else {
    btn.disabled = false;
    statusEl.textContent = 'Delete failed (HTTP ' + r.status + ').';
  }
}
```

- [ ] **Step 7: Run -> PASS**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS (serve markers present; all prior tests green).

- [ ] **Step 8: Commit**

```bash
git add static/console-biofield-reveals.html tests/test_biofield_layers.py
git commit -m "feat: reveal console shows id/name/tags + per-layer stress factors + delete"
```

---

## Verification

- Per task: the named pytest target passes (vault for Task 1; doppler+venv for Tasks 2-4); then `tests/test_biofield_layers.py` + `tests/test_biofield_trial.py` + `tests/test_biofield_cart.py` for no regression.
- Final whole-branch review (most capable model). Focus: member anti-leak (no patterns/labels in the member payload); edits preserve `patterns`/`pattern_labels` (collectLayers round-trip + `_exec_edit` keeps non-remedy fields); enrichment best-effort (no people row -> empty, never raises); delete idempotent + console-gated; XSS-safe rendering (textContent only); no emoji/em-dash.
- Ship Tasks 2-4 via one PR + merge to main (auto-deploys); gentle `/begin` + `/console/biofield-reveals` probe per the warm-up rule.
- **Post-deploy operational:** re-push id=6 from the vault (`--push`) to backfill its `pattern_labels`; then in the console, list the leftover test drafts, confirm their ids with Glen, and Delete them.

## Build order

Task 1 (vault bridge) can land first or in parallel; Tasks 2-4 are the deploy-chat PR (store/action -> endpoint -> page). The member anti-leak test (Task 2) guards the whole change.
