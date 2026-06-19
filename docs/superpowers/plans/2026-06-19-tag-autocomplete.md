# Tag Autocomplete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a contains-match autocomplete dropdown to the People detail-card add-tag input.

**Architecture:** A pure `distinct_tags` helper aggregates all tags in use; a CONSOLE_SECRET-gated `GET /api/people/tags` exposes them; the frontend fetches the list once, filters client-side to tags containing the typed substring (excluding ones the person already has), and shows a click-to-add dropdown. Builds directly on the PR #183 tag editor helpers in `static/console.html`.

**Tech Stack:** Python 3.11 (Flask, sqlite3), vanilla JS in `static/console.html`, pytest.

**Worktree:** `/tmp/wt-deploy-chat-5f407e90`, branch `sess/5f407e90-tag-autocomplete` (off `origin/main`).

## Global Constraints

- **Test invocation:** pure tests — `~/.venvs/deploy-chat311/bin/python -m pytest <file>`; app/endpoint tests — `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <file>` (DATA_DIR after `--`).
- **Auth:** new route gated by `CONSOLE_SECRET` (`X-Console-Key` header or `?key=`), matching `/api/people/<id>/tags`.
- **No emoji** in UI (portal convention).
- **Additive:** no migration, no feature flag. Ships on merge to main.
- **Reuse** the existing `static/console.html` helpers from PR #183: `_renderTagEditor(id, tags)`, `_postTagDelta(id, add, remove)`, `_rerenderTags(id, tags)`, `addPersonTag(id)`, `_esc`, `_ptagCls`, globals `BASE`, `consoleKey`.

## File Structure

- **Modify** `dashboard/people.py` — add pure `distinct_tags(tag_lists)` beside `set_person_tags`.
- **Modify** `tests/test_people_tags.py` — add `distinct_tags` unit tests.
- **Modify** `app.py` — add `distinct_tags` to the existing `from dashboard.people import ...`; add `GET /api/people/tags` route beside `POST /api/people/<id>/tags`.
- **Modify** `tests/test_people_tags_api.py` — add endpoint tests.
- **Modify** `static/console.html` — track the detail card's current tags; add fetch+cache, the `oninput` filter, dropdown render/dismiss, and click-to-add; wire the input built in `_renderTagEditor`.

---

### Task 1: `distinct_tags` pure helper

**Files:**
- Modify: `dashboard/people.py`
- Test: `tests/test_people_tags.py`

**Interfaces:**
- Produces: `distinct_tags(tag_lists) -> list[str]`. Input is an iterable where each item is either a `list[str]` or a JSON-encoded string of one; output is the de-duplicated, stripped, sorted-ascending (case-sensitive) union.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_people_tags.py`)

```python
from dashboard.people import distinct_tags


def test_distinct_union_and_sorted():
    rows = [["type:client", "OD"], ["type:client", "tier:pro-influencer"]]
    assert distinct_tags(rows) == ["OD", "tier:pro-influencer", "type:client"]


def test_distinct_accepts_json_strings():
    rows = ['["a", "b"]', '["b", "c"]']
    assert distinct_tags(rows) == ["a", "b", "c"]


def test_distinct_mixed_list_and_string():
    rows = [["a"], '["b"]']
    assert distinct_tags(rows) == ["a", "b"]


def test_distinct_skips_malformed_json():
    rows = ['["a"]', "not json", None, 5]
    assert distinct_tags(rows) == ["a"]


def test_distinct_strips_and_drops_empty():
    rows = [["  a  ", "", "   "]]
    assert distinct_tags(rows) == ["a"]


def test_distinct_case_sensitive_ascii_order():
    # uppercase sorts before lowercase in ASCII
    assert distinct_tags([["od", "OD"]]) == ["OD", "od"]


def test_distinct_empty():
    assert distinct_tags([]) == []
```

- [ ] **Step 2: Run to verify fail**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags.py -q -k distinct`
Expected: FAIL — `ImportError: cannot import name 'distinct_tags'`.

- [ ] **Step 3: Implement** (append to `dashboard/people.py`)

```python
import json as _json


def distinct_tags(tag_lists):
    """Union of all tags across people, de-duplicated, stripped, sorted ascending.

    Each item in tag_lists is either a list[str] or a JSON-encoded string of one;
    malformed/None/non-iterable entries are skipped, as are empty/whitespace tags.
    """
    out = set()
    for entry in tag_lists:
        if isinstance(entry, str):
            try:
                entry = _json.loads(entry)
            except (ValueError, TypeError):
                continue
        if not isinstance(entry, list):
            continue
        for t in entry:
            if isinstance(t, str):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags.py -q`
Expected: PASS (existing 12 + new 7).

- [ ] **Step 5: Commit**

```bash
git add dashboard/people.py tests/test_people_tags.py
git commit -m "feat: distinct_tags pure helper for tag autocomplete"
```

---

### Task 2: `GET /api/people/tags` endpoint

**Files:**
- Modify: `app.py` (import line `from dashboard.people import set_person_tags`; new route beside `update_person_tags_route`)
- Test: `tests/test_people_tags_api.py`

**Interfaces:**
- Consumes: `distinct_tags` (Task 1).
- Produces: `GET /api/people/tags` → `200 {"tags": [<sorted distinct>]}`; `401` without key.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_people_tags_api.py`)

```python
def test_tags_list_requires_key(client, db_path):
    _seed(db_path, "a@b.com", ["type:client"])
    r = client.get("/api/people/tags")
    assert r.status_code == 401


def test_tags_list_returns_sorted_distinct(client, db_path):
    _seed(db_path, "a@b.com", ["type:client", "OD"])
    _seed(db_path, "b@b.com", ["type:client", "tier:pro-influencer"])
    r = client.get("/api/people/tags", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    assert r.get_json()["tags"] == ["OD", "tier:pro-influencer", "type:client"]
```

NOTE: the route path `/api/people/tags` must be registered so Flask's `<int:person_id>` converter on `/api/people/<int:person_id>` does NOT shadow it — `tags` is not an int, so `/api/people/<int:person_id>` won't match `/api/people/tags`; the static route wins. No ordering hazard.

- [ ] **Step 2: Run to verify fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags_api.py -q -k tags_list`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Implement**

Update the import in `app.py`:

```python
from dashboard.people import set_person_tags, distinct_tags
```

Add the route immediately before `update_person_tags_route` (so it reads top-down with the other people routes):

```python
@app.route("/api/people/tags", methods=["GET"])
def list_person_tags_route():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("SELECT tags FROM people").fetchall()
    return jsonify({"tags": distinct_tags([r[0] for r in rows])})
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags_api.py tests/test_people_tags.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_people_tags_api.py
git commit -m "feat: GET /api/people/tags distinct-tags endpoint"
```

---

### Task 3: Autocomplete dropdown UI

**Files:**
- Modify: `static/console.html` (CSS near `.tag-add`; JS near `_renderTagEditor`; the input markup inside `_renderTagEditor`)

**Interfaces:**
- Consumes: `GET /api/people/tags`; existing `_renderTagEditor`, `_postTagDelta`, `_rerenderTags`, `addPersonTag`, `_esc`, `consoleKey`, `BASE`.

- [ ] **Step 1: Add CSS** (after the `.tag-add button` rule added in PR #183)

```css
  .tag-suggest { position:relative; display:inline-block; }
  .tag-suggest-list { position:absolute; left:0; top:100%; z-index:50; min-width:160px; max-height:200px; overflow-y:auto; background:var(--bg, #111); border:1px solid var(--border); border-radius:4px; margin-top:2px; box-shadow:0 4px 12px rgba(0,0,0,.4); }
  .tag-suggest-list .tsi { padding:4px 8px; font-size:11px; cursor:pointer; white-space:nowrap; }
  .tag-suggest-list .tsi:hover { background:var(--border); }
```

- [ ] **Step 2: Track current detail tags + wrap the input in a suggest container**

In `_renderTagEditor(id, tags)`, record the current tags and wrap the input so the dropdown can anchor to it. Replace the function body's `return` so the `.tag-add` becomes:

```javascript
function _renderTagEditor(id, tags) {
  _detailTags = tags.slice();
  const chips = tags.map(t =>
    `<span class="ptag ${_ptagCls(t)}">${_esc(t)}<button class="ptag-x" title="Remove" onclick="removePersonTag(${id}, '${_escAttr(t)}')">×</button></span>`
  ).join('');
  return `${chips}<span class="tag-add"><span class="tag-suggest"><input id="tag-add-input" type="text" placeholder="e.g. tier:pro-influencer" autocomplete="off" oninput="_onTagInput(${id})" onkeydown="if(event.key==='Enter'){event.preventDefault();addPersonTag(${id});}else if(event.key==='Escape'){_hideTagSuggest();}" onblur="setTimeout(_hideTagSuggest, 150)"><div class="tag-suggest-list" id="tag-suggest-list" style="display:none"></div></span><button onclick="addPersonTag(${id})">Add</button></span>`;
}
```

Add a module-level declaration near the other tag-editor state (top of the same `<script>`, beside other `let` globals):

```javascript
let _detailTags = [];
let _allTagsCache = null;
```

- [ ] **Step 3: Add fetch/cache + filter + dropdown handlers** (near `_renderTagEditor`)

```javascript
async function _fetchAllTags() {
  if (_allTagsCache) return _allTagsCache;
  try {
    const res = await fetch(`${BASE}/api/people/tags`, { headers:{'X-Console-Key':consoleKey} });
    _allTagsCache = res.ok ? ((await res.json()).tags || []) : [];
  } catch { _allTagsCache = []; }
  return _allTagsCache;
}
function _hideTagSuggest() {
  const el = document.getElementById('tag-suggest-list');
  if (el) el.style.display = 'none';
}
async function _onTagInput(id) {
  const inp = document.getElementById('tag-add-input');
  const list = document.getElementById('tag-suggest-list');
  if (!inp || !list) return;
  const q = (inp.value || '').trim().toLowerCase();
  if (!q) { list.style.display = 'none'; return; }
  const all = await _fetchAllTags();
  const have = new Set(_detailTags);
  const matches = all.filter(t => t.toLowerCase().includes(q) && !have.has(t)).slice(0, 10);
  if (!matches.length) { list.style.display = 'none'; return; }
  list.innerHTML = matches.map(t =>
    `<div class="tsi" onmousedown="event.preventDefault();_pickTagSuggest(${id}, '${_escAttr(t)}')">${_esc(t)}</div>`
  ).join('');
  list.style.display = 'block';
}
async function _pickTagSuggest(id, tag) {
  _hideTagSuggest();
  const tags = await _postTagDelta(id, [tag], []);
  if (tags) { _allTagsCache = null; _rerenderTags(id, tags); }
}
```

NOTE on `onmousedown`+`preventDefault`: the input's `onblur` hides the list after 150ms; using `mousedown` (which fires before blur) plus `preventDefault` keeps the click from being lost to the blur-hide.

- [ ] **Step 4: Clear cache after freeform adds**

In the existing `addPersonTag(id)` (PR #183), after a successful add, invalidate the cache so a brand-new tag appears in later lookups. Change its success branch:

```javascript
async function addPersonTag(id) {
  const inp = document.getElementById('tag-add-input');
  const val = (inp && inp.value || '').trim();
  if (!val) return;
  const tags = await _postTagDelta(id, [val], []);
  if (tags) { _allTagsCache = null; _hideTagSuggest(); _rerenderTags(id, tags); }
}
```

- [ ] **Step 5: JS syntax check**

```bash
~/.venvs/deploy-chat311/bin/python - <<'PY'
import re
html=open('static/console.html').read()
scripts=re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html, re.S)
open('/tmp/console_extracted.js','w').write("\n;\n".join(scripts))
print("blocks:", len(scripts))
PY
node --check /tmp/console_extracted.js && echo "JS OK"
```
Expected: `JS OK`.

- [ ] **Step 6: Manual verify** (running console)

People tab → person → focus the add-tag input → type `tier` → dropdown shows existing `tier:*` tags not already on the person → click one → it's added (chip appears), dropdown closes, persists on reselect. Type a brand-new tag + Enter → added, and it appears in the dropdown next time you type its substring.

- [ ] **Step 7: Commit**

```bash
git add static/console.html
git commit -m "feat: contains-match tag autocomplete dropdown on People detail card"
```

---

## Verification (end-to-end)

1. **Tests:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags.py tests/test_people_tags_api.py -q` — all pass.
2. **Endpoint (deployed, after merge):** `curl -s "https://illtowell.com/api/people/tags" -H "X-Console-Key: $CONSOLE_SECRET"` → JSON `{"tags":[...]}`; without key → 401.
3. **UI:** type-to-filter dropdown, click-to-add, freeform add still works, no dropdown when input empty.

## Rollout

Merge `sess/5f407e90-tag-autocomplete` to main via PR. Additive, no flag/migration; live on deploy (Doppler-synced).
