# Biofield Editor — Stress-Pattern Findings + Unbalanced Workspace — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show, per layer, the stress findings each remedy balances; list the unbalanced stresses in a bottom panel; let Glen select unbalanced stresses and add a layer to balance them.

**Architecture:** The local synthesis carries a `content.findings` list (code/name/description/rank/clinical_notes) into portal content; the editor (one HTML file) renders per-layer balanced stresses + an unbalanced panel derived live from layer membership, and adds a layer for selected unbalanced stresses. Synthesis is pure + unit-tested in the vault; the editor is JS (node --check + the Python served-page suite).

**Tech Stack:** Python (vault `e4l_synthesis.py` + `e4l-portal-import.py`), JS/HTML (deploy-chat `static/console-biofield-portal.html`). Spec: `docs/superpowers/specs/2026-06-17-editor-stress-findings-design.md`.

---

## Two repos
- **VAULT** `~/AI-Training/02 Skills/` (Tasks 1–2, auto-snapshot, no PR). Vault tests:
  `cd "/Users/remedymatch/AI-Training/02 Skills" && PYTHONPATH=. ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_e4l_synthesis.py -q`
- **DEPLOY-CHAT** worktree `/tmp/wt-deploy-chat-5326cc61` branch `sess/5326cc61-findings` (Tasks 3–5, PR). Suite:
  `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`

**Finding shape (consistent everywhere):** `{code, name, description, rank, clinical_notes}` — `name` from `full_name`, `description` from `e4l_description`, `rank` from `priority_rank`. A finding is **balanced** if its `code` is in some layer's `patterns`.

---

## Task 1: synthesis carries findings — `build_findings` + `to_portal_content`

**Files:** Modify `02 Skills/e4l_synthesis.py`; Test `02 Skills/tests/test_e4l_synthesis.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_build_findings_and_to_portal_content_carries_them():
    from e4l_synthesis import build_findings, to_portal_content
    patterns = [
        {"item_code": "ED4", "full_name": "Nerve Driver", "name": "Nerve",
         "e4l_description": "supports nerve impulses", "priority_rank": 1, "clinical_notes": ""},
        {"item_code": "EI8", "full_name": "Microbes-Liver Integrator", "name": "Microbes-Liver",
         "e4l_description": "links microbial terrain + liver", "priority_rank": 2, "clinical_notes": "x"},
    ]
    findings = build_findings(patterns)
    assert findings[0] == {"code": "ED4", "name": "Nerve Driver",
                           "description": "supports nerve impulses", "rank": 1, "clinical_notes": ""}
    assert findings[1]["clinical_notes"] == "x"
    content = to_portal_content({"greeting": "Aloha", "layers": []}, [], findings=findings)
    assert content["findings"] == findings
    assert "findings" not in to_portal_content({"greeting": "A", "layers": []}, [])
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — add `build_findings` to `e4l_synthesis.py`:

```python
def build_findings(patterns):
    """Scan patterns -> editor display findings: code, readable name, plain-English
    description, priority rank, clinical notes. Order preserved (already by rank)."""
    out = []
    for p in patterns or []:
        code = (p.get("item_code") or "").strip()
        if not code:
            continue
        out.append({
            "code": code,
            "name": (p.get("full_name") or p.get("name") or code).strip(),
            "description": (p.get("e4l_description") or "").strip(),
            "rank": p.get("priority_rank"),
            "clinical_notes": (p.get("clinical_notes") or "").strip(),
        })
    return out
```

  Then add a `findings=None` parameter to `to_portal_content` (extend its signature `def to_portal_content(synth, catalog, dosing_map=None, overrides=None, findings=None):`) and, where it currently does `return {"greeting": ...}`, build the dict into a variable and attach findings:

```python
    content = {"greeting": synth.get("greeting", ""),
               "video": {"url": "", "label": "Watch your message from Dr. Glen"},
               "layers": layers, "reorder_items": reorder, "pricing_note": ""}
    if findings:
        content["findings"] = findings
    return content
```

- [ ] **Step 4: Run → PASS.** **Step 5:** full vault file green. **Step 6:** No git commit (vault auto-snapshots).

---

## Task 2: importer passes findings — `e4l-portal-import.py`

**Files:** Modify `02 Skills/e4l-portal-import.py` (vault); no unit test (py_compile + Task 5 validates)

- [ ] **Step 1:** In `main()`, the importer already computes `patterns = E.pull_patterns(...)` and builds `content = E.to_portal_content(synth, catalog, dosing_map=dosing_map, overrides=overrides)`. Add `findings`:

```python
    content = E.to_portal_content(synth, catalog, dosing_map=dosing_map,
                                  overrides=overrides, findings=E.build_findings(patterns))
```

- [ ] **Step 2:** Syntax check (do NOT run live):
```
/Library/Developer/CommandLineTools/usr/bin/python3 -m py_compile "/Users/remedymatch/AI-Training/02 Skills/e4l-portal-import.py" && echo "py OK"
```

- [ ] **Step 3:** No git commit (vault auto-snapshots).

---

## Task 3: editor — findings storage + per-layer stresses + unbalanced panel + persist

**Files:** Modify `static/console-biofield-portal.html` (deploy-chat); no unit test (node --check + suite + manual)

- [ ] **Step 1: CSS.** Before `</style>`, add:

```css
  .stresses{margin-top:8px;border-top:1px dashed var(--border);padding-top:6px}
  .slabel{color:var(--muted);font-size:.74rem;margin-bottom:3px}
  .finding{font-size:13px;padding:3px 0}
  .fhead{cursor:pointer} .fcode{color:var(--gold);font-weight:700}
  .fx{color:var(--muted);margin-left:4px;font-size:.8em}
  .fdesc{display:none;color:var(--muted);font-size:.86em;margin:2px 0 4px 12px;white-space:pre-wrap}
  .finding.open .fdesc{display:block}
  #unbalanced .finding .fhead input{margin-right:6px;width:auto}
```

- [ ] **Step 2: The unbalanced panel element.** Immediately AFTER the "Healing path layers" card (the `<div class="card">` containing `<div id="layers">` and the "+ Add layer" button), insert:

```html
    <div id="unbalanced" class="card" style="display:none"></div>
```

- [ ] **Step 3: Findings state + render.** Near the other top-level `let` declarations (e.g. by `let CAT_BY_NAME = {};`), add `let FINDINGS = {}; let FINDINGS_LIST = [];`. Add these functions (place by `populate`):

```javascript
function findingLine(f, checkbox){
  const cb = checkbox ? `<input type="checkbox" class="ufx" value="${esc(f.code)}">` : '';
  const note = f.clinical_notes ? ('\n' + f.clinical_notes) : '';
  return `<div class="finding"><div class="fhead" onclick="this.parentNode.classList.toggle('open')">`
       + `${cb}<span class="fcode">[${esc(f.code)}]</span> ${esc(f.name||'')}`
       + `${f.rank!=null?(' · #'+f.rank):''}<span class="fx">▾</span></div>`
       + `<div class="fdesc">${esc((f.description||'(no description)') + note)}</div></div>`;
}
function renderFindings(){
  const layers = [...$('layers').children];
  if(!FINDINGS_LIST.length){
    layers.forEach(el=>{ const b=el.querySelector('.stresses'); if(b) b.remove(); });
    $('unbalanced').style.display='none'; $('unbalanced').innerHTML=''; return;
  }
  const balanced = new Set();
  layers.forEach(el=>{
    const codes = JSON.parse(el.dataset.patterns||'[]');
    codes.forEach(c=>balanced.add(c));
    let box = el.querySelector('.stresses');
    if(!box){ box=document.createElement('div'); box.className='stresses'; el.appendChild(box); }
    box.innerHTML = codes.length
      ? '<div class="slabel">Stresses this balances:</div>'
        + codes.map(c=>FINDINGS[c]?findingLine(FINDINGS[c],false):'').join('')
      : '';
  });
  const unb = FINDINGS_LIST.filter(f=>!balanced.has(f.code));
  const panel = $('unbalanced');
  if(!unb.length){ panel.style.display='none'; panel.innerHTML=''; return; }
  panel.style.display='block';
  panel.innerHTML = `<h2>Unbalanced stresses (${unb.length})</h2>`
    + unb.map(f=>findingLine(f,true)).join('')
    + `<div style="margin-top:.6rem">`
    + `<button class="btn ghost sm" onclick="addLayerForSelected()">Add a layer for selected</button> `
    + `<button class="btn ghost sm" onclick="document.querySelectorAll('.ufx').forEach(c=>c.checked=true)">Select all</button></div>`;
}
```

- [ ] **Step 4: Store findings on load + render.** In `populate(d)`, after the layers are added (after the `(d.layers||[]).forEach(addLayer);` line and the reitems line), add:

```javascript
  FINDINGS_LIST = d.findings || []; FINDINGS = {};
  FINDINGS_LIST.forEach(f=>{ FINDINGS[f.code]=f; });
  renderFindings();
```

- [ ] **Step 5: Persist findings through publish.** In `buildContent()`, before `return c;`, add:

```javascript
  if(FINDINGS_LIST.length) c.findings = FINDINGS_LIST;
```

- [ ] **Step 6: Verify.** Extract the main `<script>`, `node --check` it. Run the full Python suite (served-page test still 200): baseline green. Manual note for reviewer: load a portal that has `findings` → per-layer "Stresses this balances" lines appear, click expands the description; the bottom "Unbalanced stresses (N)" panel lists the rest.

- [ ] **Step 7: Commit**
```bash
cd /tmp/wt-deploy-chat-5326cc61
git add static/console-biofield-portal.html
git commit -m "editor: stress-pattern findings — per-layer + unbalanced panel"
```

---

## Task 4: editor — "Add a layer for selected" + refresh on layer changes

**Files:** Modify `static/console-biofield-portal.html`; no unit test (node --check + suite + manual)

- [ ] **Step 1: The add action.** Add (by `renderFindings`):

```javascript
function addLayerForSelected(){
  const codes = [...document.querySelectorAll('.ufx:checked')].map(c=>c.value);
  if(!codes.length){ setStatus('Select one or more stresses first.', false); return; }
  addLayer({patterns: codes, title: '', meaning: ''});
  renderFindings();
}
```

(`addLayer(L)` already sets `el.dataset.patterns` from `L.patterns`, so the new layer owns those codes → they leave the unbalanced panel and show under the new layer. Glen then types its remedy.)

- [ ] **Step 2: Refresh on delete.** The layer delete button is inline `onclick="this.closest('.layer').remove()"`. Change it (in `addLayer`'s template) to also refresh:

```html
      <button class="x" onclick="this.closest('.layer').remove(); renderFindings()">delete</button>
```

- [ ] **Step 3: Refresh after a manual "+ Add layer".** The "+ Add layer" button calls `addLayer()`. Make it refresh too — change that button to `onclick="addLayer(); renderFindings()"` (so a manually-added empty layer participates; it has no patterns, so it changes nothing but keeps state consistent).

- [ ] **Step 4: Verify.** `node --check` the main script. Run the full Python suite → green. Manual for reviewer: check one or more unbalanced stresses → "Add a layer for selected" → a new layer appears pre-loaded with those stresses, and they disappear from the unbalanced panel; deleting that layer returns them to the panel.

- [ ] **Step 5: Commit**
```bash
cd /tmp/wt-deploy-chat-5326cc61
git add static/console-biofield-portal.html
git commit -m "editor: add-a-layer-for-selected unbalanced stresses + live refresh"
```

---

## Task 5: validate on Othon + finish

**Files:** none (validation + PR)

- [ ] **Step 1:** Dry re-synth (no publish) to confirm `content.findings` is produced:
  `cd "/Users/remedymatch/AI-Training" && doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python "02 Skills/e4l-portal-import.py" --email backdoc.molina@gmail.com`
  Then check the written seed JSON in `~/AI-Training/05 Clients/`: `content.findings` is a non-empty list of `{code,name,description,rank,clinical_notes}`, and each layer's `patterns` codes appear among the findings' codes.
- [ ] **Step 2:** Push the branch + open a PR:
```bash
cd /tmp/wt-deploy-chat-5326cc61 && git push -u origin sess/5326cc61-findings
gh pr create --base main --title "Biofield editor: stress-pattern findings + unbalanced workspace" --body "Per-layer balanced stresses (compact + expandable) + a bottom unbalanced-stresses panel with 'Add a layer for selected'. Synthesis carries content.findings (vault, auto-snapshot); editor is this PR. Legacy reports (no findings) render unchanged."
```
- [ ] **Step 3:** Live smoke after deploy (manual): load a client whose latest scan was synthesized post-change → per-layer stresses + unbalanced panel render; add-a-layer works; publish preserves findings.

---

## Self-Review notes
- **Spec coverage:** `content.findings` data model (T1); synthesis/importer carry it (T1–T2); per-layer balanced stresses compact+expand (T3); unbalanced panel + count + expand (T3); add-a-layer-for-selected + live balanced/unbalanced derivation (T4); persist through publish (T3 step 5); back-compat no-findings (T3 `renderFindings` early-out); Othon validation (T5). Editor-scoped; client-portal display correctly out of scope.
- **Type consistency:** finding `{code,name,description,rank,clinical_notes}` identical in `build_findings`, `to_portal_content`, and the editor's `FINDINGS`/`findingLine`; `to_portal_content(synth, catalog, dosing_map=None, overrides=None, findings=None)`; layer `patterns` (codes) is the balanced/unbalanced key; `FINDINGS_LIST` persisted verbatim.
- **Verify during impl:** the existing `to_portal_content` tests still pass (findings is additive/optional); the manual "+ Add layer" button's current `onclick` text (T4 step 3) — match what's there; `buildContent` returns `c` (confirm the variable name) before adding `c.findings`.
