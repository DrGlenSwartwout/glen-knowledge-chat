# Product Page Images — Phase C3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An LLM authors new, distinct prompt variations into a `review` state; Glen edits/approves them in the console (→ `candidate`, C2-eligible) or rejects them; on-demand + a daily bench top-up keep the prompt pool fed — closing the evolution loop.

**Architecture:** A new generator module (`dashboard/sales_image_prompt_gen.py`) calls Claude (injectable for tests) for N distinct scene prompts grounded in the scene rules + leaderboard winners, parses JSON robustly, and inserts them as `review` variations. Review actions flip state (approve→candidate / reject→retired / edit). A console section + 2 routes + a daily top-up job wire it in. All under `SALES_PAGES_IMAGE_EVOLUTION`, dark.

**Tech Stack:** Python 3 / Flask, SQLite (`LOG_DB`), anthropic SDK (existing), APScheduler (existing), pytest.

## Global Constraints

- Reuses the existing flag `SALES_PAGES_IMAGE_EVOLUTION` → `_SALES_IMAGE_EVOLUTION_ENABLED` (no new flag). OFF = no generation, no candidates, routes 400, leaderboard page omits the prompt-review section, top-up no-ops.
- Generated prompts land in a NEW registry state `review` — NOT C2-eligible until approved (→ `candidate`). Reject → `retired`.
- Stored `prompt_template`s carry NO product/brand names and NO text/letters instructions (the no-text rule is appended later by Phase A's `build_generation_jobs`). Seeds + candidates already follow this.
- LLM: `anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))` + `cli.messages.create(model=_MODEL, max_tokens=…, messages=[{"role":"user","content":prompt}])`; `_MODEL = os.environ.get("IMAGE_PROMPT_GEN_MODEL", "claude-haiku-4-5-20251001")`. The LLM call is an **injectable** `llm(prompt)->str` so tests never hit the network.
- Top-up threshold: generate when a kind's `(candidate + review)` count `< 2`.
- Tests: pytest, `sqlite3.connect(":memory:")` per test, import `dashboard.*` directly, no live network (inject `llm`/`generate`). New file `tests/test_sales_pages_phase_c3.py`. Follow `tests/test_sales_pages_phase4b.py` style.
- Sandbox: use `python3` (no `python`). `import app` CANNOT run here (Pinecone at import) — verify app.py edits with `python3 -m py_compile app.py` + the unit-tested helpers.
- Work in worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at C2 tip `edfb8e6` which is in `main`). Commit per task. No edits to `main`.

---

### Task 1: Registry — review state helpers

**Files:**
- Modify: `dashboard/sales_prompt_variations.py`
- Test: `tests/test_sales_pages_phase_c3.py`

**Interfaces:**
- Produces: `review_variations(cx, kind) -> [dict{id,kind,label,prompt_template}]`; `insert_variation(cx, kind, label, prompt_template, state="review") -> int` (new id).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase_c3.py
import sqlite3
from dashboard import sales_prompt_variations as pv

def _cx(): return sqlite3.connect(":memory:")

def test_insert_and_review_variations():
    cx = _cx()
    vid = pv.insert_variation(cx, "botanical", "lbl", "a fresh herb scene")
    assert isinstance(vid, int)
    revs = pv.review_variations(cx, "botanical")
    assert [r["id"] for r in revs] == [vid]
    assert revs[0]["label"] == "lbl" and revs[0]["prompt_template"] == "a fresh herb scene"
    pv.set_state(cx, vid, "candidate")               # set_state from C2
    assert pv.review_variations(cx, "botanical") == []
    assert vid in {v["id"] for v in pv.candidate_variations(cx, "botanical")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: FAIL (`insert_variation`/`review_variations` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_prompt_variations.py`:

```python
def review_variations(cx, kind):
    init_table(cx)
    rows = cx.execute("SELECT id, kind, label, prompt_template FROM sales_prompt_variations "
                      "WHERE kind=? AND state='review' ORDER BY id", (kind,)).fetchall()
    return [{"id": r[0], "kind": r[1], "label": r[2], "prompt_template": r[3]} for r in rows]

def insert_variation(cx, kind, label, prompt_template, state="review"):
    init_table(cx)
    cur = cx.execute("INSERT INTO sales_prompt_variations (kind, label, prompt_template, state, created_at) "
                     "VALUES (?,?,?,?,?)", (kind, label, prompt_template, state, _now()))
    cx.commit()
    return cur.lastrowid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_prompt_variations.py tests/test_sales_pages_phase_c3.py
git commit -m "feat(sales-img): prompt review state + insert_variation (Phase C3 task 1)"
```

---

### Task 2: Generator — generate_candidates

**Files:**
- Create: `dashboard/sales_image_prompt_gen.py`
- Test: `tests/test_sales_pages_phase_c3.py` (append)

**Interfaces:**
- Consumes: `sales_prompt_variations.active_variations`/`candidate_variations`/`review_variations`/`insert_variation` (Task 1); `sales_image_prompts._BODY`/`IMAGE_KINDS`; `sales_image_leaderboard.leaderboard` (best-effort winners).
- Produces: `generate_candidates(cx, kind, n=2, *, llm=None) -> [dict{id,kind,label,prompt_template}]`; helpers `_parse_json_array(text)`, `_default_llm(prompt)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c3.py
from dashboard import sales_image_prompt_gen as gen

def test_generate_inserts_review_candidates_robust_parse_and_dedupe():
    cx = _cx()
    pv.seed(cx)
    existing_tmpl = pv.active_variations(cx, "botanical")[0]["prompt_template"]
    # fake LLM: JSON wrapped in prose + code fences; one item duplicates an existing template
    fake = ('Sure! Here are the prompts:\n```json\n'
            '[{"label":"new-a","prompt_template":"a brand new sunny herb garden scene"},'
            f'{{"label":"dupe","prompt_template":{existing_tmpl!r}}}]\n```')
    out = gen.generate_candidates(cx, "botanical", 2, llm=lambda p: fake)
    assert len(out) == 1 and out[0]["label"] == "new-a"          # dupe skipped
    revs = {r["prompt_template"] for r in pv.review_variations(cx, "botanical")}
    assert "a brand new sunny herb garden scene" in revs

def test_generate_malformed_response_returns_empty():
    cx = _cx(); pv.seed(cx)
    assert gen.generate_candidates(cx, "botanical", 2, llm=lambda p: "no json here") == []
    assert pv.review_variations(cx, "botanical") == []

def test_parse_json_array_slices_and_tolerates():
    assert gen._parse_json_array('prefix [ {"a":1} ] suffix') == [{"a": 1}]
    assert gen._parse_json_array("garbage") == []
    assert gen._parse_json_array("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: FAIL (`sales_image_prompt_gen` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_image_prompt_gen.py
import os, json

_MODEL = os.environ.get("IMAGE_PROMPT_GEN_MODEL", "claude-haiku-4-5-20251001")

def _default_llm(prompt):
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    resp = cli.messages.create(model=_MODEL, max_tokens=1500,
                               messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in resp.content
                   if getattr(b, "type", "") == "text").strip()

def _parse_json_array(text):
    if not text:
        return []
    s = text.find("["); e = text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return []
    try:
        data = json.loads(text[s:e + 1])
    except Exception:
        return []
    return data if isinstance(data, list) else []

def _winners_for_kind(cx, kind, limit=3):
    from dashboard import sales_image_leaderboard as _lb
    try:
        ids = {r[0] for r in cx.execute(
            "SELECT id FROM sales_prompt_variations WHERE kind=?", (kind,)).fetchall()}
        rows = _lb.leaderboard(cx, min_volume=0)["variations"]
        return [r["label"] for r in rows if r["key"] in ids][:limit]
    except Exception:
        return []

def _build_prompt(kind, n, existing, winners):
    from dashboard import sales_image_prompts as _sip
    body = _sip._BODY.get(kind, "")
    ex = "\n".join(f"- {t}" for t in existing) or "(none)"
    win = ", ".join(winners) if winners else "(no data yet)"
    return (
        f"You write image-generation SCENE prompts for a product's '{kind}' imagery.\n"
        f"Scene family: {body}\n"
        f"Currently winning variations (lean toward what works): {win}\n"
        f"Existing prompts — make NEW ones VISIBLY DISTINCT from all of these:\n{ex}\n\n"
        f"Write {n} new, distinct scene prompt_templates in the same family. Rules: NO product or brand "
        "names; NO instructions about text/letters/labels in the image (added later); each one a single "
        "vivid scene description. Return a STRICT JSON array ONLY (no prose, no code fences): "
        '[{"label": "short-kebab-label", "prompt_template": "the full scene description."}]'
    )

def generate_candidates(cx, kind, n=2, *, llm=None):
    from dashboard import sales_prompt_variations as _pv
    llm = llm or _default_llm
    existing = [v["prompt_template"] for v in (
        _pv.active_variations(cx, kind) + _pv.candidate_variations(cx, kind) + _pv.review_variations(cx, kind))]
    winners = _winners_for_kind(cx, kind)
    items = _parse_json_array(llm(_build_prompt(kind, n, existing, winners)))
    seen = set(existing)
    inserted = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tmpl = (it.get("prompt_template") or "").strip()
        label = (it.get("label") or "ai-candidate").strip()
        if not tmpl or tmpl in seen:
            continue
        vid = _pv.insert_variation(cx, kind, label, tmpl, "review")
        seen.add(tmpl)
        inserted.append({"id": vid, "kind": kind, "label": label, "prompt_template": tmpl})
    return inserted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_prompt_gen.py tests/test_sales_pages_phase_c3.py
git commit -m "feat(sales-img): LLM prompt-candidate generator (Phase C3 task 2)"
```

---

### Task 3: Review actions + bench top-up

**Files:**
- Modify: `dashboard/sales_image_prompt_gen.py`
- Test: `tests/test_sales_pages_phase_c3.py` (append)

**Interfaces:**
- Consumes: `sales_prompt_variations.set_state`/`candidate_variations`/`review_variations` (Tasks 1, C2); `generate_candidates` (Task 2); `sales_image_prompts.IMAGE_KINDS`.
- Produces: `review_action(cx, variation_id, decision, prompt_template=None) -> dict`; `topup(cx, *, threshold=2, generate=None) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c3.py
def test_review_action_approve_reject_edit():
    cx = _cx()
    vid = pv.insert_variation(cx, "botanical", "x", "scene one")
    assert gen.review_action(cx, vid, "edit", prompt_template="scene one edited")["ok"]
    assert pv.review_variations(cx, "botanical")[0]["prompt_template"] == "scene one edited"
    assert gen.review_action(cx, vid, "approve")["ok"]
    assert vid in {v["id"] for v in pv.candidate_variations(cx, "botanical")}
    v2 = pv.insert_variation(cx, "mechanism", "y", "cell scene")
    gen.review_action(cx, v2, "reject")
    assert pv.review_variations(cx, "mechanism") == []           # moved to retired
    assert gen.review_action(cx, 99999, "approve")["ok"] is False

def test_topup_only_when_bench_low():
    cx = _cx()
    calls = []
    fakegen = lambda c, kind, k: calls.append((kind, k))
    # empty bench -> generates for both kinds
    gen.topup(cx, threshold=2, generate=fakegen)
    assert {k for k, _ in calls} == {"botanical", "mechanism"}
    # fill botanical bench to 2 candidates -> botanical skipped next time
    pv.insert_variation(cx, "botanical", "a", "t1", "candidate")
    pv.insert_variation(cx, "botanical", "b", "t2", "candidate")
    calls.clear()
    gen.topup(cx, threshold=2, generate=fakegen)
    assert "botanical" not in {k for k, _ in calls}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: FAIL (`review_action`/`topup` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_prompt_gen.py`:

```python
def review_action(cx, variation_id, decision, prompt_template=None):
    from dashboard import sales_prompt_variations as _pv
    row = cx.execute("SELECT id FROM sales_prompt_variations WHERE id=?", (variation_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "not found"}
    if prompt_template is not None and decision in ("approve", "edit"):
        cx.execute("UPDATE sales_prompt_variations SET prompt_template=? WHERE id=?",
                   (prompt_template, variation_id))
        cx.commit()
    if decision == "approve":
        _pv.set_state(cx, variation_id, "candidate")
    elif decision == "reject":
        _pv.set_state(cx, variation_id, "retired")
    elif decision == "edit":
        pass
    else:
        return {"ok": False, "error": "bad decision"}
    return {"ok": True}

def topup(cx, *, threshold=2, generate=None):
    from dashboard import sales_prompt_variations as _pv, sales_image_prompts as _sip
    generate = generate or generate_candidates
    done = {}
    for kind in _sip.IMAGE_KINDS:
        bench = len(_pv.candidate_variations(cx, kind)) + len(_pv.review_variations(cx, kind))
        if bench < threshold:
            generate(cx, kind, threshold - bench)
            done[kind] = True
    return done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_prompt_gen.py tests/test_sales_pages_phase_c3.py
git commit -m "feat(sales-img): prompt review actions + bench top-up (Phase C3 task 3)"
```

---

### Task 4: Review console HTML

**Files:**
- Modify: `dashboard/sales_image_prompt_gen.py`
- Test: `tests/test_sales_pages_phase_c3.py` (append)

**Interfaces:**
- Produces: `review_console_html(cx) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c3.py
def test_review_console_html_lists_and_has_buttons():
    cx = _cx()
    pv.insert_variation(cx, "botanical", "x", "a reviewable herb scene")
    html = gen.review_console_html(cx)
    assert "a reviewable herb scene" in html
    assert "Generate" in html and "Approve" in html and "Reject" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: FAIL (`review_console_html` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_prompt_gen.py`:

```python
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

def review_console_html(cx):
    from dashboard import sales_prompt_variations as _pv, sales_image_prompts as _sip
    parts = ["<h2>Prompt candidates (review)</h2>"]
    for kind in _sip.IMAGE_KINDS:
        parts.append(f"<h3>{_esc(kind)} "
                     f"<button onclick=\"pg('generate',{{kind:'{_esc(kind)}',n:2}})\">Generate 2</button></h3>")
        revs = _pv.review_variations(cx, kind)
        if not revs:
            parts.append("<p>(none in review)</p>")
        for v in revs:
            tid = f"pg{v['id']}"
            parts.append(
                f"<div class='pg-rev'><textarea id='{tid}' rows='2' cols='80'>{_esc(v['prompt_template'])}</textarea><br>"
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'approve',prompt_template:document.getElementById('{tid}').value}})\">Approve</button> "
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'edit',prompt_template:document.getElementById('{tid}').value}})\">Save edit</button> "
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'reject'}})\">Reject</button></div>")
    parts.append("<script>function pg(op,body){fetch('/console/image-prompts/'+op,{method:'POST',"
                 "headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})"
                 ".then(function(){location.reload();});}</script>")
    return "".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_prompt_gen.py tests/test_sales_pages_phase_c3.py
git commit -m "feat(sales-img): prompt review console HTML (Phase C3 task 4)"
```

---

### Task 5: app.py — routes + scheduler + page append

**Files:**
- Modify: `app.py` (2 routes near the C2 evolution routes; a scheduler job near `_run_image_evolution`; the leaderboard route's flag block)

**Interfaces:**
- Consumes: `sales_image_prompt_gen.generate_candidates`/`review_action`/`topup`/`review_console_html` (Tasks 2-4); `_sales_console_ok()`; `_SALES_IMAGE_EVOLUTION_ENABLED`.

- [ ] **Step 1: Add the two console routes**

Add near the `/console/image-evolution/*` handlers in `app.py`:

```python
@app.route("/console/image-prompts/generate", methods=["POST"])
def console_image_prompts_generate():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return jsonify({"ok": False, "error": "evolution disabled"}), 400
    d = request.get_json(silent=True) or {}
    kind = (d.get("kind") or "").strip()
    try:
        n = int(d.get("n") or 2)
    except (TypeError, ValueError):
        n = 2
    from dashboard import sales_image_prompt_gen as _pg
    with sqlite3.connect(LOG_DB) as cx:
        out = _pg.generate_candidates(cx, kind, n)
    return jsonify({"ok": True, "count": len(out)})

@app.route("/console/image-prompts/review", methods=["POST"])
def console_image_prompts_review():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return jsonify({"ok": False, "error": "evolution disabled"}), 400
    d = request.get_json(silent=True) or {}
    from dashboard import sales_image_prompt_gen as _pg
    with sqlite3.connect(LOG_DB) as cx:
        res = _pg.review_action(cx, d.get("id"), (d.get("decision") or "").strip(),
                                prompt_template=d.get("prompt_template"))
    return jsonify(res)
```

- [ ] **Step 2: Add the top-up scheduler job**

Add a job function near `_run_image_evolution`:

```python
def _run_prompt_topup():
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return
    from dashboard import sales_image_prompt_gen as _pg
    try:
        with sqlite3.connect(LOG_DB) as cx:
            _pg.topup(cx)
    except Exception as e:
        print(f"[sales-img] prompt topup failed: {e}", flush=True)
```

Register it next to the other `scheduler.add_job(...)` calls:

```python
        scheduler.add_job(_run_prompt_topup, "interval", hours=24, id="sales_image_prompt_topup")
```

- [ ] **Step 3: Append the review section to the leaderboard page**

In the `console_image_leaderboard` route's flag block (where `_evo_html` is built), also build `_pg_html` and append it. Change:

```python
        _evo_html = ""
        if _SALES_IMAGE_EVOLUTION_ENABLED:
            from dashboard import sales_image_evolution as _ev
            _evo_html = _ev.console_section_html(cx)
    if request.args.get("format") == "json":
        return jsonify(data)
    return Response(_lb.render_html(data) + _evo_html, mimetype="text/html")
```

to:

```python
        _evo_html = ""
        _pg_html = ""
        if _SALES_IMAGE_EVOLUTION_ENABLED:
            from dashboard import sales_image_evolution as _ev
            from dashboard import sales_image_prompt_gen as _pg
            _evo_html = _ev.console_section_html(cx)
            _pg_html = _pg.review_console_html(cx)
    if request.args.get("format") == "json":
        return jsonify(data)
    return Response(_lb.render_html(data) + _evo_html + _pg_html, mimetype="text/html")
```

- [ ] **Step 4: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds. (Do NOT run `import app`.)

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): prompt-gen routes + topup job + console section (Phase C3 task 5)"
```

---

### Task 6: Regression + flag-off parity

**Files:** none (verification)

- [ ] **Step 1: Run the Phase C3 + C2 + C1 + B + A unit suites**

Run: `python3 -m pytest tests/test_sales_pages_phase_c3.py tests/test_sales_pages_phase_c2.py tests/test_sales_pages_phase_c.py tests/test_sales_pages_phase_b.py tests/test_sales_pages_phase_a.py -q`
Expected: all PASS (8 C3 + 11 C2 + 5 C1 + 5 B + 18 A = 47).

- [ ] **Step 2: Confirm flag-off parity + compile**

With `SALES_PAGES_IMAGE_EVOLUTION` unset: the prompt routes 400; `_run_prompt_topup` no-ops; the leaderboard page omits `_pg_html` (and `_evo_html`) → identical to C1. Confirm:
Run: `python3 -m py_compile app.py` → clean.
Run: `python3 -c "from dashboard import sales_image_prompt_gen as g; import sqlite3; cx=sqlite3.connect(':memory:'); print(g._parse_json_array('[]'), g.topup(cx, threshold=0, generate=lambda *a: None))"`
Expected: `[] {}` (threshold 0 → no kind is below it → no generation; no crash).

- [ ] **Step 3: Commit (if any fixups)**

```bash
git add -A && git commit -m "test(sales-img): Phase C3 regression pass" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** review state + insert (T1), generator (T2), review actions + top-up (T3), console HTML (T4), routes + scheduler + page append (T5), regression + flag-off (T6). Spec sections 1-5, data flow, testing, and the flag-reuse / app-import notes all map to tasks. "Out of scope" (model-gen, auto-approve) correctly has no task. ✔

**Placeholder scan:** all code complete; the LLM prompt text is concrete; `_default_llm` matches the codebase pattern. Tests inject `llm`/`generate` so no network. No "TODO/handle edge cases" in code.

**Type consistency:** `insert_variation(...) -> int` / `review_variations -> [dict]` (T1) consumed by `generate_candidates` + `topup` + `review_console_html` (T2-4); `generate_candidates(cx, kind, n=2, *, llm=None) -> [dict]` matches the route call `generate_candidates(cx, kind, n)` (T5) and the `topup` injected `generate(cx, kind, k)` signature (T3 test passes `(c, kind, k)`); `review_action(cx, id, decision, prompt_template=None) -> {ok}` matches the route + tests; `topup(cx, *, threshold=2, generate=None)` matches its test; `review_console_html(cx)`/`console_section_html(cx)` both appended in T5. The `pg()` JS (T4) targets `/console/image-prompts/<op>` matching the T5 routes; it's distinct from C2's `evo()` (no collision).
