# Product Page Images — Phase C2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A human-in-the-loop champion-challenger engine that, off C1's leaderboard, proposes retiring a confidently-worst active prompt/model and promoting a benched candidate; Glen approves in the console; the swap flips registry state so new products generate with the evolved set.

**Architecture:** A new `candidate` registry state + a model candidate bench. An engine module (`dashboard/sales_image_evolution.py`) computes proposals (Wilson-interval separation + min impressions), persists them, and applies approved swaps (retire-1 + promote-1, set-size preserved) with an audit log + undo. A daily scheduler proposes; the C1 console leaderboard page gains proposals/candidates + Approve/Reject/Trial/Undo. All behind `SALES_PAGES_IMAGE_EVOLUTION`, dark.

**Tech Stack:** Python 3 / Flask, SQLite (`LOG_DB`), APScheduler (existing), pytest.

## Global Constraints

- New behavior gated by env flag `SALES_PAGES_IMAGE_EVOLUTION` (truthy = `1`/`true`/`yes`) → module global `_SALES_IMAGE_EVOLUTION_ENABLED` in `app.py`. OFF = no proposals, no candidates seeded, no console additions, scheduler no-ops.
- **Propose & approve (human-in-the-loop):** the engine never changes the active set on its own; only `decide(approve)`/`trial`/`undo` (console-driven) mutate state.
- **Confidence rule:** propose a swap only when, among active items (with impressions ≥ `min_impressions`, default 50), the weakest's `wilson_upper` < the best's `wilson_lower` (intervals separated) AND a candidate is benched.
- **Set-size invariant:** every mutation is a swap (retire 1 + promote 1); `_apply_swap` asserts the active count is unchanged. Prompt swaps stay within one `kind`.
- **Registry states:** `active` (in rotation), `candidate` (benched, eligible to trial), `retired` (removed). Phase A generation reads `active_*` — changing state changes what NEW products generate.
- Candidate model Replicate refs are best-known; **confirm refs + pricing at build** (seeding is flag-gated → safe to merge dark).
- Tests: pytest, `sqlite3.connect(":memory:")` per test, import `dashboard.*` directly, no live network. New file `tests/test_sales_pages_phase_c2.py`. Follow `tests/test_sales_pages_phase4b.py` style.
- Sandbox: use `python3` (no `python`). `import app` CANNOT run here (Pinecone at import) — verify app.py edits with `python3 -m py_compile app.py` + the unit-tested engine. Full route/scheduler behavior = manual.
- Work in worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at C1 tip `753f4b9` which is in `main`). Commit per task. No edits to `main`.

---

### Task 1: Registry state helpers + model candidate bench

**Files:**
- Modify: `dashboard/sales_prompt_variations.py`, `dashboard/sales_image_models.py`
- Test: `tests/test_sales_pages_phase_c2.py`

**Interfaces:**
- Produces (variations): `set_state(cx, id, state)`, `candidate_variations(cx, kind) -> [dict{id,kind,label,prompt_template}]`.
- Produces (models): `set_state(cx, id, state)`, `candidate_models(cx) -> [dict{id,label,engine,engine_ref}]`, `seed_candidates(cx)` (idempotent INSERT OR IGNORE of 3 candidate models).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase_c2.py
import sqlite3
from dashboard import sales_image_models as mods
from dashboard import sales_prompt_variations as pv

def _cx(): return sqlite3.connect(":memory:")

def test_model_candidates_seed_and_setstate():
    cx = _cx(); mods.seed(cx)                 # 3 active
    mods.seed_candidates(cx)                  # + 3 candidate
    cands = {m["id"] for m in mods.candidate_models(cx)}
    assert cands == {"ideogram-v3", "flux-ultra", "sd-3.5-large"}
    assert {m["id"] for m in mods.active_models(cx)} == {"flux-1.1-pro", "imagen-4", "recraft-v3"}
    mods.seed_candidates(cx)                  # idempotent
    assert len(mods.candidate_models(cx)) == 3
    mods.set_state(cx, "ideogram-v3", "active")
    assert "ideogram-v3" in {m["id"] for m in mods.active_models(cx)}

def test_variation_setstate_and_candidates():
    cx = _cx(); pv.seed(cx)
    first = pv.active_variations(cx, "botanical")[0]["id"]
    pv.set_state(cx, first, "candidate")
    assert first in {v["id"] for v in pv.candidate_variations(cx, "botanical")}
    assert first not in {v["id"] for v in pv.active_variations(cx, "botanical")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: FAIL (`seed_candidates`/`set_state`/`candidate_*` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_prompt_variations.py`:

```python
def set_state(cx, id, state):
    init_table(cx)
    cx.execute("UPDATE sales_prompt_variations SET state=? WHERE id=?", (state, int(id)))
    cx.commit()

def candidate_variations(cx, kind):
    init_table(cx)
    rows = cx.execute("SELECT id, kind, label, prompt_template FROM sales_prompt_variations "
                      "WHERE kind=? AND state='candidate' ORDER BY id", (kind,)).fetchall()
    return [{"id": r[0], "kind": r[1], "label": r[2], "prompt_template": r[3]} for r in rows]
```

Append to `dashboard/sales_image_models.py`:

```python
_CANDIDATES = [
    ("ideogram-v3",  "Ideogram V3",          "ideogram-ai/ideogram-v3-quality"),
    ("flux-ultra",   "Flux 1.1 Pro Ultra",   "black-forest-labs/flux-1.1-pro-ultra"),
    ("sd-3.5-large", "Stable Diffusion 3.5 L","stability-ai/stable-diffusion-3.5-large"),
]

def seed_candidates(cx):
    init_table(cx); now = _now()
    for mid, label, ref in _CANDIDATES:
        cx.execute("INSERT OR IGNORE INTO sales_image_models (id, label, engine, engine_ref, state, created_at) "
                   "VALUES (?,?, 'replicate', ?, 'candidate', ?)", (mid, label, ref, now))
    cx.commit()

def candidate_models(cx):
    init_table(cx)
    rows = cx.execute("SELECT id, label, engine, engine_ref FROM sales_image_models "
                      "WHERE state='candidate' ORDER BY rowid").fetchall()
    return [{"id": r[0], "label": r[1], "engine": r[2], "engine_ref": r[3]} for r in rows]

def set_state(cx, id, state):
    init_table(cx)
    cx.execute("UPDATE sales_image_models SET state=? WHERE id=?", (state, id))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_prompt_variations.py dashboard/sales_image_models.py tests/test_sales_pages_phase_c2.py
git commit -m "feat(sales-img): registry candidate state + model bench (Phase C2 task 1)"
```

---

### Task 2: Wilson upper bound

**Files:**
- Modify: `dashboard/sales_image_leaderboard.py`
- Test: `tests/test_sales_pages_phase_c2.py` (append)

**Interfaces:**
- Produces: `wilson_upper(pos, n, z=1.96) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c2.py
from dashboard import sales_image_leaderboard as lb

def test_wilson_upper_brackets_rate():
    assert lb.wilson_upper(0, 0) == 0.0
    lo, hi = lb.wilson_lower(5, 10), lb.wilson_upper(5, 10)
    assert lo < 0.5 < hi                       # interval brackets the 0.5 rate
    assert lb.wilson_upper(5, 10) > lb.wilson_upper(50, 100)   # less data -> wider/higher upper
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py::test_wilson_upper_brackets_rate -q`
Expected: FAIL (`wilson_upper` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_leaderboard.py`:

```python
def wilson_upper(pos, n, z=1.96):
    if n <= 0:
        return 0.0
    phat = pos / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    return (centre + margin) / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: PASS (3 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_leaderboard.py tests/test_sales_pages_phase_c2.py
git commit -m "feat(sales-img): wilson_upper bound for evolution (Phase C2 task 2)"
```

---

### Task 3: Evolution engine — tables + propose

**Files:**
- Create: `dashboard/sales_image_evolution.py`
- Test: `tests/test_sales_pages_phase_c2.py` (append)

**Interfaces:**
- Consumes: `sales_image_leaderboard.leaderboard`/`wilson_lower`/`wilson_upper`; `sales_image_models.active_models`/`candidate_models`/`seed_candidates`; `sales_prompt_variations.active_variations`/`candidate_variations`; `sales_image_prompts.IMAGE_KINDS`.
- Produces: `init_tables(cx)`; `propose(cx, *, min_impressions=50) -> [dict]` (persists pending, deduped + 14d cooldown on rejected); `pending_proposals(cx) -> [dict{id,axis,kind,retire_key,promote_key,stats}]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c2.py
from dashboard import sales_image_evolution as ev
from dashboard import sales_images as si
from dashboard import sales_votes as sv
from dashboard import sales_image_exposures as ex

def _seed_model_field(cx, *, loser_votes, winner_votes, impressions_each):
    # two products so flux & recraft each appear; give exposures + lopsided votes
    si.record_image(cx, "p1", "botanical", 1, "p1b.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "p1", "mechanism", 1, "p1m.png", prompt_variant_id=5, model_id="recraft-v3")
    for i in range(impressions_each): ex.record(cx, "p1", f"s{i}")
    for i in range(winner_votes): sv.record_pick(cx, "p1", "botanical", 1, f"w{i}", model_id="flux-1.1-pro", prompt_variant_id=1)
    for i in range(loser_votes):  sv.record_pick(cx, "p1", "mechanism", 1, f"l{i}", model_id="recraft-v3", prompt_variant_id=5)

def test_propose_fires_on_confident_loser_with_candidate():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    # recraft is the confident loser (0/60), flux the winner (55/60); imagen has no data
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    props = ev.propose(cx, min_impressions=20)
    model_props = [p for p in props if p["axis"] == "model"]
    assert any(p["retire_key"] == "recraft-v3" for p in model_props)
    p = next(p for p in model_props if p["retire_key"] == "recraft-v3")
    assert p["promote_key"] in {"ideogram-v3", "flux-ultra", "sd-3.5-large"}
    # persisted as pending, and idempotent (no duplicate pending)
    n1 = len(ev.pending_proposals(cx)); ev.propose(cx, min_impressions=20)
    assert len(ev.pending_proposals(cx)) == n1

def test_propose_no_fire_when_intervals_overlap():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=28, winner_votes=32, impressions_each=60)  # 28/60 vs 32/60 -> overlap
    props = [p for p in ev.propose(cx, min_impressions=20) if p["axis"] == "model"]
    assert props == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: FAIL (`sales_image_evolution` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_image_evolution.py
import datetime, json

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_evolution_proposals ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, axis TEXT, kind TEXT DEFAULT '', "
               "retire_key TEXT, promote_key TEXT, stats_json TEXT, "
               "state TEXT DEFAULT 'pending', created_at TEXT DEFAULT '', decided_at TEXT DEFAULT '')")
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_evolution_log ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, axis TEXT, kind TEXT DEFAULT '', "
               "retired_key TEXT, promoted_key TEXT, actor TEXT DEFAULT '', "
               "created_at TEXT DEFAULT '', undone_at TEXT DEFAULT '')")
    cx.commit()

def _active_rows(cx, axis, kind, lb_rows, min_impressions):
    """Leaderboard rows for the currently-active items of this axis/kind, keyed by str(key)."""
    from dashboard import sales_image_models as _mods, sales_prompt_variations as _pv
    if axis == "model":
        active_keys = {m["id"] for m in _mods.active_models(cx)}
    else:
        active_keys = {str(v["id"]) for v in _pv.active_variations(cx, kind)}
    out = []
    for r in lb_rows:
        if str(r["key"]) in active_keys:
            out.append(r)
    return out

def _candidate_keys(cx, axis, kind):
    from dashboard import sales_image_models as _mods, sales_prompt_variations as _pv
    if axis == "model":
        return [m["id"] for m in _mods.candidate_models(cx)]
    return [str(v["id"]) for v in _pv.candidate_variations(cx, kind)]

def _exists_pending(cx, axis, kind, retire_key, promote_key):
    r = cx.execute("SELECT 1 FROM sales_image_evolution_proposals WHERE axis=? AND kind=? AND "
                   "retire_key=? AND promote_key=? AND state='pending'",
                   (axis, kind, str(retire_key), str(promote_key))).fetchone()
    return r is not None

def _on_cooldown(cx, axis, kind, retire_key, promote_key, days=14):
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    r = cx.execute("SELECT 1 FROM sales_image_evolution_proposals WHERE axis=? AND kind=? AND "
                   "retire_key=? AND promote_key=? AND state='rejected' AND decided_at>=?",
                   (axis, kind, str(retire_key), str(promote_key), cutoff)).fetchone()
    return r is not None

def _evaluate(cx, axis, kind, min_impressions):
    from dashboard import sales_image_leaderboard as _lb
    lb = _lb.leaderboard(cx, min_volume=0)
    lb_rows = lb["models"] if axis == "model" else lb["variations"]
    rows = _active_rows(cx, axis, kind, lb_rows, min_impressions)
    rows = [r for r in rows if r["impressions"] > 0]
    if len(rows) < 2:
        return None
    weakest = min(rows, key=lambda r: r["wilson"])
    best = max(rows, key=lambda r: r["wilson"])
    if weakest["key"] == best["key"]:
        return None
    if weakest["impressions"] < min_impressions:
        return None
    w_upper = _lb.wilson_upper(weakest["votes"], weakest["impressions"])
    if not (w_upper < best["wilson"]):
        return None
    cands = _candidate_keys(cx, axis, kind)
    if not cands:
        return None
    return {"axis": axis, "kind": kind, "retire_key": str(weakest["key"]),
            "promote_key": str(cands[0]),
            "stats": {"retire_label": weakest["label"], "promote_key": str(cands[0]),
                      "retire_wilson": weakest["wilson"], "retire_wilson_upper": w_upper,
                      "best_wilson": best["wilson"], "best_label": best["label"],
                      "retire_votes": weakest["votes"], "retire_impressions": weakest["impressions"]}}

def propose(cx, *, min_impressions=50):
    from dashboard import sales_image_models as _mods, sales_image_prompts as _sip
    init_tables(cx)
    _mods.seed_candidates(cx)
    targets = [("model", "")] + [("variation", k) for k in _sip.IMAGE_KINDS]
    created = []
    for axis, kind in targets:
        p = _evaluate(cx, axis, kind, min_impressions)
        if not p:
            continue
        if _exists_pending(cx, axis, kind, p["retire_key"], p["promote_key"]):
            created.append(p); continue
        if _on_cooldown(cx, axis, kind, p["retire_key"], p["promote_key"]):
            continue
        cx.execute("INSERT INTO sales_image_evolution_proposals "
                   "(axis, kind, retire_key, promote_key, stats_json, state, created_at) "
                   "VALUES (?,?,?,?,?, 'pending', ?)",
                   (axis, kind, p["retire_key"], p["promote_key"], json.dumps(p["stats"]), _now()))
        created.append(p)
    cx.commit()
    return created

def pending_proposals(cx):
    init_tables(cx)
    rows = cx.execute("SELECT id, axis, kind, retire_key, promote_key, stats_json "
                      "FROM sales_image_evolution_proposals WHERE state='pending' ORDER BY id").fetchall()
    out = []
    for r in rows:
        try:
            stats = json.loads(r[5] or "{}")
        except Exception:
            stats = {}
        out.append({"id": r[0], "axis": r[1], "kind": r[2], "retire_key": r[3],
                    "promote_key": r[4], "stats": stats})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_evolution.py tests/test_sales_pages_phase_c2.py
git commit -m "feat(sales-img): evolution engine — propose swaps (Phase C2 task 3)"
```

---

### Task 4: Evolution engine — apply / decide / trial / undo

**Files:**
- Modify: `dashboard/sales_image_evolution.py`
- Test: `tests/test_sales_pages_phase_c2.py` (append)

**Interfaces:**
- Consumes: registry `set_state`/`active_models`/`active_variations`/`candidate_*` (Task 1); `propose`/`pending_proposals` (Task 3).
- Produces: `decide(cx, proposal_id, decision, actor="console") -> dict`; `trial(cx, axis, kind, candidate_key, actor="console") -> dict`; `undo(cx, log_id, actor="console") -> dict`. (`_apply_swap` internal.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c2.py
def test_decide_approve_swaps_and_keeps_count():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    pid = next(p["id"] for p in ev.pending_proposals(cx) if p["axis"] == "model")
    before = len(mods.active_models(cx))
    res = ev.decide(cx, pid, "approve", actor="t")
    assert res["ok"] and res["applied"]
    active = {m["id"] for m in mods.active_models(cx)}
    assert "recraft-v3" not in active                  # retired
    assert len(active) == before                       # set-size preserved
    assert not ev.pending_proposals(cx)                # proposal consumed

def test_decide_reject_no_state_change():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    pid = next(p["id"] for p in ev.pending_proposals(cx) if p["axis"] == "model")
    ev.decide(cx, pid, "reject", actor="t")
    assert "recraft-v3" in {m["id"] for m in mods.active_models(cx)}
    assert not ev.pending_proposals(cx)

def test_trial_and_undo():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    res = ev.trial(cx, "model", "", "ideogram-v3", actor="t")
    assert res["ok"]
    assert "ideogram-v3" in {m["id"] for m in mods.active_models(cx)}
    log_id = res["log_id"]
    ev.undo(cx, log_id, actor="t")
    assert "ideogram-v3" not in {m["id"] for m in mods.active_models(cx)}   # back to candidate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: FAIL (`decide`/`trial`/`undo` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_evolution.py`:

```python
def _registry(axis):
    from dashboard import sales_image_models as _mods, sales_prompt_variations as _pv
    return _mods if axis == "model" else _pv

def _active_count(cx, axis, kind):
    from dashboard import sales_image_models as _mods, sales_prompt_variations as _pv
    return len(_mods.active_models(cx)) if axis == "model" else len(_pv.active_variations(cx, kind))

def _key(axis, key):
    return key if axis == "model" else int(key)   # variation ids are ints

def _apply_swap(cx, axis, kind, retire_key, promote_key, actor):
    reg = _registry(axis)
    before = _active_count(cx, axis, kind)
    reg.set_state(cx, _key(axis, retire_key), "retired")
    reg.set_state(cx, _key(axis, promote_key), "active")
    after = _active_count(cx, axis, kind)
    if after != before:
        raise ValueError(f"swap changed active count {before}->{after}")
    cur = cx.execute("INSERT INTO sales_image_evolution_log "
                     "(axis, kind, retired_key, promoted_key, actor, created_at) VALUES (?,?,?,?,?,?)",
                     (axis, kind, str(retire_key), str(promote_key), actor, _now()))
    cx.commit()
    return cur.lastrowid

def decide(cx, proposal_id, decision, actor="console"):
    init_tables(cx)
    r = cx.execute("SELECT axis, kind, retire_key, promote_key, state FROM "
                   "sales_image_evolution_proposals WHERE id=?", (proposal_id,)).fetchone()
    if not r or r[4] != "pending":
        return {"ok": False, "error": "not pending"}
    axis, kind, retire_key, promote_key, _ = r
    applied = False; log_id = None
    if decision == "approve":
        log_id = _apply_swap(cx, axis, kind, retire_key, promote_key, actor); applied = True
        new_state = "approved"
    elif decision == "reject":
        new_state = "rejected"
    else:
        return {"ok": False, "error": "bad decision"}
    cx.execute("UPDATE sales_image_evolution_proposals SET state=?, decided_at=? WHERE id=?",
               (new_state, _now(), proposal_id))
    cx.commit()
    return {"ok": True, "applied": applied, "log_id": log_id}

def _weakest_active_key(cx, axis, kind):
    from dashboard import sales_image_leaderboard as _lb, sales_image_models as _mods, sales_prompt_variations as _pv
    lb = _lb.leaderboard(cx, min_volume=0)
    rows = _active_rows(cx, axis, kind, (lb["models"] if axis == "model" else lb["variations"]), 0)
    if rows:
        return str(min(rows, key=lambda r: r["wilson"])["key"])
    # no data yet -> fall back to the first active item
    actives = _mods.active_models(cx) if axis == "model" else _pv.active_variations(cx, kind)
    return str(actives[0]["id"]) if actives else None

def trial(cx, axis, kind, candidate_key, actor="console"):
    init_tables(cx)
    if str(candidate_key) not in set(_candidate_keys(cx, axis, kind)):
        return {"ok": False, "error": "not a candidate"}
    retire_key = _weakest_active_key(cx, axis, kind)
    if not retire_key:
        return {"ok": False, "error": "no active item"}
    log_id = _apply_swap(cx, axis, kind, retire_key, candidate_key, actor)
    return {"ok": True, "log_id": log_id, "retired": retire_key, "promoted": str(candidate_key)}

def undo(cx, log_id, actor="console"):
    init_tables(cx)
    r = cx.execute("SELECT axis, kind, retired_key, promoted_key, undone_at FROM "
                   "sales_image_evolution_log WHERE id=?", (log_id,)).fetchone()
    if not r or r[4]:
        return {"ok": False, "error": "not undoable"}
    axis, kind, retired_key, promoted_key, _ = r
    reg = _registry(axis)
    reg.set_state(cx, _key(axis, promoted_key), "candidate")
    reg.set_state(cx, _key(axis, retired_key), "active")
    cx.execute("UPDATE sales_image_evolution_log SET undone_at=? WHERE id=?", (_now(), log_id))
    cx.commit()
    return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_evolution.py tests/test_sales_pages_phase_c2.py
git commit -m "feat(sales-img): evolution apply/decide/trial/undo (Phase C2 task 4)"
```

---

### Task 5: Console section HTML

**Files:**
- Modify: `dashboard/sales_image_evolution.py` (add `console_section_html`)
- Test: `tests/test_sales_pages_phase_c2.py` (append)

**Interfaces:**
- Produces: `console_section_html(cx) -> str` — HTML for pending proposals (with Approve/Reject), the benched candidates per axis/kind (with Trial), and a tiny inline JS that POSTs to the action routes and reloads.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c2.py
def test_console_section_html_lists_proposals_and_candidates():
    cx = _cx()
    from dashboard import sales_image_models as mods
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    html = ev.console_section_html(cx)
    assert "recraft-v3" in html              # the proposed retire
    assert "Approve" in html and "Reject" in html
    assert "ideogram-v3" in html             # a benched candidate
    assert "Trial" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: FAIL (`console_section_html` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_evolution.py`:

```python
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def console_section_html(cx):
    from dashboard import sales_image_models as _mods, sales_prompt_variations as _pv
    from dashboard import sales_image_prompts as _sip
    props = pending_proposals(cx)
    parts = ["<h2>Evolution — pending proposals</h2>"]
    if not props:
        parts.append("<p>No pending proposals.</p>")
    for p in props:
        s = p["stats"]
        scope = "model" if p["axis"] == "model" else f"variation/{p['kind']}"
        parts.append(
            f"<div class='evo-prop'>[{_esc(scope)}] retire <b>{_esc(s.get('retire_label', p['retire_key']))}</b> "
            f"(wilson {s.get('retire_wilson', 0):.3f}, {s.get('retire_votes',0)}/{s.get('retire_impressions',0)}) "
            f"→ promote <b>{_esc(p['promote_key'])}</b> "
            f"<button onclick=\"evo('decide',{{proposal_id:{p['id']},decision:'approve'}})\">Approve</button> "
            f"<button onclick=\"evo('decide',{{proposal_id:{p['id']},decision:'reject'}})\">Reject</button></div>")
    parts.append("<h2>Benched candidates</h2>")
    rows = [("model", "")] + [("variation", k) for k in _sip.IMAGE_KINDS]
    for axis, kind in rows:
        cands = _mods.candidate_models(cx) if axis == "model" else _pv.candidate_variations(cx, kind)
        scope = "model" if axis == "model" else f"variation/{kind}"
        for c in cands:
            ck = c["id"]
            parts.append(
                f"<div class='evo-cand'>[{_esc(scope)}] {_esc(c['label'])} "
                f"<button onclick=\"evo('trial',{{axis:'{axis}',kind:'{_esc(kind)}',candidate_key:'{_esc(ck)}'}})\">Trial</button></div>")
    parts.append(
        "<script>function evo(op,body){fetch('/console/image-'+(op==='decide'?'evolution/decide':"
        "'evolution/'+op),{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify(body)}).then(function(){location.reload();});}</script>")
    return "".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py -q`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_evolution.py tests/test_sales_pages_phase_c2.py
git commit -m "feat(sales-img): evolution console section HTML (Phase C2 task 5)"
```

---

### Task 6: Flag + scheduler + console action routes

**Files:**
- Modify: `app.py` (flag near line 2544; scheduler job near ~16820; 3 routes near the C1 leaderboard route)

**Interfaces:**
- Consumes: `_SALES_IMAGE_VOTE_ENABLED` style flag; `sales_image_evolution.propose`/`decide`/`trial`/`undo`; `_sales_console_ok()`.

- [ ] **Step 1: Add the flag**

In `app.py`, after the line `_SALES_IMAGE_VOTE_ENABLED = ...` (~line 2544), add:

```python
_SALES_IMAGE_EVOLUTION_ENABLED = os.environ.get("SALES_PAGES_IMAGE_EVOLUTION", "").strip().lower() in ("1", "true", "yes")
```

- [ ] **Step 2: Add the scheduler job**

Add a job function near `_drain_sales_image_queue` (e.g. right after `_run_image_tournament` if present, else near the other `_run_*`/`_drain_*` defs):

```python
def _run_image_evolution():
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return
    from dashboard import sales_image_evolution as _ev
    try:
        with sqlite3.connect(LOG_DB) as cx:
            _ev.propose(cx)
    except Exception as e:
        print(f"[sales-img] evolution propose failed: {e}", flush=True)
```

Register it next to the other `scheduler.add_job(...)` calls (~line 16820):

```python
        scheduler.add_job(_run_image_evolution, "interval", hours=24, id="sales_image_evolution")
```

- [ ] **Step 3: Add the three console action routes**

Add near the `@app.route("/console/image-leaderboard")` handler:

```python
@app.route("/console/image-evolution/decide", methods=["POST"])
def console_image_evolution_decide():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return jsonify({"ok": False, "error": "evolution disabled"}), 400
    d = request.get_json(silent=True) or {}
    from dashboard import sales_image_evolution as _ev
    with sqlite3.connect(LOG_DB) as cx:
        res = _ev.decide(cx, d.get("proposal_id"), (d.get("decision") or "").strip(), actor="console")
    return jsonify(res)

@app.route("/console/image-evolution/trial", methods=["POST"])
def console_image_evolution_trial():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return jsonify({"ok": False, "error": "evolution disabled"}), 400
    d = request.get_json(silent=True) or {}
    from dashboard import sales_image_evolution as _ev
    with sqlite3.connect(LOG_DB) as cx:
        res = _ev.trial(cx, (d.get("axis") or "").strip(), (d.get("kind") or "").strip(),
                        (d.get("candidate_key") or "").strip(), actor="console")
    return jsonify(res)

@app.route("/console/image-evolution/undo", methods=["POST"])
def console_image_evolution_undo():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    if not _SALES_IMAGE_EVOLUTION_ENABLED:
        return jsonify({"ok": False, "error": "evolution disabled"}), 400
    d = request.get_json(silent=True) or {}
    from dashboard import sales_image_evolution as _ev
    with sqlite3.connect(LOG_DB) as cx:
        res = _ev.undo(cx, d.get("log_id"), actor="console")
    return jsonify(res)
```

- [ ] **Step 4: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds. (Do NOT run `import app`.)

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): evolution flag + scheduler + console action routes (Phase C2 task 6)"
```

---

### Task 7: Render the evolution section on the leaderboard page

**Files:**
- Modify: `app.py` — `console_image_leaderboard` route (the C1 route)

**Interfaces:**
- Consumes: `_SALES_IMAGE_EVOLUTION_ENABLED`; `sales_image_evolution.console_section_html` (Task 5); existing `sales_image_leaderboard.render_html`.

- [ ] **Step 1: Append the evolution section to the page**

In `app.py`, the C1 route currently is:

```python
@app.route("/console/image-leaderboard")
def console_image_leaderboard():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    from dashboard import sales_image_leaderboard as _lb
    with sqlite3.connect(LOG_DB) as cx:
        data = _lb.leaderboard(cx)
    if request.args.get("format") == "json":
        return jsonify(data)
    return Response(_lb.render_html(data), mimetype="text/html")
```

Replace its body's tail (the `data` build + return) so that, when the flag is on, the evolution section is appended:

```python
    from dashboard import sales_image_leaderboard as _lb
    with sqlite3.connect(LOG_DB) as cx:
        data = _lb.leaderboard(cx)
        _evo_html = ""
        if _SALES_IMAGE_EVOLUTION_ENABLED:
            from dashboard import sales_image_evolution as _ev
            _evo_html = _ev.console_section_html(cx)
    if request.args.get("format") == "json":
        return jsonify(data)
    return Response(_lb.render_html(data) + _evo_html, mimetype="text/html")
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): show evolution proposals on the leaderboard page (Phase C2 task 7)"
```

---

### Task 8: Regression + flag-off parity

**Files:** none (verification)

- [ ] **Step 1: Run the Phase C2 + C + B + A unit suites**

Run: `python3 -m pytest tests/test_sales_pages_phase_c2.py tests/test_sales_pages_phase_c.py tests/test_sales_pages_phase_b.py tests/test_sales_pages_phase_a.py -q`
Expected: all PASS (9 C2 + 6 C1 + 5 B + 18 A = 38).

- [ ] **Step 2: Confirm flag-off parity + compile**

With `SALES_PAGES_IMAGE_EVOLUTION` unset: `_run_image_evolution` no-ops; the action routes 400; the leaderboard page omits the evolution section (`_evo_html=""`), i.e. C1 behavior unchanged; `seed_candidates` is never called (no candidate models added). Confirm:
Run: `python3 -m py_compile app.py` → clean.
Run: `python3 -c "from dashboard import sales_image_evolution as ev; import sqlite3; cx=sqlite3.connect(':memory:'); print(ev.propose(cx))"`
Expected: `[]` (no data → no proposals, no crash; note this DOES seed model candidates because propose() always seeds — that's fine, candidates are inert until activated).

- [ ] **Step 3: Commit (if any fixups)**

```bash
git add -A && git commit -m "test(sales-img): Phase C2 regression pass" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** registry candidate state + bench (T1), wilson_upper (T2), engine propose (T3), apply/decide/trial/undo (T4), console section HTML (T5), flag + scheduler + action routes (T6), leaderboard-page render (T7), regression + flag-off (T8). Spec sections 1-6, data flow, testing, and the set-size-invariant / flag / app-import notes all map to tasks. C3 (prompt generation) correctly out of scope. ✔

**Placeholder scan:** all code complete; candidate Replicate refs are best-known with an explicit build-time confirm (a constraint, not a code gap). Console render + scheduler are manual-verified (app can't boot) but the underlying engine helpers are unit-tested. No "TODO/handle edge cases" in code.

**Type consistency:** keys are stored as TEXT and cast on registry writes via `_key(axis, key)` (int for variations, str for models) — consistent across `_apply_swap`/`undo`/`trial`; `propose` emits `{axis,kind,retire_key,promote_key,stats}` consumed by `pending_proposals`/`console_section_html`/`decide`; `decide`/`trial`/`undo` return `{ok,...}` consumed by the routes' `jsonify`; `wilson_upper(pos,n,z=1.96)` (T2) used in `_evaluate` (T3). Registry `set_state`/`candidate_*` (T1) used by the engine (T3/T4). The leaderboard row shape `{key,label,votes,impressions,rate,wilson,...}` (C1) is what `_active_rows`/`_evaluate` read.
