# Biofield CovExt — Non-Scan Stress Coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give non-scan stresses (voice/tag/comm) a remedy-coverage path via historical `stress_suggestions`, so they join B1 auto-balance and B4 minimal-remedy set-cover.

**Architecture:** A `historical_remedies(cx, label)` helper (lowercased remedies Glen historically used for that stress name, from the FMP snapshot) feeds two extensions in `dashboard/biofield_stress.py`: `list_stresses` gains a historical-coverage auto-balance path for non-scan stresses; `suggest_minimal_remedies` generalizes its cover from E4L codes to "tokens" (code for scan, normalized label for non-scan) and includes non-scan stresses.

**Tech Stack:** Python 3.11, sqlite3, pytest.

## Global Constraints

- Only `dashboard/biofield_stress.py` changes (+ tests). Reuse `dashboard/biofield_authoring.py:stress_suggestions(cx, stress, limit=8) -> [{"remedy","count"}]` (FMP historical stress→remedies). `minimal_remedies` (pure greedy set-cover) is UNCHANGED — it already takes opaque tokens.
- Association source = `stress_suggestions` only (no LLM). Empty when no history / FMP snapshot absent — never raises.
- "Non-scan" stress = `source != 'scan'`. Scan-stress behavior (code-coverage path, scan-only set-cover) MUST be unchanged.
- `balanced_by` precedence: covering remedy (scan code) → label-match remedy (B2) → **historical remedy (new)** → "manual" → "".
- Remedy-name comparisons are lowercased (consistent with `covered_codes`/coverage which store/lower remedy names).
- Local-only; no prod deploy, no feature flag.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v`. Existing B1/B2/B4 stress tests must stay green (`tests/test_biofield_stress_derive.py`, `tests/test_biofield_stress_labelmatch.py`, `tests/test_biofield_suggest_remedies.py`).

---

### Task 1: `historical_remedies` + non-scan auto-balance in `list_stresses`

**Files:**
- Modify: `dashboard/biofield_stress.py`
- Test: `tests/test_biofield_covext_balance.py`

**Interfaces:**
- Produces: `historical_remedies(cx, label) -> set[str]` (lowercased remedy names from `stress_suggestions`; `{}`/empty on no history or missing FMP tables). `list_stresses` return shape unchanged, but a non-scan active stress is now `balanced` when a current chain remedy is in its historical set, with `balanced_by` = that remedy (precedence below code-coverage and label-match).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_covext_balance.py
import sqlite3
from dashboard.biofield_stress import (
    add_stress, add_voice_stress, historical_remedies, init_stress_tables, list_stresses)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    # FMP snapshot tables that stress_suggestions joins on
    cx.executescript("""
        CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT);
        CREATE TABLE fmp_snap_client_causal_chain(id_pk INTEGER, id_fk_active_stress INTEGER);
        CREATE TABLE fmp_snap_client_remedy(id_fk_causal_chain INTEGER, remedy TEXT);
    """)
    # history: "Adrenal Fatigue" was balanced with "Adaptogen Blend"
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress VALUES(1,'Adrenal Fatigue')")
    cx.execute("INSERT INTO fmp_snap_client_causal_chain VALUES(10,1)")
    cx.execute("INSERT INTO fmp_snap_client_remedy VALUES(10,'Adaptogen Blend')")
    cx.commit()
    return cx


def test_historical_remedies_lowercased(tmp_path):
    cx = _cx(tmp_path)
    assert historical_remedies(cx, "Adrenal Fatigue") == {"adaptogen blend"}
    assert historical_remedies(cx, "Unknown Thing") == set()


def test_nonscan_stress_balanced_by_history(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")                       # source=voice, no E4L code
    # a chain row whose remedy is historically used for the stress (head unrelated, so NOT a B2 label-match)
    res = list_stresses(cx, "a5", [{"head": "unrelated layer", "remedy": "Adaptogen Blend"}])
    bal = {s["label"]: s["balanced_by"] for s in res["balanced"]}
    assert bal == {"Adrenal Fatigue": "adaptogen blend"}
    assert res["active"] == []
    # removing the remedy reactivates it
    assert [s["label"] for s in list_stresses(cx, "a5", [])["active"]] == ["Adrenal Fatigue"]


def test_unrelated_remedy_does_not_balance(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")
    res = list_stresses(cx, "a5", [{"head": "x", "remedy": "Something Else"}])
    assert [s["label"] for s in res["active"]] == ["Adrenal Fatigue"]
```

- [ ] **Step 2: Run** → FAIL (`historical_remedies` undefined / non-scan not balanced).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_covext_balance.py -v`

- [ ] **Step 3: Implement** — in `dashboard/biofield_stress.py`:

(a) Add the helper (near `covered_codes`):

```python
def historical_remedies(cx, label):
    """Lowercased remedies Glen historically used for this stress name (FMP snapshot).
    Empty set on no history or missing snapshot. Never raises."""
    try:
        from dashboard.biofield_authoring import stress_suggestions
        return {(s.get("remedy") or "").strip().lower()
                for s in stress_suggestions(cx, label) if (s.get("remedy") or "").strip()}
    except Exception:
        return set()
```

(b) Replace the per-row loop in `list_stresses` to add the non-scan historical path. Inside `list_stresses`, after `head_map` is built, compute the lowercased chain remedy set, then in the loop add the historical branch:

```python
    chain_rem_lower = {(n or "").strip().lower() for n in remedy_names if (n or "").strip()}
    active, balanced = [], []
    for r in rows:
        is_cov = r["code"] in covered
        lbl_rem = head_map.get(_norm(r["label"]))
        hist_rem = None
        if not is_cov and lbl_rem is None and r["source"] != "scan" and chain_rem_lower:
            hist = historical_remedies(cx, r["label"]) & chain_rem_lower
            if hist:
                hist_rem = sorted(hist)[0]                # deterministic
        is_bal = bool(r["manual_balanced"]) or is_cov or (lbl_rem is not None) or (hist_rem is not None)
        if is_cov:
            cvs = _coverers(cx, tid, r["code"], remedy_names)
            by = cvs[0] if cvs else ""
        elif lbl_rem is not None:
            by = lbl_rem
        elif hist_rem is not None:
            by = hist_rem
        elif r["manual_balanced"]:
            by = "manual"
        else:
            by = ""
        item = {"id": r["id"], "code": r["code"], "label": r["label"],
                "source": r["source"], "balance": r["balance"],
                "balanced": is_bal, "balanced_by": by}
        (balanced if is_bal else active).append(item)
    return {"active": active, "balanced": balanced}
```

(Keep the lines above the loop — `init_stress_tables`, `cx.row_factory`, `_chain_parts`, `covered_codes`, `head_map`, the `rows` SELECT — exactly as they are; only the loop + the new `chain_rem_lower` line change.)

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_covext_balance.py tests/test_biofield_stress_derive.py tests/test_biofield_stress_labelmatch.py -v` → PASS (B1/B2 derive + label-match tests still green — scan + label-match paths unchanged; non-scan stresses with no chain remedy / no history behave as before).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_stress.py tests/test_biofield_covext_balance.py
git commit -m "feat(covext): historical_remedies + non-scan auto-balance in list_stresses"
```

---

### Task 2: Token-generalized `suggest_minimal_remedies` (include non-scan)

**Files:**
- Modify: `dashboard/biofield_stress.py` (`suggest_minimal_remedies`)
- Test: `tests/test_biofield_covext_setcover.py`

**Interfaces:**
- Consumes: `historical_remedies` (Task 1), `minimal_remedies` (unchanged).
- Produces: `suggest_minimal_remedies` now covers active+required **scan AND non-scan** stresses. Cover token = E4L `code` for scan, `_norm(label)` for non-scan. Coverage = scan map (remedy→codes) merged with non-scan historical (remedy→norm-label). Output shape unchanged (`{"picks":[{remedy,covers:[labels]}], "uncovered":[labels]}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_covext_setcover.py
import sqlite3
from dashboard.biofield_stress import (
    add_voice_stress, init_stress_tables, seed_from_scan, suggest_minimal_remedies)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    cx.executescript("""
        CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT);
        CREATE TABLE fmp_snap_client_causal_chain(id_pk INTEGER, id_fk_active_stress INTEGER);
        CREATE TABLE fmp_snap_client_remedy(id_fk_causal_chain INTEGER, remedy TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress VALUES(1,'Adrenal Fatigue')")
    cx.execute("INSERT INTO fmp_snap_client_causal_chain VALUES(10,1)")
    cx.execute("INSERT INTO fmp_snap_client_remedy VALUES(10,'Adaptogen Blend')")
    cx.commit()
    return cx


def test_nonscan_stress_in_setcover(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Adrenal Fatigue")
    res = suggest_minimal_remedies(cx, "a5", [])
    assert {"remedy": "adaptogen blend", "covers": ["Adrenal Fatigue"]} in res["picks"]
    assert res["uncovered"] == []


def test_nonscan_no_history_uncovered(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Mystery Stress")          # no FMP history
    res = suggest_minimal_remedies(cx, "a5", [])
    assert res["picks"] == [] and res["uncovered"] == ["Mystery Stress"]


def test_scan_and_nonscan_covered_together(tmp_path):
    cx = _cx(tmp_path)
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Membrane"}],
                   {"neuro magnesium": {"ED1"}})           # scan stress, required
    add_voice_stress(cx, "a5", "Adrenal Fatigue")          # non-scan, historical
    res = suggest_minimal_remedies(cx, "a5", [])
    by = {p["remedy"]: p["covers"] for p in res["picks"]}
    assert by.get("neuro magnesium") == ["Membrane"]
    assert by.get("adaptogen blend") == ["Adrenal Fatigue"]
    assert res["uncovered"] == []
```

- [ ] **Step 2: Run** → FAIL (non-scan not in picks).

- [ ] **Step 3: Implement** — replace `suggest_minimal_remedies` in `dashboard/biofield_stress.py` with:

```python
def suggest_minimal_remedies(cx, tid, chain_rows):
    """Fewest remedies covering active+required stresses (scan via the coverage map,
    non-scan via historical stress_suggestions). Cover token = E4L code (scan) or
    _norm(label) (non-scan). Returns picks (remedy + covered LABELS) + uncovered labels."""
    from dashboard.biofield_setcover import minimal_remedies
    data = list_stresses(cx, tid, chain_rows)
    token_label, active_tokens, coverage = {}, set(), {}
    # scan coverage from the persisted map
    for remedy, code in cx.execute(
            "SELECT remedy, code FROM biofield_auth_remedy_coverage WHERE test_id=?",
            (_num(tid),)).fetchall():
        coverage.setdefault(remedy, set()).add(code)
    for s in data["active"]:
        if s.get("balance") != "required":
            continue
        if s.get("source") == "scan":
            code = s.get("code") or ""
            if code:
                active_tokens.add(code)
                token_label[code] = s.get("label") or code
        else:                                            # non-scan: token = norm-label
            tok = _norm(s.get("label") or "")
            if not tok:
                continue
            active_tokens.add(tok)
            token_label[tok] = s.get("label") or tok
            for rem in historical_remedies(cx, s.get("label") or ""):
                coverage.setdefault(rem, set()).add(tok)
    res = minimal_remedies(active_tokens, coverage)
    picks = [{"remedy": p["remedy"], "covers": [token_label.get(c, c) for c in p["covers"]]}
             for p in res["picks"]]
    uncovered = [token_label.get(c, c) for c in res["uncovered"]]
    return {"picks": picks, "uncovered": uncovered}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_covext_setcover.py tests/test_biofield_suggest_remedies.py -v` → PASS (existing B4 scan-only suggest tests stay green: with no non-scan active stresses, `active_tokens`/`coverage` are exactly the scan codes/map as before).

- [ ] **Step 5: Run the CovExt + adjacent suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_biofield_covext_balance.py tests/test_biofield_covext_setcover.py \
  tests/test_biofield_stress_derive.py tests/test_biofield_stress_labelmatch.py \
  tests/test_biofield_suggest_remedies.py tests/test_biofield_stress_seed.py \
  tests/test_biofield_setcover.py -q
git add dashboard/biofield_stress.py tests/test_biofield_covext_setcover.py
git commit -m "feat(covext): token-generalized set-cover includes non-scan stresses"
```

---

## Self-Review

**Spec coverage:**
- `historical_remedies` from FMP `stress_suggestions` → Task 1. ✓
- Non-scan auto-balance path + precedence (code > label-match > historical > manual) → Task 1. ✓
- Token-generalized set-cover incl. non-scan; no-history → uncovered → Task 2. ✓
- Scan behavior unchanged; `minimal_remedies` untouched → both tasks (scan path identical when no non-scan active). ✓
- Local-only, only biofield_stress.py changes → confirmed. ✓

**Placeholder scan:** No TBDs; complete code; Task 1 Step 3 explicitly says keep the pre-loop lines unchanged.

**Type consistency:** `historical_remedies(cx, label) -> set[str]` (T1) used in T1 loop + T2 coverage build; `_norm` (existing) used for non-scan tokens in both; `minimal_remedies(active_tokens, coverage)` token contract matches (opaque strings). `balanced_by` lowercased remedies consistent with `_coverers`.

## Verification (manual, after both tasks)

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```
For a client with voice/tag/comm stresses whose names have FMP history: putting a historically-used remedy on a chain layer moves the matching non-scan stress to Balanced; "Suggest minimal remedies" now lists remedies for non-scan stresses too (and flags ones with no history as uncovered).
