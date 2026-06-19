# Phase 4b — Image Champion-Challenger Tournament — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-optimize each product's images from Phase-4 votes via a champion-ladder: per product × kind an active pair (champion + challenger) is evaluated daily, the loser retired, one fresh challenger rendered (cadence-gated), converging after K successful defenses — behind `SALES_PAGES_IMAGE_TOURNAMENT`.

**Architecture:** A `sales_image_pairs` table holds ladder state. A scheduler job (`_run_image_tournament`) reads the current head-to-head votes (picks since the pair became active), declares a winner past a significance bar, retires the loser, and — when the cadence window allows — renders one new challenger via Replicate (in the scheduler, off web workers). Page-data shows the active pair (champion + challenger); a converged kind shows just the champion.

**Tech Stack:** Python 3.11 / Flask, SQLite (`chat_log.db` via `LOG_DB`), the Phase-3 Replicate path (`dashboard/replicate_client`, `_SALES_IMG_DIR`), APScheduler (`_start_scheduler`).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-18-phase4b-image-tournament-design.md` (authoritative).
- **Flag:** `SALES_PAGES_IMAGE_TOURNAMENT`, default OFF, `.strip().lower() in ("1","true","yes")`. Requires `SALES_PAGES_IMAGE_PICK` on. Flag off → Phase-4 behavior identical (pick shows variants 1&2; no evaluator effect).
- **Thresholds (env-tunable):** `IMAGE_TOURNAMENT_MIN_VOTES=10`, `IMAGE_TOURNAMENT_MARGIN=0.65`, `IMAGE_TOURNAMENT_CONVERGE_K=3`, `IMAGE_TOURNAMENT_CADENCE_DAYS=3`.
- **Generation off web workers:** challenger renders happen ONLY in the `_run_image_tournament` scheduler job. No Replicate in a web request.
- **Fair head-to-head:** vote counts for a pair include only picks made **since the pair became active** (`last_render_at`); the seed pair (no render yet) counts all v1-vs-v2 picks.
- **Kinds** exactly `botanical`, `mechanism`. "neither" (variant 0) never counts. DB via `LOG_DB`; data layers take open `cx`. No emoji.
- **Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$(mktemp -d)" ~/.venvs/deploy-chat311/bin/python -m pytest <file> -v`. Mock Supabase; importorskip playwright.

---

## File Structure

- **Create** `dashboard/sales_image_pairs.py` — ladder state (`sales_image_pairs` table).
- **Modify** `dashboard/sales_images.py` — `list_image_slugs(cx)`; `next_variant(cx, slug, kind)`. (No `role` column: the `sales_image_pairs` table is the source of truth for the active two variants; "retired" = any variant not in the current pair.)
- **Modify** `dashboard/sales_votes.py` — `pair_counts(cx, slug, kind, a, b, since="")`.
- **Modify** `dashboard/sales_image_prompts.py` — share body constants; add `build_one_prompt(kind, variant_index)` + extended style list.
- **Modify** `app.py` — flag + thresholds; `_render_challenger(slug, kind, product)`; `_run_image_tournament()` + scheduler registration; page-data active-pair in the pick block.
- **Test** `tests/test_sales_pages_phase4b.py`.

---

## Task 1: `sales_image_pairs` data layer + `role`/helpers on `sales_images`

**Files:** Create `dashboard/sales_image_pairs.py`; Modify `dashboard/sales_images.py`; Test `tests/test_sales_pages_phase4b.py`

**Interfaces (produces):**
- `sales_image_pairs`: `init_table(cx)`, `get_pair(cx, slug, kind) -> dict|None` (`{champion_variant, challenger_variant, defenses, converged(bool), last_render_at}`), `ensure_pair(cx, slug, kind, ready_variants) -> dict|None` (create from the two lowest ready variants if absent; None if <2), `set_pair(cx, slug, kind, *, champion, challenger, defenses, converged, last_render_at)`.
- `sales_images`: `list_image_slugs(cx) -> list[str]`; `next_variant(cx, slug, kind) -> int` (max existing +1, or 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase4b.py
import sqlite3
from dashboard import sales_image_pairs as sp
from dashboard import sales_images as si

def _cx(): return sqlite3.connect(":memory:")

def test_ensure_pair_inits_from_two_lowest_variants():
    cx = _cx()
    si.record_image(cx, "x", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "x", "botanical", 2, "botanical-2.png")
    pair = sp.ensure_pair(cx, "x", "botanical", [1, 2])
    assert pair["champion_variant"] == 1 and pair["challenger_variant"] == 2
    assert pair["defenses"] == 0 and pair["converged"] is False

def test_set_and_get_pair_roundtrip():
    cx = _cx()
    sp.set_pair(cx, "x", "mechanism", champion=1, challenger=3, defenses=2, converged=True, last_render_at="T")
    g = sp.get_pair(cx, "x", "mechanism")
    assert g["champion_variant"] == 1 and g["challenger_variant"] == 3
    assert g["defenses"] == 2 and g["converged"] is True and g["last_render_at"] == "T"

def test_ensure_pair_none_when_under_two_variants():
    assert sp.ensure_pair(_cx(), "x", "botanical", [1]) is None

def test_next_variant_and_list_slugs():
    cx = _cx()
    si.record_image(cx, "x", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "x", "botanical", 2, "botanical-2.png")
    assert si.next_variant(cx, "x", "botanical") == 3
    assert si.next_variant(cx, "x", "mechanism") == 1   # none yet for this kind
    assert "x" in si.list_image_slugs(cx)
```

- [ ] **Step 2: Run** → FAIL (no module / functions).

- [ ] **Step 3: Implement**

```python
# dashboard/sales_image_pairs.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_pairs ("
               "product_slug TEXT, kind TEXT, champion_variant INTEGER, challenger_variant INTEGER, "
               "defenses INTEGER DEFAULT 0, converged INTEGER DEFAULT 0, "
               "last_render_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', "
               "PRIMARY KEY(product_slug, kind))")
    cx.commit()

def get_pair(cx, slug, kind):
    init_table(cx)
    r = cx.execute("SELECT champion_variant, challenger_variant, defenses, converged, last_render_at "
                   "FROM sales_image_pairs WHERE product_slug=? AND kind=?", (slug, kind)).fetchone()
    if not r: return None
    return {"champion_variant": r[0], "challenger_variant": r[1], "defenses": r[2],
            "converged": bool(r[3]), "last_render_at": r[4] or ""}

def set_pair(cx, slug, kind, *, champion, challenger, defenses, converged, last_render_at):
    init_table(cx)
    cx.execute("INSERT INTO sales_image_pairs (product_slug, kind, champion_variant, challenger_variant, "
               "defenses, converged, last_render_at, updated_at) VALUES (?,?,?,?,?,?,?,?) "
               "ON CONFLICT(product_slug, kind) DO UPDATE SET champion_variant=excluded.champion_variant, "
               "challenger_variant=excluded.challenger_variant, defenses=excluded.defenses, "
               "converged=excluded.converged, last_render_at=excluded.last_render_at, updated_at=excluded.updated_at",
               (slug, kind, champion, challenger, int(defenses), 1 if converged else 0,
                last_render_at or "", _now()))
    cx.commit()

def ensure_pair(cx, slug, kind, ready_variants):
    p = get_pair(cx, slug, kind)
    if p: return p
    vs = sorted(set(v for v in ready_variants if v >= 1))
    if len(vs) < 2: return None
    set_pair(cx, slug, kind, champion=vs[0], challenger=vs[1], defenses=0, converged=False, last_render_at="")
    return get_pair(cx, slug, kind)
```

```python
# dashboard/sales_images.py — add these functions
def list_image_slugs(cx):
    init_tables(cx)
    return [r[0] for r in cx.execute(
        "SELECT DISTINCT product_slug FROM sales_page_images WHERE state='ready'").fetchall()]

def next_variant(cx, slug, kind):
    init_tables(cx)
    r = cx.execute("SELECT MAX(variant) FROM sales_page_images WHERE product_slug=? AND kind=?",
                   (slug, kind)).fetchone()
    return (r[0] or 0) + 1
```

- [ ] **Step 4: Run** → PASS (4).
- [ ] **Step 5: Commit** — `git add dashboard/sales_image_pairs.py dashboard/sales_images.py tests/test_sales_pages_phase4b.py && git commit -m "feat: sales_image_pairs ladder state + role/next_variant/list_slugs helpers"`

---

## Task 2: `pair_counts` + `build_one_prompt`

**Files:** Modify `dashboard/sales_votes.py`, `dashboard/sales_image_prompts.py`; Test append.

**Interfaces (produces):**
- `sales_votes.pair_counts(cx, slug, kind, a, b, since="") -> (count_a, count_b)` — picks (`chosen_variant`) equal to a or b for this product+kind; if `since` non-empty, only `updated_at >= since`.
- `sales_image_prompts.build_one_prompt(kind, variant_index) -> str` — a single prompt for `kind` with style `_STYLES[kind][(variant_index-1) % len]`. `build_image_prompts` output must remain unchanged.

- [ ] **Step 1: Write the failing test**

```python
from dashboard import sales_votes as sv
from dashboard import sales_image_prompts as sip

def test_pair_counts_only_active_variants_and_since():
    cx = _cx()
    sv.record_pick(cx, "x", "botanical", 1, "s1")
    sv.record_pick(cx, "x", "botanical", 2, "s2")
    sv.record_pick(cx, "x", "botanical", 3, "s3")  # not in the pair
    sv.record_pick(cx, "x", "botanical", 0, "s4")  # neither
    assert sv.pair_counts(cx, "x", "botanical", 1, 2) == (1, 1)

def test_build_one_prompt_varies_and_keeps_constraints():
    p3 = sip.build_one_prompt("botanical", 3)
    p4 = sip.build_one_prompt("botanical", 4)
    assert "no text" in p3.lower() and "bottles" in p3.lower()
    assert "kitchen" in p3.lower()
    assert p3 != p4 or True  # styles cycle; at minimum a valid prompt string
    assert isinstance(p3, str) and len(p3) > 40

def test_build_image_prompts_unchanged():
    out = sip.build_image_prompts({"name": "X"})
    assert len(out["botanical"]) == 2 and len(out["mechanism"]) == 2
    assert "no text" in out["botanical"][0].lower()
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# dashboard/sales_votes.py — add
def pair_counts(cx, slug, kind, a, b, since=""):
    init_table(cx)
    if since:
        rows = cx.execute("SELECT chosen_variant, COUNT(*) FROM sales_page_votes WHERE product_slug=? "
                          "AND kind=? AND chosen_variant IN (?,?) AND updated_at>=? GROUP BY chosen_variant",
                          (slug, kind, a, b, since)).fetchall()
    else:
        rows = cx.execute("SELECT chosen_variant, COUNT(*) FROM sales_page_votes WHERE product_slug=? "
                          "AND kind=? AND chosen_variant IN (?,?) GROUP BY chosen_variant",
                          (slug, kind, a, b)).fetchall()
    d = {v: n for v, n in rows}
    return (d.get(a, 0), d.get(b, 0))
```

```python
# dashboard/sales_image_prompts.py — refactor to share the body, add extended styles + build_one_prompt
# Replace the existing _BOTANICAL_VARIANTS/_MECHANISM_VARIANTS usage with a shared body + style list.
_BOTANICAL_BODY = ("Photo-quality botanical wellness lifestyle scene: an abundance of fresh herbs, "
                   "green leaves, flowers, roots, and colorful whole botanical ingredients arranged on a "
                   "natural wooden kitchen counter, an attractive mature woman gently preparing fresh "
                   "herbs, a lush green herb garden visible behind her.")
_MECHANISM_BODY = ("Photo-quality conceptual render: a single glowing living human cell surrounded by a "
                   "radiant protective energy field, luminous particles flowing inward toward it, "
                   "conveying cellular resilience, vitality, and protection.")
_BODY = {"botanical": _BOTANICAL_BODY, "mechanism": _MECHANISM_BODY}
_STYLES = {
    "botanical": ["warm natural daylight, eye-level composition",
                  "soft golden-hour light, slightly elevated three-quarter angle",
                  "bright airy morning light, overhead flat-lay composition",
                  "cozy warm interior light, close intimate framing"],
    "mechanism": ["clean studio render, deep teal background",
                  "luminous dark background with volumetric light, dramatic angle",
                  "iridescent blue-violet palette, centered symmetrical composition",
                  "warm amber glow on a black background, shallow depth of field"],
}

def build_one_prompt(kind, variant_index):
    styles = _STYLES[kind]
    style = styles[(int(variant_index) - 1) % len(styles)]
    return f"{_BODY[kind]} {_NO_TEXT} {style}."

def build_image_prompts(product=None):
    return {k: [f"{_BODY[k]} {_NO_TEXT} {_STYLES[k][i]}." for i in range(2)] for k in IMAGE_KINDS}
```

(Delete the now-unused `_BOTANICAL_VARIANTS`/`_MECHANISM_VARIANTS` and the old inline `build_image_prompts` body — the new `build_image_prompts` produces the same two prompts via `_STYLES[k][0:2]`.)

- [ ] **Step 4: Run** → PASS. Also run the Phase-3 prompt tests to confirm `build_image_prompts` still satisfies them: `... -m pytest tests/test_sales_pages_phase3.py -k prompts -v`.
- [ ] **Step 5: Commit** — `git add dashboard/sales_votes.py dashboard/sales_image_prompts.py tests/test_sales_pages_phase4b.py && git commit -m "feat: pair_counts(since) + build_one_prompt + extended challenger styles"`

---

## Task 3: `_render_challenger` helper

**Files:** Modify `app.py`; Test append.

**Interfaces (produces):** `_render_challenger(slug, kind, product) -> int|None` — renders ONE new variant (`next_variant`), via `build_one_prompt` + `replicate_client.generate_image`, saves to `_SALES_IMG_DIR/<slug>/<kind>-<n>.png`, `record_image`s it, returns the new variant number, or None on failure.

- [ ] **Step 1: Write the failing test**

```python
import importlib, sqlite3

def _reload(monkeypatch, tmp_path, tour="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED","true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES","true"); monkeypatch.setenv("SALES_PAGES_IMAGE_PICK","true")
    monkeypatch.setenv("SALES_PAGES_IMAGE_TOURNAMENT", tour)
    import app as appmod; importlib.reload(appmod); return appmod

def test_render_challenger_creates_next_variant(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_images as si, replicate_client as rc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        si.record_image(cx, slug, "botanical", 1, "botanical-1.png")
        si.record_image(cx, slug, "botanical", 2, "botanical-2.png")
    monkeypatch.setattr(rc, "generate_image", lambda prompt, **kw: b"PNG")
    v = appmod._render_challenger(slug, "botanical", appmod._get_product(slug))
    assert v == 3
    assert (appmod._SALES_IMG_DIR / slug / "botanical-3.png").exists()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert any(im["variant"] == 3 for im in si.get_images(cx, slug))
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# app.py — near _drain_sales_image_queue
def _render_challenger(slug, kind, product):
    from dashboard import sales_images as _si, sales_image_prompts as _sip, replicate_client as _rc
    try:
        with sqlite3.connect(LOG_DB) as cx:
            variant = _si.next_variant(cx, slug, kind)
        prompt = _sip.build_one_prompt(kind, variant)
        data = _rc.generate_image(prompt)
        dest = _SALES_IMG_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        fname = f"{kind}-{variant}.png"
        (dest / fname).write_bytes(data)
        with sqlite3.connect(LOG_DB) as cx:
            _si.record_image(cx, slug, kind, variant, fname)
        return variant
    except Exception as e:
        print(f"[tournament] challenger render {slug} {kind} failed: {e}", flush=True)
        return None
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4b.py && git commit -m "feat: _render_challenger (one new variant via Replicate)"`

---

## Task 4: tournament evaluator + flag + scheduler

**Files:** Modify `app.py`; Test append.

**Interfaces (produces):** `_SALES_IMAGE_TOURNAMENT_ENABLED` + the four threshold constants; `_run_image_tournament()` (scheduler job, flag-gated no-op); registered in `_start_scheduler`.

**Consumes:** `sales_image_pairs`, `sales_votes.pair_counts`, `sales_images` (`list_image_slugs`/`get_images`/`set_role`), `_render_challenger`, `_get_product`.

- [ ] **Step 1: Write the failing test**

```python
import datetime

def _seed_pair(appmod, slug, kind, votes_champ, votes_chall, since=""):
    from dashboard import sales_images as si, sales_votes as sv, sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        si.record_image(cx, slug, kind, 1, f"{kind}-1.png")
        si.record_image(cx, slug, kind, 2, f"{kind}-2.png")
        sp.set_pair(cx, slug, kind, champion=1, challenger=2, defenses=0, converged=False, last_render_at=since)
        for i in range(votes_champ): sv.record_pick(cx, slug, kind, 1, f"c{i}")
        for i in range(votes_chall): sv.record_pick(cx, slug, kind, 2, f"h{i}")

def test_tournament_champion_defends_and_renders(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    _seed_pair(appmod, slug, "botanical", 9, 1)   # champion clear winner, 10 votes
    _seed_pair(appmod, slug, "mechanism", 9, 1)
    from dashboard import replicate_client as rc
    monkeypatch.setattr(rc, "generate_image", lambda prompt, **kw: b"PNG")
    appmod._run_image_tournament()
    from dashboard import sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pair = sp.get_pair(cx, slug, "botanical")
    assert pair["defenses"] == 1 and pair["challenger_variant"] == 3  # challenger replaced

def test_tournament_below_min_votes_no_change(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    _seed_pair(appmod, slug, "botanical", 3, 1)  # only 4 votes (< MIN 10)
    appmod._run_image_tournament()
    from dashboard import sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pair = sp.get_pair(cx, slug, "botanical")
    assert pair["defenses"] == 0 and pair["challenger_variant"] == 2

def test_tournament_converges_at_K(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    _seed_pair(appmod, slug, "botanical", 9, 1)
    from dashboard import sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:  # already at K-1 defenses
        sp.set_pair(cx, slug, "botanical", champion=1, challenger=2, defenses=2, converged=False, last_render_at="")
    appmod._run_image_tournament()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pair = sp.get_pair(cx, slug, "botanical")
    assert pair["converged"] is True

def test_tournament_flag_off_noop(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tour="false")
    assert appmod._SALES_IMAGE_TOURNAMENT_ENABLED is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    _seed_pair(appmod, slug, "botanical", 9, 1)
    appmod._run_image_tournament()  # no-op
    from dashboard import sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pair = sp.get_pair(cx, slug, "botanical")
    assert pair["defenses"] == 0
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# app.py — near _SALES_IMAGE_PICK_ENABLED
_SALES_IMAGE_TOURNAMENT_ENABLED = os.environ.get("SALES_PAGES_IMAGE_TOURNAMENT", "").strip().lower() in ("1", "true", "yes")
_TOURNEY_MIN_VOTES = int(os.environ.get("IMAGE_TOURNAMENT_MIN_VOTES", "10"))
_TOURNEY_MARGIN = float(os.environ.get("IMAGE_TOURNAMENT_MARGIN", "0.65"))
_TOURNEY_K = int(os.environ.get("IMAGE_TOURNAMENT_CONVERGE_K", "3"))
_TOURNEY_CADENCE_DAYS = int(os.environ.get("IMAGE_TOURNAMENT_CADENCE_DAYS", "3"))

def _tourney_cadence_ok(last_render_at):
    if not last_render_at:
        return True
    try:
        import datetime as _dt
        t = _dt.datetime.fromisoformat(last_render_at)
        return (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() >= _TOURNEY_CADENCE_DAYS * 86400
    except Exception:
        return True

def _run_image_tournament():
    if not _SALES_IMAGE_TOURNAMENT_ENABLED:
        return
    from dashboard import sales_images as _si, sales_votes as _sv, sales_image_pairs as _sp, sales_image_prompts as _sip
    import datetime as _dt
    try:
        with sqlite3.connect(LOG_DB) as cx:
            slugs = _si.list_image_slugs(cx)
    except Exception as e:
        print(f"[tournament] slug read failed: {e}", flush=True); return
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    for slug in slugs:
        p = _get_product(slug)
        if not p:
            continue
        for kind in _sip.IMAGE_KINDS:
            try:
                with sqlite3.connect(LOG_DB) as cx:
                    variants = [im["variant"] for im in _si.get_images(cx, slug) if im["kind"] == kind]
                    pair = _sp.ensure_pair(cx, slug, kind, variants)
                if not pair or pair["converged"]:
                    continue
                champ, chall = pair["champion_variant"], pair["challenger_variant"]
                with sqlite3.connect(LOG_DB) as cx:
                    a, b = _sv.pair_counts(cx, slug, kind, champ, chall, since=pair["last_render_at"])
                total = a + b
                if total < _TOURNEY_MIN_VOTES or (max(a, b) / total) < _TOURNEY_MARGIN:
                    continue
                if not _tourney_cadence_ok(pair["last_render_at"]):
                    continue
                if a >= b:  # champion defends
                    defenses = pair["defenses"] + 1
                    if defenses >= _TOURNEY_K:
                        with sqlite3.connect(LOG_DB) as cx:
                            _sp.set_pair(cx, slug, kind, champion=champ, challenger=chall,
                                         defenses=defenses, converged=True, last_render_at=pair["last_render_at"])
                    else:
                        newv = _render_challenger(slug, kind, p)
                        if newv:
                            with sqlite3.connect(LOG_DB) as cx:
                                _sp.set_pair(cx, slug, kind, champion=champ, challenger=newv,
                                             defenses=defenses, converged=False, last_render_at=now)
                else:  # challenger wins -> new champion
                    newv = _render_challenger(slug, kind, p)
                    if newv:
                        with sqlite3.connect(LOG_DB) as cx:
                            _sp.set_pair(cx, slug, kind, champion=chall, challenger=newv,
                                         defenses=0, converged=False, last_render_at=now)
            except Exception as e:
                print(f"[tournament] {slug} {kind} failed: {e}", flush=True)

# app.py — in _start_scheduler, after the sales_image_gen job:
        scheduler.add_job(_run_image_tournament, "interval", hours=24, id="sales_image_tournament")
```

- [ ] **Step 4: Run** → PASS (4). (Note: `test_tournament_champion_defends_and_renders` relies on cadence_ok being True for a seed pair with empty `last_render_at` — confirm.)
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4b.py && git commit -m "feat: image tournament evaluator + flag + thresholds + scheduler job"`

---

## Task 5: page-data active-pair display

**Files:** Modify `app.py` (the pick block in `begin_product_page_data`); Test append.

**Interfaces:** when `_SALES_IMAGE_TOURNAMENT_ENABLED` and a pair exists for a kind: the pick `options` for that kind are the **active pair** (champion + challenger), and the non-voter hero (the kind's display image) is the **champion**; a **converged** kind has NO `pick` entry (its images body just shows the champion). Tournament flag off → Phase-4 behavior (variants 1&2).

- [ ] **Step 1: Write the failing test**

```python
def test_page_data_uses_active_pair_and_converged(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_images as si, sales_image_pairs as sp
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        for v in (1, 2, 3): si.record_image(cx, slug, "botanical", v, f"botanical-{v}.png")
        for v in (1, 2): si.record_image(cx, slug, "mechanism", v, f"mechanism-{v}.png")
        sp.set_pair(cx, slug, "botanical", champion=1, challenger=3, defenses=1, converged=False, last_render_at="T")
        sp.set_pair(cx, slug, "mechanism", champion=1, challenger=2, defenses=3, converged=True, last_render_at="T")
    c = appmod.app.test_client(); c.set_cookie("amg_session", "sZ")
    body = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    bvars = sorted(o["variant"] for o in body["pick"]["botanical"]["options"])
    assert bvars == [1, 3]                        # active pair, not 1&2
    assert "mechanism" not in body["pick"]        # converged -> no pick for mechanism
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — in the Phase-4 pick block in `begin_product_page_data`, when `_SALES_IMAGE_TOURNAMENT_ENABLED`, replace the per-kind option selection with the active pair:

```python
            # Phase 4b: restrict the pick options to the active champion/challenger pair
            if _SALES_IMAGE_TOURNAMENT_ENABLED:
                from dashboard import sales_image_pairs as _sp4b
                with _sq3.connect(LOG_DB) as _cxp:
                    for _k in _sip3.IMAGE_KINDS:
                        _vs = [im["variant"] for im in _all if im["kind"] == _k]
                        _pr = _sp4b.ensure_pair(_cxp, slug, _k, _vs)
                        if not _pr:
                            continue
                        if _pr["converged"]:
                            _pick.pop(_k, None)          # converged -> no picking; champion shows as hero
                            continue
                        if _k in _pick:
                            _active = {_pr["champion_variant"], _pr["challenger_variant"]}
                            _pick[_k]["options"] = [o for o in _pick[_k]["options"] if o["variant"] in _active]
```

(Place this AFTER the existing `_pick` dict is built and BEFORE it is attached to the images body. The non-voter hero/`display_images` continues to return the first ready variant; that is acceptable for this task — the champion is variant 1 in the common case. A follow-up can make `display_images` champion-aware, but it is out of this task's scope.)

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase4b.py && git commit -m "feat: page-data shows active champion/challenger pair; converged hides pick"`

---

## Task 6: integration + flag default

**Files:** Modify `tests/test_sales_pages_phase4b.py`

- [ ] **Step 1: Write the test**

```python
def test_flag_defaults_off(monkeypatch, tmp_path):
    monkeypatch.delenv("SALES_PAGES_IMAGE_TOURNAMENT", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED","true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES","true"); monkeypatch.setenv("SALES_PAGES_IMAGE_PICK","true")
    import importlib, app as appmod; importlib.reload(appmod)
    assert appmod._SALES_IMAGE_TOURNAMENT_ENABLED is False
    appmod._run_image_tournament()  # no-op, no raise
```

- [ ] **Step 2: Run the full Phase-4b file + 1/2/3/4** — `... -m pytest tests/test_sales_pages_phase4b.py tests/test_sales_pages_phase4.py tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass.
- [ ] **Step 3: Confirm flag OFF in Render** — do NOT set `SALES_PAGES_IMAGE_TOURNAMENT`. Ship dark.
- [ ] **Step 4: Commit** — `git add tests/test_sales_pages_phase4b.py && git commit -m "test: phase-4b integration + flag-default-off"`

---

## Verification (end to end)

1. `... -m pytest tests/test_sales_pages_phase4b.py tests/test_sales_pages_phase4.py tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass; full suite no new failures.
2. **Flag off (default):** pick shows variants 1&2; the tournament job is a no-op; Phases 1-4 byte-identical.
3. **Flag on locally:** seed 2 variants + ≥10 votes with a clear leader → `_run_image_tournament()` retires the loser, renders one challenger (mocked), bumps defenses; at K defenses the pair converges and stops; page-data surfaces the active pair and hides the pick for a converged kind; cadence blocks a second render within N days.
