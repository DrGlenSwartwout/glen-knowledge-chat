# Chat Page-Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a `/begin` chat message is about a topic that already has a published page, surface an active link to that page as a card below the answer — deterministically, with no extra LLM call.

**Architecture:** A pure, unit-tested `dashboard/page_links.py` (a phrase-index builder + a word-boundary matcher) plus a small hand-maintained `data/page-aliases.json`. `app.py` builds a short-TTL-cached index from approved topic/ingredient/product pages and, in the `/chat` SSE done handler, matches the query + answer text and merges link cards into the existing `surfaced_cards`. Ships dark behind `CHAT_PAGE_LINKS_ENABLED`.

**Tech Stack:** Python 3, Flask, sqlite3 (`LOG_DB`), pytest. No new dependencies. Reuses the existing `surfaced_cards` card shape (`{key, title, sub, href}`) so the client needs no change.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-21-chat-page-links-design.md`. Every task inherits its requirements.
- **Never `import app` in tests** — Pinecone is built at import and fails in the sandbox. Test `dashboard/*` helpers directly; verify `app.py` with `python3 -m py_compile app.py`. Use `python3`, not `python`.
- `dashboard/page_links.py` is **pure**: no Flask import, no DB access, no network. Functions take plain data and return plain data; they must not raise on normal input.
- Match method is **deterministic name/alias** over the user query AND the AI answer text. No embeddings, no LLM call.
- Eligible pages are **approved** only: topic (`/learn/<slug>`, `kind="topic"`, `gated=False`), ingredient (`/begin/ingredient/<slug>`, `kind="ingredient"`, `gated=True`), product (`/begin/product/<slug>`, `kind="product"`, `gated=False`).
- Matching is **word-boundary** (so `"iron"` does NOT match inside `"environment"`) and **longest-phrase-first** (so `"magnesium glycinate"` wins over `"magnesium"`, and a claimed span is not double-matched).
- Card shape: `{"key": f"{kind}:{slug}", "title": <page name>, "sub": <type label>, "href": <url>}`. Sub labels: topic→`"Read the guide"`, ingredient→`"See the ingredient"`, product→`"View product"`.
- Link cards are **deduped by href** and capped at **2**. The combined `surfaced_cards` list is capped at **3 total ONLY when link cards fired** (flag off, or zero link matches → existing card behavior is byte-identical).
- Ingredient-page links **show to everyone** (no member gating in the matcher).
- Ships **DARK behind `CHAT_PAGE_LINKS_ENABLED`** (default off). Flag off → the matcher is never invoked.

---

### Task 1: `dashboard/page_links.py` — pure index + matcher (+ alias seed file)

**Files:**
- Create: `dashboard/page_links.py`
- Create: `data/page-aliases.json`
- Test: `tests/test_page_links.py`

**Interfaces:**
- Produces:
  - `load_aliases(path) -> dict` — read `{phrase: slug}` JSON; missing/unreadable file → `{}` (never raises).
  - `build_index(pages, *, alias_map=None) -> dict` — `pages` is a list of `{slug, name, kind, href, gated}`; returns `{phrase_lower: {"title", "href", "kind", "gated"}}`. Phrases per page = its `name`, its slug-as-words (`"low-energy"`→`"low energy"`), and any alias phrases mapping to that slug. On phrase collision the first page wins; a later alias never overwrites a real page name.
  - `match_page_links(text, index, *, limit=2) -> list[dict]` — returns ordered, href-deduped link-card dicts `{key, title, sub, href}`, longest-phrase-first, word-boundary, capped at `limit`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_page_links.py
import json
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import page_links
        return page_links
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"page_links not importable: {e}")


def _pages():
    return [
        {"slug": "low-energy", "name": "Low Energy", "kind": "topic",
         "href": "/learn/low-energy", "gated": False},
        {"slug": "brain-fog", "name": "Brain Fog", "kind": "topic",
         "href": "/learn/brain-fog", "gated": False},
        {"slug": "magnesium", "name": "Magnesium", "kind": "ingredient",
         "href": "/begin/ingredient/magnesium", "gated": True},
        {"slug": "magnesium-glycinate", "name": "Magnesium Glycinate", "kind": "ingredient",
         "href": "/begin/ingredient/magnesium-glycinate", "gated": True},
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "kind": "product",
         "href": "/begin/product/neuro-magnesium", "gated": False},
    ]


def _idx(aliases=None):
    pl = _mod()
    return pl.build_index(_pages(), alias_map=aliases or {})


def test_matches_page_named_in_text():
    pl = _mod()
    cards = pl.match_page_links("I keep struggling with low energy lately", _idx())
    assert cards and cards[0]["href"] == "/learn/low-energy"
    assert cards[0]["key"] == "topic:low-energy"
    assert cards[0]["title"] == "Low Energy"
    assert cards[0]["sub"] == "Read the guide"


def test_matches_term_only_in_answer_text():
    # the AI answer mentions "brain fog" even though the user didn't
    pl = _mod()
    cards = pl.match_page_links("why am I so tired? Many people experience brain fog too.", _idx())
    hrefs = [c["href"] for c in cards]
    assert "/learn/brain-fog" in hrefs


def test_word_boundary_no_substring_false_positive():
    # "iron" must NOT match inside "environment"; add an iron page to prove it
    pl = _mod()
    pages = _pages() + [{"slug": "iron", "name": "Iron", "kind": "ingredient",
                         "href": "/begin/ingredient/iron", "gated": True}]
    idx = pl.build_index(pages, alias_map={})
    cards = pl.match_page_links("a calm environment helps recovery", idx)
    assert all(c["href"] != "/begin/ingredient/iron" for c in cards)


def test_longest_phrase_wins_no_double_match():
    pl = _mod()
    cards = pl.match_page_links("I take magnesium glycinate daily", _idx(), limit=5)
    hrefs = [c["href"] for c in cards]
    assert "/begin/ingredient/magnesium-glycinate" in hrefs
    # the substring "magnesium" page must NOT also be surfaced from the same span
    assert "/begin/ingredient/magnesium" not in hrefs


def test_dedupe_by_href_and_cap():
    pl = _mod()
    cards = pl.match_page_links("low energy, low energy, brain fog, magnesium", _idx(), limit=2)
    assert len(cards) == 2
    assert len({c["href"] for c in cards}) == 2


def test_alias_maps_paraphrase_to_page():
    pl = _mod()
    idx = _idx({"can't focus": "brain-fog"})
    cards = pl.match_page_links("I just can't focus these days", idx)
    assert any(c["href"] == "/learn/brain-fog" for c in cards)


def test_sub_labels_per_kind():
    pl = _mod()
    cards = pl.match_page_links("neuro magnesium and low energy", _idx(), limit=5)
    by_href = {c["href"]: c for c in cards}
    assert by_href["/begin/product/neuro-magnesium"]["sub"] == "View product"
    assert by_href["/learn/low-energy"]["sub"] == "Read the guide"


def test_load_aliases_missing_file_is_empty():
    pl = _mod()
    assert pl.load_aliases("/no/such/file.json") == {}


def test_load_aliases_reads_seed_file():
    pl = _mod()
    seed = Path(__file__).resolve().parent.parent / "data" / "page-aliases.json"
    data = pl.load_aliases(str(seed))
    assert isinstance(data, dict)


def test_no_match_returns_empty():
    pl = _mod()
    assert pl.match_page_links("the weather is nice today", _idx()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_page_links.py -q`
Expected: SKIP/FAIL — `page_links not importable`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/page_links.py
"""Pure deterministic matcher: surface already-published pages as chat link cards.

No Flask, no DB, no network. build_index() turns a list of approved-page records
into a phrase lookup; match_page_links() finds whole-word phrase hits in text
(longest-first) and returns ready-to-render card dicts. Never raises on normal input.
"""
import json
import re

_SUB_BY_KIND = {
    "topic": "Read the guide",
    "ingredient": "See the ingredient",
    "product": "View product",
}


def load_aliases(path):
    """Read a {phrase: slug} JSON map. Missing/unreadable/invalid -> {}."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v).strip()
                    for k, v in data.items() if str(k).strip() and str(v).strip()}
    except Exception:  # noqa: BLE001 - missing/invalid alias file must never break the caller
        pass
    return {}


def _slug_words(slug):
    return re.sub(r"[-_]+", " ", str(slug or "")).strip().lower()


def build_index(pages, *, alias_map=None):
    """Map phrase(lower) -> {title, href, kind, gated}. First page wins on collision."""
    index = {}
    by_slug = {}
    for p in (pages or []):
        slug = str(p.get("slug") or "").strip()
        if not slug:
            continue
        rec = {
            "title": p.get("name") or slug,
            "href": p.get("href") or "",
            "kind": p.get("kind") or "topic",
            "gated": bool(p.get("gated")),
        }
        by_slug[slug] = rec
        for phrase in (str(p.get("name") or "").strip().lower(), _slug_words(slug)):
            if phrase and phrase not in index:
                index[phrase] = rec
    # aliases point at a slug; only add if the slug is a real page and the phrase is free
    for phrase, slug in (alias_map or {}).items():
        ph = str(phrase or "").strip().lower()
        rec = by_slug.get(str(slug or "").strip())
        if ph and rec and ph not in index:
            index[ph] = rec
    return index


def match_page_links(text, index, *, limit=2):
    """Return up to `limit` deduped link cards for phrases present in text (longest first)."""
    if not text or not index:
        return []
    low = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
    # longest phrases first so "magnesium glycinate" claims its span before "magnesium"
    phrases = sorted(index.keys(), key=len, reverse=True)
    cards = []
    seen_hrefs = set()
    claimed = []  # list of (start, end) spans already consumed by a longer phrase

    def _overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    for phrase in phrases:
        if not phrase:
            continue
        # word-boundary search: phrase must sit between non-word chars
        pat = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
        for m in re.finditer(pat, low):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            claimed.append((s, e))
            rec = index[phrase]
            href = rec.get("href")
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            cards.append({
                "key": f"{rec.get('kind', 'topic')}:" + href.rstrip("/").rsplit("/", 1)[-1],
                "title": rec.get("title") or phrase,
                "sub": _SUB_BY_KIND.get(rec.get("kind"), "Read the guide"),
                "href": href,
            })
            break  # one card per phrase
        if len(cards) >= limit:
            break
    return cards[:limit]
```

- [ ] **Step 4: Create the alias seed file**

```json
// data/page-aliases.json
{
  "can't focus": "brain-fog",
  "cant focus": "brain-fog",
  "foggy": "brain-fog",
  "trouble sleeping": "poor-sleep",
  "can't sleep": "poor-sleep",
  "exhausted": "low-energy",
  "tired all the time": "low-energy",
  "stressed out": "everyday-stress",
  "bloated": "digestive-discomfort",
  "achy joints": "joint-discomfort"
}
```

(JSON does not allow `//` comments — create the file with the object only, no comment line.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_page_links.py -q`
Expected: PASS (10 passed).

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/page_links.py data/page-aliases.json tests/test_page_links.py
git commit -m "feat(chat-links): pure page-link matcher + alias seed"
```

---

### Task 2: `app.py` wiring — flag, cached index, `/chat` merge

**Files:**
- Modify: `app.py` (flag near line 2902 by `TOPIC_PAGES_ENABLED`; cached-index helper near the funnel helpers; merge block in the `/chat` done handler after line 2398, before `_done_payload`)
- Test: `tests/test_chat_page_links_wiring.py`

**Interfaces:**
- Consumes: `dashboard.page_links.build_index/match_page_links/load_aliases` (Task 1); existing `surfaced_cards`, `query`, `answer`, `_db_lock`, `LOG_DB` in the `/chat` handler.
- Produces: route behavior only (no new public function other than the module-local `_chat_page_link_index()`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_page_links_wiring.py
import subprocess
import sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_flag_defined_and_default_off():
    src = (_repo() / "app.py").read_text()
    assert "CHAT_PAGE_LINKS_ENABLED" in src
    # default must be off: the env default string is falsy
    assert 'os.environ.get("CHAT_PAGE_LINKS_ENABLED"' in src


def test_chat_merges_page_links_when_flag_on():
    src = (_repo() / "app.py").read_text()
    # the done handler must call the matcher and cap at 3 when link cards fire
    assert "match_page_links" in src
    assert "_chat_page_link_index" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_chat_page_links_wiring.py -q`
Expected: FAIL — `CHAT_PAGE_LINKS_ENABLED`/`match_page_links` not present.

- [ ] **Step 3a: Add the flag**

In `app.py`, immediately after the `TOPIC_PAGES_ENABLED` line (~2902), add:

```python
CHAT_PAGE_LINKS_ENABLED = os.environ.get("CHAT_PAGE_LINKS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 3b: Add the cached index builder**

In `app.py`, add this helper near the other funnel helpers (anywhere at module level after `LOG_DB`/`_db_lock` are defined; e.g. just above the `/chat` route). It enumerates approved pages and caches for 60s:

```python
_CHAT_PAGE_LINK_CACHE = {"at": 0.0, "index": {}}


def _chat_page_link_index():
    """Approved-page phrase index for chat link surfacing. 60s TTL; failure -> empty."""
    import time as _time
    now = _time.time()
    if _CHAT_PAGE_LINK_CACHE["index"] and (now - _CHAT_PAGE_LINK_CACHE["at"]) < 60:
        return _CHAT_PAGE_LINK_CACHE["index"]
    pages = []
    try:
        from dashboard import topic_pages as _tp, ingredient_pages as _ip
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _tp.init_table(cx)
            for r in _tp.list_pages(cx):
                if r.get("state") == "approved":
                    pages.append({"slug": r["slug"], "name": r["name"], "kind": "topic",
                                  "href": f"/learn/{r['slug']}", "gated": False})
            _ip.init_table(cx)
            for slug, name in cx.execute(
                    "SELECT ingredient_slug, name FROM ingredient_pages WHERE state='approved'").fetchall():
                pages.append({"slug": slug, "name": name or slug, "kind": "ingredient",
                              "href": f"/begin/ingredient/{slug}", "gated": True})
            try:
                cx.execute("CREATE TABLE IF NOT EXISTS sales_pages (product_slug TEXT PRIMARY KEY, state TEXT)")
                approved_products = [row[0] for row in cx.execute(
                    "SELECT product_slug FROM sales_pages WHERE state='approved'").fetchall()]
            except Exception:
                approved_products = []
        # reuse the already-loaded products catalog global for names
        _prod_names = {k: (v.get("name") or k)
                       for k, v in (_PRODUCTS.get("products", {}) or {}).items()}
        for slug in approved_products:
            pages.append({"slug": slug, "name": _prod_names.get(slug, slug), "kind": "product",
                          "href": f"/begin/product/{slug}", "gated": False})
        from dashboard import page_links as _pl
        aliases = _pl.load_aliases(str(DATA_DIR / "page-aliases.json"))
        index = _pl.build_index(pages, alias_map=aliases)
    except Exception as _e:  # noqa: BLE001 - never break chat
        print(f"[chat-page-links] index build failed: {_e}", flush=True)
        index = {}
    _CHAT_PAGE_LINK_CACHE["at"] = now
    _CHAT_PAGE_LINK_CACHE["index"] = index
    return index
```

> Path symbols confirmed present in app.py: `DATA_DIR = Path(__file__).parent / "data"` (line ~89), the `_PRODUCTS` catalog global `{"products": {slug: {...}}}` (line ~2878), `_db_lock`, `LOG_DB`, and `Path`/`sqlite3` imports. The helper must be defined AFTER line 2878 so `_PRODUCTS` is in scope (placing it just above the `/chat` route satisfies this).

- [ ] **Step 3c: Add the merge block in the `/chat` done handler**

In `app.py`, in the `/chat` generate() done handler, immediately AFTER the clip-surface block (the `surface_approved_clips` try/except ending ~line 2398) and BEFORE `_done_payload = {`, insert:

```python
        if CHAT_PAGE_LINKS_ENABLED:
            try:
                from dashboard import page_links as _pl
                _links = _pl.match_page_links(f"{query or ''} {answer or ''}",
                                              _chat_page_link_index(), limit=2)
                if _links:
                    merged, seen = [], set()
                    for c in _links + (surfaced_cards or []):
                        h = c.get("href")
                        if h and h in seen:
                            continue
                        if h:
                            seen.add(h)
                        merged.append(c)
                    surfaced_cards = merged[:3]
            except Exception as _ple:  # noqa: BLE001 - never break chat
                print(f"[chat-page-links] {_ple!r}", flush=True)
```

> Note: the 3-card cap applies ONLY inside `if _links:` — so when the flag is off OR no link matched, `surfaced_cards` is untouched and existing behavior is byte-identical.

- [ ] **Step 4: Verify**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m py_compile app.py && python3 -m pytest tests/test_chat_page_links_wiring.py -q`
Expected: compile OK, 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add app.py tests/test_chat_page_links_wiring.py
git commit -m "feat(chat-links): flag + cached index + /chat merge (dark behind CHAT_PAGE_LINKS_ENABLED)"
```

---

### Task 3: Regression sanity + dark-default check

**Files:** none (verification only)

- [ ] **Step 1: Run the new suites**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_page_links.py tests/test_chat_page_links_wiring.py -q`
Expected: all green (13 passed).

- [ ] **Step 2: Confirm the funnel helpers still pass and nothing imports app**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/ -q -k "page_link or begin_funnel or topic" --continue-on-collection-errors`
Expected: the page_link/topic/begin_funnel dashboard tests pass; any `PineconeConfigurationError` collection errors are environmental (tests that `import app`), not regressions — note them, don't fix.

- [ ] **Step 3: Confirm dark-by-default**

Read the diff: `CHAT_PAGE_LINKS_ENABLED` default is `"false"` and the merge block is fully inside `if CHAT_PAGE_LINKS_ENABLED:`. No code change — a read-check that the feature is inert when the flag is off.

- [ ] **Step 4: Commit (only if fixups were needed)**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add -A && git commit -m "test(chat-links): regression pass" || echo "nothing to commit"
```

---

## Rollout (post-merge, not a code task)

- Open a PR from `chat-page-links` → `main`; merge (direct-push to main is guarded).
- Go-live: set `CHAT_PAGE_LINKS_ENABLED=true` in Doppler `remedy-match/prd` after a quick live check that a chat message about a published topic (e.g. "I have low energy") surfaces its `/learn/low-energy` link card. Reversible by flipping back.

## Self-Review notes (author)

- **Spec coverage:** §4 `page_links.py` (build_index/match_page_links/load_aliases) → Task 1; `data/page-aliases.json` → Task 1; flag + cached index + `/chat` merge → Task 2; §5 safety (wrapped, no hallucinated URLs, dedupe-by-href, cap, flag-off-inert) → Task 2 (merge inside `if _links:` + try/except); §6 testing → Tasks 1–2; §7 rollout → Task 2 flag + rollout section. No gaps.
- **Type consistency:** card dict `{key,title,sub,href}` produced by `match_page_links` (Task 1) and consumed unchanged by the Task 2 merge; `build_index` output shape `{phrase: {title,href,kind,gated}}` consumed only by `match_page_links`. Page-record shape `{slug,name,kind,href,gated}` produced by `_chat_page_link_index` (Task 2) and consumed by `build_index` (Task 1) — fields match.
- **Resolved:** Task 2 Step 3b uses `DATA_DIR` (line ~89) and reuses the `_PRODUCTS` catalog global (line ~2878); the helper must be defined after line 2878 so `_PRODUCTS` is in scope.
