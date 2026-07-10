# E4L Parser: capture BFA and classify recommendation sections — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every E4L scan's infoceutical recommendations complete and correctly
classified, so "the five infoceuticals from this scan" is a query rather than a guess.

**Architecture:** `02 Skills/parse-e4l-scans.py` reads scan PDFs via `pdftotext` and writes
`e4l_scan_results`. Two defects: the item regex requires a digit, so the bare `BFA` line is
skipped; and every row is stamped `section_context = "Recommendations"`, so the `ER`/`MR`
rows under the `MIHEALTH FUNCTIONS` heading are indistinguishable from the infoceuticals
above it. We capture `BFA` and populate `section_context` from the document's own headings.
`e4l-reparse-results.py` then rebuilds existing rows from the PDFs already on disk.

**Tech Stack:** Python 3, `sqlite3`, `pdftotext` (poppler), `pytest`.

## Global Constraints

- Repository for all code changes: **the vault** (`~/AI-Training`), not `deploy-chat`.
  This spec and plan live in `deploy-chat` because the rest of the feature does.
- `parse-e4l-scans.py` calls `require_active_mac()` at import. Tests must load it with
  `importlib.util.spec_from_file_location`, exactly as `test_clinical_tags_sweep.py` does.
  The filename contains dashes and cannot be imported normally.
- **Colour/band is NOT extractable.** `pdftotext` discards it; the module's own header says
  so. Do not add a `match_band` column. The spec's Slice 0 is corrected by this plan.
- `priority_rank` stays a single sequence in document order across the whole
  RECOMMENDATIONS region. Adding `BFA` at rank 1 **shifts every later rank by one** for
  scans that have it. Consumers use ordering, never absolute rank values.
- Never mutate `~/AI-Training/e4l.db` before the same operation has been proved on a copy.

## Evidence this plan is built on

Measured over 250 randomly sampled PDFs from `~/e4l-scans` (3,481 total):

| zone | families found |
|---|---|
| before the `INFOCEUTICALS` heading | `ED` 324, `EI` 219, `ET` 214, `ES` 178, `MB` 131 |
| under `MIHEALTH FUNCTIONS` | `ER` 1025, `MR` 440 |

- **Zero** `ER`/`MR` rows appear before the heading.
- The **only** bare three-letter code in any zone is `BFA`, seen 63 times, always in the
  infoceutical zone.
- `item_code = 'BFA'` currently appears in **0 of 5,764** rows of `e4l_scan_results`.

So the classification is exact, not heuristic. `protocol_days` (15 vs 2) correlates but is
not the rule; the headings are.

## File Structure

| file | responsibility |
|---|---|
| `02 Skills/parse-e4l-scans.py` (modify) | extract + persist. `extract_recommendations` becomes the single source of truth for *what a scan recommends and in which section*. |
| `02 Skills/e4l-reparse-results.py` (modify) | backfill. Stops duplicating parse logic; imports `extract_recommendations`. |
| `02 Skills/test_parse_e4l_recommendations.py` (create) | unit tests over fixture text. No PDFs, no DB. |
| `02 Skills/setup-e4l-db.py` (modify) | seed the `BFA` row in `e4l_items`. |

---

### Task 1: `BFA` exists as a real catalog item

Today the parser's fallback would insert `('BFA', 'Unknown', 'BFA')`. `BFA` is the Big Field
Aligner; the PDF's own glossary calls it "BIG FIELD ALIGNER". Glen's display rule: **"Big
Field Aligner (BFA)"** — the label says BFA, and "aligner" both expands the acronym and
describes the benefit.

**Files:**
- Modify: `02 Skills/setup-e4l-db.py` (the `e4l_items` seed block)
- Test: `02 Skills/test_parse_e4l_recommendations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an `e4l_items` row `('BFA', 'BFA', 'Big Field Aligner', 'Big Field Aligner (BFA)')`
  as `(code, category, name, full_name)`.

- [ ] **Step 1: Write the failing test**

```python
# 02 Skills/test_parse_e4l_recommendations.py
import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "e4l.db"


def test_bfa_is_a_known_item_not_an_unknown_placeholder():
    cx = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT code, category, name, full_name FROM e4l_items WHERE code='BFA'").fetchone()
    assert row is not None, "BFA missing from e4l_items"
    assert row["category"] == "BFA"
    assert row["name"] == "Big Field Aligner"
    assert row["full_name"] == "Big Field Aligner (BFA)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py::test_bfa_is_a_known_item_not_an_unknown_placeholder" -q`
Expected: FAIL with `AssertionError: BFA missing from e4l_items`

- [ ] **Step 3: Seed the row**

Add to the `e4l_items` seed in `02 Skills/setup-e4l-db.py`, and run it once against
`e4l.db` (the seed is `INSERT OR IGNORE`, so it is safe to re-run):

```python
    # BFA ("Big Fields") is recommended as a bare code with no digit, so it has no
    # numbered sibling. Seed it explicitly, or the parser's placeholder branch files it
    # under category 'Unknown'.
    cx.execute(
        "INSERT OR IGNORE INTO e4l_items (code, category, name, full_name) VALUES (?,?,?,?)",
        ("BFA", "BFA", "Big Field Aligner", "Big Field Aligner (BFA)"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/AI-Training && python3 "02 Skills/setup-e4l-db.py" && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/AI-Training
git add "02 Skills/setup-e4l-db.py" "02 Skills/test_parse_e4l_recommendations.py"
git commit -m "feat(e4l): seed BFA (Big Field Aligner) as a known item"
```

---

### Task 2: `extract_recommendations` captures BFA and reports a section

**Files:**
- Modify: `02 Skills/parse-e4l-scans.py:39` (regexes) and `:69-106` (`extract_recommendations`)
- Test: `02 Skills/test_parse_e4l_recommendations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `extract_recommendations(text: str) -> list[tuple[str, int | None, str]]`,
  a list of `(item_code, protocol_days, section)` in document order. `section` is exactly
  `"Infoceuticals"` or `"miHealth Functions"`. **This changes the return arity from 2 to 3**;
  Task 3 updates both call sites.

- [ ] **Step 1: Write the failing test**

Real text, trimmed from a live scan PDF. Note the blank lines and the bare `BFA`.

```python
# append to 02 Skills/test_parse_e4l_recommendations.py
import importlib.util
from pathlib import Path


def _load(stem):
    p = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # require_active_mac() runs here; this is Glen's Mac
    return m


PARSER = _load("parse-e4l-scans")

SCAN_TEXT = """
RECOMMENDATIONS

BFA

15

ED6 - Heart

15

ES7 - Muscle

15

INFOCEUTICALS

How to Take Infoceuticals:

MIHEALTH FUNCTIONS

ER2 - Large Intestine

2

MR4 - Microtubule

2

ENERGY SOURCE
"""


def test_bfa_is_captured_first():
    recs = PARSER.extract_recommendations(SCAN_TEXT)
    assert recs[0] == ("BFA", 15, "Infoceuticals")


def test_infoceuticals_and_mihealth_are_separated():
    recs = PARSER.extract_recommendations(SCAN_TEXT)
    info = [c for c, _, s in recs if s == "Infoceuticals"]
    mih = [c for c, _, s in recs if s == "miHealth Functions"]
    assert info == ["BFA", "ED6", "ES7"]
    assert mih == ["ER2", "MR4"]


def test_er_and_mr_are_never_infoceuticals():
    recs = PARSER.extract_recommendations(SCAN_TEXT)
    assert not [c for c, _, s in recs if c.startswith(("ER", "MR")) and s == "Infoceuticals"]


def test_protocol_days_still_attach_to_the_right_item():
    recs = PARSER.extract_recommendations(SCAN_TEXT)
    days = {c: d for c, d, _ in recs}
    assert days == {"BFA": 15, "ED6": 15, "ES7": 15, "ER2": 2, "MR4": 2}


def test_the_heading_words_are_not_mistaken_for_item_codes():
    recs = PARSER.extract_recommendations(SCAN_TEXT)
    codes = [c for c, _, _ in recs]
    assert "INFOCEUTICALS" not in codes
    assert "MIHEALTH" not in codes


def test_a_scan_without_bfa_is_unchanged():
    text = SCAN_TEXT.replace("BFA\n\n15\n\n", "", 1)
    recs = PARSER.extract_recommendations(text)
    assert [c for c, _, _ in recs] == ["ED6", "ES7", "ER2", "MR4"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py" -q`
Expected: FAIL — `ValueError: too many values to unpack` (the function returns 2-tuples),
and `test_bfa_is_captured_first` fails because `BFA` is absent.

- [ ] **Step 3: Implement**

Replace the regex block near `02 Skills/parse-e4l-scans.py:39`:

```python
ITEM_CODE_RE = re.compile(r"\b([A-Z]{2,3}\d+)\s*[-–]\s*[\w\s&()/]+")
# "Big Fields" is recommended as a bare code with no digit and no ' - Name' suffix, so
# ITEM_CODE_RE never saw it: 0 of 5,764 stored rows had item_code='BFA'. Across 250
# sampled PDFs it is the ONLY bare code that ever appears, so whitelist it rather than
# matching any bare 3-letter word (which would swallow headings).
BARE_CODE_RE = re.compile(r"^(BFA)$")

# Sub-section headings WITHIN the recommendations region. ER/MR live under
# MIHEALTH FUNCTIONS and are miHealth cycles, not infoceuticals ("ER's are not
# infoceuticals" — Glen). Measured over 250 PDFs: no ER/MR ever precedes the heading,
# and only ED/EI/ES/ET/MB/BFA precede it.
INFOCEUTICALS_H_RE = re.compile(r"^INFOCEUTICALS\s*$")
MIHEALTH_H_RE = re.compile(r"^MIHEALTH FUNCTIONS\s*$")

SECTION_INFOCEUTICAL = "Infoceuticals"
SECTION_MIHEALTH = "miHealth Functions"
```

Replace `extract_recommendations` (`02 Skills/parse-e4l-scans.py:69-106`):

```python
def extract_recommendations(text: str) -> list[tuple[str, int | None, str]]:
    """Return (item_code, protocol_days, section) in document order from the
    RECOMMENDATIONS region.

    `section` comes from the document's own headings, not from protocol_days. The
    15-vs-2 day split correlates with it, but the headings are the ground truth.
    """
    lines = text.splitlines()
    in_rec = False
    section = SECTION_INFOCEUTICAL
    recs: list[tuple[str, int | None, str]] = []

    def _days_after(idx: int) -> int | None:
        for j in range(idx + 1, min(idx + 5, len(lines))):
            d = lines[j].strip()
            if re.match(r"^\d+$", d):
                return int(d)
        return None

    for i, raw in enumerate(lines):
        line = raw.strip()

        if not in_rec:
            if REC_SECTION_RE.match(line):
                in_rec = True
            continue

        if END_SECTION_RE.match(line):
            break
        if MIHEALTH_H_RE.match(line):
            section = SECTION_MIHEALTH
            continue
        if INFOCEUTICALS_H_RE.match(line):
            continue                      # a heading over prose; the section is unchanged

        m = ITEM_CODE_RE.match(line)
        b = BARE_CODE_RE.match(line)
        code = m.group(1) if m else (b.group(1) if b else None)
        if code:
            recs.append((code, _days_after(i), section))

    return recs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py" -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/AI-Training
git add "02 Skills/parse-e4l-scans.py" "02 Skills/test_parse_e4l_recommendations.py"
git commit -m "fix(e4l): capture the bare BFA code and classify recommendation sections"
```

---

### Task 3: persist `section_context`, and make the reparse tool share the parser

`section_context` is already a column, hardcoded to `"Recommendations"` on all 5,764 rows.
Its schema comment reads `-- which section of report: e.g. "Priority", "Secondary"`, so it
was always meant to hold this. `02 Skills/e4l_synthesis.py:63` reads it; after this task it
reads `"Infoceuticals"` / `"miHealth Functions"` instead of a constant, which is strictly
more information for the synthesis prompt.

**Files:**
- Modify: `02 Skills/parse-e4l-scans.py` (the `INSERT INTO e4l_scan_results` block, ~:205)
- Modify: `02 Skills/e4l-reparse-results.py:26-33`
- Test: `02 Skills/test_parse_e4l_recommendations.py`

**Interfaces:**
- Consumes: `extract_recommendations(text) -> list[(code, days, section)]` from Task 2.
- Produces: `store_recommendations(c, scan_id: int, recs: list[tuple[str, int|None, str]]) -> None`
  in `parse-e4l-scans.py`. Rows in `e4l_scan_results` carry `section_context` of
  `"Infoceuticals"` or `"miHealth Functions"`.
- `e4l-reparse-results.py` keeps its existing signature
  `reparse_scan(cx, scan_id, pdf_path, extract_fn)` — only its body changes. Its CLI is
  `--apply` (dry-run is the default); there is **no** `--all` flag.

- [ ] **Step 1: Write the failing test**

```python
# append to 02 Skills/test_parse_e4l_recommendations.py
import sqlite3


def _memdb():
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE e4l_items(code TEXT PRIMARY KEY, category TEXT, name TEXT, full_name TEXT)")
    cx.execute("CREATE TABLE e4l_scan_results(id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "scan_id INTEGER, item_code TEXT, priority_rank INTEGER, protocol_days INTEGER, "
               "section_context TEXT, score REAL, notes TEXT)")
    cx.execute("INSERT INTO e4l_items VALUES ('BFA','BFA','Big Field Aligner','Big Field Aligner (BFA)')")
    return cx


def test_rows_are_written_with_their_section():
    cx = _memdb()
    PARSER.store_recommendations(cx, scan_id=1, recs=PARSER.extract_recommendations(SCAN_TEXT))
    rows = cx.execute("SELECT item_code, priority_rank, section_context FROM e4l_scan_results "
                      "ORDER BY priority_rank").fetchall()
    assert rows == [("BFA", 1, "Infoceuticals"),
                    ("ED6", 2, "Infoceuticals"),
                    ("ES7", 3, "Infoceuticals"),
                    ("ER2", 4, "miHealth Functions"),
                    ("MR4", 5, "miHealth Functions")]


def test_the_infoceutical_set_is_a_query_not_a_guess():
    """On a real scan this LIMIT 5 returns BFA, ED6, ED8, ED9, ES7. The fixture is
    trimmed to three items, so the query returns three; the point is that the ER/MR rows
    are excluded by the WHERE clause rather than by a protocol_days heuristic."""
    cx = _memdb()
    PARSER.store_recommendations(cx, scan_id=1, recs=PARSER.extract_recommendations(SCAN_TEXT))
    got = [r[0] for r in cx.execute(
        "SELECT item_code FROM e4l_scan_results WHERE scan_id=1 AND section_context='Infoceuticals' "
        "ORDER BY priority_rank LIMIT 5")]
    assert got == ["BFA", "ED6", "ES7"]


def test_an_unknown_code_still_gets_a_placeholder_item():
    cx = _memdb()
    PARSER.store_recommendations(cx, scan_id=2, recs=[("ZZ9", 15, "Infoceuticals")])
    assert cx.execute("SELECT category FROM e4l_items WHERE code='ZZ9'").fetchone()[0] == "Unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py" -q`
Expected: FAIL with `AttributeError: module has no attribute 'store_recommendations'`

- [ ] **Step 3: Extract the writer and use it in both scripts**

Add to `02 Skills/parse-e4l-scans.py` (and replace the inline INSERT loop at ~:199-207 with
a call to it):

```python
def store_recommendations(c, scan_id: int, recs: list[tuple[str, int | None, str]]) -> None:
    """Write a scan's recommendations. Single writer, shared with e4l-reparse-results.py,
    so the two can never drift on section labelling."""
    known = {r[0] for r in c.execute("SELECT code FROM e4l_items")}
    for rank, (code, days, section) in enumerate(recs, start=1):
        if code not in known:
            c.execute("INSERT OR IGNORE INTO e4l_items (code, category, name) VALUES (?, 'Unknown', ?)",
                      (code, code))
        c.execute("INSERT INTO e4l_scan_results "
                  "(scan_id, item_code, priority_rank, protocol_days, section_context) "
                  "VALUES (?,?,?,?,?)", (scan_id, code, rank, days, section))
```

In `02 Skills/e4l-reparse-results.py`, replace the body of `reparse_scan` (lines 22-34).
Keep its signature: `main()` calls it as `reparse_scan(cx, sid, pdf, extract)`, and
`extract` is `lambda p: parser.extract_recommendations(parser.pdf_to_text(Path(p)))`, which
now returns 3-tuples.

```python
def reparse_scan(cx, scan_id, pdf_path, extract_fn):
    """extract_fn(pdf_path) -> [(code, days, section), ...]. Replaces this scan's results.

    Delegates the write to parse-e4l-scans.store_recommendations so the two scripts can
    never drift on section labelling — this file used to hardcode 'Recommendations'.
    """
    recs = extract_fn(pdf_path)
    cx.execute("DELETE FROM e4l_scan_results WHERE scan_id=?", (scan_id,))
    PARSER.store_recommendations(cx, scan_id, recs)
    cx.commit()
    return len(recs)
```

`PARSER` is a new module-level constant in `e4l-reparse-results.py`, defined once directly
below `_load_parser`, so the parser (and its `require_active_mac()` guard) is not
re-executed per scan:

```python
PARSER = _load_parser()
```

and `main()` uses it instead of its local `parser = _load_parser()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/test_parse_e4l_recommendations.py" -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/AI-Training
git add "02 Skills/parse-e4l-scans.py" "02 Skills/e4l-reparse-results.py" "02 Skills/test_parse_e4l_recommendations.py"
git commit -m "fix(e4l): store the recommendation section; one writer for parse and reparse"
```

---

### Task 4: backfill, on a copy first

**Files:**
- No source changes. This task runs the tools from Tasks 1-3.

**Interfaces:**
- Consumes: `reparse_scan` from Task 3.
- Produces: an `e4l.db` where `BFA` exists and every row carries a real section.

- [ ] **Step 1: Record the "before" numbers**

```bash
cd ~/AI-Training
python3 - <<'PY'
import sqlite3
cx = sqlite3.connect("file:e4l.db?mode=ro", uri=True)
q = lambda s: cx.execute(s).fetchone()[0]
print("rows              :", q("SELECT COUNT(*) FROM e4l_scan_results"))
print("BFA rows          :", q("SELECT COUNT(*) FROM e4l_scan_results WHERE item_code='BFA'"))
print("distinct sections :", [r[0] for r in cx.execute("SELECT DISTINCT section_context FROM e4l_scan_results")])
PY
```

Expected before: `5764`, `0`, `['Recommendations']`.

- [ ] **Step 2: Give the reparse tool a db override, so a copy can be targeted**

`e4l-reparse-results.py:9` hardcodes `DB = os.path.join(VAULT, "e4l.db")`. There is no way
to point it at a copy today. Replace that line with:

```python
DB = os.environ.get("E4L_DB") or os.path.join(VAULT, "e4l.db")
```

Commit it before running anything:

```bash
cd ~/AI-Training
git add "02 Skills/e4l-reparse-results.py"
git commit -m "chore(e4l): allow E4L_DB to target a copy of the database"
```

- [ ] **Step 3: Dry-run against a COPY, never the live db**

The tool is dry-run by default and writes only with `--apply`. There is no `--all` flag; it
reparses every `scan_id` already present in `e4l_scan_results`.

```bash
cd ~/AI-Training
cp e4l.db /tmp/e4l-backfill-test.db
E4L_DB=/tmp/e4l-backfill-test.db python3 "02 Skills/e4l-reparse-results.py"
```

Expected: a line like `scans=477 reparsed=… missing_pdf=… avg_codes 12 -> 13 (DRY RUN, use --apply)`.
The average must **rise**, because `BFA` is now captured. If it falls, stop.

- [ ] **Step 4: Apply to the copy**

```bash
E4L_DB=/tmp/e4l-backfill-test.db python3 "02 Skills/e4l-reparse-results.py" --apply
```

- [ ] **Step 5: Verify the copy, and only then accept it**

```bash
python3 - <<'PY'
import sqlite3
cx = sqlite3.connect("file:/tmp/e4l-backfill-test.db?mode=ro", uri=True)
q = lambda s: cx.execute(s).fetchone()[0]
print("BFA rows                 :", q("SELECT COUNT(*) FROM e4l_scan_results WHERE item_code='BFA'"))
print("sections                 :", [r for r in cx.execute("SELECT section_context, COUNT(*) FROM e4l_scan_results GROUP BY 1")])
print("ER/MR wrongly infoceutical:", q("SELECT COUNT(*) FROM e4l_scan_results WHERE section_context='Infoceuticals' AND (item_code LIKE 'ER%' OR item_code LIKE 'MR%')"))
print("scans w/ 5 infoceuticals :", q("SELECT COUNT(*) FROM (SELECT scan_id FROM e4l_scan_results WHERE section_context='Infoceuticals' GROUP BY scan_id HAVING COUNT(*)=5)"))
print("scans w/ 4 infoceuticals :", q("SELECT COUNT(*) FROM (SELECT scan_id FROM e4l_scan_results WHERE section_context='Infoceuticals' GROUP BY scan_id HAVING COUNT(*)=4)"))
PY
```

Acceptance, all four must hold:
1. `BFA rows` > 0.
2. `sections` contains only `Infoceuticals` and `miHealth Functions`.
3. `ER/MR wrongly infoceutical` **= 0**.
4. `scans w/ 5 infoceuticals` has **risen** and `scans w/ 4` has **fallen** versus the
   pre-backfill split of 269 / 205.

If any check fails, stop. Do not touch `e4l.db`.

- [ ] **Step 6: Backfill for real**

```bash
cd ~/AI-Training
cp e4l.db "e4l.db.bak-$(date +%Y%m%d-%H%M%S)"
python3 "02 Skills/e4l-reparse-results.py" --apply
```

Re-run the Step 5 verification against `e4l.db` (drop the `/tmp` path). Same four checks.

- [ ] **Step 7: Spot-check one known scan end-to-end**

Pick a scan whose PDF you can open. Its `Infoceuticals` rows, ordered by `priority_rank`,
must match the first five item lines of the PDF's RECOMMENDATIONS section exactly, `BFA`
first. Then commit the backup marker only — `e4l.db` is gitignored.

```bash
cd ~/AI-Training
git status --short   # expect: no e4l.db in the diff
```

---

## What this plan deliberately does not do

- **No `match_band` column.** Colour is not extractable from `pdftotext`; the spec is
  corrected here. `section_context` is a stronger signal anyway, because it is the
  document's own structure rather than a proxy.
- **No re-numbering of `priority_rank` per section.** Rank stays a single document-order
  sequence. "Top five infoceuticals" is `WHERE section_context='Infoceuticals' ORDER BY
  priority_rank LIMIT 5`.
- **No push to production.** That is Slice 1.
- **No change to `clinical_tagger.py`'s insert.** It writes its own rows for a different
  purpose; folding it into `store_recommendations` is out of scope and untested here.
