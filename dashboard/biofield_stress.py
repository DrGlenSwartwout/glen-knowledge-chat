"""Per-test master stress list + remedy<->stress coverage map for the local
Biofield Intake balancing loop (B1). Pure sqlite; the caller passes a connection.
Balanced state is DERIVED at read time, never stored (see list_stresses)."""
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(tid):
    return int(str(tid).lstrip("a") or 0)


def _is_operational_tag(tag):
    """Lazy wrapper: True if a CRM tag is pipeline/marketing state, not a health stress."""
    from dashboard.biofield_profile import is_operational_tag
    return is_operational_tag(tag)


def _clean_label(label, source):
    """Display cleanup for mined CRM-tag labels only: drop JSON quote/bracket
    artifacts and the `pb:` namespace, turn a slug into words, and tidy casing.
    Scan/voice/comm labels are clinical free text and returned unchanged."""
    if source != "tag":
        return label
    s = (label or "").strip()
    core = s.strip('[]"\' ')               # drop JSON quote/bracket artifacts
    is_ns = core.lower().startswith("pb:")
    if is_ns:
        core = core[3:]
    # Only reformat coded/quoted/namespaced tags. Clean free text stays as typed.
    if core == s and not is_ns:
        return label
    from dashboard.biofield_profile import prettify_condition
    return prettify_condition(core) or label   # hyphens->spaces, Title-case, acronyms UPPER


def init_stress_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_stress(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, code TEXT, label TEXT,
        source TEXT NOT NULL DEFAULT 'scan', balance TEXT NOT NULL DEFAULT 'optional',
        manual_balanced INTEGER NOT NULL DEFAULT 0, created_at TEXT, updated_at TEXT,
        UNIQUE(test_id, source, code))""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_remedy_coverage(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, remedy TEXT, code TEXT,
        UNIQUE(test_id, remedy, code))""")
    # Per-test edited/ordered minimal-remedy set (survives reload).
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_remedy_set(
        test_id INTEGER PRIMARY KEY, remedies_json TEXT, updated_at TEXT)""")
    # Reusable minimal-remedy set keyed by the active required-stress pattern.
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_remedy_pattern(
        pattern_key TEXT PRIMARY KEY, tokens_json TEXT, remedies_json TEXT,
        label TEXT, updated_at TEXT)""")
    cx.commit()


def init_custom_vocab(cx):
    """Durable, reusable stress terms Glen coins in the picker. Kept separate from the
    FMP snapshot (which is overwritten on every re-import)."""
    cx.execute("""CREATE TABLE IF NOT EXISTS custom_stress_vocab(
        term       TEXT PRIMARY KEY,
        created_at TEXT,
        created_by TEXT DEFAULT 'glen')""")
    cx.commit()


def _table_exists(cx, name):
    return cx.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


def add_custom_vocab(cx, term):
    """Persist a stress term to the reusable custom vocabulary. Idempotent and
    case-insensitive. Returns True if a new row was inserted."""
    init_custom_vocab(cx)
    t = (term or "").strip()
    if not t:
        return False
    if cx.execute("SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)=LOWER(?)", (t,)).fetchone():
        return False
    cx.execute("INSERT INTO custom_stress_vocab(term,created_at,created_by) VALUES(?,?,?)",
               (t, _now(), "glen"))
    cx.commit()
    return True


def vocab_has(cx, term):
    """True if term is already a known stress vocabulary term — in the FMP snapshot
    or the custom table (case-insensitive). Blank counts as known (never persist blank)."""
    t = (term or "").strip()
    if not t:
        return True
    if _table_exists(cx, "fmp_snap_client_active_main_stress") and cx.execute(
            "SELECT 1 FROM fmp_snap_client_active_main_stress "
            "WHERE LOWER(TRIM(main_stress))=LOWER(?) LIMIT 1", (t,)).fetchone():
        return True
    if _table_exists(cx, "custom_stress_vocab") and cx.execute(
            "SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)=LOWER(?) LIMIT 1",
            (t,)).fetchone():
        return True
    return False


def seed_from_scan(cx, tid, findings, coverage):
    init_stress_tables(cx)
    t = _num(tid)
    covered = set()
    for codes in (coverage or {}).values():
        covered |= set(codes)
    now = _now()
    req = 0
    for f in findings or []:
        code = (f.get("code") or "").strip()
        if not code:
            continue
        balance = "required" if code in covered else "optional"
        if balance == "required":
            req += 1
        cx.execute(
            "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
            "manual_balanced,created_at,updated_at) VALUES(?,?,?,'scan',?,0,?,?) "
            "ON CONFLICT(test_id,source,code) DO UPDATE SET "
            "label=excluded.label, balance=excluded.balance, updated_at=excluded.updated_at",
            (t, code, (f.get("name") or code).strip(), balance, now, now))
    cx.execute("DELETE FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,))
    for remedy, codes in (coverage or {}).items():
        for code in codes:
            cx.execute("INSERT OR IGNORE INTO biofield_auth_remedy_coverage(test_id,remedy,code) "
                       "VALUES(?,?,?)", (t, (remedy or "").strip().lower(), code))
    cx.commit()
    n = cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=? AND source='scan'", (t,)).fetchone()[0]
    c = cx.execute("SELECT COUNT(*) FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,)).fetchone()[0]
    return {"stresses": n, "required": req, "coverage": c}


def covered_codes(cx, tid, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return set()
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT DISTINCT code FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND remedy IN ({ph})", (t, *names)).fetchall()
    return {r[0] for r in rows}


def _coverers(cx, tid, code, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return []
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT remedy FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND code=? AND remedy IN ({ph})", (t, code, *names)).fetchall()
    return [r[0] for r in rows]


def _norm(s):
    """Normalize a stress label for dedup/label-match: lowercase, collapse internal
    whitespace, strip surrounding non-word characters."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def _chain_parts(chain_rows):
    """Split a mixed chain-rows list into (remedy_names, [(norm_head, remedy), ...]).
    Accepts plain remedy-name strings (no head) and {"head","remedy"} dicts."""
    names, heads = [], []
    for r in chain_rows or []:
        if isinstance(r, str):
            if r.strip():
                names.append(r)
        elif isinstance(r, dict):
            rem = (r.get("remedy") or "").strip()
            if rem:
                names.append(rem)
                h = _norm(r.get("head") or "")
                if h:
                    heads.append((h, rem))
    return names, heads


def historical_remedies(cx, label):
    """Lowercased remedies Glen historically used for this stress name (FMP snapshot).
    Empty set on no history or missing snapshot. Never raises."""
    try:
        from dashboard.biofield_authoring import stress_suggestions
        return {(s.get("remedy") or "").strip().lower()
                for s in stress_suggestions(cx, label) if (s.get("remedy") or "").strip()}
    except Exception:
        return set()


def list_stresses(cx, tid, chain_rows):
    init_stress_tables(cx)
    cx.row_factory = sqlite3.Row
    t = _num(tid)
    remedy_names, head_pairs = _chain_parts(chain_rows)
    covered = covered_codes(cx, tid, remedy_names)
    head_map = {}
    for h, rem in head_pairs:
        head_map.setdefault(h, rem)
    rows = cx.execute(
        "SELECT id, code, label, source, balance, manual_balanced "
        "FROM biofield_auth_stress WHERE test_id=? ORDER BY "
        "CASE balance WHEN 'required' THEN 0 ELSE 1 END, id", (t,)).fetchall()
    chain_rem_lower = {(n or "").strip().lower() for n in remedy_names if (n or "").strip()}
    active, balanced, items = [], [], []
    for r in rows:
        # Operational CRM/marketing tags (type:*, consent:*, "concierge", ...) are not
        # health stresses -> never show them in the balancing panel.
        if r["source"] == "tag" and _is_operational_tag(r["code"] or r["label"]):
            continue
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
        item = {"id": r["id"], "code": r["code"],
                "label": _clean_label(r["label"], r["source"]),
                "source": r["source"], "balance": r["balance"],
                "balanced": is_bal, "balanced_by": by}
        (balanced if is_bal else active).append(item)
        items.append(item)
    by_layer, unassigned = _group_by_layer(cx, tid, chain_rows, items)
    return {"active": active, "balanced": balanced,
            "by_layer": by_layer, "unassigned": unassigned}


def _group_by_layer(cx, tid, chain_rows, items):
    """Group stresses under each causal-chain layer. A stress belongs to a layer
    when the layer's remedy covers the stress's code (biofield_auth_remedy_coverage)
    OR the stress label matches the layer's head. A stress covered by several
    layers' remedies is listed under EACH. Stresses on no layer -> `unassigned`.
    Chain rows sharing a layer number are merged (a layer can need several remedies)."""
    layers, order = {}, []
    for r in chain_rows or []:
        if not isinstance(r, dict):
            continue
        try:
            ln = int(r.get("layer"))
        except (TypeError, ValueError):
            continue
        if ln not in layers:
            layers[ln] = {"head": "", "remedies": [], "remedies_disp": []}
            order.append(ln)
        head = (r.get("head") or "").strip()
        if head and not layers[ln]["head"]:
            layers[ln]["head"] = head
        rem = (r.get("remedy") or "").strip()
        if rem and rem.lower() not in layers[ln]["remedies"]:
            layers[ln]["remedies"].append(rem.lower())
            layers[ln]["remedies_disp"].append(rem)
    all_rem = [rl for L in layers.values() for rl in L["remedies"]]
    cov = {}
    if all_rem:
        ph = ",".join("?" for _ in all_rem)
        for rem, code in cx.execute(
                f"SELECT remedy, code FROM biofield_auth_remedy_coverage "
                f"WHERE test_id=? AND remedy IN ({ph})", (_num(tid), *all_rem)).fetchall():
            cov.setdefault(rem, set()).add(code)
    by_layer, assigned = [], set()
    for ln in order:
        L = layers[ln]
        head_norm = _norm(L["head"])
        rem_codes = set()
        for rl in L["remedies"]:
            rem_codes |= cov.get(rl, set())
        stresses = []
        for it in items:
            if it["code"] in rem_codes or (head_norm and _norm(it["label"]) == head_norm):
                stresses.append(it)
                assigned.add(it["id"])
        by_layer.append({"layer": ln, "head": L["head"],
                         "remedy": ", ".join(L["remedies_disp"]),
                         "remedies": L["remedies_disp"], "stresses": stresses})
    unassigned = [it for it in items if it["id"] not in assigned]
    return by_layer, unassigned


def set_manual_balanced(cx, tid, stress_id, value):
    cx.execute("UPDATE biofield_auth_stress SET manual_balanced=?, updated_at=? "
               "WHERE id=? AND test_id=?",
               (1 if value else 0, _now(), stress_id, _num(tid)))
    cx.commit()


def cover_stress(cx, tid, stress_id, rids):
    """Mark a stress as covered by a layer: link its code to each of the layer's
    remedy rows (rids) in the coverage map, so it shows as balanced under that
    layer. Returns the stress code, or None if the stress/code is missing."""
    init_stress_tables(cx)
    t = _num(tid)
    row = cx.execute("SELECT code FROM biofield_auth_stress WHERE id=? AND test_id=?",
                     (stress_id, t)).fetchone()
    if not row or not (row[0] or "").strip():
        return None
    code = row[0]
    for rid in rids or []:
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            continue
        rr = cx.execute("SELECT remedy FROM biofield_auth_chain WHERE id=? AND test_id=?",
                        (rid, t)).fetchone()
        remedy = ((rr[0] if rr else "") or "").strip().lower()
        if remedy:
            cx.execute("INSERT OR IGNORE INTO biofield_auth_remedy_coverage"
                       "(test_id,remedy,code) VALUES(?,?,?)", (t, remedy, code))
    cx.commit()
    return code


def layer_rids(chain_layers, layer_num):
    """Chain-row ids for a layer number (a layer may have several remedy rows)."""
    out = []
    for l in chain_layers or []:
        try:
            if int(l.get("layer")) == int(layer_num):
                rid = l.get("rid") if l.get("rid") is not None else l.get("id")
                if rid is not None:
                    out.append(int(rid))
        except (TypeError, ValueError):
            continue
    return out


def build_assign_prompt(stresses, layers):
    """Prompt for an LLM to assign each stress to its single best-fit causal-chain
    layer. stresses=[{id,code,label}]; layers=[{layer,head,remedy}]."""
    lyr = "\n".join(f"  Layer {L.get('layer')}: {(L.get('head') or '').strip()}"
                    f" (remedy: {(L.get('remedy') or '').strip()})" for L in (layers or []))
    strs = "\n".join(f"  id={s.get('id')}: {(s.get('code') or '').strip()} "
                     f"{(s.get('label') or '').strip()}" for s in (stresses or []))
    system = ("You assign biofield stress patterns to the most appropriate causal-chain "
              "layer for a clinical report. Match by body region / organ system / function: "
              "a Heart stress goes to a Cardiovascular layer; a Cervical Spine stress to a "
              "Cervical/Spine layer; a driver code like ED9 to the layer whose remedy carries "
              "that same code. Every stress gets exactly one layer — the single best fit. "
              'Return JSON only: {"assignments": [{"id": <stress id>, "layer": <layer number>}]}.')
    user = f"LAYERS:\n{lyr}\n\nSTRESSES:\n{strs}\n\nAssign each stress id to one layer number."
    return {"system": system, "user": user}


def parse_assignments(resp, valid_layers):
    """LLM JSON -> {stress_id: layer_num}, keeping only real layer numbers."""
    valid = {int(v) for v in valid_layers}
    out = {}
    for a in (resp or {}).get("assignments") or []:
        try:
            sid, ln = int(a.get("id")), int(a.get("layer"))
        except (TypeError, ValueError):
            continue
        if ln in valid:
            out[sid] = ln
    return out


def add_stress(cx, tid, label, *, source="voice", balance="required"):
    """Add a stress unless its normalized label already exists for this test (any
    source) -> merge. Stored with code=_norm(label) so UNIQUE(test_id,source,code)
    never collides. Returns True if inserted."""
    init_stress_tables(cx)
    t = _num(tid)
    n = _norm(label)
    if not n:
        return False
    existing = cx.execute("SELECT label FROM biofield_auth_stress WHERE test_id=?", (t,)).fetchall()
    if any(_norm(r[0]) == n for r in existing):
        return False
    now = _now()
    cx.execute(
        "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
        "manual_balanced,created_at,updated_at) VALUES(?,?,?,?,?,0,?,?)",
        (t, n, (label or "").strip(), source, balance, now, now))
    cx.commit()
    return True


def add_voice_stress(cx, tid, label):
    """Voice-captured stress (required). Thin wrapper over add_stress."""
    return add_stress(cx, tid, label, source="voice", balance="required")


def stress_id_for(cx, tid, label):
    """id of the test's stress whose label normalizes to `label` (any source), or None.
    Matches by normalized label — mirrors add_stress's dedup — so it also finds a stress
    that add_stress merged into an existing row (incl. scan-sourced rows whose `code`
    column holds the raw E4L finding code rather than the normalized label)."""
    init_stress_tables(cx)
    n = _norm(label)
    if not n:
        return None
    for rid, lbl in cx.execute(
            "SELECT id, label FROM biofield_auth_stress WHERE test_id=? ORDER BY id",
            (_num(tid),)).fetchall():
        if _norm(lbl) == n:
            return rid
    return None


def layer_chain_rids(cx, tid, layer):
    """Remedy-bearing chain-row ids on a given layer of a test (inputs to cover_stress)."""
    try:
        ln = int(layer)
    except (TypeError, ValueError):
        return []
    rows = cx.execute("SELECT id FROM biofield_auth_chain "
                      "WHERE test_id=? AND layer=? AND TRIM(COALESCE(remedy,''))<>''",
                      (_num(tid), ln)).fetchall()
    return [r[0] for r in rows]


def _remedy_context(cx, tid, chain_rows):
    """Set-cover inputs: active required tokens, token->label, remedy->codes coverage.
    Cover token = E4L code (scan) or _norm(label) (non-scan)."""
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
    return active_tokens, token_label, coverage


def _pattern_key(active_tokens):
    """Order-independent fingerprint of the active required-stress set."""
    toks = sorted(active_tokens)
    key = hashlib.sha1("\n".join(toks).encode("utf-8")).hexdigest() if toks else ""
    return key, toks


def _covers_for(remedies, active_tokens, token_label, coverage):
    """Per-remedy covered LABELS + overall uncovered LABELS for an ordered name list.

    Both lists are emitted in the order the stresses appear on the chain. `codes` is a
    set intersection and `active_tokens` is a set, so iterating either directly made
    the label order depend on the process hash seed: the same input produced
    ['Membrane', 'Lymph'] or ['Lymph', 'Membrane'] run to run. `token_label` is a dict
    filled while walking the active stresses, so its insertion order IS chain order."""
    covered_tokens, picks = set(), []
    for r in remedies:
        codes = coverage.get((r or "").strip().lower(), set()) & active_tokens
        covered_tokens |= codes
        picks.append({"remedy": r,
                      "covers": [token_label[c] for c in token_label if c in codes]})
    uncovered = [token_label[c] for c in token_label if c not in covered_tokens]
    return picks, uncovered


def _computed_set(active_tokens, coverage):
    from dashboard.biofield_setcover import minimal_remedies
    return [p["remedy"] for p in minimal_remedies(active_tokens, coverage)["picks"]]


def suggest_minimal_remedies(cx, tid, chain_rows):
    """Fewest remedies covering active+required stresses. Returns picks + uncovered
    (kept for any caller that wants the raw computed set)."""
    active_tokens, token_label, coverage = _remedy_context(cx, tid, chain_rows)
    picks, uncovered = _covers_for(_computed_set(active_tokens, coverage),
                                   active_tokens, token_label, coverage)
    return {"picks": picks, "uncovered": uncovered}


def get_saved_remedy_set(cx, tid):
    row = cx.execute("SELECT remedies_json FROM biofield_auth_remedy_set WHERE test_id=?",
                     (_num(tid),)).fetchone()
    if not row or not row[0]:
        return None
    try:
        return [r for r in json.loads(row[0]) if isinstance(r, str)]
    except Exception:
        return None


def save_remedy_set(cx, tid, remedies):
    init_stress_tables(cx)
    rems = [(r or "").strip() for r in (remedies or []) if (r or "").strip()]
    cx.execute("INSERT INTO biofield_auth_remedy_set(test_id,remedies_json,updated_at) "
               "VALUES(?,?,?) ON CONFLICT(test_id) DO UPDATE SET "
               "remedies_json=excluded.remedies_json, updated_at=excluded.updated_at",
               (_num(tid), json.dumps(rems), _now()))
    cx.commit()
    return rems


def clear_remedy_set(cx, tid):
    cx.execute("DELETE FROM biofield_auth_remedy_set WHERE test_id=?", (_num(tid),))
    cx.commit()


def _get_pattern_set(cx, key):
    if not key:
        return None
    row = cx.execute("SELECT remedies_json FROM biofield_remedy_pattern WHERE pattern_key=?",
                     (key,)).fetchone()
    if not row or not row[0]:
        return None
    try:
        return [r for r in json.loads(row[0]) if isinstance(r, str)]
    except Exception:
        return None


def save_pattern_set(cx, tid, chain_rows, remedies):
    """Persist the current ordered set as a reusable template keyed by the stress pattern."""
    init_stress_tables(cx)
    active_tokens, token_label, _cov = _remedy_context(cx, tid, chain_rows)
    key, toks = _pattern_key(active_tokens)
    if not key:
        return {"ok": False, "reason": "no active required stresses to key a pattern"}
    rems = [(r or "").strip() for r in (remedies or []) if (r or "").strip()]
    label = ", ".join(token_label.get(t, t) for t in toks)[:240]
    cx.execute("INSERT INTO biofield_remedy_pattern"
               "(pattern_key,tokens_json,remedies_json,label,updated_at) VALUES(?,?,?,?,?) "
               "ON CONFLICT(pattern_key) DO UPDATE SET tokens_json=excluded.tokens_json, "
               "remedies_json=excluded.remedies_json, label=excluded.label, "
               "updated_at=excluded.updated_at",
               (key, json.dumps(toks), json.dumps(rems), label, _now()))
    cx.commit()
    return {"ok": True, "pattern_key": key, "count": len(rems)}


def resolve_remedy_set(cx, tid, chain_rows, force_computed=False):
    """Panel source of truth. Priority: per-test saved set -> exact stress-pattern
    template -> freshly computed set-cover. force_computed skips the first two."""
    active_tokens, token_label, coverage = _remedy_context(cx, tid, chain_rows)
    key, _toks = _pattern_key(active_tokens)
    remedies, source = None, "computed"
    if not force_computed:
        saved = get_saved_remedy_set(cx, tid)
        if saved is not None:
            remedies, source = saved, "saved"
        else:
            pat = _get_pattern_set(cx, key)
            if pat is not None:
                remedies, source = pat, "pattern"
    if remedies is None:
        remedies = _computed_set(active_tokens, coverage)
    picks, uncovered = _covers_for(remedies, active_tokens, token_label, coverage)
    return {"picks": picks, "uncovered": uncovered, "remedies": remedies,
            "source": source, "pattern_key": key,
            "has_pattern": _get_pattern_set(cx, key) is not None}


def layer_candidates(cx, tid, chain_rows, fallback_by_code=None, n=5):
    """Per-layer ranked remedy pick-list that AUGMENTS the set-cover default.

    For each causal-chain layer, return the current pick(s) as `default` plus up to
    `n` candidates ordered coverage-first with a learned boost:
      - coverage: remedies in the per-test coverage map that cover >=1 of the
        layer's stress codes, ranked by how many they cover;
      - learned boost: remedies in the saved set / pattern template for this test
        are lifted to the top of their tier and tagged `used_before`;
      - blank layers (no coverer, but the layer still has codes) fall back to
        `fallback_by_code` -- a {code: [remedy names]} map the CALLER injects from
        the formulation map -- tagged source "functional", so a layer never shows
        nothing.
    Pure over `cx` (biofield_auth_* tables); e4l.db / the formulation map stay in
    the caller, keeping this function's DB boundary clean and unit-testable."""
    stresses = list_stresses(cx, tid, chain_rows)
    active_tokens, _label, coverage = _remedy_context(cx, tid, chain_rows)
    key, _toks = _pattern_key(active_tokens)
    learned = set()
    for src in (get_saved_remedy_set(cx, tid), _get_pattern_set(cx, key)):
        for r in (src or []):
            low = (r or "").strip().lower()
            if low:
                learned.add(low)
    fb = fallback_by_code or {}
    # Per-layer stress codes carried from the synthesis/reveal (biofield_auth_chain.codes).
    # The coverage-based stress ASSIGNMENT is sparse for hand-authored chains -- a layer
    # keeps its own patterns here so it can still generate candidates.
    chain_codes = {}
    try:
        _crows = cx.execute("SELECT layer, codes FROM biofield_auth_chain WHERE test_id=?",
                            (_num(tid),)).fetchall()
    except Exception:
        _crows = []                                   # table/column absent -> no per-layer codes
    for ln, cj in _crows:
        if not cj:
            continue
        try:
            for c in (json.loads(cj) or []):
                if c:
                    chain_codes.setdefault(ln, set()).add(c)
        except Exception:
            continue
    out = []
    for L in stresses["by_layer"]:
        codes = {s["code"] for s in L["stresses"] if s.get("code")}
        codes |= chain_codes.get(L["layer"], set())
        default_disp = list(L.get("remedies") or [])
        default_lower = {(d or "").strip().lower() for d in default_disp}
        scored = []
        for remedy, rcodes in coverage.items():
            hit = sorted(set(rcodes) & codes)
            if not hit:
                continue
            low = (remedy or "").strip().lower()
            scored.append({"remedy": remedy, "covers": hit, "coverage": len(hit),
                           "source": "coverage", "used_before": low in learned,
                           "is_default": low in default_lower})
        # learned first, then most coverage, then name (stable, hash-seed independent)
        scored.sort(key=lambda c: (not c["used_before"], -c["coverage"], c["remedy"].lower()))
        if not scored and codes:
            seen = set()
            for code in sorted(codes):
                for name in (fb.get(code) or []):
                    low = (name or "").strip().lower()
                    if low and low not in seen:
                        seen.add(low)
                        scored.append({"remedy": name, "covers": [code], "coverage": 0,
                                       "source": "functional", "used_before": low in learned,
                                       "is_default": low in default_lower})
        # Always surface the layer's CURRENT remedy as a (default) candidate, even when
        # the coverage map doesn't list it -- so review always shows "current + alternatives"
        # rather than alternatives with the current pick mysteriously absent.
        present = {(c["remedy"] or "").strip().lower() for c in scored}
        for dname in default_disp:
            dl = (dname or "").strip().lower()
            if dl and dl not in present:
                dcov = sorted(coverage.get(dl, set()) & codes)
                scored.insert(0, {"remedy": dname, "covers": dcov, "coverage": len(dcov),
                                  "source": "current", "used_before": dl in learned,
                                  "is_default": True})
                present.add(dl)
        capped = scored[:n]
        if scored and not any(c.get("is_default") for c in capped):
            dflt = next((c for c in scored if c.get("is_default")), None)
            if dflt:
                capped = capped[:max(0, n - 1)] + [dflt]
        out.append({"n": L["layer"], "head": L.get("head") or "",
                    "codes": sorted(codes), "default": default_disp,
                    "candidates": capped})
    return out
