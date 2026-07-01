"""Per-test master stress list + remedy<->stress coverage map for the local
Biofield Intake balancing loop (B1). Pure sqlite; the caller passes a connection.
Balanced state is DERIVED at read time, never stored (see list_stresses)."""
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
    if " " not in core:                    # a coded slug -> spaced words
        core = core.replace("-", " ").replace("_", " ")
    core = " ".join(w[:1].upper() + w[1:] if w else w for w in core.split(" "))
    return core or label


def init_stress_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_stress(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, code TEXT, label TEXT,
        source TEXT NOT NULL DEFAULT 'scan', balance TEXT NOT NULL DEFAULT 'optional',
        manual_balanced INTEGER NOT NULL DEFAULT 0, created_at TEXT, updated_at TEXT,
        UNIQUE(test_id, source, code))""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_remedy_coverage(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, remedy TEXT, code TEXT,
        UNIQUE(test_id, remedy, code))""")
    cx.commit()


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
