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
