"""Condition triage: short self-report (glaucoma pilot) -> which condition
program(s) apply -> seed those remedies into recommendations under the
`condition` source. Postgres-portable (composite PK, DELETE-then-INSERT upsert)."""
import json
from datetime import datetime, timezone

_N, _E = "glaucoma-normal-iop", "glaucoma-elevated-iop"


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS condition_triage ("
        " email TEXT, condition TEXT, iop_od TEXT, iop_os TEXT, on_meds INTEGER,"
        " med_count INTEGER, meds_names TEXT, field_loss INTEGER, category TEXT,"
        " resolved_programs TEXT, updated_at TEXT, PRIMARY KEY (email, condition))")
    cx.commit()


def _floats(*vals):
    out = []
    for v in vals:
        try:
            if v not in (None, ""):
                out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def resolve_programs(condition, answers):
    """Glaucoma triage decision table (Dr. Glen's confirmed clinical rules).
    condition != "glaucoma" -> [] (pilot: glaucoma only).
    on_meds -> [E, N] (treated IOP masks baseline, lead with Elevated).
    Else, by higher (worse) eye IOP: <20 -> [N]; 20-21 borderline -> [E, N]
    unless field_loss -> [E]; >=22 -> [E].
    No IOP numbers given -> fall back to self-reported category, or (if "not
    sure"/absent) the field-loss tiebreak."""
    if (condition or "").strip().lower() != "glaucoma":
        return []                                   # pilot: glaucoma only
    a = answers or {}
    field_loss = bool(a.get("field_loss"))
    if bool(a.get("on_meds")):
        return [_E, _N]                             # treated IOP masks baseline -> both, lead E
    iops = _floats(a.get("iop_od"), a.get("iop_os"))
    if iops:
        hi = max(iops)
        if hi < 20:
            return [_N]
        if hi <= 21:
            return [_E] if field_loss else [_E, _N]   # borderline
        return [_E]
    cat = (a.get("category") or "").strip().lower()
    if cat == "normal":
        return [_N]
    if cat == "elevated":
        return [_E]
    return [_E] if field_loss else [_E, _N]         # not sure -> field-loss tiebreak


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def upsert_triage(cx, email, condition, answers, resolved):
    email = (email or "").strip().lower()
    condition = (condition or "").strip().lower()
    a = answers or {}
    cx.execute("DELETE FROM condition_triage WHERE email=? AND condition=?", (email, condition))
    cx.execute(
        "INSERT INTO condition_triage (email, condition, iop_od, iop_os, on_meds, med_count,"
        " meds_names, field_loss, category, resolved_programs, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (email, condition, str(a.get("iop_od") or ""), str(a.get("iop_os") or ""),
         1 if a.get("on_meds") else 0, _safe_int(a.get("med_count"), 0),
         (a.get("meds_names") or "").strip(), 1 if a.get("field_loss") else 0,
         (a.get("category") or "").strip(), json.dumps(resolved), _now()))
    cx.commit()


def get_triage(cx, email, condition):
    r = cx.execute(
        "SELECT iop_od,iop_os,on_meds,med_count,meds_names,field_loss,category,resolved_programs"
        " FROM condition_triage WHERE email=? AND condition=?",
        ((email or "").strip().lower(), (condition or "").strip().lower())).fetchone()
    if not r:
        return None
    return {"iop_od": r[0], "iop_os": r[1], "on_meds": bool(r[2]), "med_count": r[3],
            "meds_names": r[4], "field_loss": bool(r[5]), "category": r[6],
            "resolved_programs": json.loads(r[7] or "[]")}


def seed_from_triage(cx, email, condition, answers):
    """Resolve programs, persist the triage answers, then seed (or re-seed,
    replacing any prior condition-sourced rows) the resolved programs' items
    into recommendation_events under source_key="condition", origin_ref=condition."""
    from dashboard import recommendation_events as _re, condition_programs as _cp
    from dashboard.related_products import DO_NOT_RECOMMEND
    init_table(cx)
    email = (email or "").strip().lower()
    condition = (condition or "").strip().lower()
    keys = resolve_programs(condition, answers)
    upsert_triage(cx, email, condition, answers, keys)
    _re.clear_events(cx, email, "condition", condition)          # replace prior seed
    seeded, seen = [], set()
    for k in keys:
        prog = _cp.get(cx, k)
        if not prog:
            continue
        for it in _cp.resolve_program_items(prog, audience="client"):
            slug = (it or {}).get("slug")
            if not slug or slug in seen:
                continue
            if slug in DO_NOT_RECOMMEND:
                continue
            seen.add(slug)
            _re.record_event(cx, email, slug, "condition",
                              occurred_at=_now(), origin_ref=condition)
            seeded.append(slug)
    return {"programs": keys, "seeded": seeded}
