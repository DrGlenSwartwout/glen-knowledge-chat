"""Condition triage: short self-report (glaucoma pilot, plus cataract and
macular sub-type triage) -> which condition program(s) apply -> seed those
remedies into recommendations under the `condition` source. Postgres-portable
(composite PK, DELETE-then-INSERT upsert)."""
import json
from datetime import datetime, timezone

from dashboard import db

_N, _E = "glaucoma-normal-iop", "glaucoma-elevated-iop"

# Conditions that map to exactly ONE program: no triage questions needed,
# resolve straight to that program.
_SINGLE_PROGRAM = {
    "dry-eye": "dry-eye",
    "retinitis-pigmentosa": "retinitis-pigmentosa",
    "diabetic-retinopathy": "diabetic-retinopathy",
    "vision-improvement": "vision-improvement",
}

# Cataract "not sure" risk flags (Dr. Glen): none of these change routing
# under rule 1 (told-PSC) or rule 2 (told-senile) -- they are captured as
# data regardless -- but under rule 3 (not sure/unspecified) at age >= 50 (or
# unknown age), any one of them present routes to BOTH programs instead of
# senile-cataract alone.
_CATARACT_RISK_FLAGS = ("steroids", "diabetes", "inflammation", "radiation", "atopy")

# Extra columns beyond the original glaucoma-pilot schema, added for cataract
# + macular sub-type triage. ALTER-added if missing so an existing prod table
# (already carrying glaucoma rows) picks them up without a destructive
# migration -- mirrors condition_programs.py's modifiers_json column add.
_NEW_COLUMNS = (
    ("cataract_type", "TEXT"), ("age", "TEXT"),
    ("steroids", "INTEGER"), ("diabetes", "INTEGER"), ("inflammation", "INTEGER"),
    ("radiation", "INTEGER"), ("atopy", "INTEGER"), ("yellow_vision", "INTEGER"),
    ("amd_type", "TEXT"), ("injections", "INTEGER"), ("distortion", "INTEGER"),
    ("other_condition", "TEXT"),
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS condition_triage ("
        " email TEXT, condition TEXT, iop_od TEXT, iop_os TEXT, on_meds INTEGER,"
        " med_count INTEGER, meds_names TEXT, field_loss INTEGER, category TEXT,"
        " resolved_programs TEXT, updated_at TEXT, PRIMARY KEY (email, condition))")
    for col, coltype in _NEW_COLUMNS:
        if not db.column_exists(cx, "condition_triage", col):
            cx.execute("ALTER TABLE condition_triage ADD COLUMN %s %s" % (col, coltype))
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


def _age(a):
    """Parse the `age` answer to a float, or None if absent/unparseable
    ("not sure"/blank age is treated the same as unknown age)."""
    v = (a or {}).get("age")
    try:
        if v not in (None, ""):
            return float(v)
    except (TypeError, ValueError):
        pass
    return None


def _resolve_cataract(answers):
    """Dr. Glen's confirmed cataract sub-type decision table.
    1. cataract_type == "psc": age > 50 -> [psc, senile] (PSC plus senile
       remedies); otherwise -> [psc] only. Risk flags do NOT change this rule.
    2. cataract_type == "senile" -> [senile] only.
    3. not sure / unspecified:
       - age < 50 -> [psc]
       - age >= 50 (or age unknown) AND any risk flag present -> [psc, senile]
       - age >= 50 (or age unknown) with NO risk flags -> [senile]
    Boundary: exactly 50 counts as ">= 50" (the senile side)."""
    a = answers or {}
    ctype = (a.get("cataract_type") or "").strip().lower()
    age = _age(a)
    if ctype == "psc":
        if age is not None and age > 50:
            return ["psc-cataract", "senile-cataract"]
        return ["psc-cataract"]
    if ctype == "senile":
        return ["senile-cataract"]
    # not sure / unspecified
    if age is not None and age < 50:
        return ["psc-cataract"]
    has_risk = any(bool(a.get(f)) for f in _CATARACT_RISK_FLAGS)
    if has_risk:
        return ["psc-cataract", "senile-cataract"]
    return ["senile-cataract"]


def _resolve_macular(answers):
    """Dr. Glen's confirmed macular sub-type decision table.
    1. amd_type == "wet" OR injections truthy -> [wet-amd]
    2. amd_type == "dry" -> [dry-amd]
    3. not sure: distortion truthy -> [wet-amd]; else -> [dry-amd]"""
    a = answers or {}
    amd_type = (a.get("amd_type") or "").strip().lower()
    if amd_type == "wet" or bool(a.get("injections")):
        return ["wet-amd"]
    if amd_type == "dry":
        return ["dry-amd"]
    return ["wet-amd"] if bool(a.get("distortion")) else ["dry-amd"]


def resolve_programs(condition, answers):
    """Single-program conditions (see _SINGLE_PROGRAM) resolve immediately with
    no triage questions. cataract/macular resolve via their own sub-type
    decision tables (see _resolve_cataract/_resolve_macular). Otherwise: the
    glaucoma triage decision table (Dr. Glen's confirmed clinical rules); any
    other condition -> [].
    on_meds -> [E, N] (treated IOP masks baseline, lead with Elevated).
    Else, by higher (worse) eye IOP: <20 -> [N]; 20-21 borderline -> [E, N]
    unless field_loss -> [E]; >=22 -> [E].
    No IOP numbers given -> fall back to self-reported category, or (if "not
    sure"/absent) the field-loss tiebreak."""
    condition = (condition or "").strip().lower()
    single = _SINGLE_PROGRAM.get(condition)
    if single:
        return [single]                             # single-program: no triage needed
    if condition == "cataract":
        return _resolve_cataract(answers)
    if condition == "macular":
        return _resolve_macular(answers)
    if condition != "glaucoma":
        return []                                   # unrecognized condition
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


def _dry_eye_facts(answers):
    """The two client facts that drive the dry-eye program's modifiers
    (see condition_programs seed: aqueous_deficiency, severe).

    aqueous_deficiency defaults TRUE when unanswered (client_default: true --
    assume aqueous tear deficiency is present unless the client says
    otherwise), driven by either the Sjogren's-triad question ("Do you also
    have dry mouth or vaginal dryness?", answers["sjogrens"]) or a direct
    "not enough tears" option (answers["not_enough_tears"]). An explicit
    "no" on the direct question (and no corroborating Sjogren's-triad "yes")
    overrides the default to False.

    severe defaults FALSE when unanswered (client_default: false -- only
    when the client reports it), driven by answers["severe"]
    ("Is your dry eye severe?")."""
    a = answers or {}
    sjogrens = bool(a.get("sjogrens"))
    direct = a.get("not_enough_tears")
    if sjogrens or direct is True:
        aqueous = True
    elif direct is False:
        aqueous = False
    else:
        aqueous = True  # unanswered -> assume present (client_default: true)
    return {"aqueous_deficiency": aqueous, "severe": bool(a.get("severe"))}


def resolve_client_facts(condition, answers):
    """client_facts dict (for condition_programs.resolve_program_items'
    client_facts= kwarg) implied by a condition's triage answers -- distinct
    from the separately-stored, persistent dashboard/client_facts.py facts
    (which cover things like on_areds2 and are edited independently of
    triage). Additive: a condition with no applicable facts returns {}.

    dry-eye: aqueous_deficiency/severe, see _dry_eye_facts.
    cataract: `brunescent` <- the "Does your vision seem more yellow, less
    blue?" answer (key `yellow_vision`), default False."""
    a = answers or {}
    condition = (condition or "").strip().lower()
    if condition == "dry-eye":
        return _dry_eye_facts(answers)
    facts = {}
    if condition == "cataract":
        facts["brunescent"] = bool(a.get("yellow_vision"))
    return facts


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
        " meds_names, field_loss, category, cataract_type, age, steroids, diabetes,"
        " inflammation, radiation, atopy, yellow_vision, amd_type, injections, distortion,"
        " other_condition, resolved_programs, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (email, condition, str(a.get("iop_od") or ""), str(a.get("iop_os") or ""),
         1 if a.get("on_meds") else 0, _safe_int(a.get("med_count"), 0),
         (a.get("meds_names") or "").strip(), 1 if a.get("field_loss") else 0,
         (a.get("category") or "").strip(),
         (a.get("cataract_type") or "").strip(), str(a.get("age") or ""),
         1 if a.get("steroids") else 0, 1 if a.get("diabetes") else 0,
         1 if a.get("inflammation") else 0, 1 if a.get("radiation") else 0,
         1 if a.get("atopy") else 0, 1 if a.get("yellow_vision") else 0,
         (a.get("amd_type") or "").strip(), 1 if a.get("injections") else 0,
         1 if a.get("distortion") else 0,
         (a.get("other_condition") or "").strip(),
         json.dumps(resolved), _now()))
    cx.commit()


def get_triage(cx, email, condition):
    r = cx.execute(
        "SELECT iop_od,iop_os,on_meds,med_count,meds_names,field_loss,category,"
        " cataract_type,age,steroids,diabetes,inflammation,radiation,atopy,"
        " yellow_vision,amd_type,injections,distortion,other_condition,resolved_programs"
        " FROM condition_triage WHERE email=? AND condition=?",
        ((email or "").strip().lower(), (condition or "").strip().lower())).fetchone()
    if not r:
        return None
    return {"iop_od": r[0], "iop_os": r[1], "on_meds": bool(r[2]), "med_count": r[3],
            "meds_names": r[4], "field_loss": bool(r[5]), "category": r[6],
            "cataract_type": r[7], "age": r[8],
            "steroids": bool(r[9]), "diabetes": bool(r[10]), "inflammation": bool(r[11]),
            "radiation": bool(r[12]), "atopy": bool(r[13]), "yellow_vision": bool(r[14]),
            "amd_type": r[15], "injections": bool(r[16]), "distortion": bool(r[17]),
            "other_condition": r[18],
            "resolved_programs": json.loads(r[19] or "[]")}


def seed_from_triage(cx, email, condition, answers):
    """Resolve programs, persist the triage answers, then seed (or re-seed,
    replacing any prior condition-sourced rows) the resolved programs' items
    into recommendation_events under source_key="condition", origin_ref=condition.
    Also reports whether ANY resolved program has consult_recommended set, so
    the portal UI can push harder toward a Biofield Analysis (e.g. wet-amd)."""
    from dashboard import recommendation_events as _re, condition_programs as _cp
    from dashboard.related_products import DO_NOT_RECOMMEND
    init_table(cx)
    email = (email or "").strip().lower()
    condition = (condition or "").strip().lower()
    keys = resolve_programs(condition, answers)
    upsert_triage(cx, email, condition, answers, keys)
    facts = resolve_client_facts(condition, answers)
    _re.clear_events(cx, email, "condition", condition)          # replace prior seed
    seeded, seen = [], set()
    consult_recommended = False
    for k in keys:
        prog = _cp.get(cx, k)
        if not prog:
            continue
        if prog.get("consult_recommended"):
            consult_recommended = True
        for it in _cp.resolve_program_items(prog, audience="client", client_facts=facts):
            slug = (it or {}).get("slug")
            if not slug or slug in seen:
                continue
            if slug in DO_NOT_RECOMMEND:
                continue
            seen.add(slug)
            _re.record_event(cx, email, slug, "condition",
                              occurred_at=_now(), origin_ref=condition)
            seeded.append(slug)
    return {"programs": keys, "seeded": seeded, "consult_recommended": consult_recommended}
