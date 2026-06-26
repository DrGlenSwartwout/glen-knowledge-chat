"""In-house canonical store for a person's clinical attributes (CTI-1 foundation).
Authoritative store + controlled-vocabulary/alias mechanism. Pure: takes a sqlite
connection; mirrors dashboard/biofield_meanings.py. Dark/non-breaking in CTI-1 —
no caller is wired yet (CTI-2 makes it authoritative)."""
import json
import re
import sqlite3
from datetime import datetime, timezone

DISCRETE_FIELDS = ("tags", "conditions", "terrain_concerns", "body_systems")
SCALAR_FIELDS = ("challenges", "goals")
ALL_FIELDS = DISCRETE_FIELDS + SCALAR_FIELDS


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(s):
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def _clean(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS canonical_vocab ("
               "field TEXT, alias_norm TEXT, canonical TEXT, "
               "PRIMARY KEY(field, alias_norm))")
    cx.execute("CREATE TABLE IF NOT EXISTS person_attributes ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, field TEXT, "
               "value TEXT, value_norm TEXT, source TEXT, added_at TEXT, "
               "UNIQUE(email, field, value_norm))")
    cx.commit()


def resolve(cx, field, value):
    v = _clean(value)
    if not v:
        return ""
    if field in DISCRETE_FIELDS:
        row = cx.execute(
            "SELECT canonical FROM canonical_vocab WHERE field=? AND alias_norm=?",
            (field, _norm(v))).fetchone()
        if row and (row[0] or "").strip():
            return row[0].strip()
    return v


def set_attr(cx, email, field, value, *, source):
    init_tables(cx)
    email = (email or "").strip().lower()
    if not email or field not in ALL_FIELDS:
        return False
    canon = resolve(cx, field, value)
    if not canon:
        return False
    now, vn = _now(), _norm(canon)
    if field in SCALAR_FIELDS:
        cx.execute("DELETE FROM person_attributes WHERE email=? AND field=?", (email, field))
        cx.execute(
            "INSERT INTO person_attributes(email,field,value,value_norm,source,added_at) "
            "VALUES(?,?,?,?,?,?)", (email, field, canon, vn, source, now))
        cx.commit()
        return True
    cur = cx.execute(
        "INSERT OR IGNORE INTO person_attributes(email,field,value,value_norm,source,added_at) "
        "VALUES(?,?,?,?,?,?)", (email, field, canon, vn, source, now))
    cx.commit()
    return cur.rowcount > 0


def get_person(cx, email):
    init_tables(cx)
    cx.row_factory = sqlite3.Row
    email = (email or "").strip().lower()
    out = {f: ([] if f in DISCRETE_FIELDS else "") for f in ALL_FIELDS}
    for r in cx.execute(
            "SELECT field, value FROM person_attributes WHERE email=?", (email,)).fetchall():
        f = r["field"]
        if f in DISCRETE_FIELDS:
            out[f].append(r["value"])
        elif f in SCALAR_FIELDS:
            out[f] = r["value"]
    for f in DISCRETE_FIELDS:
        out[f] = sorted(out[f])
    return out


def rebuild_people_columns(cx, email):
    email = (email or "").strip().lower()
    p = get_person(cx, email)
    cx.execute(
        "UPDATE people SET tags=?, conditions=?, terrain_concerns=?, body_systems=?, "
        "challenges=?, goals=? WHERE lower(email)=?",
        (json.dumps(p["tags"]), json.dumps(p["conditions"]),
         json.dumps(p["terrain_concerns"]), json.dumps(p["body_systems"]),
         p["challenges"], p["goals"], email))
    cx.commit()


def _parse_list(s):
    try:
        v = json.loads(s or "[]")
    except Exception:
        return []
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def import_from_people(cx):
    init_tables(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT email, tags, conditions, terrain_concerns, body_systems, challenges, goals "
        "FROM people WHERE TRIM(COALESCE(email,''))<>''").fetchall()
    persons, attrs = 0, 0
    for r in rows:
        persons += 1
        for f in DISCRETE_FIELDS:
            for val in _parse_list(r[f]):
                if set_attr(cx, r["email"], f, val, source="import"):
                    attrs += 1
        for f in SCALAR_FIELDS:
            if (r[f] or "").strip() and set_attr(cx, r["email"], f, r[f], source="import"):
                attrs += 1
    return {"persons": persons, "attrs": attrs}
