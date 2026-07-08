"""In-house customer records over the existing `people` table (Phase 1 of the
order-entry / proposed-invoice build). The People directory already carries
email/phone/name/city/state for imported contacts; here we add a full shipping
address and the lookup/save helpers the order-entry form needs. Pure functions
over a sqlite connection (cx) for testability; the people + orders tables live in
the same LOG_DB."""
import json
from datetime import datetime, timezone

# Address columns added to `people` (city/state/country/phone already exist).
_ADDR_COLS = ("address1", "address2", "zip")

# Columns the order-entry customer picker reads back.
PICKER_COLS = ("id", "name", "first_name", "last_name", "email", "phone",
               "address1", "address2", "city", "state", "zip", "country")


def _now():
    return datetime.now(timezone.utc).isoformat()


def add_people_address_columns(cx):
    """Additively migrate `people` to carry a full shipping address. Idempotent."""
    for col in _ADDR_COLS:
        try:
            cx.execute(f"ALTER TABLE people ADD COLUMN {col} TEXT DEFAULT ''")
        except Exception:
            pass  # already present
    cx.commit()


def _person_row(cx, person_id):
    cx.row_factory = __import__("sqlite3").Row
    return cx.execute("SELECT * FROM people WHERE id=?", (int(person_id),)).fetchone()


def get_person(cx, person_id):
    row = _person_row(cx, person_id)
    if row is None:
        return None
    d = dict(row)
    return {k: d.get(k, "") for k in PICKER_COLS}


def find_people(cx, query, limit=10):
    """Case-insensitive LIKE match over name/email/phone for the picker. Returns
    client/known contacts first (those with an order history or a saved address)."""
    q = (query or "").strip()
    if not q:
        return []
    cx.row_factory = __import__("sqlite3").Row
    like = f"%{q.lower()}%"
    rows = cx.execute(
        "SELECT * FROM people WHERE lower(name) LIKE ? OR lower(email) LIKE ? "
        "OR lower(coalesce(first_name,'')||' '||coalesce(last_name,'')) LIKE ? "
        "OR replace(coalesce(phone,''),' ','') LIKE ? "
        "ORDER BY order_count DESC, last_order_date DESC LIMIT ?",
        (like, like, like, like, int(limit))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        rec = {k: d.get(k, "") for k in PICKER_COLS}
        # A record can match a name search via first_name/last_name while the
        # `name` column is blank OR holds a placeholder email (imported contacts
        # sometimes have the email copied into `name`, e.g. Miriam Lynn Nelson =
        # "heritagecms@aol.com"). In both cases synthesize a real name from
        # first/last so the order-entry picker fills the Name field with a name,
        # not the email. Only substitute when we actually have a first/last to use
        # — never blank out a name we can't replace.
        # (See test_find_people_synthesizes_name_from_first_last_when_blank
        #  and test_find_people_replaces_email_in_name_column.)
        nm = (rec.get("name") or "").strip()
        first_last = (str(rec.get("first_name") or "") + " "
                      + str(rec.get("last_name") or "")).strip()
        if first_last and (not nm or "@" in nm):
            rec["name"] = first_last
        out.append(rec)
    return out


def upsert_person_address(cx, person_id, addr):
    """Save a shipping address back onto a person so it's on file next time.
    Only non-empty fields overwrite existing values."""
    addr = addr or {}
    field_map = {
        "address1": addr.get("address1") or addr.get("street") or "",
        "address2": addr.get("address2") or "",
        "city": addr.get("city") or "",
        "state": addr.get("state") or "",
        "zip": addr.get("zip") or addr.get("postal") or "",
        "country": (addr.get("country") or "").upper(),
        "phone": addr.get("phone") or "",
    }
    sets, vals = [], []
    for col, val in field_map.items():
        if str(val).strip():
            sets.append(f"{col}=?")
            vals.append(str(val).strip())
    if not sets:
        return False
    sets.append("updated_at=?")
    vals.append(_now())
    vals.append(int(person_id))
    cx.execute(f"UPDATE people SET {', '.join(sets)} WHERE id=?", vals)
    cx.commit()
    return True


def find_or_create_by_email(cx, *, email, name="", phone=""):
    """Return an existing person id for this email, or create a minimal record.
    Email is the unique key on `people`."""
    em = (email or "").strip().lower()
    if not em:
        return None
    row = cx.execute("SELECT id FROM people WHERE lower(email)=?", (em,)).fetchone()
    if row:
        return row[0]
    cur = cx.execute(
        "INSERT INTO people (email, name, phone, source, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (em, (name or "").strip(), (phone or "").strip(), "order-entry", _now(), _now()))
    cx.commit()
    return cur.lastrowid


def rename_by_email(cx, email, *, name, first_name=None, last_name=None):
    """Correct a customer's display name across their people record AND every order
    they have — both keyed by email. The invoice bills to the ORDER name, so a
    person-only rename wouldn't reach it. Only non-blank values are written; all
    other fields are left untouched. Returns {people_updated, orders_updated}."""
    em = (email or "").strip().lower()
    nm = (name or "").strip()
    if not em or not nm:
        raise ValueError("email and non-blank name required")
    ts = _now()
    people_updated = cx.execute(
        "UPDATE people SET name=?, updated_at=? WHERE lower(email)=?", (nm, ts, em)).rowcount
    if (first_name or "").strip():
        cx.execute("UPDATE people SET first_name=? WHERE lower(email)=?",
                   ((first_name or "").strip(), em))
    if (last_name or "").strip():
        cx.execute("UPDATE people SET last_name=? WHERE lower(email)=?",
                   ((last_name or "").strip(), em))
    orders_updated = cx.execute(
        "UPDATE orders SET name=?, updated_at=? WHERE lower(email)=?", (nm, ts, em)).rowcount
    cx.commit()
    return {"people_updated": people_updated, "orders_updated": orders_updated}


def last_address_for(cx, email):
    """The most recent shipping address this email shipped to (from orders), so a
    repeat customer without a saved people-address still autofills."""
    em = (email or "").strip().lower()
    if not em:
        return {}
    row = cx.execute(
        "SELECT address_json FROM orders WHERE lower(email)=? AND address_json IS NOT NULL "
        "AND address_json NOT IN ('', '{}') ORDER BY created_at DESC, id DESC LIMIT 1",
        (em,)).fetchone()
    if not row:
        return {}
    try:
        a = json.loads(row[0] if not hasattr(row, "keys") else row["address_json"])
    except Exception:
        return {}
    # Normalise the orders address_json shape ({street,...}) to the people shape.
    return {
        "address1": a.get("street") or a.get("address1") or "",
        "address2": a.get("address2") or "",
        "city": a.get("city") or "", "state": a.get("state") or "",
        "zip": a.get("zip") or "", "country": a.get("country") or "US",
    }
