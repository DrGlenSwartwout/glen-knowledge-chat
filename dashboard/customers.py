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
        out.append({k: d.get(k, "") for k in PICKER_COLS})
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
