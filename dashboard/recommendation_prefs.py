"""Per-client recommendation preferences: per-(client, product) notes (operator + client)
and per-(client, section) collapse state. Separate from recommendation_events (the log) and
recommendation_hidden (the hide flag). Pure: functions take an open sqlite connection."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(e):
    return (e or "").strip().lower()


def init_recommendation_prefs(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_notes (
            client_email  TEXT NOT NULL,
            product_key   TEXT NOT NULL,
            operator_note TEXT NOT NULL DEFAULT '',
            client_note   TEXT NOT NULL DEFAULT '',
            updated_at    TEXT,
            PRIMARY KEY (client_email, product_key)
        )""")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_section_state (
            client_email TEXT NOT NULL,
            section_key  TEXT NOT NULL,
            collapsed    INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT,
            PRIMARY KEY (client_email, section_key)
        )""")
    cx.commit()


def get_notes(cx, email):
    rows = cx.execute(
        "SELECT product_key, operator_note, client_note FROM recommendation_notes "
        "WHERE client_email=?", (_norm(email),)).fetchall()
    return {r[0]: {"operator_note": r[1] or "", "client_note": r[2] or ""} for r in rows}


def set_operator_note(cx, email, product_key, note):
    _set_note(cx, email, product_key, "operator_note", note)


def set_client_note(cx, email, product_key, note):
    _set_note(cx, email, product_key, "client_note", note)


def _set_note(cx, email, product_key, field, note):
    # field is a fixed internal literal ("operator_note" | "client_note"), never user input.
    assert field in ("operator_note", "client_note")
    e = _norm(email)
    pk = (product_key or "").strip()
    if not e or not pk:
        return
    val = (note or "").strip()
    cx.execute(
        f"INSERT INTO recommendation_notes (client_email, product_key, {field}, updated_at) "
        f"VALUES (?,?,?,?) ON CONFLICT(client_email, product_key) "
        f"DO UPDATE SET {field}=excluded.{field}, updated_at=excluded.updated_at",
        (e, pk, val, _now()))
    cx.commit()


def get_section_state(cx, email):
    rows = cx.execute(
        "SELECT section_key, collapsed FROM recommendation_section_state WHERE client_email=?",
        (_norm(email),)).fetchall()
    return {r[0]: bool(r[1]) for r in rows}


def set_section_state(cx, email, section_key, collapsed):
    e = _norm(email)
    sk = (section_key or "").strip()
    if not e or not sk:
        return
    cx.execute(
        "INSERT INTO recommendation_section_state (client_email, section_key, collapsed, updated_at) "
        "VALUES (?,?,?,?) ON CONFLICT(client_email, section_key) "
        "DO UPDATE SET collapsed=excluded.collapsed, updated_at=excluded.updated_at",
        (e, sk, 1 if collapsed else 0, _now()))
    cx.commit()
