"""Email suppression list: addresses that permanently fail (hard bounces) so the
app stops emailing them. Populated by the local bounce scanner via the
email_suppression.add console action. Spam-blocks are NOT stored here (the address
is valid — our sender reputation is the problem). Reversible: delete a row if an
address recovers."""
import sqlite3


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS email_suppression (
        email TEXT PRIMARY KEY, bounce_type TEXT, reason TEXT,
        source TEXT, created_at TEXT DEFAULT (datetime('now')))""")
    cx.commit()


def is_suppressed(cx, email):
    if not email:
        return False
    try:
        r = cx.execute("SELECT 1 FROM email_suppression WHERE email=lower(?)",
                       (email.strip().lower(),)).fetchone()
    except sqlite3.OperationalError:
        return False
    return bool(r)


def add(cx, email, bounce_type, reason, source):
    if not email:
        return
    cx.execute("""INSERT INTO email_suppression(email,bounce_type,reason,source)
        VALUES(lower(?),?,?,?) ON CONFLICT(email) DO UPDATE SET
        bounce_type=excluded.bounce_type, reason=excluded.reason,
        source=excluded.source""", (email.strip().lower(), bounce_type, reason, source))
    cx.commit()


def list_recent(cx, limit=200):
    cx.row_factory = sqlite3.Row
    return [dict(r) for r in cx.execute(
        "SELECT * FROM email_suppression ORDER BY created_at DESC LIMIT ?", (limit,))]
