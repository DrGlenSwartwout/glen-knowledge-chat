"""Studio-credit free month: claim store + approve/reject with a one-per-year
guard. Claims arrive from the console (manual entry) or the public self-serve form
(upsert_self_serve_claim, source='self_serve'); both flow to the same console
approval queue. The grant+notify side effect is injected at approve time so this
module stays Flask-free and unit-testable."""
import sqlite3
import uuid
from datetime import datetime, timedelta


def _now():
    return datetime.utcnow().isoformat() + "Z"


def migrate(cx) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS studio_credit_claims (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            invoice_ref   TEXT NOT NULL DEFAULT '',
            proof_note    TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT NOT NULL,
            created_by    TEXT NOT NULL DEFAULT '',
            decided_at    TEXT,
            decided_by    TEXT,
            decision_note TEXT NOT NULL DEFAULT '',
            membership_id TEXT,
            source        TEXT NOT NULL DEFAULT 'console'
        )
    """)
    cx.commit()


def _row(r):
    return dict(r) if r is not None else None


def add_claim(cx, *, email, invoice_ref="", proof_note="", source="console", created_by=""):
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    cid = str(uuid.uuid4())
    cx.execute(
        "INSERT INTO studio_credit_claims "
        "(id, email, invoice_ref, proof_note, status, created_at, created_by, source) "
        "VALUES (?,?,?,?, 'pending', ?, ?, ?)",
        (cid, email, invoice_ref or "", proof_note or "", _now(), created_by or "", source or "console"))
    cx.commit()
    return get(cx, cid)


def get(cx, claim_id):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return _row(cur.execute(
        "SELECT * FROM studio_credit_claims WHERE id=?", (claim_id,)).fetchone())


def list_claims(cx, status=None):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    if status:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims WHERE status=? ORDER BY created_at DESC",
            (status,)).fetchall()
    else:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def studio_credit_granted_within_year(cx, email):
    """Most-recent studio_credit membership for email granted in the last 365 days,
    or None. Reads the existing memberships table (same DB)."""
    email = (email or "").strip().lower()
    cutoff = (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z"
    try:
        row = cx.execute(
            "SELECT granted_at, expires_at FROM memberships "
            "WHERE email=? AND source='studio_credit' AND granted_at > ? "
            "ORDER BY granted_at DESC LIMIT 1",
            (email, cutoff)).fetchone()
    except sqlite3.OperationalError:
        return None   # memberships table absent (shouldn't happen in prod)
    if not row:
        return None
    return {"granted_at": row[0], "until": row[1]}


def approve_claim(cx, claim_id, *, decided_by, grant_fn, force=False):
    claim = get(cx, claim_id)
    if claim is None:
        raise ValueError("claim not found")
    if claim["status"] == "approved":
        return {"ok": True, "already": True, "membership_id": claim["membership_id"]}
    if claim["status"] == "rejected":
        raise ValueError("claim already rejected")
    email = claim["email"]
    if not force:
        prior = studio_credit_granted_within_year(cx, email)
        if prior is not None:
            return {"ok": False, "warning": "granted_within_year",
                    "granted_at": prior["granted_at"], "until": prior["until"]}
    granted = grant_fn(cx, email, 30)
    cx.execute(
        "UPDATE studio_credit_claims SET status='approved', decided_at=?, decided_by=?, "
        "membership_id=? WHERE id=?",
        (_now(), decided_by or "", granted["membership_id"], claim_id))
    cx.commit()
    # Only now. grant_fn's email carries a magic link, so it must not go out
    # until the token and the approval are durable. A grant_fn that does its own
    # notifying (or none) simply returns no 'notify'.
    notify = granted.get("notify")
    if callable(notify):
        notify()
    return {"ok": True, "membership_id": granted["membership_id"],
            "magic_link_url": granted.get("magic_link_url", "")}


def reject_claim(cx, claim_id, *, decided_by, reason=""):
    claim = get(cx, claim_id)
    if claim is None:
        raise ValueError("claim not found")
    if claim["status"] == "approved":
        raise ValueError("cannot reject an approved claim")
    cx.execute(
        "UPDATE studio_credit_claims SET status='rejected', decided_at=?, decided_by=?, "
        "decision_note=? WHERE id=?",
        (_now(), decided_by or "", reason or "", claim_id))
    cx.commit()
    return {"ok": True}


def upsert_self_serve_claim(cx, *, email, invoice_ref="", proof_note=""):
    """Public self-serve submission. Dedupe: if a pending self_serve claim already
    exists for this email, update it in place (refresh invoice_ref/proof_note and
    bump created_at); otherwise create one. Returns (claim, is_new). Pending-only:
    an approved/rejected email gets a fresh pending claim."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    existing = cur.execute(
        "SELECT id FROM studio_credit_claims "
        "WHERE email=? AND status='pending' AND source='self_serve' "
        "ORDER BY created_at DESC LIMIT 1",
        (email,)).fetchone()
    if existing is not None:
        cx.execute(
            "UPDATE studio_credit_claims SET invoice_ref=?, proof_note=?, created_at=? "
            "WHERE id=?",
            (invoice_ref or "", proof_note or "", _now(), existing["id"]))
        cx.commit()
        return get(cx, existing["id"]), False
    claim = add_claim(cx, email=email, invoice_ref=invoice_ref, proof_note=proof_note,
                      source="self_serve", created_by="self_serve")
    return claim, True
