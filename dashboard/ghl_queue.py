"""GHL write-queue: GHL writes are blocked from Render by the Cloudflare WAF, so
CRM write actions enqueue here (a local DB insert that works from the server) and
a local Mac drain script (sync-ghl-writes.py) pushes them to GHL via curl. The
actions are audited + governed by the dispatch spine like any other."""
import json
from datetime import datetime, timezone

from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA

OP_TYPES = ("tag_add", "tag_remove", "note", "opportunity", "workflow")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_ghl_queue_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ghl_write_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            op TEXT NOT NULL,
            email TEXT,
            payload_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            actor TEXT,
            processed_at TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_ghl_queue_status ON ghl_write_queue(status)")
    cx.commit()


def enqueue(cx, *, op, email, payload, actor=""):
    if op not in OP_TYPES:
        raise ValueError(f"unknown op: {op}")
    if not (email or "").strip():
        raise ValueError("email required")
    cur = cx.execute(
        "INSERT INTO ghl_write_queue (created_at, op, email, payload_json, status, actor) "
        "VALUES (?,?,?,?, 'pending', ?)",
        (_now(), op, email.strip(), json.dumps(payload or {}), actor or ""))
    cx.commit()
    return cur.lastrowid


def list_pending(cx, limit=100):
    cur = cx.execute(
        "SELECT * FROM ghl_write_queue WHERE status='pending' ORDER BY id ASC LIMIT ?",
        (limit,))
    return [dict(r) for r in cur.fetchall()]


def mark_result(cx, qid, status, result=""):
    cx.execute("UPDATE ghl_write_queue SET status=?, result=?, processed_at=? WHERE id=?",
               (status, str(result)[:500], _now(), qid))
    cx.commit()
    return True


def _enqueue_action(op, label):
    def _exec(params, ctx):
        cx = (ctx or {}).get("cx") or (params or {}).get("cx")
        if cx is None:
            raise ValueError("no db connection")
        init_ghl_queue_table(cx)
        email = (params.get("email") or "").strip()
        if not email:
            raise ValueError("email required")
        payload = {k: v for k, v in (params or {}).items()
                   if k not in ("email", "cx", "confirmed")}
        actor = (ctx or {}).get("actor")
        qid = enqueue(cx, op=op, email=email, payload=payload,
                      actor=getattr(actor, "name", "") if actor else "")
        return {"queue_id": qid, "op": op, "email": email,
                "message": f"{label} for {email} queued for GHL sync (#{qid})."}
    return _exec


action(key="crm.add_tag", module="crm", title="Add GHL tag",
       description="Tag a contact in GHL (queued; pushes on the next sync).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_enqueue_action("tag_add", "Tag"))
action(key="crm.log_outreach", module="crm", title="Log outreach note",
       description="Add a note to a GHL contact (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS, VA))(_enqueue_action("note", "Note"))
action(key="crm.create_opportunity", module="crm", title="Create opportunity",
       description="Create a pipeline opportunity in GHL (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS))(_enqueue_action("opportunity", "Opportunity"))
action(key="crm.enroll_workflow", module="crm", title="Enroll in workflow",
       description="Enroll a contact in the onboarding workflow (queued).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS))(_enqueue_action("workflow", "Workflow enrollment"))
