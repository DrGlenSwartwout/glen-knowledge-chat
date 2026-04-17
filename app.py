#!/usr/bin/env python3
"""
RAG Chat Server — Glen Swartwout Knowledge Base (Production)
"""

import os
import re
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
import anthropic

# ── Load .env if present ──────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if not _env.exists():
    _env = Path.home() / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

# ── App setup ─────────────────────────────────────────────────────────────────
STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC))
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_INDEX    = "remedy-match-llc"
NAMESPACES        = ["mentors", "ingredients", "e4l-protocols", ""]
TOP_K_PER_NS      = 8
MAX_CONTEXT_CHARS = 18000
FEEDBACK_SUBMIT_URL = os.environ.get("FEEDBACK_SUBMIT_URL", "https://Truly.VIP/Results")
FEEDBACK_VIEW_URL   = os.environ.get("FEEDBACK_VIEW_URL",   "https://Truly.VIP/Feedback")

# ── Module-level API clients (initialized once at startup) ────────────────────
_oa  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
_pc  = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
_idx = _pc.Index(PINECONE_INDEX)
_cl  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Query log DB ──────────────────────────────────────────────────────────────
LOG_DB   = Path(__file__).parent / "chat_log.db"
_db_lock = threading.Lock()

def _init_log_db():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                query       TEXT    NOT NULL,
                level       TEXT,
                answer      TEXT,
                rating      INTEGER,
                rated_at    TEXT
            )
        """)
        cx.commit()

_init_log_db()


def log_query(query: str, level: str, answer: str) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cur = cx.execute(
            "INSERT INTO query_log (ts, query, level, answer) VALUES (?,?,?,?)",
            (ts, query, level, answer[:2000])
        )
        cx.commit()
        return cur.lastrowid


_SYSTEM_BASE = """You are Glen Swartwout's knowledge assistant — a deeply informed synthesis engine for his Clinical Theory of Everything, which integrates:
- BEV terrain medicine (Louis-Claude Vincent's 5 Phases of Health)
- Bioenergetic diagnostics (EAV/Voll, Vegatest, NES body-field, O-Ring/BDORT)
- Syntonic Optometry and Behavioral Optometry
- Orthomolecular and nutritional medicine
- Spirit Minerals / ORMUS (monatomic elements, Bose-Einstein Condensates)
- Electromagnetic medicine (PEMF, biophotons, living matrix, EMF sensitivity)
- Living Universe cosmology (Electric Universe, plasma cosmology)
- Consciousness science (IONS, HeartMath, morphic resonance)

Your task:
1. Synthesize the provided source snippets into a unified, coherent answer to the user's question.
2. When multiple mentors or concepts connect, explicitly show how they reinforce each other.
3. At the end of your response, list the source references used (name + field).
4. Do NOT fabricate information not present in the snippets. If the snippets don't fully answer the question, say so clearly.
5. Keep responses focused and readable — prefer synthesis over exhaustive lists."""

_LEVEL_INSTRUCTIONS = {
    "self-healing": """
LANGUAGE LEVEL: Self-Healing (general public)
Write in warm, empowering, accessible language. Avoid clinical jargon — use everyday words and intuitive analogies. Speak to the reader's own inner healing intelligence. Focus on what they can feel, experience, and do for themselves. Use "you" and "your body." Make the information feel practical and hopeful, not overwhelming.""",

    "health-care": """
LANGUAGE LEVEL: Health Care (practitioner)
Write at a clinical practitioner level — naturopathic, integrative, or functional medicine context. Use anatomical and physiological terminology, meridian names, clinical protocols, dosage ranges, and mechanism-of-action language. Assume the reader can interpret lab values, treatment timelines, and nutrient biochemistry. Be precise and protocol-oriented.""",

    "science": """
LANGUAGE LEVEL: Science (researcher / academic)
Write at a scientific research level. Use precise biochemical and biophysical terminology: receptor names, signaling cascades, molecular pathways, quantitative parameters. Cite mechanisms referenced in the source material. Be rigorous — distinguish between established findings and hypotheses. Appropriate for a peer-reviewed or academic audience.""",
}


def get_system_prompt(level: str) -> str:
    instruction = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS["self-healing"])
    return _SYSTEM_BASE + "\n" + instruction


SYSTEM_PROMPT = get_system_prompt("self-healing")


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed(text):
    return _oa.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding


def query_ns(vec, ns, k):
    try:
        return _idx.query(vector=vec, top_k=k, namespace=ns, include_metadata=True).matches
    except Exception:
        return []


def query_all_namespaces(vec):
    """Query all namespaces in parallel."""
    all_matches = []
    with ThreadPoolExecutor(max_workers=len(NAMESPACES)) as pool:
        futures = {pool.submit(query_ns, vec, ns, TOP_K_PER_NS): ns for ns in NAMESPACES}
        for future in as_completed(futures):
            all_matches.extend(future.result())
    return all_matches


def build_context(matches):
    seen, sources, parts, total = set(), {}, [], 0
    for m in sorted(matches, key=lambda x: -x.score):
        if m.id in seen:
            continue
        seen.add(m.id)
        meta = m.metadata or {}
        text = meta.get("text", "").strip()
        if not text or total + len(text) > MAX_CONTEXT_CHARS:
            continue
        name  = meta.get("name", "Unknown")
        field = meta.get("field", "")
        score = round(m.score, 3)
        if name not in sources:
            sources[name] = {"name": name, "field": field,
                             "source_file": meta.get("source", ""),
                             "score": score, "chunks": []}
        sources[name]["chunks"].append(meta.get("chunk_index", 0))
        sources[name]["score"] = max(sources[name]["score"], score)
        parts.append(f"[SOURCE: {name} | {field} | score {score}]\n{text}")
        total += len(text)
    return "\n\n---\n\n".join(parts), sorted(sources.values(), key=lambda x: -x["score"])


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = send_from_directory(STATIC, "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/embed")
def embed_page():
    resp = send_from_directory(STATIC, "embed.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/widget.js")
def widget_js():
    resp = send_from_directory(STATIC, "widget.js")
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC, filename)


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    data    = request.get_json() or {}
    query   = (data.get("query") or "").strip()
    history = data.get("history") or []
    level   = (data.get("level") or "self-healing").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        try:
            q_vec = embed(query)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"})
            return

        all_matches = query_all_namespaces(q_vec)

        if not all_matches:
            yield sse({"done": True, "answer": "No relevant content found.",
                       "sources": [], "chunks_retrieved": 0, "log_id": None})
            return

        context_str, sources_list = build_context(all_matches)

        messages = []
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content":
            f"USER QUESTION: {query}\n\nRETRIEVED SNIPPETS:\n{context_str}\n\n"
            "Synthesize these into a comprehensive answer. List sources used at the end."
        })

        full_answer = []
        try:
            with _cl.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=get_system_prompt(level),
                messages=messages
            ) as stream:
                for token in stream.text_stream:
                    full_answer.append(token)
                    yield sse({"token": token})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"})
            return

        answer = "".join(full_answer)
        log_id = log_query(query, level, answer)
        yield sse({"done": True, "log_id": log_id,
                   "sources": sources_list, "chunks_retrieved": len(all_matches)})

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/rate", methods=["POST", "OPTIONS"])
def rate():
    if request.method == "OPTIONS":
        return "", 200
    data   = request.get_json() or {}
    log_id = data.get("log_id")
    rating = data.get("rating")
    if not log_id or rating not in (1, 2, 3, 4, 5):
        return jsonify({"error": "Invalid"}), 400
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE query_log SET rating=?, rated_at=? WHERE id=?",
                   (rating, ts, log_id))
        cx.commit()
    return jsonify({"ok": True})


@app.route("/feedback-url")
def feedback_url():
    return jsonify({"submit": FEEDBACK_SUBMIT_URL, "view": FEEDBACK_VIEW_URL})


# ── GHL Integration ──────────────────────────────────────────────────────────
GHL_API_KEY      = os.environ.get("GHL_API_KEY", "")
GHL_BASE         = "https://rest.gohighlevel.com/v1"
GHL_PIPELINE_ID  = "A6LWJMBoIsOFBMeCa6NY"   # E4L Onboarding pipeline
GHL_STAGE_NEW    = "397c5fb2-1612-4b7a-aa14-f0dac42a7fda"  # E4L Account Invite
GHL_WORKFLOW_ID  = "0b02dd3e-b82a-4032-a575-f9269afbd3ac"  # E4L Onboarding Workflow

import urllib.request as _urllib_req

def _ghl_headers():
    return {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Content-Type":  "application/json",
    }

def _ghl_post(path, payload):
    req = _urllib_req.Request(
        f"{GHL_BASE}{path}",
        data=json.dumps(payload).encode(),
        headers=_ghl_headers(),
        method="POST"
    )
    try:
        with _urllib_req.urlopen(req, timeout=10) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)

def _ghl_put(path, payload):
    req = _urllib_req.Request(
        f"{GHL_BASE}{path}",
        data=json.dumps(payload).encode(),
        headers=_ghl_headers(),
        method="PUT"
    )
    try:
        with _urllib_req.urlopen(req, timeout=10) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)

def _ghl_get(path, params=None):
    url = f"{GHL_BASE}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = _urllib_req.Request(url, headers=_ghl_headers())
    try:
        with _urllib_req.urlopen(req, timeout=10) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


def ghl_upsert_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
    """Find or create a GHL contact. Returns (contact_id, created_bool, error)."""
    if not GHL_API_KEY:
        return None, False, "GHL_API_KEY not set"

    all_new_tags = set()
    if source_tag:
        all_new_tags.add(source_tag)
    if extra_tags:
        all_new_tags.update(extra_tags)

    # Try to find existing contact
    data, err = _ghl_get("/contacts/", {"email": email})
    if not err:
        contacts = data.get("contacts", [])
        if contacts:
            contact_id = contacts[0]["id"]
            if all_new_tags:
                existing_tags = set(contacts[0].get("tags", []))
                existing_tags.update(all_new_tags)
                _ghl_put(f"/contacts/{contact_id}", {"tags": list(existing_tags)})
            return contact_id, False, None

    # Create new contact
    payload = {"email": email, "firstName": first_name, "lastName": last_name}
    if phone:
        payload["phone"] = phone
    if all_new_tags:
        payload["tags"] = list(all_new_tags)

    data, err = _ghl_post("/contacts/", payload)
    if err:
        return None, False, err

    contact_id = data.get("contact", {}).get("id") or data.get("id")
    return contact_id, True, None


def ghl_add_to_pipeline(contact_id, name="", email=""):
    """Create an opportunity in the E4L Onboarding pipeline at stage 1."""
    if not contact_id:
        return None, "No contact_id"
    payload = {
        "pipelineId":      GHL_PIPELINE_ID,
        "pipelineStageId": GHL_STAGE_NEW,
        "contactId":       contact_id,
        "name":            name or email or contact_id,
        "status":          "open",
    }
    data, err = _ghl_post("/opportunities/", payload)
    if err:
        return None, err
    opp_id = data.get("opportunity", {}).get("id") or data.get("id")
    return opp_id, None


def ghl_enroll_workflow(contact_id):
    """Enroll a contact in the E4L Onboarding Workflow."""
    if not contact_id:
        return None, "No contact_id"
    data, err = _ghl_post(f"/contacts/{contact_id}/workflow/{GHL_WORKFLOW_ID}", {})
    return data, err


def ghl_onboard_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
    """Full onboarding: upsert contact → pipeline → workflow. Returns result dict."""
    result = {"email": email, "source_tag": source_tag}

    contact_id, created, err = ghl_upsert_contact(email, first_name, last_name, phone, source_tag, extra_tags)
    result["contact_id"] = contact_id
    result["contact_created"] = created
    if err:
        result["contact_error"] = err
        return result

    opp_id, err = ghl_add_to_pipeline(contact_id, f"{first_name} {last_name}".strip(), email)
    result["opportunity_id"] = opp_id
    if err:
        result["pipeline_error"] = err

    _, err = ghl_enroll_workflow(contact_id)
    if err:
        result["workflow_error"] = err
    else:
        result["workflow_enrolled"] = True

    return result


# ── Inbound lead DB table ─────────────────────────────────────────────────────
def _init_leads_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS inbound_leads (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                email       TEXT,
                first_name  TEXT,
                last_name   TEXT,
                phone       TEXT,
                raw_json    TEXT,
                ghl_contact_id TEXT,
                ghl_opp_id     TEXT,
                ghl_error      TEXT
            )
        """)
        cx.commit()

_init_leads_table()


def _log_inbound_lead(source, email, first_name, last_name, phone, raw, ghl_result):
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO inbound_leads
              (received_at, source, email, first_name, last_name, phone, raw_json,
               ghl_contact_id, ghl_opp_id, ghl_error)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (ts, source, email, first_name, last_name, phone, raw,
              ghl_result.get("contact_id"),
              ghl_result.get("opportunity_id"),
              ghl_result.get("contact_error") or ghl_result.get("pipeline_error")))
        cx.commit()


# ── Practice Better Webhook ───────────────────────────────────────────────────
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

def _init_pb_events_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS pb_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT    NOT NULL,
                event_type  TEXT,
                pb_email    TEXT,
                pb_name     TEXT,
                raw_json    TEXT,
                synced      INTEGER DEFAULT 0
            )
        """)
        cx.commit()

_init_pb_events_table()


@app.route("/webhook/practice-better", methods=["POST"])
def pb_webhook():
    if WEBHOOK_SECRET:
        incoming = request.headers.get("X-Webhook-Secret", "")
        if incoming != WEBHOOK_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    data       = request.get_json(force=True) or {}
    event_type = data.get("event_type", "unknown")
    pb_email   = data.get("email", "")
    pb_name    = data.get("name", data.get("full_name", ""))
    raw        = json.dumps(data)
    ts         = datetime.now(timezone.utc).isoformat()

    # Log to pb_events table (existing)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO pb_events (received_at, event_type, pb_email, pb_name, raw_json)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, event_type, pb_email, pb_name, raw))
        cx.commit()

    # For new member signups → push to GHL E4L pipeline
    if event_type in ("client.created", "client.signup", "member.created", "unknown") and pb_email:
        parts = pb_name.split(" ", 1) if pb_name else ["", ""]
        first = parts[0]
        last  = parts[1] if len(parts) > 1 else ""
        ghl_result = ghl_onboard_contact(
            email=pb_email, first_name=first, last_name=last,
            source_tag="source:pb-signup"
        )
        _log_inbound_lead("practice-better", pb_email, first, last, "", raw, ghl_result)

    return jsonify({"ok": True, "event": event_type}), 200


# Maps question text fragments → GHL tag prefix
_SCOREAPP_Q_PREFIX = {
    "which system is most in need": "system",
    "what's the main challenge":    "phase",
    "what is the main challenge":   "phase",
    "what's your top concern":      "concern",
    "what is your top concern":     "concern",
    "how well you heal when you try": "regulation",
}

def _scoreapp_answer_tags(quiz_questions):
    """Convert ScoreApp quiz_questions list → list of GHL tags like ['system:immune', ...]."""
    tags = []
    for q in quiz_questions:
        q_text  = (q.get("question") or "").lower().strip().rstrip("?:")
        answers = q.get("answers") or []
        prefix  = next((v for k, v in _SCOREAPP_Q_PREFIX.items() if k in q_text), None)
        if not prefix:
            continue
        for a in answers:
            val = (a.get("answer") or "").strip()
            if val:
                slug = re.sub(r"[^a-z0-9]+", "-", val.lower()).strip("-")
                tags.append(f"{prefix}:{slug}")
    return tags


@app.route("/webhook/scoreapp", methods=["POST"])
def scoreapp_webhook():
    """ScoreApp QUIZ_FINISHED → GHL E4L pipeline with per-answer tags."""
    payload = request.get_json(force=True) or {}
    raw     = json.dumps(payload)

    # ScoreApp wraps everything in {"event_name": "...", "data": {...}}
    data  = payload.get("data", payload)
    email = (data.get("email") or "").strip()
    first = (data.get("first_name") or "").strip()
    last  = (data.get("last_name")  or "").strip()
    phone = (data.get("phone")      or "").strip()
    score = (data.get("total_score") or {}).get("percent") or data.get("score") or ""

    # Only process QUIZ_FINISHED (ignore QUIZ_STARTED)
    event = payload.get("event_name", "")
    if event == "QUIZ_STARTED":
        return jsonify({"ok": True, "skipped": "quiz_started"}), 200

    if not email:
        return jsonify({"error": "No email in payload"}), 400

    # Build answer tags from quiz_questions
    quiz_questions = data.get("quiz_questions") or []
    answer_tags    = _scoreapp_answer_tags(quiz_questions)

    ghl_result = ghl_onboard_contact(
        email=email, first_name=first, last_name=last, phone=phone,
        source_tag="source:scoreapp",
        extra_tags=answer_tags,
    )

    # Store quiz answers + score as a note
    if ghl_result.get("contact_id"):
        note_lines = [f"ScoreApp quiz — {event or 'QUIZ_FINISHED'}"]
        if score:
            note_lines.append(f"Score: {score}%")
        for q in quiz_questions:
            q_text = q.get("question", "")
            ans    = ", ".join(a.get("answer", "") for a in (q.get("answers") or []))
            if q_text and ans:
                note_lines.append(f"{q_text}: {ans}")
        _ghl_post(f"/contacts/{ghl_result['contact_id']}/notes",
                  {"body": "\n".join(note_lines)})

    _log_inbound_lead("scoreapp", email, first, last, phone, raw, ghl_result)
    return jsonify({"ok": True, "tags": answer_tags, "ghl": ghl_result}), 200


@app.route("/webhook/groovekart", methods=["POST"])
def groovekart_webhook():
    """GrooveKart order → GHL E4L pipeline with purchase tag."""
    data  = request.get_json(force=True) or {}

    # Support multiple GrooveKart webhook payload formats
    customer = data.get("customer") or data.get("billing_address") or data
    email    = (customer.get("email") or data.get("email") or "").strip()
    first    = (customer.get("first_name") or customer.get("firstName") or "").strip()
    last     = (customer.get("last_name")  or customer.get("lastName")  or "").strip()
    phone    = (customer.get("phone")      or customer.get("telephone") or "").strip()
    product  = data.get("line_items", [{}])[0].get("name", "") if data.get("line_items") else ""
    raw      = json.dumps(data)

    if not email:
        return jsonify({"error": "No email in payload"}), 400

    ghl_result = ghl_onboard_contact(
        email=email, first_name=first, last_name=last, phone=phone,
        source_tag="source:gk-purchase"
    )
    # Add product note
    if ghl_result.get("contact_id") and product:
        _ghl_post(f"/contacts/{ghl_result['contact_id']}/notes",
                  {"body": f"GrooveKart purchase: {product}"})

    _log_inbound_lead("groovekart", email, first, last, phone, raw, ghl_result)
    return jsonify({"ok": True, "ghl": ghl_result}), 200


@app.route("/inbound-leads", methods=["GET"])
def get_inbound_leads():
    """Review recent inbound leads and their GHL sync status."""
    if WEBHOOK_SECRET:
        incoming = request.headers.get("X-Webhook-Secret", "")
        if incoming != WEBHOOK_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    limit = int(request.args.get("limit", 100))
    source = request.args.get("source", "")

    with sqlite3.connect(LOG_DB) as cx:
        if source:
            rows = cx.execute("""
                SELECT received_at, source, email, first_name, last_name,
                       ghl_contact_id, ghl_opp_id, ghl_error
                FROM inbound_leads WHERE source = ? ORDER BY id DESC LIMIT ?
            """, (source, limit)).fetchall()
        else:
            rows = cx.execute("""
                SELECT received_at, source, email, first_name, last_name,
                       ghl_contact_id, ghl_opp_id, ghl_error
                FROM inbound_leads ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()

    leads = [{"received_at": r[0], "source": r[1], "email": r[2],
              "name": f"{r[3]} {r[4]}".strip(),
              "ghl_contact_id": r[5], "ghl_opp_id": r[6], "error": r[7]}
             for r in rows]
    return jsonify({"leads": leads, "count": len(leads)})


@app.route("/pb-events", methods=["GET"])
def get_pb_events():
    if WEBHOOK_SECRET:
        incoming = request.headers.get("X-Webhook-Secret", "")
        if incoming != WEBHOOK_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    limit = int(request.args.get("limit", 500))

    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, received_at, event_type, pb_email, pb_name, raw_json
            FROM pb_events WHERE synced = 0
            ORDER BY id ASC LIMIT ?
        """, (limit,)).fetchall()

    events = [
        {"id": r[0], "received_at": r[1], "event_type": r[2],
         "email": r[3], "name": r[4], "data": json.loads(r[5])}
        for r in rows
    ]
    return jsonify({"events": events, "count": len(events)})


@app.route("/pb-events/mark-synced", methods=["POST"])
def mark_pb_synced():
    if WEBHOOK_SECRET:
        incoming = request.headers.get("X-Webhook-Secret", "")
        if incoming != WEBHOOK_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    ids = request.get_json(force=True).get("ids", [])
    if not ids:
        return jsonify({"ok": True, "marked": 0})

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.executemany("UPDATE pb_events SET synced=1 WHERE id=?", [(i,) for i in ids])
        cx.commit()

    return jsonify({"ok": True, "marked": len(ids)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
