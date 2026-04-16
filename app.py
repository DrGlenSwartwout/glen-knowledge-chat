#!/usr/bin/env python3
"""
RAG Chat Server — Glen Swartwout Knowledge Base (Production)
"""

import os
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
    return send_from_directory(STATIC, "index.html")


@app.route("/embed")
def embed_page():
    return send_from_directory(STATIC, "embed.html")


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

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO pb_events (received_at, event_type, pb_email, pb_name, raw_json)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, event_type, pb_email, pb_name, raw))
        cx.commit()

    return jsonify({"ok": True, "event": event_type}), 200


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
