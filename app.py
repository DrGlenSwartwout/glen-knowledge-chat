#!/usr/bin/env python3
"""
RAG Chat Server — Glen Swartwout Knowledge Base (Production)
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
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
CORS(app)  # Allow cross-origin requests (needed for embedded widget)

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_INDEX    = "remedy-match-llc"
NAMESPACES        = ["mentors", "ingredients", "e4l-protocols", ""]
TOP_K_PER_NS      = 8
MAX_CONTEXT_CHARS = 18000
FEEDBACK_SUBMIT_URL = os.environ.get("FEEDBACK_SUBMIT_URL", "https://Truly.VIP/Results")
FEEDBACK_VIEW_URL   = os.environ.get("FEEDBACK_VIEW_URL",   "https://Truly.VIP/Feedback")

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
    """Insert a query row and return its id."""
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


# Default for backward compatibility
SYSTEM_PROMPT = get_system_prompt("self-healing")


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed(text, client):
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding


def query_ns(vec, idx, ns, k):
    try:
        return idx.query(vector=vec, top_k=k, namespace=ns, include_metadata=True).matches
    except Exception:
        return []


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

    try:
        oa  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        pc  = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        idx = pc.Index(PINECONE_INDEX)
        cl  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        return jsonify({"error": f"Init failed: {e}"}), 500

    try:
        q_vec = embed(query, oa)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {e}"}), 500

    all_matches = []
    for ns in NAMESPACES:
        all_matches.extend(query_ns(q_vec, idx, ns, TOP_K_PER_NS))

    if not all_matches:
        return jsonify({"answer": "No relevant content found.", "sources": [], "query": query})

    context_str, sources_list = build_context(all_matches)

    messages = []
    for turn in history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content":
        f"USER QUESTION: {query}\n\nRETRIEVED SNIPPETS:\n{context_str}\n\n"
        "Synthesize these into a comprehensive answer. List sources used at the end."
    })

    try:
        answer = cl.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=get_system_prompt(level),
            messages=messages
        ).content[0].text
    except Exception as e:
        return jsonify({"error": f"Claude error: {e}"}), 500

    log_id = log_query(query, level, answer)

    return jsonify({
        "answer": answer,
        "sources": sources_list,
        "query": query,
        "chunks_retrieved": len(all_matches),
        "log_id": log_id
    })


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
