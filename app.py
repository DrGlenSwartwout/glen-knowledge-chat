#!/usr/bin/env python3
"""
RAG Chat Server — Glen Swartwout Knowledge Base (Production)
"""

import os
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
PINECONE_INDEX   = "remedy-match-llc"
NAMESPACES       = ["mentors", ""]
TOP_K_PER_NS     = 8
MAX_CONTEXT_CHARS = 18000

SYSTEM_PROMPT = """You are Glen Swartwout's knowledge assistant — a deeply informed synthesis engine for his Clinical Theory of Everything, which integrates:
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
2. Write in an authoritative, integrative voice — as if explaining the connections Glen himself would draw.
3. When multiple mentors or concepts connect, explicitly show how they reinforce each other.
4. At the end of your response, list the source references used (name + field) so the reader can identify which profiles to explore.
5. Do NOT fabricate information not present in the snippets. If the snippets don't fully answer the question, say so clearly.
6. Keep responses focused and readable — prefer synthesis over exhaustive lists."""


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
            system=SYSTEM_PROMPT,
            messages=messages
        ).content[0].text
    except Exception as e:
        return jsonify({"error": f"Claude error: {e}"}), 500

    return jsonify({
        "answer": answer,
        "sources": sources_list,
        "query": query,
        "chunks_retrieved": len(all_matches)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
