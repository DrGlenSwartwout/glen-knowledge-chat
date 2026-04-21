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
NAMESPACES        = ["mentors", "ingredients", "e4l-protocols", "consultations", "training", "business", ""]
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
LOG_DB   = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "chat_log.db"
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


@app.route("/debug-ghl")
def debug_ghl():
    key = GHL_API_KEY
    result = {
        "key_length": len(key),
        "key_first10": key[:10],
        "key_last10": key[-10:],
        "has_key": bool(key),
        "curl_available": bool(_CURL),
    }
    # Live test via curl (bypasses Cloudflare JA3 blocking)
    try:
        r = _subprocess.run(
            [_CURL, "-s", "--max-time", "10",
             "https://rest.gohighlevel.com/v1/contacts/?limit=1",
             "-H", f"Authorization: Bearer {key}"],
            capture_output=True, text=True, timeout=15
        )
        body = json.loads(r.stdout)
        result["ghl_test"] = "ok"
        result["ghl_contacts_total"] = body.get("meta", {}).get("total", "unknown")
    except Exception as e:
        result["ghl_test"] = f"Error: {str(e)[:100]}"
    return jsonify(result)


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

import subprocess as _subprocess
import shutil as _shutil

_CURL = _shutil.which("curl")  # Use curl to bypass Cloudflare JA3 fingerprint blocking

def _curl_args(method="GET", data=None):
    args = [_CURL, "-s", "--max-time", "10"]
    if method != "GET":
        args += ["-X", method]
    h = _ghl_headers()
    for k, v in h.items():
        args += ["-H", f"{k}: {v}"]
    if data is not None:
        args += ["-d", json.dumps(data)]
    return args

def _ghl_post(path, payload):
    if not _CURL:
        return None, "curl not available"
    try:
        r = _subprocess.run(_curl_args("POST", payload) + [f"{GHL_BASE}{path}"],
                            capture_output=True, text=True, timeout=15)
        return (json.loads(r.stdout) if r.stdout.strip() else {}), None
    except Exception as e:
        return None, str(e)

def _ghl_put(path, payload):
    if not _CURL:
        return None, "curl not available"
    try:
        r = _subprocess.run(_curl_args("PUT", payload) + [f"{GHL_BASE}{path}"],
                            capture_output=True, text=True, timeout=15)
        return (json.loads(r.stdout) if r.stdout.strip() else {}), None
    except Exception as e:
        return None, str(e)

def _ghl_get(path, params=None):
    if not _CURL:
        return None, "curl not available"
    url = f"{GHL_BASE}{path}"
    if params:
        import urllib.parse as _urlparse
        url += "?" + "&".join(f"{k}={_urlparse.quote(str(v))}" for k, v in params.items())
    try:
        r = _subprocess.run(_curl_args("GET") + [url],
                            capture_output=True, text=True, timeout=15)
        return (json.loads(r.stdout) if r.stdout.strip() else {}), None
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

    # Try to find existing contact (GHL v1 email filter is fuzzy — check exact match)
    data, err = _ghl_get("/contacts/", {"email": email, "limit": "20"})
    if not err:
        contacts = data.get("contacts", [])
        match = next((c for c in contacts if (c.get("email") or "").lower() == email.lower()), None)
        if match:
            contact_id = match["id"]
            if all_new_tags:
                existing_tags = set(match.get("tags", []))
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
        "stageId":   GHL_STAGE_NEW,
        "contactId": contact_id,
        "title":     name or email or contact_id,
        "status":    "open",
    }
    data, err = _ghl_post(f"/pipelines/{GHL_PIPELINE_ID}/opportunities", payload)
    if err:
        return None, err
    # GHL returns error if contact already has an opportunity in this pipeline — treat as OK
    if data.get("contactId", {}).get("rule") == "invalid":
        return "already_exists", None
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


# ── Referral tracking ─────────────────────────────────────────────────────────
def _init_referral_tables():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS affiliate_signups (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT NOT NULL,
                name         TEXT NOT NULL,
                email        TEXT NOT NULL UNIQUE,
                organization TEXT DEFAULT '',
                website      TEXT DEFAULT '',
                promo_method TEXT DEFAULT '',
                slug         TEXT NOT NULL UNIQUE,
                token        TEXT NOT NULL UNIQUE,
                status       TEXT DEFAULT 'approved',
                notes        TEXT DEFAULT ''
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS referral_sources (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                name        TEXT NOT NULL,
                slug        TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                utm_source  TEXT NOT NULL,
                utm_medium  TEXT DEFAULT 'referral',
                utm_campaign TEXT DEFAULT '',
                active      INTEGER DEFAULT 1
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS referral_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at  TEXT NOT NULL,
                lead_id      INTEGER,
                email        TEXT,
                first_name   TEXT,
                last_name    TEXT,
                utm_source   TEXT DEFAULT '',
                utm_medium   TEXT DEFAULT '',
                utm_campaign TEXT DEFAULT '',
                utm_content  TEXT DEFAULT '',
                utm_term     TEXT DEFAULT '',
                quiz_score   TEXT DEFAULT '',
                raw_json     TEXT DEFAULT ''
            )
        """)
        # Seed AllHeal as first referral source if not exists
        existing = cx.execute("SELECT id FROM referral_sources WHERE slug='allheal'").fetchone()
        if not existing:
            ts = datetime.now(timezone.utc).isoformat()
            cx.execute("""
                INSERT INTO referral_sources (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, "AllHeal Nonprofit", "allheal",
                  "AllHeal nonprofit referrals to Accelerate Self-Healing quiz",
                  "allheal", "referral", "scoreapp-quiz"))
        cx.commit()

_init_referral_tables()


@app.route("/api/referral-sources", methods=["GET"])
def get_referral_sources():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, created_at, name, slug, description, utm_source, utm_medium, utm_campaign, active
            FROM referral_sources ORDER BY name ASC
        """).fetchall()
    cols = ["id","created_at","name","slug","description","utm_source","utm_medium","utm_campaign","active"]
    return jsonify({"sources": [dict(zip(cols, r)) for r in rows]})


@app.route("/api/referral-sources", methods=["POST"])
def post_referral_source():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True) or {}
    name     = (data.get("name") or "").strip()
    slug     = re.sub(r"[^a-z0-9]+", "-", (data.get("slug") or name).lower()).strip("-")
    desc     = (data.get("description") or "").strip()
    utm_src  = (data.get("utm_source") or slug).strip()
    utm_med  = (data.get("utm_medium") or "referral").strip()
    utm_camp = (data.get("utm_campaign") or "scoreapp-quiz").strip()
    if not name or not slug:
        return jsonify({"error": "name required"}), 400
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("""
                INSERT INTO referral_sources (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, name, slug, desc, utm_src, utm_med, utm_camp))
            cx.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": f"slug '{slug}' already exists"}), 409
    return jsonify({"ok": True, "slug": slug,
                    "tracking_url": f"https://healing.scoreapp.com?utm_source={utm_src}&utm_medium={utm_med}&utm_campaign={utm_camp}"}), 201


@app.route("/api/referrals", methods=["GET"])
def get_referrals():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    # Stats per utm_source
    with sqlite3.connect(LOG_DB) as cx:
        stats = cx.execute("""
            SELECT utm_source,
                   COUNT(*) as total,
                   MAX(received_at) as last_lead
            FROM referral_events
            GROUP BY utm_source
            ORDER BY total DESC
        """).fetchall()
        recent = cx.execute("""
            SELECT received_at, utm_source, utm_medium, utm_campaign,
                   first_name, last_name, email, quiz_score
            FROM referral_events
            ORDER BY received_at DESC LIMIT 50
        """).fetchall()
    stat_list = [{"utm_source": r[0], "total": r[1], "last_lead": r[2]} for r in stats]
    recent_list = [{"received_at": r[0], "utm_source": r[1], "utm_medium": r[2],
                    "utm_campaign": r[3], "name": f"{r[4] or ''} {r[5] or ''}".strip(),
                    "email": r[6], "quiz_score": r[7]} for r in recent]
    return jsonify({"stats": stat_list, "recent": recent_list})


QUIZ_URL = "https://healing.scoreapp.com"


@app.route("/affiliate")
def affiliate_page():
    resp = send_from_directory(STATIC, "affiliate.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/affiliate/portal")
def affiliate_portal_page():
    resp = send_from_directory(STATIC, "affiliate-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/affiliate/apply", methods=["POST", "OPTIONS"])
def affiliate_apply():
    if request.method == "OPTIONS":
        return "", 200
    data  = request.get_json(force=True) or {}
    name  = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    org   = (data.get("organization") or "").strip()
    site  = (data.get("website") or "").strip()
    promo = (data.get("promo_method") or "").strip()

    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400

    # Generate slug from org or name
    base = re.sub(r"[^a-z0-9]+", "-", (org or name).lower()).strip("-")[:30]
    slug = base
    # Generate secure token
    import secrets as _secrets
    token = _secrets.token_urlsafe(24)

    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            # Ensure slug uniqueness
            existing = cx.execute("SELECT id FROM affiliate_signups WHERE slug=?", (slug,)).fetchone()
            if existing:
                slug = f"{base}-{token[:6]}"
            cx.execute("""
                INSERT INTO affiliate_signups
                  (created_at, name, email, organization, website, promo_method, slug, token, status)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (ts, name, email, org, site, promo, slug, token, "approved"))
            # Also create referral_sources entry so tracking is live immediately
            cx.execute("""
                INSERT OR IGNORE INTO referral_sources
                  (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, org or name, slug,
                  f"Affiliate: {name}" + (f" ({org})" if org else ""),
                  slug, "affiliate", "scoreapp-quiz"))
            cx.commit()
    except sqlite3.IntegrityError:
        # Email already exists — return their existing portal link
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute("SELECT token, slug FROM affiliate_signups WHERE email=?", (email,)).fetchone()
        if row:
            token, slug = row
        else:
            return jsonify({"error": "Email already registered"}), 409

    tracking_url = (
        f"{QUIZ_URL}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    )
    portal_url = f"/affiliate/portal?token={token}"
    return jsonify({
        "ok": True,
        "portal_url": portal_url,
        "tracking_url": tracking_url,
        "slug": slug,
    }), 201


@app.route("/affiliate/portal-data", methods=["GET"])
def affiliate_portal_data():
    token = request.args.get("token", "").strip()
    if not token:
        return jsonify({"error": "token required"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("""
            SELECT id, name, email, organization, slug, status, created_at
            FROM affiliate_signups WHERE token=?
        """, (token,)).fetchone()
    if not row:
        return jsonify({"error": "Invalid token"}), 404
    aff_id, name, email, org, slug, status, created_at = row
    if status != "approved":
        return jsonify({"error": "Application pending review"}), 403

    tracking_url = f"{QUIZ_URL}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"

    with sqlite3.connect(LOG_DB) as cx:
        stats = cx.execute("""
            SELECT COUNT(*) as total, MAX(received_at) as last_lead
            FROM referral_events WHERE utm_source=?
        """, (slug,)).fetchone()
        recent = cx.execute("""
            SELECT received_at, first_name, last_name, quiz_score
            FROM referral_events WHERE utm_source=?
            ORDER BY received_at DESC LIMIT 10
        """, (slug,)).fetchall()

    return jsonify({
        "name": name,
        "organization": org,
        "slug": slug,
        "tracking_url": tracking_url,
        "total_leads": stats[0] if stats else 0,
        "last_lead": stats[1] if stats else None,
        "recent": [{"received_at": r[0],
                    "name": f"{r[1] or ''} {r[2] or ''}".strip(),
                    "score": r[3]} for r in recent],
        "member_since": created_at,
    })


@app.route("/api/affiliates", methods=["GET"])
def get_affiliates():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT a.id, a.created_at, a.name, a.email, a.organization,
                   a.website, a.promo_method, a.slug, a.status,
                   COUNT(r.id) as lead_count
            FROM affiliate_signups a
            LEFT JOIN referral_events r ON r.utm_source = a.slug
            GROUP BY a.id
            ORDER BY a.created_at DESC
        """).fetchall()
    cols = ["id","created_at","name","email","organization","website","promo_method","slug","status","lead_count"]
    return jsonify({"affiliates": [dict(zip(cols, r)) for r in rows]})


@app.route("/api/affiliates/<int:aff_id>", methods=["PATCH"])
def patch_affiliate(aff_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    data   = request.get_json(force=True) or {}
    status = data.get("status", "")
    if status not in ("approved", "rejected", "suspended"):
        return jsonify({"error": "status must be approved, rejected, or suspended"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE affiliate_signups SET status=? WHERE id=?", (status, aff_id))
        if status == "approved":
            row = cx.execute("SELECT name, organization, slug, created_at FROM affiliate_signups WHERE id=?", (aff_id,)).fetchone()
            if row:
                name, org, slug, ts = row
                cx.execute("""
                    INSERT OR IGNORE INTO referral_sources
                      (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                    VALUES (?,?,?,?,?,?,?)
                """, (ts, org or name, slug,
                      f"Affiliate: {name}" + (f" ({org})" if org else ""),
                      slug, "affiliate", "scoreapp-quiz"))
        cx.commit()
    return jsonify({"ok": True, "status": status})


def _log_referral_event(lead_id, email, first_name, last_name, utm_source, utm_medium,
                        utm_campaign, utm_content, utm_term, quiz_score, raw):
    if not utm_source:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO referral_events
              (received_at, lead_id, email, first_name, last_name,
               utm_source, utm_medium, utm_campaign, utm_content, utm_term, quiz_score, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, lead_id, email, first_name, last_name,
              utm_source, utm_medium, utm_campaign, utm_content, utm_term, quiz_score, raw))
        cx.commit()


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


# ── GHL sync endpoint (for local-machine sync when Cloudflare blocks Render→GHL) ──
@app.route("/leads/pending-ghl", methods=["GET"])
def leads_pending_ghl():
    secret = request.headers.get("X-Webhook-Secret", "")
    ws = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("""
            SELECT id, received_at, source, email, first_name, last_name, phone, raw_json, ghl_error
            FROM inbound_leads
            WHERE ghl_contact_id IS NULL AND email IS NOT NULL AND email != ''
            ORDER BY received_at ASC LIMIT 100
        """).fetchall()
    return jsonify({"leads": [dict(r) for r in rows], "count": len(rows)})


@app.route("/leads/mark-ghl-synced", methods=["POST"])
def leads_mark_ghl_synced():
    secret = request.headers.get("X-Webhook-Secret", "")
    ws = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    lead_id = data.get("id")
    contact_id = data.get("contact_id")
    if not lead_id or not contact_id:
        return jsonify({"error": "id and contact_id required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE inbound_leads SET ghl_contact_id=?, ghl_error=NULL WHERE id=?",
                   (contact_id, lead_id))
        cx.commit()
    return jsonify({"ok": True, "id": lead_id, "contact_id": contact_id})


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

    lead_id = None
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT id FROM inbound_leads WHERE email=? ORDER BY id DESC LIMIT 1", (email,)).fetchone()
        if row:
            lead_id = row[0]
    _log_inbound_lead("scoreapp", email, first, last, phone, raw, ghl_result)

    # Referral tracking — extract UTM params from ScoreApp payload
    utm_source   = (data.get("utm_source")   or payload.get("utm_source")   or "").strip()
    utm_medium   = (data.get("utm_medium")   or payload.get("utm_medium")   or "").strip()
    utm_campaign = (data.get("utm_campaign") or payload.get("utm_campaign") or "").strip()
    utm_content  = (data.get("utm_content")  or payload.get("utm_content")  or "").strip()
    utm_term     = (data.get("utm_term")     or payload.get("utm_term")     or "").strip()
    if utm_source:
        _log_referral_event(lead_id, email, first, last,
                            utm_source, utm_medium, utm_campaign,
                            utm_content, utm_term, str(score), raw)

    return jsonify({"ok": True, "tags": answer_tags, "ghl": ghl_result,
                    "utm_source": utm_source or None}), 200


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


# ── Team Console ──────────────────────────────────────────────────────────────
CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", os.environ.get("WEBHOOK_SECRET", ""))

def _init_todos_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT NOT NULL,
                owner          TEXT NOT NULL,
                category       TEXT DEFAULT 'General',
                title          TEXT NOT NULL,
                body           TEXT DEFAULT '',
                priority       TEXT DEFAULT 'normal',
                status         TEXT DEFAULT 'open',
                delegated_to   TEXT DEFAULT '',
                delegated_at   TEXT DEFAULT '',
                done_at        TEXT DEFAULT '',
                source         TEXT DEFAULT '',
                dedup_key      TEXT UNIQUE,
                ai_summary      TEXT DEFAULT '',
                suggested_reply TEXT DEFAULT '',
                action_note     TEXT DEFAULT '',
                received_at     TEXT DEFAULT ''
            )
        """)
        # Migrate existing tables that predate these columns
        for col, ddl in [("ai_summary", "TEXT DEFAULT ''"), ("suggested_reply", "TEXT DEFAULT ''"),
                         ("action_note", "TEXT DEFAULT ''"), ("received_at", "TEXT DEFAULT ''")] :
            try:
                cx.execute(f"ALTER TABLE todos ADD COLUMN {col} {ddl}")
            except Exception:
                pass
        cx.commit()

_init_todos_table()


def _init_calendar_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pushed_at       TEXT NOT NULL,
                google_cal_id   TEXT NOT NULL,
                google_event_id TEXT NOT NULL,
                calendar_name   TEXT DEFAULT '',
                summary         TEXT NOT NULL,
                start           TEXT NOT NULL,
                end             TEXT DEFAULT '',
                location        TEXT DEFAULT '',
                owner           TEXT DEFAULT 'glen',
                status          TEXT DEFAULT 'visible',
                cal_alert       INTEGER DEFAULT 0,
                UNIQUE(google_cal_id, google_event_id)
            )
        """)
        try:
            cx.execute("ALTER TABLE calendar_events ADD COLUMN cal_alert INTEGER DEFAULT 0")
        except Exception:
            pass
        cx.commit()

_init_calendar_table()


@app.route("/api/calendar", methods=["GET"])
def get_calendar():
    owner  = request.args.get("owner", "glen").lower()
    status = request.args.get("status", "visible")
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, google_cal_id, google_event_id, calendar_name,
                   summary, start, end, location, owner, status, cal_alert
            FROM calendar_events
            WHERE owner=? AND status=?
            ORDER BY start ASC
        """, (owner, status)).fetchall()
    cols = ["id","google_cal_id","google_event_id","calendar_name",
            "summary","start","end","location","owner","status","cal_alert"]
    return jsonify({"events": [dict(zip(cols, r)) for r in rows]})


@app.route("/api/calendar", methods=["POST"])
def post_calendar():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    items = request.get_json(force=True) or []
    if isinstance(items, dict):
        items = [items]

    ts = datetime.now(timezone.utc).isoformat()
    upserted = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for ev in items:
            try:
                cx.execute("""
                    INSERT INTO calendar_events
                      (pushed_at, google_cal_id, google_event_id, calendar_name,
                       summary, start, end, location, owner)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(google_cal_id, google_event_id) DO UPDATE SET
                      pushed_at=excluded.pushed_at,
                      calendar_name=excluded.calendar_name,
                      summary=excluded.summary,
                      start=excluded.start,
                      end=excluded.end,
                      location=excluded.location
                    WHERE status='visible'
                """, (ts,
                      ev.get("google_cal_id",""),
                      ev.get("google_event_id",""),
                      ev.get("calendar_name",""),
                      ev.get("summary","(no title)"),
                      ev.get("start",""),
                      ev.get("end",""),
                      ev.get("location",""),
                      ev.get("owner","glen")))
                upserted += 1
            except Exception:
                pass
        cx.commit()
    return jsonify({"ok": True, "upserted": upserted}), 201


@app.route("/api/calendar/<int:event_id>", methods=["PATCH"])
def patch_calendar(event_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    action = (request.get_json(force=True) or {}).get("action", "hide")
    new_status = "delete_requested" if action == "delete" else "hidden"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE calendar_events SET status=? WHERE id=?", (new_status, event_id))
        cx.commit()
    return jsonify({"ok": True, "status": new_status})


@app.route("/api/calendar/<int:event_id>/alert", methods=["PATCH"])
def patch_calendar_alert(event_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    enabled = (request.get_json(force=True) or {}).get("alert", True)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE calendar_events SET cal_alert=? WHERE id=?", (1 if enabled else 0, event_id))
        cx.commit()
    return jsonify({"ok": True, "cal_alert": 1 if enabled else 0})


@app.route("/api/calendar/alerts", methods=["GET"])
def get_calendar_alerts():
    """Return events with cal_alert=1 whose start is within the next 90 minutes."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(minutes=90)
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, summary, start, owner
            FROM calendar_events
            WHERE cal_alert=1 AND status='visible'
              AND start > ? AND start <= ?
            ORDER BY start ASC
        """, (now.strftime("%Y-%m-%dT%H:%M:%SZ"), window_end.strftime("%Y-%m-%dT%H:%M:%SZ"))).fetchall()
    return jsonify({"alerts": [{"id":r[0],"summary":r[1],"start":r[2],"owner":r[3]} for r in rows]})


@app.route("/api/calendar/delete-queue", methods=["GET"])
def calendar_delete_queue():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, google_cal_id, google_event_id, summary
            FROM calendar_events WHERE status='delete_requested'
        """).fetchall()
    return jsonify({"queue": [{"id":r[0],"cal_id":r[1],"event_id":r[2],"summary":r[3]} for r in rows]})


@app.route("/api/calendar/delete-queue/clear", methods=["POST"])
def clear_delete_queue():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    ids = (request.get_json(force=True) or {}).get("ids", [])
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.executemany("UPDATE calendar_events SET status='deleted' WHERE id=?", [(i,) for i in ids])
        cx.commit()
    return jsonify({"ok": True, "cleared": len(ids)})


@app.route("/console")
def console_page():
    resp = send_from_directory(STATIC, "console.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/todos", methods=["GET"])
def get_todos():
    owner  = request.args.get("owner", "glen").lower()
    status = request.args.get("status", "open")
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, created_at, owner, category, title, body, priority,
                   status, delegated_to, delegated_at, done_at, source, dedup_key,
                   ai_summary, suggested_reply, action_note, received_at
            FROM todos
            WHERE owner=? AND status=?
            ORDER BY
                CASE priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                created_at DESC
        """, (owner, status)).fetchall()
    cols = ["id","created_at","owner","category","title","body","priority",
            "status","delegated_to","delegated_at","done_at","source","dedup_key",
            "ai_summary","suggested_reply","action_note","received_at"]
    return jsonify({"todos": [dict(zip(cols, r)) for r in rows]})


@app.route("/api/todos", methods=["POST"])
def post_todos():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    items = request.get_json(force=True) or {}
    # Accept single item or list
    if isinstance(items, dict):
        items = [items]

    ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for item in items:
            owner           = (item.get("owner") or "glen").lower()
            category        = item.get("category") or "General"
            title           = (item.get("title") or "").strip()
            body            = item.get("body") or ""
            priority        = item.get("priority") or "normal"
            source          = item.get("source") or ""
            dedup           = item.get("dedup_key") or None
            ai_summary      = item.get("ai_summary") or ""
            suggested_reply = item.get("suggested_reply") or ""
            action_note     = item.get("action_note") or ""
            received_at     = item.get("received_at") or ""
            if not title:
                continue
            try:
                cx.execute("""
                    INSERT INTO todos
                      (created_at, owner, category, title, body, priority, source, dedup_key,
                       ai_summary, suggested_reply, action_note, received_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(dedup_key) DO UPDATE SET
                      ai_summary=excluded.ai_summary,
                      suggested_reply=excluded.suggested_reply,
                      action_note=excluded.action_note,
                      received_at=CASE WHEN excluded.received_at != '' THEN excluded.received_at ELSE received_at END
                """, (ts, owner, category, title, body, priority, source, dedup,
                      ai_summary, suggested_reply, action_note, received_at))
                if cx.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception:
                pass
        cx.commit()
    return jsonify({"ok": True, "inserted": inserted}), 201


@app.route("/api/todos/<int:todo_id>", methods=["PATCH"])
def patch_todo(todo_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    data   = request.get_json(force=True) or {}
    action = data.get("action", "")
    ts     = datetime.now(timezone.utc).isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        if action == "done":
            cx.execute("UPDATE todos SET status='done', done_at=? WHERE id=?", (ts, todo_id))
        elif action == "delegate":
            to = (data.get("to") or "").lower()
            if to not in ("glen", "rae", "shaira"):
                return jsonify({"error": "Invalid delegate target"}), 400
            # Create a copy for the delegate, mark original as delegated
            row = cx.execute(
                "SELECT owner, category, title, body, priority, source, ai_summary, suggested_reply FROM todos WHERE id=?",
                (todo_id,)
            ).fetchone()
            if row:
                cx.execute("UPDATE todos SET status='delegated', delegated_to=?, delegated_at=? WHERE id=?",
                           (to, ts, todo_id))
                new_title = f"[From {row[0].title()}] {row[2]}"
                cx.execute("""
                    INSERT INTO todos (created_at, owner, category, title, body, priority, source,
                                       ai_summary, suggested_reply)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (ts, to, row[1], new_title, row[3], row[4], row[5], row[6], row[7]))
        elif action == "reopen":
            cx.execute("UPDATE todos SET status='open', done_at='', delegated_to='', delegated_at=? WHERE id=?",
                       (ts, todo_id))
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/todos/<int:todo_id>", methods=["DELETE"])
def delete_todo(todo_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE todos SET status='dismissed' WHERE id=?", (todo_id,))
        cx.commit()
    return jsonify({"ok": True})


# ── Audio Generation ──────────────────────────────────────────────────────────
_EL_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
_EL_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
_EL_BASE     = "https://api.elevenlabs.io/v1"

_AUDIO_SCRIPT_PROMPT = """You are preparing a spoken audio summary for Dr. Glen Swartwout's students.
Convert the following content into a natural, engaging spoken script of approximately {max_words} words
(about {minutes} minutes when read aloud at 150 words/minute).

Write in first person as Dr. Glen. Keep his warm, knowledgeable, mentor tone.
No bullet points — flowing spoken prose only. No stage directions or labels.
Start speaking immediately (no "Hello" or "Welcome" opener).

Content:
{content}"""

def _el_tts(script: str) -> tuple[bytes | None, str | None]:
    """Call ElevenLabs TTS. Returns (audio_bytes, error)."""
    if not _EL_API_KEY or not _EL_VOICE_ID:
        return None, "ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID not set"
    import urllib.request as _ur
    payload = json.dumps({
        "text":           script,
        "model_id":       "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.80, "style": 0.20},
    }).encode()
    req = _ur.Request(
        f"{_EL_BASE}/text-to-speech/{_EL_VOICE_ID}",
        data=payload,
        headers={
            "xi-api-key":   _EL_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        },
        method="POST",
    )
    try:
        with _ur.urlopen(req, timeout=60) as resp:
            return resp.read(), None
    except Exception as e:
        return None, str(e)


@app.route("/generate-audio", methods=["POST"])
def generate_audio():
    """
    Generate a spoken audio summary from text content using Claude + ElevenLabs.

    POST body (JSON):
      text      — raw content to summarize (required)
      title     — label for this audio piece (optional)
      max_words — target script length in words, default 450 (~3 min)
      raw       — if true, send text directly to ElevenLabs without Claude summarization

    Returns JSON:
      { "ok": true, "title": "...", "script": "...", "audio_base64": "...",
        "audio_bytes": <int>, "estimated_minutes": <float> }
    """
    secret = request.headers.get("X-Webhook-Secret", "")
    ws     = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401

    body      = request.get_json(force=True) or {}
    text      = (body.get("text") or "").strip()
    title     = (body.get("title") or "Audio Summary").strip()
    max_words = int(body.get("max_words") or 450)
    raw_mode  = bool(body.get("raw"))

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not _EL_API_KEY or not _EL_VOICE_ID:
        return jsonify({"error": "ElevenLabs not configured (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)"}), 503

    # Step 1 — summarize with Claude (skip if raw mode or text is already short)
    if raw_mode or len(text.split()) <= max_words:
        script = text
    else:
        minutes = round(max_words / 150, 1)
        prompt  = _AUDIO_SCRIPT_PROMPT.format(
            max_words=max_words, minutes=minutes, content=text[:12000]
        )
        try:
            msg    = _cl.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            script = msg.content[0].text.strip()
        except Exception as e:
            return jsonify({"error": f"Claude error: {str(e)[:200]}"}), 500

    # Step 2 — render with ElevenLabs
    audio_bytes, err = _el_tts(script)
    if err:
        return jsonify({"error": f"ElevenLabs error: {err}"}), 500

    import base64
    word_count        = len(script.split())
    estimated_minutes = round(word_count / 150, 1)

    return jsonify({
        "ok":                True,
        "title":             title,
        "script":            script,
        "word_count":        word_count,
        "estimated_minutes": estimated_minutes,
        "audio_bytes":       len(audio_bytes),
        "audio_base64":      base64.b64encode(audio_bytes).decode(),
    })


# ── Transcript Ingest Endpoint ───────────────────────────────────────────────
import re as _re
import time as _time
from concurrent.futures import ThreadPoolExecutor as _TPE

_INGEST_CHUNK_SIZE    = 500
_INGEST_CHUNK_OVERLAP = 50
_INGEST_MIN_WORDS     = 20
_INGEST_BATCH_SIZE    = 50

_TRAINING_KEYWORDS = ["cert", "certification", "training", "class", "workshop", "cohort"]
_BUSINESS_KEYWORDS = ["marketing", "copywriting", "copy", "funnel", "launch", "ads", "advertising",
                      "strategy", "business", "ai tool", "automation", "make.com", "campaign",
                      "sales", "conversion", "brand", "seo", "social media"]


def _clean_zoom_transcript(text):
    """Strip VTT/SRT timestamps, WEBVTT headers, and Otter branding."""
    # WEBVTT header
    text = _re.sub(r'^WEBVTT.*?\n\n', '', text, flags=_re.DOTALL)
    # VTT/SRT cue timestamps  e.g. 00:01:23.456 --> 00:01:27.890
    text = _re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', text)
    # Plain timestamps  00:04:12 or 4:12
    text = _re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
    # Cue numeric identifiers (lines that are just a number)
    text = _re.sub(r'^\d+\s*$', '', text, flags=_re.MULTILINE)
    # Otter branding
    text = _re.sub(r'Transcript by Otter\.ai.*', '', text, flags=_re.IGNORECASE | _re.DOTALL)
    # Unknown Speaker lines
    text = _re.sub(r'^Unknown Speaker\s*$', '', text, flags=_re.MULTILINE)
    # Collapse whitespace
    text = _re.sub(r'[ \t]{2,}', ' ', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _chunk_text(text):
    words = text.split()
    if len(words) <= _INGEST_CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + _INGEST_CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += _INGEST_CHUNK_SIZE - _INGEST_CHUNK_OVERLAP
    return chunks


def _ingest_to_pinecone(text, title, namespace, speakers=""):
    chunks = [c for c in _chunk_text(text) if len(c.split()) >= _INGEST_MIN_WORDS]
    if not chunks:
        return 0, "No usable content after chunking"

    ns_prefix = {"consultations": "consult", "training": "train"}.get(namespace, namespace[:6])
    slug      = _re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:60]

    for i in range(0, len(chunks), _INGEST_BATCH_SIZE):
        batch  = chunks[i:i + _INGEST_BATCH_SIZE]
        texts  = batch
        resp   = _oa.embeddings.create(input=texts, model="text-embedding-3-small")
        vecs   = []
        for j, (chunk, emb) in enumerate(zip(batch, resp.data)):
            vecs.append({
                "id":     f"{ns_prefix}-{slug}-{i+j:03d}",
                "values": emb.embedding,
                "metadata": {
                    "text":      chunk,
                    "source":    "zoom-transcript",
                    "namespace": namespace,
                    "title":     title,
                    "speakers":  speakers,
                    "chunk":     i + j,
                }
            })
        _idx.upsert(vectors=vecs, namespace=namespace)
        _time.sleep(0.2)

    return len(chunks), None


@app.route("/ingest-transcript", methods=["POST"])
def ingest_transcript():
    """
    Ingest a Zoom transcript into Pinecone.
    Called by Make.com after a Zoom recording is ready.

    POST body (JSON):
      text       — transcript text (required)
      title      — meeting title (used for routing + metadata)
      namespace  — override auto-routing: "consultations" or "training"
      speakers   — comma-separated speaker names (optional)
      secret     — webhook secret (or pass as X-Webhook-Secret header)
    """
    secret = request.headers.get("X-Webhook-Secret", "") or (request.get_json(force=True) or {}).get("secret", "")
    ws     = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401

    body         = request.get_json(force=True) or {}
    text         = (body.get("text") or "").strip()
    download_url = (body.get("download_url") or "").strip()
    token        = (body.get("download_token") or "").strip()
    title        = (body.get("title") or "Untitled Session").strip()
    speakers     = (body.get("speakers") or "").strip()
    namespace    = (body.get("namespace") or "").strip().lower()

    # Fetch transcript from Zoom if download_url provided instead of text
    if not text and download_url:
        import urllib.request as _ur2
        try:
            req = _ur2.Request(download_url, headers={"Authorization": f"Bearer {token}"} if token else {})
            with _ur2.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return jsonify({"error": f"Failed to fetch transcript: {str(e)[:200]}"}), 500

    if not text:
        return jsonify({"error": "text or download_url is required"}), 400

    # Auto-route by title if namespace not specified
    if not namespace:
        title_lower = title.lower()
        if any(k in title_lower for k in _TRAINING_KEYWORDS):
            namespace = "training"
        elif any(k in title_lower for k in _BUSINESS_KEYWORDS):
            namespace = "business"
        else:
            namespace = "consultations"

    if namespace not in ["consultations", "training", "business", "default"]:
        return jsonify({"error": f"invalid namespace: {namespace}"}), 400

    cleaned      = _clean_zoom_transcript(text)
    n, err       = _ingest_to_pinecone(cleaned, title, namespace, speakers)

    if err:
        return jsonify({"error": err}), 500

    return jsonify({
        "ok":        True,
        "title":     title,
        "namespace": namespace,
        "chunks":    n,
        "words":     len(cleaned.split()),
    })


# ── Clip hosting (temporary storage for Creatomate) ──────────────────────────
_CLIPS_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "clips"
_CLIPS_DIR.mkdir(exist_ok=True)

@app.route("/clips/upload", methods=["PUT"])
def clips_upload():
    secret = request.headers.get("X-Webhook-Secret", "")
    ws     = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401
    filename = request.args.get("filename", "")
    if not filename or not re.match(r'^[\w\-]+\.mp4$', filename):
        return jsonify({"error": "invalid filename (alphanumeric, hyphens, .mp4 only)"}), 400
    dest = _CLIPS_DIR / filename
    dest.write_bytes(request.data)
    base_url = os.environ.get("RENDER_EXTERNAL_URL", "https://glen-knowledge-chat.onrender.com")
    return jsonify({"ok": True, "url": f"{base_url}/clips/{filename}"})


@app.route("/clips/<filename>")
def clips_serve(filename):
    if not re.match(r'^[\w\-]+\.mp4$', filename):
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(str(_CLIPS_DIR), filename, mimetype="video/mp4")


@app.route("/clips/<filename>", methods=["DELETE"])
def clips_delete(filename):
    secret = request.headers.get("X-Webhook-Secret", "")
    ws     = os.environ.get("WEBHOOK_SECRET", "")
    if ws and secret != ws:
        return jsonify({"error": "unauthorized"}), 401
    f = _CLIPS_DIR / filename
    if f.exists():
        f.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
