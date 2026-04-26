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
1. OPEN WITH A HOOK: Begin with a single compelling sentence — a surprising research finding, a thought-provoking reframe, or a striking quote from the source material. This hook is your first line, before any explanation.
2. Synthesize the provided source snippets into a unified, coherent answer to the user's question.
3. When multiple mentors or concepts connect, explicitly show how they reinforce each other.
4. At the end of your response, list the source references used (name + field).
5. Do NOT fabricate information not present in the snippets. If the snippets don't fully answer the question, say so clearly.
6. Keep responses focused and readable — prefer synthesis over exhaustive lists."""

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

    data      = request.get_json() or {}
    query     = (data.get("query") or "").strip()
    history   = data.get("history") or []
    level     = (data.get("level") or "self-healing").strip().lower()
    name      = (data.get("name") or "").strip()
    email     = (data.get("email") or "").strip()
    frequency = (data.get("frequency") or "").strip()

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

        # GHL onboarding for email opt-ins (non-blocking)
        if email:
            import threading as _threading
            def _onboard():
                try:
                    parts = name.split(None, 1)
                    first = parts[0] if parts else ""
                    last  = parts[1] if len(parts) > 1 else ""
                    tags  = ["chatbot-lead"]
                    if frequency:
                        tags.append(f"frequency-{frequency}")
                    ghl_onboard_contact(email, first, last, source_tag="chatbot", extra_tags=tags)
                except Exception:
                    pass
            _threading.Thread(target=_onboard, daemon=True).start()

        # Generate next Socratic question
        next_question = ""
        try:
            nq_msg = _cl.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system="Generate a single Socratic follow-up question for a healing wisdom chatbot. Output only the question, nothing else.",
                messages=[{"role": "user", "content":
                    f"Question: {query}\nAnswer summary: {answer[:400]}\n\n"
                    "What one deeper question would guide the person further into this exploration? "
                    "Make it personally evocative and specific to what was just discussed."}]
            )
            next_question = nq_msg.content[0].text.strip().strip('"')
        except Exception:
            pass

        yield sse({"done": True, "log_id": log_id,
                   "sources": sources_list, "chunks_retrieved": len(all_matches),
                   "next_question": next_question})

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
                notes        TEXT DEFAULT '',
                referred_by  TEXT DEFAULT ''
            )
        """)
        for col in ["referred_by TEXT DEFAULT ''", "short_url TEXT DEFAULT ''"]:
            try:
                cx.execute(f"ALTER TABLE affiliate_signups ADD COLUMN {col}")
            except Exception:
                pass
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
        cx.execute("""
            CREATE TABLE IF NOT EXISTS affiliate_offers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sort_order  INTEGER DEFAULT 0,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                url_template TEXT NOT NULL,
                active      INTEGER DEFAULT 1
            )
        """)
        # Seed quiz as first offer
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Accelerate Self-Healing Quiz'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, active)
                VALUES (1, 'Accelerate Self-Healing Quiz',
                    'Free quiz — discover your top healing opportunities. Share with anyone curious about natural healing.',
                    'https://healing.scoreapp.com?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz',
                    1)
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


QUIZ_URL            = "https://healing.scoreapp.com"
REBRANDLY_API_KEY   = os.environ.get("REBRANDLY_API_KEY", "")
REBRANDLY_VIP       = "truly.vip"   # affiliate / referral tracking links
REBRANDLY_SO        = "truly.so"    # general short links


def _rebrandly_create(slashtag, destination, domain=REBRANDLY_VIP, title=""):
    """Create (or fetch existing) Rebrandly short link. Returns shortUrl string or None."""
    if not REBRANDLY_API_KEY:
        return None
    import urllib.request as _ur
    import urllib.error as _ue
    headers = {"apikey": REBRANDLY_API_KEY, "Content-Type": "application/json"}
    payload = json.dumps({
        "destination": destination,
        "slashtag":    slashtag,
        "domain":      {"fullName": domain},
        "title":       title,
    }).encode()
    req = _ur.Request("https://api.rebrandly.com/v1/links", data=payload, headers=headers, method="POST")
    try:
        resp = json.loads(_ur.urlopen(req, timeout=10).read())
        return "https://" + resp.get("shortUrl", "")
    except _ue.HTTPError as e:
        # 403 = slashtag already taken (Rebrandly quirk) — fetch the existing link
        try:
            get_req = _ur.Request(
                f"https://api.rebrandly.com/v1/links?domain.fullName={domain}&slashtag={slashtag}",
                headers={"apikey": REBRANDLY_API_KEY}, method="GET"
            )
            links = json.loads(_ur.urlopen(get_req, timeout=10).read())
            if links:
                return "https://" + links[0].get("shortUrl", "")
        except Exception as e2:
            print(f"[rebrandly] fetch existing error: {e2}")
        print(f"[rebrandly] create error {e.code}")
        return None
    except Exception as e:
        print(f"[rebrandly] create error: {e}")
        return None


@app.route("/affiliate")
def affiliate_page():
    resp = send_from_directory(STATIC, "affiliate.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/affiliate/hub/<slug>")
def affiliate_hub_page(slug):
    resp = send_from_directory(STATIC, "affiliate-hub.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/affiliate/hub-data/<slug>")
def affiliate_hub_data(slug):
    with sqlite3.connect(LOG_DB) as cx:
        aff = cx.execute(
            "SELECT name, organization, slug FROM affiliate_signups WHERE slug=? AND status='approved'",
            (slug,)
        ).fetchone()
    if not aff:
        return jsonify({"error": "Not found"}), 404
    name, org, slug = aff
    with sqlite3.connect(LOG_DB) as cx:
        offers = cx.execute(
            "SELECT name, description, url_template FROM affiliate_offers WHERE active=1 ORDER BY sort_order ASC"
        ).fetchall()
    return jsonify({
        "name": name,
        "organization": org,
        "slug": slug,
        "offers": [
            {
                "name": o[0],
                "description": o[1],
                "url": o[2].replace("{slug}", slug),
            }
            for o in offers
        ]
    })


@app.route("/affiliate/portal")
def affiliate_portal_page():
    from flask import redirect as _redir
    import urllib.parse as _up
    # Allow ?email= as a convenience — look up token and redirect
    email = request.args.get("email", "").strip().lower()
    if email and not request.args.get("token"):
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute("SELECT token FROM affiliate_signups WHERE LOWER(email)=?", (email,)).fetchone()
        if row:
            return _redir(f"/affiliate/portal?token={row[0]}")
        return _redir("/affiliate?error=" + _up.quote("No affiliate account found for that email. Apply below."))
    resp = send_from_directory(STATIC, "affiliate-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/affiliate/apply-form", methods=["POST"])
def affiliate_apply_form():
    """HTML form POST — processes signup and does a 302 redirect to the portal."""
    from flask import redirect as _redirect
    import urllib.parse as _urlparse
    name        = (request.form.get("name") or "").strip()
    email       = (request.form.get("email") or "").strip().lower()
    org         = (request.form.get("organization") or "").strip()
    site        = (request.form.get("website") or "").strip()
    referred_by = (request.form.get("referred_by") or "").strip()
    if site and not site.startswith(("http://", "https://")):
        site = "https://" + site
    promo = (request.form.get("promo_method") or "").strip()

    if not name or not email:
        return _redirect("/affiliate?error=" + _urlparse.quote("Name and email are required"))

    base = re.sub(r"[^a-z0-9]+", "-", (org or name).lower()).strip("-")[:30]
    import secrets as _sec
    token = _sec.token_urlsafe(24)
    slug  = base
    ts    = datetime.now(timezone.utc).isoformat()

    # Return existing portal if email already registered
    with sqlite3.connect(LOG_DB) as cx:
        existing = cx.execute("SELECT token FROM affiliate_signups WHERE email=?", (email,)).fetchone()
    if existing:
        return _redirect(f"/affiliate/portal?token={existing[0]}")

    # Ensure unique slug
    with sqlite3.connect(LOG_DB) as cx:
        if cx.execute("SELECT id FROM affiliate_signups WHERE slug=?", (slug,)).fetchone():
            slug = f"{base}-{token[:6]}"

    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            # Generate Rebrandly short link → points to affiliate hub
            _hub_dest = f"{request.host_url.rstrip('/')}/affiliate/hub/{slug}"
            short_url = _rebrandly_create(
                slashtag=slug, destination=_hub_dest,
                title=f"Affiliate: {org or name}"
            ) or ""

            cx.execute("""
                INSERT INTO affiliate_signups
                  (created_at, name, email, organization, website, promo_method, slug, token, status, referred_by, short_url)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (ts, name, email, org, site, promo, slug, token, "approved", referred_by, short_url))
            cx.execute("""
                INSERT OR IGNORE INTO referral_sources
                  (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, org or name, slug,
                  f"Affiliate: {name}" + (f" ({org})" if org else ""),
                  slug, "affiliate", "scoreapp-quiz"))
            cx.commit()
    except Exception as e:
        return _redirect("/affiliate?error=" + _urlparse.quote(f"Signup failed: {str(e)[:80]}"))

    return _redirect(f"/affiliate/portal?token={token}")


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
    # Check if email already exists — if so, return their existing portal
    with sqlite3.connect(LOG_DB) as cx:
        existing_row = cx.execute("SELECT token, slug FROM affiliate_signups WHERE email=?", (email,)).fetchone()
    if existing_row:
        token, slug = existing_row
    else:
        # Ensure slug uniqueness
        with sqlite3.connect(LOG_DB) as cx:
            if cx.execute("SELECT id FROM affiliate_signups WHERE slug=?", (slug,)).fetchone():
                slug = f"{base}-{token[:6]}"
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute("""
                    INSERT INTO affiliate_signups
                      (created_at, name, email, organization, website, promo_method, slug, token, status)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (ts, name, email, org, site, promo, slug, token, "approved"))
                cx.execute("""
                    INSERT OR IGNORE INTO referral_sources
                      (created_at, name, slug, description, utm_source, utm_medium, utm_campaign)
                    VALUES (?,?,?,?,?,?,?)
                """, (ts, org or name, slug,
                      f"Affiliate: {name}" + (f" ({org})" if org else ""),
                      slug, "affiliate", "scoreapp-quiz"))
                cx.commit()
        except sqlite3.IntegrityError as e:
            return jsonify({"error": f"Signup failed: {str(e)[:100]}"}), 409

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
            SELECT id, name, email, organization, slug, status, created_at, short_url
            FROM affiliate_signups WHERE token=?
        """, (token,)).fetchone()
    if not row:
        return jsonify({"error": "Invalid token"}), 404
    aff_id, name, email, org, slug, status, created_at, short_url = row
    if status != "approved":
        return jsonify({"error": "Application pending review"}), 403

    long_url       = f"{QUIZ_URL}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    tracking_url   = short_url if short_url else long_url
    recruit_url    = f"https://glen-knowledge-chat.onrender.com/affiliate?ref={slug}"

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
        recruited_count = cx.execute("""
            SELECT COUNT(*) FROM affiliate_signups WHERE referred_by=? AND status='approved'
        """, (slug,)).fetchone()[0]

    return jsonify({
        "name": name,
        "organization": org,
        "slug": slug,
        "tracking_url": tracking_url,
        "recruit_url": recruit_url,
        "total_leads": stats[0] if stats else 0,
        "last_lead": stats[1] if stats else None,
        "recruited_count": recruited_count,
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


@app.route("/api/affiliates/backfill-links", methods=["POST"])
def backfill_affiliate_links():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    results = []
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute(
            "SELECT id, slug, name, organization FROM affiliate_signups WHERE (short_url IS NULL OR short_url='') AND status='approved'"
        ).fetchall()
    base_url = request.host_url.rstrip("/")
    for aff_id, slug, name, org in rows:
        destination = f"{base_url}/affiliate/hub/{slug}"
        short_url = _rebrandly_create(slug, destination, title=f"Affiliate: {org or name}")
        if short_url:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute("UPDATE affiliate_signups SET short_url=? WHERE id=?", (short_url, aff_id))
                cx.commit()
            results.append({"slug": slug, "short_url": short_url, "status": "created"})
        else:
            results.append({"slug": slug, "status": "failed"})
    return jsonify({"backfilled": len(results), "results": results})


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
        cx.execute("""
            CREATE TABLE IF NOT EXISTS calendar_suppressed (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                owner         TEXT NOT NULL DEFAULT 'glen',
                title_pattern TEXT NOT NULL,
                day_of_week   INTEGER,
                hour          INTEGER,
                created_at    TEXT NOT NULL,
                UNIQUE(owner, title_pattern, day_of_week, hour)
            )
        """)
        # Migrate old schema if needed
        for col in ["day_of_week INTEGER", "hour INTEGER", "title_pattern TEXT DEFAULT ''"]:
            try:
                cx.execute(f"ALTER TABLE calendar_suppressed ADD COLUMN {col}")
            except Exception:
                pass
        cx.commit()

_init_calendar_table()


def _normalize_cal_title(title: str) -> str:
    """Strip session numbers and minor variation markers from event titles."""
    import re as _re
    t = title
    t = _re.sub(r'\s+\d+\s*:', ':', t)               # "Training 4:" → "Training:"
    t = _re.sub(r':\s*\d+\s*:', ':', t)               # ": 4:" → ":"
    t = _re.sub(r'\s+\d+\s*$', '', t)                 # trailing numbers
    t = _re.sub(r'\[\d+\s+of\s+\d+[^\]]*\]', '', t)  # "[11 of 100 spots filled]"
    t = _re.sub(r'\s+', ' ', t).strip()
    return t.lower()


def _parse_event_start(start_iso: str):
    """Return (day_of_week 0=Mon, hour) from ISO start string, or (None, None)."""
    if not start_iso or 'T' not in start_iso:
        return None, None
    try:
        d = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        return d.weekday(), d.hour
    except Exception:
        return None, None



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
              AND substr(start, 1, 10) >= date('now')
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
    upserted = skipped = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Build suppression rules per owner: list of (title_pattern, day_of_week, hour)
        sup_rows = cx.execute(
            "SELECT owner, title_pattern, day_of_week, hour FROM calendar_suppressed"
        ).fetchall()
        suppressed = {}  # owner → list of (pattern, dow, hour)
        for row_owner, pattern, dow, hr in sup_rows:
            suppressed.setdefault(row_owner, []).append((pattern, dow, hr))

        for ev in items:
            owner   = ev.get("owner", "glen")
            summary = ev.get("summary", "(no title)")
            start   = ev.get("start", "")
            norm    = _normalize_cal_title(summary)
            ev_dow, ev_hr = _parse_event_start(start)
            # Check suppression: title_pattern match + same day-of-week + same hour
            is_suppressed = False
            for pattern, dow, hr in suppressed.get(owner, []):
                if norm == pattern:
                    if (dow is None or dow == ev_dow) and (hr is None or hr == ev_hr):
                        is_suppressed = True
                        break
            if is_suppressed:
                skipped += 1
                continue
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
                      location=excluded.location,
                      owner=excluded.owner
                    WHERE status='visible'
                """, (ts,
                      ev.get("google_cal_id",""),
                      ev.get("google_event_id",""),
                      ev.get("calendar_name",""),
                      summary,
                      ev.get("start",""),
                      ev.get("end",""),
                      ev.get("location",""),
                      owner))
                upserted += 1
            except Exception:
                pass
        cx.commit()
    return jsonify({"ok": True, "upserted": upserted, "skipped": skipped}), 201


@app.route("/api/calendar/<int:event_id>", methods=["PATCH"])
def patch_calendar(event_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    action = (request.get_json(force=True) or {}).get("action", "hide")
    if action == "show":
        new_status = "visible"
    elif action == "delete":
        new_status = "delete_requested"
    else:
        new_status = "hidden"
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


@app.route("/api/calendar/<int:event_id>/suppress", methods=["DELETE"])
def unsuppress_calendar_event(event_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT summary, owner, start FROM calendar_events WHERE id=?", (event_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        summary, owner, start = row
        pattern = _normalize_cal_title(summary)
        dow, hr  = _parse_event_start(start)
        cx.execute(
            "DELETE FROM calendar_suppressed WHERE owner=? AND title_pattern=? AND day_of_week=? AND hour=?",
            (owner, pattern, dow, hr)
        )
        cx.execute("UPDATE calendar_events SET status='visible' WHERE id=?", (event_id,))
        cx.commit()
    return jsonify({"ok": True, "unsuppressed": summary})


@app.route("/api/calendar/<int:event_id>/suppress", methods=["POST"])
def suppress_calendar_event(event_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT summary, owner, start FROM calendar_events WHERE id=?", (event_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        summary, owner, start = row
        pattern = _normalize_cal_title(summary)
        dow, hr  = _parse_event_start(start)
        ts = datetime.now(timezone.utc).isoformat()
        cx.execute(
            """INSERT OR IGNORE INTO calendar_suppressed
               (owner, title_pattern, day_of_week, hour, created_at)
               VALUES (?,?,?,?,?)""",
            (owner, pattern, dow, hr, ts)
        )
        cx.execute("UPDATE calendar_events SET status='hidden' WHERE id=?", (event_id,))
        # Also hide all existing visible events matching the same pattern/day/hour for this owner
        existing = cx.execute(
            "SELECT id, summary, start FROM calendar_events WHERE owner=? AND status='visible'",
            (owner,)
        ).fetchall()
        to_hide = [
            r[0] for r in existing
            if _normalize_cal_title(r[1]) == pattern and _parse_event_start(r[2]) == (dow, hr)
        ]
        if to_hide:
            cx.executemany(
                "UPDATE calendar_events SET status='hidden' WHERE id=?",
                [(i,) for i in to_hide]
            )
        cx.commit()
    return jsonify({"ok": True, "suppressed": summary, "pattern": pattern,
                    "day_of_week": dow, "hour": hr, "owner": owner})


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
            if to not in ("glen", "rae", "shaira", "justus"):
                return jsonify({"error": "Invalid delegate target"}), 400
            note = (data.get("note") or "").strip()
            # Create a copy for the delegate, mark original as delegated
            row = cx.execute(
                "SELECT owner, category, title, body, priority, source, ai_summary, suggested_reply FROM todos WHERE id=?",
                (todo_id,)
            ).fetchone()
            if row:
                cx.execute("UPDATE todos SET status='delegated', delegated_to=?, delegated_at=? WHERE id=?",
                           (to, ts, todo_id))
                new_title = f"[From {row[0].title()}] {row[2]}"
                extra_body = f"\n\n📝 Glen's note: {note}" if note else ""
                cx.execute("""
                    INSERT INTO todos (created_at, owner, category, title, body, priority, source,
                                       ai_summary, suggested_reply)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (ts, to, row[1], new_title, (row[3] or "") + extra_body, row[4], row[5], row[6], row[7]))
        elif action == "undelegated":
            # Undo a delegate: restore original to open, remove the delegated copy
            cx.execute(
                "UPDATE todos SET status='open', delegated_to='', delegated_at='' WHERE id=?",
                (todo_id,)
            )
            cx.execute(
                "DELETE FROM todos WHERE source=(SELECT source FROM todos WHERE id=?) "
                "AND title LIKE '[From %' AND id != ?",
                (todo_id, todo_id)
            )
        elif action == "reopen":
            cx.execute("UPDATE todos SET status='open', done_at='', delegated_to='', delegated_at=? WHERE id=?",
                       (ts, todo_id))
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/todos/<int:todo_id>/draft-reply", methods=["POST"])
def draft_reply_endpoint(todo_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    data     = request.get_json(force=True) or {}
    guidance = (data.get("guidance") or "").strip()
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT title, body, category FROM todos WHERE id=?", (todo_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    title, body, category = row
    guidance_block = f"\n\nGlen's guidance: {guidance}" if guidance else ""
    prompt = (
        "You are drafting a reply on behalf of Dr. Glen Swartwout, naturopathic physician "
        "and biofield scientist in Hilo, Hawaiʻi. Be warm, concise, and professional. "
        "Sign off naturally as Dr. Glen.\n\n"
        f"Email subject: {title}\n"
        f"Email content:\n{(body or '')[:2000]}"
        f"{guidance_block}\n\n"
        "Draft the reply now:"
    )
    try:
        msg = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"draft": msg.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


# ── Rae Feedback (humor / speech monitoring) ──────────────────────────────────

def _init_rae_feedback_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS rae_feedback (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT NOT NULL,
                event_type       TEXT NOT NULL,  -- 'laugh', 'speech', 'greeting_played'
                greeting_index   INTEGER DEFAULT -1,
                greeting_style   TEXT DEFAULT '',
                amplitude_peak   REAL DEFAULT 0,
                duration_ms      INTEGER DEFAULT 0,
                transcript       TEXT DEFAULT '',
                notes            TEXT DEFAULT ''
            )
        """)
        cx.commit()

_init_rae_feedback_table()


@app.route("/api/rae-feedback", methods=["POST"])
def post_rae_feedback():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    events = request.get_json(force=True) or {}
    # Accept either a single event dict or a list
    if isinstance(events, dict):
        events = [events]

    ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for e in events:
            cx.execute("""
                INSERT INTO rae_feedback
                    (ts, event_type, greeting_index, greeting_style,
                     amplitude_peak, duration_ms, transcript, notes)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                e.get("ts", ts),
                e.get("event_type", "unknown"),
                e.get("greeting_index", -1),
                e.get("greeting_style", ""),
                e.get("amplitude_peak", 0),
                e.get("duration_ms", 0),
                e.get("transcript", "")[:500],
                e.get("notes", ""),
            ))
            inserted += 1
        cx.commit()
    return jsonify({"ok": True, "inserted": inserted}), 201


@app.route("/api/rae-feedback", methods=["GET"])
def get_rae_feedback():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    event_type = request.args.get("event_type")       # optional filter
    limit      = min(int(request.args.get("limit", 200)), 1000)

    query  = "SELECT * FROM rae_feedback"
    params = []
    if event_type:
        query += " WHERE event_type = ?"
        params.append(event_type)
    query += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(query, params).fetchall()

    return jsonify({"events": [dict(r) for r in rows]})


@app.route("/api/rae-feedback/summary", methods=["GET"])
def get_rae_feedback_summary():
    """Laugh counts grouped by greeting_style — reveals which humor lands best."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        by_style = cx.execute("""
            SELECT greeting_style,
                   COUNT(*)                          AS laugh_count,
                   AVG(amplitude_peak)               AS avg_amplitude,
                   AVG(duration_ms)                  AS avg_duration_ms,
                   MAX(ts)                           AS last_seen
            FROM   rae_feedback
            WHERE  event_type = 'laugh'
            GROUP  BY greeting_style
            ORDER  BY laugh_count DESC
        """).fetchall()

        recent = cx.execute("""
            SELECT ts, event_type, greeting_style, transcript, amplitude_peak
            FROM   rae_feedback
            ORDER  BY ts DESC
            LIMIT  20
        """).fetchall()

    return jsonify({
        "laugh_by_style": [dict(r) for r in by_style],
        "recent_events":  [dict(r) for r in recent],
    })


# ── People / CRM ──────────────────────────────────────────────────────────────
# GHL custom field IDs that have actual data
GHL_FIELD_MAP = {
    "1lkRpyfPcZNrTBpCzJnk": "terrain_concerns",
    "6Z8AK3c4Z56HcJpV5bft": "request",
    "BywF1IMDoVyg9kEvLOBL": "birth_time",
    "HwkLqsLPUrpPzsjKu38Q": "surgeries",
    "I4Enwr40l0s9vW5auWMK": "challenges",
    "Icll73HcO6QFyCbqLGPS": "budget",
    "PPomHxQW6jaj5vf0r8Sx": "personal_history",
    "UIoZLhStWzI84krSl0tZ": "roles",
    "bwLAZCPo7hByZ7xQEKvN": "title",
    "cTtOuUiZN8lQjBzyrwb4": "birthplace",
    "fx3khczY6JEhAODOV9os": "gender",
    "ghiyQnT354WRKL1csRfm": "resources",
    "h91bcznkcDa2994aNfwb": "healing_response",
    "icNJnKoS1OW0r4apHmbs": "form_completed_by",
    "kmIvkDLwMTvogkWkkF4X": "family_history",
    "q12KidO5toCrpPtSY3Mj": "medications",
    "quRxBSJr4S6XF4gRAGsC": "goals",
    "uk6jYxfE45gKT2FBsqPo": "conditions",
    "vR79NGSTFxn3WZ34VXGW": "body_systems",
    "zW4bdPaR6GMUKt1jtR7U": "issue_duration",
    "eE8sWQAEy4stBPMS1jV3": "investment",
    "xyGLzfZyHSw26rxlEbRl": "interests",
    # New fields
    "DsbMjwrQqecAsShUJ49b": "profession",
    "FFChZTwhu9nqlFKTULjB": "organizations",
    "Hu7x2xN60nOG3fMT0uZY": "island",
}

def _init_people_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                email            TEXT UNIQUE NOT NULL,
                first_name       TEXT DEFAULT '',
                last_name        TEXT DEFAULT '',
                name             TEXT DEFAULT '',
                phone            TEXT DEFAULT '',
                dob              TEXT DEFAULT '',
                birth_time       TEXT DEFAULT '',
                birthplace       TEXT DEFAULT '',
                gender           TEXT DEFAULT '',
                city             TEXT DEFAULT '',
                state            TEXT DEFAULT '',
                country          TEXT DEFAULT '',
                island           TEXT DEFAULT '',
                profession       TEXT DEFAULT '',
                title            TEXT DEFAULT '',
                organizations    TEXT DEFAULT '[]',
                ghl_id           TEXT DEFAULT '',
                pb_id            TEXT DEFAULT '',
                source           TEXT DEFAULT '',
                tags             TEXT DEFAULT '[]',
                roles            TEXT DEFAULT '[]',
                challenges       TEXT DEFAULT '',
                goals            TEXT DEFAULT '',
                terrain_concerns TEXT DEFAULT '[]',
                body_systems     TEXT DEFAULT '[]',
                conditions       TEXT DEFAULT '[]',
                healing_response TEXT DEFAULT '[]',
                interests        TEXT DEFAULT '[]',
                request          TEXT DEFAULT '[]',
                personal_history TEXT DEFAULT '',
                family_history   TEXT DEFAULT '',
                medications      TEXT DEFAULT '',
                surgeries        TEXT DEFAULT '',
                budget           TEXT DEFAULT '',
                investment       TEXT DEFAULT '',
                resources        TEXT DEFAULT '',
                issue_duration   TEXT DEFAULT '',
                form_completed_by TEXT DEFAULT '',
                order_count      INTEGER DEFAULT 0,
                last_order_date  TEXT DEFAULT '',
                session_count    INTEGER DEFAULT 0,
                last_session_date TEXT DEFAULT '',
                last_contact_date TEXT DEFAULT '',
                notes            TEXT DEFAULT '',
                created_at       TEXT DEFAULT '',
                updated_at       TEXT DEFAULT '',
                synced_at        TEXT DEFAULT ''
            )
        """)
        cx.commit()

_init_people_table()


def _people_search_query(params):
    """Build WHERE clause from search params. Returns (where_str, args)."""
    clauses, args = [], []
    q = params.get("q", "").strip()
    if q:
        clauses.append("(name LIKE ? OR email LIKE ? OR first_name LIKE ? OR last_name LIKE ?)")
        like = f"%{q}%"
        args += [like, like, like, like]
    for fld in ("state", "island", "profession", "gender", "source", "city"):
        val = params.get(fld, "").strip()
        if val:
            clauses.append(f"{fld} LIKE ?")
            args.append(f"%{val}%")
    # Multi-tag filter: all tags must match (AND logic)
    for tag in [t.strip() for t in params.get("tags", "").split(",") if t.strip()]:
        clauses.append("tags LIKE ?")
        args.append(f"%{tag}%")
    if params.get("has_orders") == "1":
        clauses.append("order_count > 0")
    if params.get("has_sessions") == "1":
        clauses.append("session_count > 0")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, args


@app.route("/api/people", methods=["GET"])
def get_people():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    limit  = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    where, args = _people_search_query(request.args)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        total = cx.execute(f"SELECT COUNT(*) FROM people {where}", args).fetchone()[0]
        rows  = cx.execute(
            f"SELECT id,email,name,first_name,last_name,phone,city,state,country,island,"
            f"profession,title,organizations,ghl_id,source,tags,roles,challenges,goals,"
            f"terrain_concerns,body_systems,conditions,order_count,last_order_date,"
            f"session_count,last_session_date,last_contact_date,synced_at "
            f"FROM people {where} ORDER BY last_contact_date DESC, name ASC "
            f"LIMIT ? OFFSET ?",
            args + [limit, offset]
        ).fetchall()
    return jsonify({"total": total, "people": [dict(r) for r in rows]})


@app.route("/api/people/<int:person_id>", methods=["GET"])
def get_person(person_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return jsonify({"error":"Not found"}), 404
    return jsonify(dict(row))


@app.route("/api/people", methods=["POST"])
def upsert_people():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    items = request.get_json(force=True) or []
    if isinstance(items, dict):
        items = [items]
    ts = datetime.now(timezone.utc).isoformat()
    inserted = updated = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for p in items:
            email = (p.get("email") or "").strip().lower()
            if not email:
                continue
            existing = cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()
            fields = {k: p.get(k, "") for k in [
                "first_name","last_name","name","phone","dob","birth_time","birthplace",
                "gender","city","state","country","island","profession","title",
                "ghl_id","pb_id","source","challenges","goals","personal_history",
                "family_history","medications","surgeries","budget","investment",
                "resources","issue_duration","form_completed_by",
                "last_order_date","last_session_date","last_contact_date","notes",
            ]}
            # JSON array fields
            for jf in ["organizations","tags","roles","terrain_concerns","body_systems",
                        "conditions","healing_response","interests","request"]:
                v = p.get(jf, [])
                fields[jf] = json.dumps(v) if isinstance(v, list) else v
            # Integer fields
            fields["order_count"]   = int(p.get("order_count", 0) or 0)
            fields["session_count"] = int(p.get("session_count", 0) or 0)
            fields["synced_at"]     = ts
            if existing:
                fields["updated_at"] = ts
                set_clause = ", ".join(f"{k}=?" for k in fields)
                cx.execute(f"UPDATE people SET {set_clause} WHERE email=?",
                           list(fields.values()) + [email])
                updated += 1
            else:
                fields["email"]      = email
                fields["created_at"] = ts
                fields["updated_at"] = ts
                cols = ", ".join(fields.keys())
                vals = ", ".join("?" * len(fields))
                cx.execute(f"INSERT INTO people ({cols}) VALUES ({vals})",
                           list(fields.values()))
                inserted += 1
        cx.commit()
    return jsonify({"inserted": inserted, "updated": updated, "ok": True})


@app.route("/api/people/<int:person_id>/note", methods=["POST"])
def add_person_note(person_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    note = (request.get_json(force=True) or {}).get("note","").strip()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            UPDATE people SET notes = CASE
              WHEN notes='' THEN ?
              ELSE notes || char(10) || ?
            END WHERE id=?
        """, (f"[{ts}] {note}", f"[{ts}] {note}", person_id))
        cx.commit()
    return jsonify({"ok": True})


# ── Console AI chat (context-aware) ───────────────────────────────────────────
@app.route("/api/console-ask", methods=["POST"])
def console_ask():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    data    = request.get_json(force=True) or {}
    query   = (data.get("query") or "").strip()
    owner   = (data.get("owner") or "glen").lower()
    context = (data.get("context") or "")   # page context string
    history = data.get("history") or []
    if not query:
        return jsonify({"error":"No query"}), 400

    owner_desc = {
        "glen":   "Dr. Glen Swartwout, naturopathic optometrist, solopreneur — full access to all systems and data.",
        "rae":    "Rae (Susan Luscombe), business owner and operations partner — full access, handles orders/fulfillment/finance/scheduling.",
        "shaira": "Shaira, technical VA — focused on implementation tasks, GHL/tech integrations.",
    }.get(owner, owner)

    system = (
        f"You are the AI assistant in the Remedy Match business console. "
        f"You are speaking with: {owner_desc}\n"
        f"Be concise and action-oriented. Use bullet points for lists. "
        f"Answer questions about clients, business, health protocols, operations, or anything relevant.\n"
    )
    if context:
        system += f"\nCurrent context:\n{context}"

    msgs = []
    for h in history[-6:]:
        msgs.append({"role": h.get("role","user"), "content": h.get("content","")})
    msgs.append({"role": "user", "content": query})

    def _stream():
        with _cl.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system,
            messages=msgs,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(_stream()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── Token storage (OAuth tokens persisted in DB for cloud cron) ───────────────
def _init_tokens_table():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                name       TEXT PRIMARY KEY,
                token_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cx.commit()

_init_tokens_table()


@app.route("/api/tokens/<name>", methods=["GET"])
def get_token(name):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT token_json, updated_at FROM oauth_tokens WHERE name=?", (name,)).fetchone()
    if not row:
        return jsonify({"error":"Not found"}), 404
    return jsonify({"name": name, "token_json": row[0], "updated_at": row[1]})


@app.route("/api/tokens/<name>", methods=["PUT"])
def put_token(name):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key","") or request.args.get("key","")
        if key != CONSOLE_SECRET:
            return jsonify({"error":"Unauthorized"}), 401
    data = request.get_json(force=True) or {}
    token_json = data.get("token_json","")
    if not token_json:
        return jsonify({"error":"Missing token_json"}), 400
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            INSERT INTO oauth_tokens (name, token_json, updated_at) VALUES (?,?,?)
            ON CONFLICT(name) DO UPDATE SET token_json=excluded.token_json, updated_at=excluded.updated_at
        """, (name, token_json, ts))
        cx.commit()
    return jsonify({"ok": True, "updated_at": ts})


# ── Background cron scheduler ─────────────────────────────────────────────────
def _run_cron():
    """Run the console push logic in-process on Render (no Mac needed)."""
    import importlib.util, sys as _sys, tempfile, base64 as _b64
    from pathlib import Path as _Path

    print(f"[CRON] Starting scheduled push at {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load tokens from DB → temp files
    token_map = {
        "glen_gmail":  "/tmp/token_glen.json",
        "rae_gmail":   "/tmp/token_rae.json",
        "calendar":    "/tmp/token_calendar.json",
    }
    with sqlite3.connect(LOG_DB) as cx:
        for name, path in token_map.items():
            row = cx.execute("SELECT token_json FROM oauth_tokens WHERE name=?", (name,)).fetchone()
            if row:
                _Path(path).write_text(row[0])

    # Import and run console-push logic with Render token paths
    try:
        import requests as _req

        base_url  = f"http://localhost:{os.environ.get('PORT','10000')}"
        headers   = {"X-Console-Key": CONSOLE_SECRET, "Content-Type": "application/json"}

        # Run the push script as a subprocess so it uses the right paths
        import subprocess as _sp
        env = os.environ.copy()
        env["GLEN_TOKEN_PATH"]     = token_map["glen_gmail"]
        env["RAE_TOKEN_PATH"]      = token_map["rae_gmail"]
        env["CALENDAR_TOKEN_PATH"] = token_map["calendar"]
        env["RENDER_BASE"]         = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME','glen-knowledge-chat.onrender.com')}"
        result = _sp.run(
            ["python3", "/opt/render/project/src/console_push_cron.py"],
            capture_output=True, text=True, timeout=300, env=env
        )
        print(result.stdout[-3000:] if result.stdout else "(no output)")
        if result.returncode != 0:
            print(f"[CRON] Error: {result.stderr[-1000:]}")

        # Save any refreshed tokens back to DB
        for name, path in token_map.items():
            p = _Path(path)
            if p.exists():
                with _db_lock, sqlite3.connect(LOG_DB) as cx:
                    cx.execute("""
                        INSERT INTO oauth_tokens (name, token_json, updated_at) VALUES (?,?,?)
                        ON CONFLICT(name) DO UPDATE SET token_json=excluded.token_json, updated_at=excluded.updated_at
                    """, (name, p.read_text(), datetime.now(timezone.utc).isoformat()))
                    cx.commit()
    except Exception as e:
        print(f"[CRON] Exception: {e}")


def _start_scheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(_run_cron, "interval", hours=1, id="console_push",
                          next_run_time=datetime.now(timezone.utc))
        scheduler.start()
        print("[CRON] Scheduler started — hourly push active")
    except Exception as e:
        print(f"[CRON] Scheduler failed to start: {e}")

# Only start scheduler on Render (DATA_DIR is set), not in local dev
if os.environ.get("DATA_DIR"):
    _start_scheduler()


# ── Command Center Dashboard ──────────────────────────────────────────────────
from dashboard import require_console_key, ok, fail
from dashboard import money as _money
from dashboard import brain as _brain
from dashboard import pinecone_stats as _pc_stats
from dashboard import ghl as _ghl
from dashboard import scoreapp as _scoreapp
from dashboard import heygen as _heygen
from dashboard import facebook as _fb
from dashboard import health as _health


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(STATIC, "dashboard.html")


@app.route("/api/money/today")
@require_console_key
def api_money_today():
    try: return ok(_money.today_summary())
    except Exception as e: return fail(e)


@app.route("/api/money/week")
@require_console_key
def api_money_week():
    try: return ok(_money.week_summary())
    except Exception as e: return fail(e)


@app.route("/api/money/banks")
@require_console_key
def api_money_banks():
    try: return ok(_money.qb_banks())
    except Exception as e: return fail(e)


@app.route("/api/money/wise")
@require_console_key
def api_money_wise():
    try: return ok(_money.wise_data())
    except Exception as e: return fail(e)


@app.route("/api/brain")
@require_console_key
def api_brain_get():
    try: return ok(_brain.read_brain())
    except Exception as e: return fail(e)


@app.route("/api/brain/upload", methods=["POST"])
@require_console_key
def api_brain_upload():
    try: return ok(_brain.write_brain(request.get_data()))
    except Exception as e: return fail(e, status=400)


@app.route("/api/pinecone/stats")
@require_console_key
def api_pinecone_stats():
    try: return ok(_pc_stats.index_stats())
    except Exception as e: return fail(e)


@app.route("/api/ghl/pipeline")
@require_console_key
def api_ghl_pipeline():
    try: return ok(_ghl.opportunities_by_stage())
    except Exception as e: return fail(e)


@app.route("/api/scoreapp/recent")
@require_console_key
def api_scoreapp_recent():
    try: return ok(_scoreapp.recent_signups())
    except Exception as e: return fail(e)


@app.route("/api/heygen/recent")
@require_console_key
def api_heygen_recent():
    try: return ok(_heygen.recent_videos())
    except Exception as e: return fail(e)


@app.route("/api/facebook/boulder-test")
@require_console_key
def api_facebook_boulder():
    try: return ok(_fb.boulder_test_stats())
    except Exception as e: return fail(e)


@app.route("/api/health")
@require_console_key
def api_dashboard_health():
    try: return ok(_health.status_grid())
    except Exception as e: return fail(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
