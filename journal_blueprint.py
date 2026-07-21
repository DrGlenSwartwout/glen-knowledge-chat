"""
Voice Journal — Flask blueprint to bolt onto glen-knowledge-chat (T2 build).

Pipeline:
    audio  -->  Whisper (verbose_json + word timestamps)  -->  transcript + words[]
                                                                |
                                            Lexical features  <-+
                                                                |
                Haiku 4.5 (cached system prompt) <--- transcript + lexical context
                                                                |
                                       48 emotions + 5 elements + 3 treasures
                                       + polyvagal + congruence + themes
                                                                |
                              tcm_mapper.compare_haiku_to_mapper  (QA cross-check)
                                                                |
                       OpenAI ada-002 embedding of transcript --+
                                                                v
                                                    Supabase journal_entries

Endpoints:
    POST /journal/analyze   — multipart audio upload → full pipeline → JSON
    GET  /journal/today     — most recent entry (last 24h)
    GET  /journal/history   — last 30 days of entries (heatmap data)
    GET  /journal           — serves journal.html

Integration (glen-knowledge-chat side):
    1. Copy this file + tcm_mapper.py + journal.html to the Render repo.
    2. In app.py (after gevent monkey-patch and after `app = Flask(...)` is defined):
           from journal_blueprint import journal_bp
           app.register_blueprint(journal_bp)
    3. Doppler env vars required:
           OPENAI_API_KEY        (existing — for Whisper + ada-002)
           ANTHROPIC_API_KEY     (existing — for Haiku 4.5)
           SUPABASE_URL          (existing — intelligence-engine project)
           SUPABASE_KEY          (existing — sb_secret_* service role key)
    4. Apply schema.sql in the intelligence-engine Supabase project (creates
       the vector extension and the journal_entries table).

Note on Hume: T1 scoped Hume Prosody as the central signal source. Hume is
sunsetting Expression Measurement on 2026-06-14, so we jumped straight to T2.
Hume's 48-emotion vocabulary is preserved as Haiku's output schema, which
keeps tcm_mapper.py useful (QA cross-check) and future-proofs a successor swap.
"""

import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from flask import Blueprint, jsonify, request, send_from_directory

from tcm_mapper import compare_haiku_to_mapper
from dashboard import journal_store, db
from dashboard.tcm_analysis import (
    ANTHROPIC_MESSAGES,
    HAIKU_MODEL,
    HUME_48_EMOTIONS,
    HAIKU_SYSTEM_PROMPT,
    ANALYSIS_TOOL,
    _haiku_analyze,
    _extract_json,
)

log = logging.getLogger(__name__)

journal_bp = Blueprint("journal", __name__)

OPENAI_TRANSCRIPTIONS = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_EMBEDDINGS     = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL       = "text-embedding-ada-002"

# D7 — Remedy match config
PINECONE_INDEX_NAME   = "remedy-match-llc"
REMEDY_NAMESPACES     = ["e4l-protocols", "specific-formulations"]
REMEDY_TOP_K_PER_NS   = 4
REMEDY_FINAL_TOP_K    = 5

HERE = Path(__file__).parent
# Journal entries persist in the app's local sqlite (same chat_log.db as the rest
# of the app), re-homed from the now-dead Supabase project. Mirrors app.py's LOG_DB.
LOG_DB = Path(os.environ.get("DATA_DIR", str(HERE))) / "chat_log.db"


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@journal_bp.route("/journal", methods=["GET"])
def journal_page():
    return send_from_directory(HERE, "journal.html")


@journal_bp.route("/journal/trends", methods=["GET"])
def journal_trends_page():
    return send_from_directory(HERE, "journal_trends.html")


# ---------------------------------------------------------------------------
# POST /journal/analyze
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "missing audio file"}), 400

    audio_file = request.files["audio"]
    duration = float(request.form.get("duration_seconds", 0) or 0)
    retain_audio = request.form.get("retain_audio", "false").lower() == "true"
    is_test = request.form.get("test", "false").lower() == "true"
    parent_entry_id = request.form.get("parent_entry_id") or None
    entry_type = request.form.get("entry_type") or ("affirmation_reading" if parent_entry_id else "journal")

    suffix = Path(audio_file.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        audio_file.save(tf.name)
        audio_path = tf.name

    try:
        whisper = _whisper_transcribe(audio_path)
    except Exception as e:
        log.exception("Whisper failed")
        os.unlink(audio_path)
        return jsonify({"error": f"transcription failed: {e}"}), 502

    transcript = whisper.get("text", "").strip()
    words = whisper.get("words", []) or []

    if not retain_audio:
        os.unlink(audio_path)

    if not transcript:
        return jsonify({"error": "empty transcript"}), 422

    lexical = _lexical_features(words, duration)

    # Haiku analysis (best-effort; degrade to lexical-only if it fails)
    try:
        haiku = _haiku_analyze(transcript, lexical)
    except Exception as e:
        log.exception("Haiku analysis failed")
        haiku = {"error": str(e)}

    # ada-002 embedding (best-effort)
    embedding = None
    try:
        embedding = _embed_ada002(transcript)
    except Exception as e:
        log.exception("Embedding failed")

    # QA cross-check (only if Haiku returned both pieces)
    mapper_check = None
    if (isinstance(haiku, dict)
            and isinstance(haiku.get("emotions"), dict)
            and isinstance(haiku.get("elements"), dict)):
        try:
            mapper_check = compare_haiku_to_mapper(haiku["emotions"], haiku["elements"])
        except Exception:
            log.exception("Mapper QA cross-check failed")

    # Build response payload (also what we persist)
    elements = (haiku or {}).get("elements") or {}
    treasures = (haiku or {}).get("treasures") or {}
    dominant_element = max(elements, key=elements.get) if elements else None
    dominant_treasure = max(treasures, key=treasures.get) if treasures else None

    top_emotions = _top_n_emotions(haiku.get("emotions") or {}, n=3)

    recorded_at_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "user_id": "glen",
        "recorded_at": recorded_at_iso,
        "duration_seconds": duration,
        "transcript": transcript,
        "emotion_scores": haiku.get("emotions"),
        "tcm_scores": {
            "elements": elements,
            "treasures": treasures,
            "treasure_confidence": haiku.get("treasure_confidence") or {},
        },
        "dominant_element": dominant_element,
        "dominant_treasure": dominant_treasure,
        "top_emotions": top_emotions,
        "polyvagal_state": haiku.get("polyvagal_state"),
        "congruence": haiku.get("congruence"),
        "lexical_metrics": lexical,
        "top_themes": haiku.get("top_themes"),
        "transcript_embedding": embedding,
        "mapper_check": mapper_check,
        "metadata": _build_metadata(words, is_test, parent_entry_id, entry_type),
    }

    save_error = None
    saved_id = None
    try:
        with db.connect(LOG_DB) as cx:
            saved = journal_store.insert(cx, record)
        if isinstance(saved, list) and saved:
            saved_id = saved[0].get("id")
    except Exception as e:
        log.exception("journal insert failed")
        save_error = str(e)

    response = {
        "id": saved_id,
        "transcript": transcript,
        "recorded_at": recorded_at_iso,
        "duration_seconds": duration,
        "entry_type": entry_type,
        "parent_entry_id": parent_entry_id,
        "top_emotions": top_emotions,
        "element_scores": elements,
        "dominant_element": dominant_element,
        "treasure_scores": treasures,
        "treasure_confidence": haiku.get("treasure_confidence") or {},
        "dominant_treasure": dominant_treasure,
        "polyvagal_state": haiku.get("polyvagal_state"),
        "congruence": haiku.get("congruence"),
        "lexical_metrics": lexical,
        "top_themes": haiku.get("top_themes"),
        "mapper_qa": mapper_check,
    }
    if save_error:
        response["save_error"] = save_error
    if "error" in (haiku or {}):
        response["analysis_error"] = haiku["error"]
    if is_test:
        response["test"] = True
    return jsonify(response)


def _build_metadata(words, is_test, parent_entry_id=None, entry_type=None):
    md = {}
    if words:
        md["word_timestamps"] = words
    if is_test:
        md["test"] = True
    if parent_entry_id:
        md["parent_entry_id"] = parent_entry_id
    if entry_type and entry_type != "journal":
        md["entry_type"] = entry_type
    return md or None


# ---------------------------------------------------------------------------
# GET /journal/today
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/today", methods=["GET"])
def today():
    include_test = request.args.get("include_test", "false").lower() == "true"
    include_followups = request.args.get("include_followups", "false").lower() == "true"
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with db.connect(LOG_DB) as cx:
        rows = journal_store.select(cx, since_iso=cutoff, order="desc", limit=10)
    if not include_test:
        rows = [r for r in rows if not (r.get("metadata") or {}).get("test")]
    if not include_followups:
        rows = [r for r in rows if (r.get("metadata") or {}).get("entry_type") != "affirmation_reading"]
    return jsonify(rows[0] if rows else {})


# ---------------------------------------------------------------------------
# GET /journal/history
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/history", methods=["GET"])
def history():
    include_test      = request.args.get("include_test", "false").lower() == "true"
    include_followups = request.args.get("include_followups", "false").lower() == "true"
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with db.connect(LOG_DB) as cx:
        rows = journal_store.select(cx, since_iso=cutoff, order="asc")
    test_rows     = [r for r in rows if (r.get("metadata") or {}).get("test")]
    followup_rows = [r for r in rows if (r.get("metadata") or {}).get("entry_type") == "affirmation_reading"]
    if not include_test:
        rows = [r for r in rows if not (r.get("metadata") or {}).get("test")]
    if not include_followups:
        rows = [r for r in rows if (r.get("metadata") or {}).get("entry_type") != "affirmation_reading"]
    return jsonify({
        "entries": rows,
        "count": len(rows),
        "test_count": len(test_rows),
        "followup_count": len(followup_rows),
        "include_test": include_test,
        "include_followups": include_followups,
    })


# ---------------------------------------------------------------------------
# D7 — POST /journal/match (Pinecone remedy match)
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/match", methods=["POST"])
def match_remedies():
    """Given an entry's transcript + state, returns top remedy matches from
    Pinecone (e4l-protocols + specific-formulations namespaces, dedup'd, ranked)."""
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    dom_element = data.get("dominant_element")
    dom_treasure = data.get("dominant_treasure")

    if not transcript:
        return jsonify({"error": "missing transcript"}), 400

    # Build query — embed transcript + state context for richer match
    parts = [transcript]
    if dom_element:
        parts.append(f"Current TCM state: {dom_element}-dominant element pattern.")
    if dom_treasure:
        parts.append(f"{dom_treasure}-prominent treasure (San Bao depth axis).")
    query_text = "\n".join(parts)

    try:
        query_vec = _embed_ada002(query_text)
    except Exception as e:
        log.exception("Match embed failed")
        return jsonify({"error": f"embedding failed: {e}"}), 502

    try:
        idx = _pinecone_index()
    except Exception as e:
        log.exception("Pinecone init failed")
        return jsonify({"error": f"pinecone init failed: {e}"}), 502

    raw_matches = []
    for ns in REMEDY_NAMESPACES:
        try:
            res = idx.query(
                vector=query_vec,
                top_k=REMEDY_TOP_K_PER_NS,
                namespace=ns,
                include_metadata=True,
            )
            for m in (res.matches or []):
                raw_matches.append({
                    "id": m.id,
                    "namespace": ns,
                    "score": float(m.score),
                    "metadata": dict(m.metadata or {}),
                })
        except Exception as e:
            log.exception(f"Pinecone query failed in {ns}")

    # Dedup by id (preserving best score) and take top N
    raw_matches.sort(key=lambda m: m["score"], reverse=True)
    seen, unique = set(), []
    for m in raw_matches:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        unique.append(m)
        if len(unique) >= REMEDY_FINAL_TOP_K:
            break

    return jsonify({
        "matches": unique,
        "query_state": {"dominant_element": dom_element, "dominant_treasure": dom_treasure},
        "namespaces_searched": REMEDY_NAMESPACES,
    })


def _pinecone_index():
    from pinecone import Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=api_key)
    return pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# D7 — POST /journal/affirmations (Haiku-generated, per remedy)
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/affirmations", methods=["POST"])
def generate_affirmations_endpoint():
    """For a list of remedy matches, generates 3 first-person affirmations per
    remedy using Haiku. Best-effort — failures per-remedy degrade gracefully."""
    data = request.get_json(silent=True) or {}
    remedies = data.get("remedies") or []
    transcript = data.get("transcript") or ""
    dom_element = data.get("dominant_element")
    dom_treasure = data.get("dominant_treasure")

    if not remedies or not isinstance(remedies, list):
        return jsonify({"error": "missing remedies[]"}), 400

    out = []
    for r in remedies:
        try:
            affirmations = _haiku_affirmations(r, transcript, dom_element, dom_treasure)
        except Exception as e:
            log.exception(f"Affirmation generation failed for {r.get('id')}")
            affirmations = []
        out.append({
            "remedy_id": r.get("id"),
            "namespace": r.get("namespace"),
            "affirmations": affirmations,
        })

    return jsonify({"affirmations_per_remedy": out})


AFFIRMATION_SYSTEM_PROMPT = """You are generating personalized therapeutic affirmations to be read aloud by a user, immediately after a voice-journal session has revealed their current Traditional Chinese Medicine state.

Each affirmation must:
- Be first person, present tense ("I am...", "I feel...", "My...")
- Be 8-20 words
- Be embodied, sensory, specific (NOT abstract or generic wellness slogans)
- Bridge the user's current TCM state to the remedy's clinical purpose
- Read aloud naturally — vocal cadence and breath rhythm matter
- Land in the user's body, not their head

Generate 3 affirmations per remedy. Each affirmation should target a different angle of the remedy's purpose.

OUTPUT STRICT JSON, NO PROSE: {"affirmations": ["...", "...", "..."]}"""


def _haiku_affirmations(remedy, transcript, dom_element, dom_treasure):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    md = remedy.get("metadata") or {}
    remedy_label = (md.get("name") or md.get("title") or md.get("formulation")
                    or md.get("formulation_name") or remedy.get("id"))
    purpose_keys = ["purpose", "description", "clinical_purpose", "summary", "text", "indication"]
    remedy_purpose = ""
    for k in purpose_keys:
        if md.get(k):
            remedy_purpose = str(md[k])[:600]
            break

    user_message = (
        f"Remedy: {remedy_label}\n"
        f"Remedy purpose / clinical text:\n{remedy_purpose or '(no purpose text in metadata)'}\n\n"
        f"User's current state:\n"
        f"  Dominant element:  {dom_element or '—'}\n"
        f"  Dominant treasure: {dom_treasure or '—'}\n\n"
        f"User's recent transcript: \"{(transcript or '').strip()[:600]}\"\n\n"
        f"Generate 3 first-person affirmations the user will read aloud. JSON only."
    )

    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 512,
        "system": [
            {"type": "text", "text": AFFIRMATION_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    resp = requests.post(
        ANTHROPIC_MESSAGES,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Haiku affirmations {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text").strip()
    parsed = _extract_json(text)
    if not parsed or not isinstance(parsed.get("affirmations"), list):
        raise RuntimeError(f"Haiku returned non-JSON or missing affirmations: {text[:300]}")
    return [str(a).strip() for a in parsed["affirmations"] if str(a).strip()]


# ---------------------------------------------------------------------------
# Whisper (verbose_json + word timestamps)
# ---------------------------------------------------------------------------
def _whisper_transcribe(audio_path: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    with open(audio_path, "rb") as fh:
        resp = requests.post(
            OPENAI_TRANSCRIPTIONS,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": fh},
            data=[
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
            ],
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"Whisper {resp.status_code}: {resp.text[:300]}")
    return resp.json()


# ---------------------------------------------------------------------------
# Lexical features from Whisper word timestamps
# ---------------------------------------------------------------------------
PAUSE_MIN_SEC   = 0.25   # gap above which counts as a pause for density
PAUSE_BREATH_SEC = 0.50  # gap above which we treat as breath-marker


def _lexical_features(words: list, duration_sec: float) -> dict:
    """Compute speech-rate / pause / lexical-diversity features from Whisper
    word-level timestamps.

    Returns:
        {
            "word_count":         int,
            "wpm":                float,    # words / minute
            "pause_density":      float,    # 0–1, fraction of duration in >250ms pauses
            "pause_count":        int,      # number of >500ms pauses (~breath markers)
            "breath_proxy":       float,    # avg words per breath-bounded run
            "type_token_ratio":   float,    # 0–1, lexical diversity
            "median_word_dur_ms": float,    # rough articulation pace marker
        }
    """
    word_count = len(words)
    if word_count == 0 or duration_sec <= 0:
        return {
            "word_count": 0, "wpm": 0.0, "pause_density": 0.0,
            "pause_count": 0, "breath_proxy": 0.0,
            "type_token_ratio": 0.0, "median_word_dur_ms": 0.0,
        }

    wpm = word_count / (duration_sec / 60.0)

    pause_total = 0.0
    pause_count = 0
    runs: list[int] = []
    current_run = 0
    word_durs: list[float] = []

    prev_end = 0.0
    for w in words:
        start = float(w.get("start") or 0)
        end = float(w.get("end") or start)
        word_durs.append(max(0.0, end - start))
        gap = start - prev_end
        if gap >= PAUSE_MIN_SEC:
            pause_total += gap
        if gap >= PAUSE_BREATH_SEC:
            pause_count += 1
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
        current_run += 1
        prev_end = end
    if current_run > 0:
        runs.append(current_run)

    pause_density = min(1.0, pause_total / duration_sec)
    breath_proxy = (sum(runs) / len(runs)) if runs else float(word_count)

    tokens = [
        (w.get("word") or "").strip().lower().strip(".,!?;:'\"")
        for w in words
    ]
    tokens = [t for t in tokens if t]
    ttr = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    median_word_dur_ms = 0.0
    if word_durs:
        sorted_durs = sorted(word_durs)
        mid = len(sorted_durs) // 2
        median_word_dur_ms = round(sorted_durs[mid] * 1000, 1)

    return {
        "word_count": word_count,
        "wpm": round(wpm, 1),
        "pause_density": round(pause_density, 3),
        "pause_count": pause_count,
        "breath_proxy": round(breath_proxy, 1),
        "type_token_ratio": round(ttr, 3),
        "median_word_dur_ms": median_word_dur_ms,
    }


def _top_n_emotions(emotions: dict, n: int = 3) -> list:
    if not isinstance(emotions, dict):
        return []
    items = sorted(
        ((k, float(v)) for k, v in emotions.items() if isinstance(v, (int, float))),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [{"name": k, "score": round(v, 3)} for k, v in items[:n] if v > 0]


# ---------------------------------------------------------------------------
# OpenAI ada-002 embedding (1536 dims)
# ---------------------------------------------------------------------------
def _embed_ada002(transcript: str) -> list:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = requests.post(
        OPENAI_EMBEDDINGS,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": transcript},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Embedding {resp.status_code}: {resp.text[:300]}")
    return resp.json()["data"][0]["embedding"]


# Journal persistence now lives in dashboard/journal_store.py (local sqlite,
# LOG_DB). The former Supabase REST helpers were removed when that project went
# dark — see analyze()/today()/history() which call journal_store directly.
