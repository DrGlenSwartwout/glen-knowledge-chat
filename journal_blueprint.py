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
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from flask import Blueprint, jsonify, request, send_from_directory

from tcm_mapper import compare_haiku_to_mapper

log = logging.getLogger(__name__)

journal_bp = Blueprint("journal", __name__)

OPENAI_TRANSCRIPTIONS = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_EMBEDDINGS     = "https://api.openai.com/v1/embeddings"
ANTHROPIC_MESSAGES    = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL           = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL       = "text-embedding-ada-002"

HERE = Path(__file__).parent


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

    record = {
        "user_id": "glen",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
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
        "metadata": {"word_timestamps": words} if words else None,
    }

    save_error = None
    try:
        _supabase_insert(record)
    except Exception as e:
        log.exception("Supabase insert failed")
        save_error = str(e)

    response = {
        "transcript": transcript,
        "duration_seconds": duration,
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
    return jsonify(response)


# ---------------------------------------------------------------------------
# GET /journal/today
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/today", methods=["GET"])
def today():
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    rows = _supabase_select(
        f"recorded_at=gte.{cutoff}&order=recorded_at.desc&limit=1"
        "&select=id,recorded_at,duration_seconds,transcript,tcm_scores,"
        "dominant_element,dominant_treasure,top_emotions,polyvagal_state,"
        "congruence,lexical_metrics,top_themes"
    )
    return jsonify(rows[0] if rows else {})


# ---------------------------------------------------------------------------
# GET /journal/history
# ---------------------------------------------------------------------------
@journal_bp.route("/journal/history", methods=["GET"])
def history():
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rows = _supabase_select(
        f"recorded_at=gte.{cutoff}&order=recorded_at.asc"
        "&select=recorded_at,duration_seconds,tcm_scores,"
        "dominant_element,dominant_treasure,top_emotions,polyvagal_state"
    )
    return jsonify({"entries": rows, "count": len(rows)})


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


# ---------------------------------------------------------------------------
# Haiku 4.5 — structured TCM analysis
# ---------------------------------------------------------------------------

# 48-emotion vocabulary inherited from Hume's Expression Measurement taxonomy.
# Locked here so future Haiku model upgrades produce comparable score vectors,
# AND so swapping in a Hume-successor (if one emerges) is a single function.
HUME_48_EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness",
    "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment",
    "Craving", "Desire", "Determination", "Disappointment", "Disgust",
    "Distress", "Doubt", "Ecstasy", "Embarrassment", "Empathic Pain",
    "Entrancement", "Envy", "Excitement", "Fear", "Guilt",
    "Horror", "Interest", "Joy", "Love", "Nostalgia",
    "Pain", "Pride", "Realization", "Relief", "Romance",
    "Sadness", "Satisfaction", "Shame", "Surprise (negative)",
    "Surprise (positive)", "Sympathy", "Tiredness", "Triumph",
]

HAIKU_SYSTEM_PROMPT = f"""You are a clinical analysis engine for daily voice-journal entries. Your output is ingested by a Traditional Chinese Medicine (TCM) practitioner's longitudinal-tracking dashboard.

You read the entry through FIVE LAYERS:

1. EMOTION VECTOR — score each of the 48 emotion dimensions on 0.0–1.0 (continuous, not categorical). Most should be near 0; only the genuinely-present emotions register.
   The 48 dimensions: {", ".join(HUME_48_EMOTIONS)}.

2. FIVE ELEMENTS (horizontal axis — emotional/organ resonance).
   • Wood / Liver–Gallbladder — anger, frustration, irritation, determination, drive
   • Fire / Heart–Small Intestine–Pericardium — joy, mania, ecstasy, romance, anxiety-as-overstimulation
   • Earth / Spleen–Stomach — worry, contemplation, rumination, sympathy
   • Metal / Lung–Large Intestine — sadness, grief, regret, nostalgia, disappointment
   • Water / Kidney–Bladder — fear, dread, distress, shame, deep depletion
   Score 0–100 each, sum to 100.

3. THREE TREASURES (vertical axis — constitutional depth).
   • Jing (Essence) — Kidney-rooted hereditary substance. Signals: fundamental (not situational) fatigue, depletion language ("running on fumes for years"), bone/teeth/hair/sexual-vitality complaints, generational/family-line themes, voice low-pitched/monotone/breathy.
   • Qi (Vital Force) — current functional vitality. Signals: situational energy, breath, digestion, immunity, current-stress level. Voice: speech rate, breath support (sentence-length-before-breath), volume.
   • Shen (Spirit) — presence, awareness, joy, meaning, eyes-lit-up. Signals: clarity, awe, contemplative depth, capacity for joy, prosodic vibrancy.
   Score 0–100 each, sum to 100.
   Confidence note: Jing is constitutional and slow-moving. From a single 30s–5min entry, Jing confidence is typically 0.30–0.60 unless multiple deep-fatigue indicators stack. Qi and Shen are confidently scored from one entry (typical 0.70–0.90).

4. POLYVAGAL STATE (autonomic).
   • ventral_vagal — safe, connected, social-engaged, regulated
   • sympathetic — mobilized, activated, fight/flight, urgent
   • dorsal_vagal — collapsed, shut-down, freeze, dissociated
   Score 0–100 each, sum to 100.
   Read both content cues ("I feel safe" / "I'm shutting down" / "I'm wired") AND lexical-pace cues from the metrics provided (high wpm + low pause_density → sympathetic; very low wpm + high pause_density + low type_token_ratio → dorsal; balanced metrics → ventral).

5. INTERNAL CONGRUENCE (transcript-internal).
   Does the speaker contradict themselves within this entry — saying one thing then immediately undermining it? Are stated feelings congruent with the felt-sense the language carries?
   Output: {{"score": 0.0–1.0, "self_contradictions": [...quoted phrases...], "notes": "short clinical observation"}}.

OUTPUT STRICTLY AS A SINGLE JSON OBJECT, NO PROSE BEFORE OR AFTER, with this exact shape:

{{
  "emotions": {{ "<name>": <0–1>, ... all 48 ... }},
  "elements": {{ "Wood": <0–100>, "Fire": <0–100>, "Earth": <0–100>, "Metal": <0–100>, "Water": <0–100> }},
  "treasures": {{ "Jing": <0–100>, "Qi": <0–100>, "Shen": <0–100> }},
  "treasure_confidence": {{ "Jing": <0–1>, "Qi": <0–1>, "Shen": <0–1> }},
  "polyvagal_state": {{ "ventral_vagal": <0–100>, "sympathetic": <0–100>, "dorsal_vagal": <0–100> }},
  "congruence": {{ "score": <0–1>, "self_contradictions": [...], "notes": "..." }},
  "top_themes": [<3–6 short strings>]
}}

Be precise, not poetic. The dashboard renders these numbers directly.
"""


def _haiku_analyze(transcript: str, lexical: dict) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    user_message = (
        f"Lexical metrics from Whisper word timestamps:\n"
        f"  wpm: {lexical.get('wpm')}\n"
        f"  pause_density: {lexical.get('pause_density')}\n"
        f"  pause_count: {lexical.get('pause_count')}\n"
        f"  breath_proxy: {lexical.get('breath_proxy')}  (avg words per breath-bounded run)\n"
        f"  type_token_ratio: {lexical.get('type_token_ratio')}\n"
        f"  word_count: {lexical.get('word_count')}\n"
        f"  median_word_dur_ms: {lexical.get('median_word_dur_ms')}\n\n"
        f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
        f"Produce the JSON now."
    )

    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 2048,
        "system": [
            {
                "type": "text",
                "text": HAIKU_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
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
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Haiku {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    text_blocks = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    raw = "".join(text_blocks).strip()

    parsed = _extract_json(raw)
    if parsed is None:
        raise RuntimeError(f"Haiku returned non-JSON: {raw[:300]}")
    return parsed


def _extract_json(text: str):
    """Tolerant JSON extractor — handles fenced blocks and stray prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


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


# ---------------------------------------------------------------------------
# Supabase REST helpers (intelligence-engine project)
# ---------------------------------------------------------------------------
def _supabase_headers():
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("SUPABASE_KEY not set")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _supabase_url(path: str) -> str:
    base = os.environ.get("SUPABASE_URL")
    if not base:
        raise RuntimeError("SUPABASE_URL not set")
    return f"{base.rstrip('/')}/rest/v1/{path.lstrip('/')}"


def _supabase_insert(record: dict):
    resp = requests.post(
        _supabase_url("journal_entries"),
        headers=_supabase_headers(),
        json=record,
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Supabase insert {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _supabase_select(query: str) -> list:
    resp = requests.get(
        _supabase_url(f"journal_entries?{query}"),
        headers=_supabase_headers(),
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Supabase select {resp.status_code}: {resp.text[:300]}")
    return resp.json()
