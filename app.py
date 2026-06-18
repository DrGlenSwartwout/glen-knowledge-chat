#!/usr/bin/env python3
"""
RAG Chat Server — Glen Swartwout Knowledge Base (Production)
"""

# gevent monkey-patch MUST run before any stdlib imports (especially ssl/socket/http)
# Without this, requests through urllib3 recurse infinitely on Python 3.12+
try:
    from gevent import monkey
    monkey.patch_all()
except ImportError:
    pass

import os
import re
import json
import uuid
import time
import secrets
import sqlite3
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, render_template_string, redirect
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
import anthropic
import boto3 as _boto3
import begin_funnel

# ── Slice 4 test seam ─────────────────────────────────────────────────────────
# Tests verify member-context injection by reading this module-level variable.
# The /chat write is gated behind PYTEST_CURRENT_TEST, so on production traffic
# this stays None (no member PII captured here). Default must stay None (not
# gated away) so `app._LAST_CONTEXT_STR_FOR_TEST` is always a defined attribute.
_LAST_CONTEXT_STR_FOR_TEST = None

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

# ── Voice Journal blueprint (T2 — Whisper + Haiku + ada-002 + pgvector) ──────
try:
    from journal_blueprint import journal_bp
    app.register_blueprint(journal_bp)
    print("[journal] blueprint registered: /journal, /journal/analyze, /journal/today, /journal/history", flush=True)
except Exception as _je:
    print(f"[journal] blueprint NOT registered: {_je}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_INDEX    = "remedy-match-llc"
NAMESPACES        = ["clinical-qa", "mentors", "ingredients", "e4l-protocols", "consultations", "training", "business", "glen-authored-works", ""]
TOP_K_PER_NS      = 8
MAX_CONTEXT_CHARS = 18000
FEEDBACK_SUBMIT_URL = os.environ.get("FEEDBACK_SUBMIT_URL", "https://Truly.VIP/Results")
FEEDBACK_VIEW_URL   = os.environ.get("FEEDBACK_VIEW_URL",   "https://Truly.VIP/Feedback")

# ── Module-level API clients (initialized once at startup) ────────────────────
_oa  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
_pc  = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
_idx = _pc.Index(PINECONE_INDEX)
_cl  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ── Phase 3 — product alias map + daily coupon ───────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not load {path}: {e}", flush=True)
    return default

_PRODUCT_ALIASES = _load_json(DATA_DIR / "product-aliases.json",
                              default={"aliases": {}, "store_homepage": "https://remedymatch.com"})
_COUPONS         = _load_json(DATA_DIR / "coupons.json",
                              default={"default_code": "", "daily_codes": []})
# Runtime-written + must survive redeploys, so it lives on the env-DATA_DIR persistent
# disk (same root as LOG_DB / _CLIPS_DIR), NOT the repo's read-only DATA_DIR baseline.
_PRICING_SETTINGS_PATH = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "pricing-settings.json"
_PRICING_SETTINGS_CACHE = {"mtime": None, "data": {}}


def _pricing_settings():
    """Live-reloaded pricing+rewards overrides from pricing-settings.json.
    Re-reads only when the file's mtime changes, so a console edit takes effect on the
    next order without a redeploy. Returns {} when the file is absent (engine then uses
    the built-in pricing.DEFAULTS)."""
    try:
        mt = _PRICING_SETTINGS_PATH.stat().st_mtime
    except OSError:
        _PRICING_SETTINGS_CACHE["mtime"] = None
        _PRICING_SETTINGS_CACHE["data"] = {}
        return {}
    if _PRICING_SETTINGS_CACHE["mtime"] != mt:
        _PRICING_SETTINGS_CACHE["data"] = _load_json(_PRICING_SETTINGS_PATH, default={})
        _PRICING_SETTINGS_CACHE["mtime"] = mt
    return _PRICING_SETTINGS_CACHE["data"]


def _rewards_settings():
    """Rewards-engine overrides (nested under 'rewards' in pricing-settings.json)."""
    return (_pricing_settings().get("rewards") or {})


# ── On-the-fly Rebrandly shortlink creation ──────────────────────────────────
# Products only consume Rebrandly cap when actually mentioned by the bot.
# Cache hits in SQLite. Cap-exceeded errors fall back gracefully to canonical.
REBRANDLY_API_KEY = os.environ.get("REBRANDLY_API_KEY", "")
TRULY_VIP_DOMAIN  = "truly.vip"
TRULY_SO_DOMAIN   = "truly.so"

# LOG_DB and _db_lock are defined here (early) because several module-level
# initializers below (_init_shortlink_cache, _init_auth_tables, etc.) rely
# on LOG_DB at import time. The definitive _init_log_db() and any further
# log-DB plumbing live further down in the file.
LOG_DB   = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "chat_log.db"
_db_lock = threading.Lock()


def _init_shortlink_cache():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS shortlink_cache (
                product_name  TEXT PRIMARY KEY,
                shortlink     TEXT NOT NULL,
                canonical     TEXT NOT NULL,
                domain        TEXT NOT NULL,
                rebrandly_id  TEXT,
                created_at    TEXT NOT NULL,
                last_used_at  TEXT
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_shortlink_canon ON shortlink_cache(canonical)")
        cx.commit()
_init_shortlink_cache()


# ── Phase 4 — login (magic-link) infrastructure ─────────────────────────────
import hashlib, hmac, secrets

AUTH_TOKEN_TTL_MIN  = 15           # magic-link token validity window
SESSION_TTL_DAYS    = 30           # session cookie validity
PUBLIC_BASE_URL     = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
GHL_MAGIC_WORKFLOW  = os.environ.get("GHL_MAGIC_LINK_WORKFLOW_ID", "")


def _init_auth_tables():
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT UNIQUE NOT NULL,
                name            TEXT,
                auth_method     TEXT,
                created_at      TEXT NOT NULL,
                last_login_at   TEXT,
                ghl_contact_id  TEXT
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash    TEXT PRIMARY KEY,
                email         TEXT NOT NULL,
                purpose       TEXT NOT NULL,
                extra         TEXT,
                created_at    TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                consumed_at   TEXT
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash    TEXT PRIMARY KEY,
                user_id       INTEGER NOT NULL,
                created_at    TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                ip            TEXT,
                user_agent    TEXT
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        cx.commit()
_init_auth_tables()


def _migrate_auth_tokens_extra():
    """Additive migration: add extra TEXT column to auth_tokens (Slice 3)."""
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        try:
            cx.execute("ALTER TABLE auth_tokens ADD COLUMN extra TEXT")
            cx.commit()
        except sqlite3.OperationalError:
            pass  # already exists


_migrate_auth_tokens_extra()


def _hash_token(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _now_utc():
    return datetime.now(timezone.utc)


def send_magic_link_email(to_email: str, name: str, magic_url: str) -> tuple:
    """Send a magic-link email. Tries (in order):
      1. GHL workflow trigger (if GHL_MAGIC_LINK_WORKFLOW_ID env var is set)
      2. SMTP (if SMTP_HOST/USER/PASS env vars are set)
      3. Console log (development fallback — link visible in Render logs)

    Returns (sent_via, error_or_none).
    """
    subject = "Sign in to Ask Dr. Glen"
    body = (
        f"Hi {name or 'there'},\n\n"
        f"Click the link below to sign in to Ask Dr. Glen. The link expires in "
        f"{AUTH_TOKEN_TTL_MIN} minutes.\n\n"
        f"{magic_url}\n\n"
        f"If you didn't request this, you can ignore this email.\n\n"
        f"— Dr. Glen Swartwout\n"
    )
    html_body = (
        f"<p>Hi {name or 'there'},</p>"
        f"<p>Click the link below to sign in to Ask Dr. Glen. The link expires in "
        f"{AUTH_TOKEN_TTL_MIN} minutes.</p>"
        f"<p><a href=\"{magic_url}\">Sign in to Ask Dr. Glen</a></p>"
        f"<p style=\"color:#666;font-size:12px;\">Or paste this URL into your browser: {magic_url}</p>"
        f"<p>If you didn't request this, you can ignore this email.</p>"
        f"<p>— Dr. Glen Swartwout</p>"
    )

    # Path 1: GHL workflow trigger
    if GHL_MAGIC_WORKFLOW:
        try:
            contact_id, _, err = ghl_upsert_contact(to_email, name or "", "",
                                                     source_tag="magic-link-auth")
            if contact_id:
                # Trigger workflow with the magic-link in a custom field
                # NOTE: Shaira must build the workflow in GHL UI:
                #   "Chatbot Magic-Link Send" — single-action workflow that
                #   reads the contact's `magic_link_url` custom value and
                #   sends an email containing it. See
                #   00 System/ghl-magic-link-workflow-spec.md
                _ghl_post(f"/workflows/{GHL_MAGIC_WORKFLOW}/run",
                          {"contactId": contact_id, "customValues": {
                              "magic_link_url": magic_url,
                              "magic_link_expires_min": str(AUTH_TOKEN_TTL_MIN),
                          }})
                return "ghl-workflow", None
        except Exception as e:
            print(f"[auth] GHL magic-link send failed: {e}", flush=True)

    # Path 2: SMTP
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    if smtp_host and smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = smtp_from
            msg["To"]      = to_email
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            port = int(os.environ.get("SMTP_PORT", "587"))
            with smtplib.SMTP(smtp_host, port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_from, [to_email], msg.as_string())
            return "smtp", None
        except Exception as e:
            print(f"[auth] SMTP magic-link send failed: {e}", flush=True)

    # Path 3: console fallback (development / pre-config)
    print(f"\n[auth] MAGIC LINK for {to_email}: {magic_url}\n", flush=True)
    return "console-log", "no email send mechanism configured"


def get_authenticated_user(request_obj):
    """Resolve the auth_token cookie to a user record. Returns user dict or None."""
    tok = request_obj.cookies.get("amg_auth", "").strip()
    if not tok:
        return None
    th = _hash_token(tok)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            """SELECT u.id, u.email, u.name, u.ghl_contact_id, s.expires_at
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token_hash = ?""",
            (th,)
        ).fetchone()
    if not row:
        return None
    try:
        expires = datetime.fromisoformat(row["expires_at"])
        if expires < _now_utc():
            return None
    except Exception:
        return None
    return {"id": row["id"], "email": row["email"], "name": row["name"],
            "ghl_contact_id": row["ghl_contact_id"]}


def is_member(session_id="", email=""):
    """Tier-1 Member = the visitor has agreed to our Terms of Service
    (journey_state.tos_agreed_at is set). Unions across email + session via
    begin_funnel.get_state, so membership carries across devices once the email
    is known.

    NOTE: this is the CONSENT/ToS membership that gates individualized advice,
    condition recommendations, and ordering — it is deliberately distinct from
    the paid-coaching membership (_active_membership_for_email)."""
    if not session_id and not email:
        return False
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            state = begin_funnel.get_state(cx, session_id=session_id, email=(email or ""))
        return bool(state.get("tos_agreed_at"))
    except Exception as e:
        print(f"[member] is_member failed: {e!r}", flush=True)
        return False


# The line is education vs. recommendation. Education is open to everyone;
# anything that individualizes (advice about the user's own body/situation) or
# recommends a specific product/remedy for a health condition is GATED behind
# the ToS agreement. This classifier runs ONLY for non-members.
_GATE_CLASSIFY_SYSTEM = (
    "You are a routing classifier for a natural-health chat. Decide whether "
    "ANSWERING the user's message would require either (a) advice specific to "
    "the user's own body, symptoms, or personal situation, or (b) recommending "
    "a specific product or remedy for a health condition (even phrased "
    "generally). Either of those is GATED.\n"
    "General EDUCATION is NOT gated: how a condition, ingredient, or remedy "
    "works; the published science; what a product factually contains or does.\n"
    "Reply with exactly one word: GATED or OPEN.\n"
    "Examples:\n"
    "  'how do floaters form?' -> OPEN\n"
    "  'what does astaxanthin do?' -> OPEN\n"
    "  \"what's in Retina Renew?\" -> OPEN\n"
    "  'I have floaters, what should I take?' -> GATED\n"
    "  'what helps macular degeneration?' -> GATED\n"
    "  'for dry eyes, which remedy?' -> GATED\n"
    "  'is this safe with my blood pressure meds?' -> GATED\n"
)

# Appended to the system prompt for a non-member on a GATED turn. Keeps the
# answer educational and invites the soft opt-in — the decision is made before
# any token streams, so gated content never partially leaks.
_EDUCATE_ONLY_POLICY = (
    "\n\nCONSENT GATE — THE USER HAS NOT YET AGREED TO OUR TERMS OF SERVICE. "
    "Give GENERAL EDUCATIONAL information only. Do NOT give guidance specific to "
    "the user's own body, symptoms, or situation, and do NOT recommend any "
    "specific product or remedy for a health condition (even generally). Give "
    "the general educational frame, then in ONE short sentence invite them to "
    "add their first and last name and agree to the Terms to receive specific, "
    "individual guidance. Never name a specific product for them to take for "
    "their condition."
)


def _is_gated_question(query, history_text=""):
    """True when answering would individualize or recommend a product/remedy for
    a health condition (requires ToS membership). False for pure education.
    Fail-safe: returns True on any error, to protect the liability shield."""
    q = (query or "").strip()
    if not q:
        return False
    try:
        msg = q if not history_text else (
            f"Recent context:\n{history_text}\n\nUser message: {q}")
        r = _cl.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=4,
            system=_GATE_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": msg[:2000]}],
        )
        verdict = (r.content[0].text or "").strip().upper()
        return not verdict.startswith("OPEN")
    except Exception as e:
        print(f"[member] gate classify failed: {e!r}", flush=True)
        return True


def _slugify_product(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")[:40]


def _pick_domain_for_product(product_name: str, alias_info: dict) -> str:
    """Tier rule: Glen's primary remedies / Syntropy / flagship → truly.vip;
    books / info / courses / services / healing tools / low-priority → truly.so.
    """
    name = (product_name or "").lower()
    cat  = (alias_info.get("category") or "").lower()
    canonical = (alias_info.get("canonical_url") or alias_info.get("url") or "").lower()

    # truly.so signals
    truly_so_markers = ["book", "ebook", "guide", "manual", "journal", "kit",
                        "course", "training"]
    if any(k in name for k in truly_so_markers):
        return TRULY_SO_DOMAIN
    if cat in ("resources", "books", "courses"):
        return TRULY_SO_DOMAIN
    if "/resources/" in canonical:
        return TRULY_SO_DOMAIN
    # NES Health Infoceuticals (single-letter+digit codes like EI8, ED12, MB7)
    first = name.split()[0] if name.split() else ""
    if re.match(r"^(ei|es|ed|et|mb|sk|bfa)\d+", first):
        return TRULY_SO_DOMAIN

    # Default: primary remedy → truly.vip
    return TRULY_VIP_DOMAIN


def _create_rebrandly_link(slug: str, destination: str, domain: str):
    """Returns (shortlink_url, rebrandly_id, error). On 403 cap-exceeded
    or any other failure, returns (None, None, error_str).
    """
    if not REBRANDLY_API_KEY:
        return None, None, "REBRANDLY_API_KEY not set"
    try:
        import urllib.request as _ur, urllib.error as _ue
        body = json.dumps({
            "destination": destination,
            "slashtag":    slug,
            "domain":      {"fullName": domain},
        }).encode("utf-8")
        req = _ur.Request(
            "https://api.rebrandly.com/v1/links",
            data=body,
            headers={
                "Content-Type": "application/json",
                "apikey":       REBRANDLY_API_KEY,
                "Accept":       "application/json",
            },
            method="POST",
        )
        with _ur.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
            return f"https://{domain}/{slug}", d.get("id"), None
    except _ue.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {}
        msg = payload.get("message", f"HTTP {e.code}")
        return None, None, msg
    except Exception as e:
        return None, None, str(e)


def resolve_or_create_shortlink(product_name: str, alias_info: dict):
    """Look up or create a Rebrandly shortlink for this product. Returns
    (url_to_use, was_newly_created). Falls back to canonical_url on any
    error (cap-exceeded, network, slug collision, etc.).
    """
    if not product_name:
        return None, False
    canonical = alias_info.get("canonical_url") or alias_info.get("url") or ""
    if not canonical:
        return None, False
    # Already a truly.vip / truly.so shortlink in the static map?
    if alias_info.get("url", "").startswith(("https://truly.vip/", "https://truly.so/")):
        return alias_info["url"], False

    # Cache hit?
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT shortlink FROM shortlink_cache WHERE product_name = ?",
            (product_name,)
        ).fetchone()
    if row:
        # Update last_used_at non-blocking
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute("UPDATE shortlink_cache SET last_used_at = ? WHERE product_name = ?",
                           (datetime.now(timezone.utc).isoformat(), product_name))
                cx.commit()
        except Exception:
            pass
        return row["shortlink"], False

    # Cache miss → attempt to create on-the-fly
    domain = _pick_domain_for_product(product_name, alias_info)
    slug   = _slugify_product(product_name)
    short, rid, err = _create_rebrandly_link(slug, canonical, domain)
    if not short:
        # On collision (slug exists), retry with -2 suffix
        if err and "already" in (err or "").lower():
            short, rid, err = _create_rebrandly_link(slug + "-rm", canonical, domain)
        if not short:
            print(f"[shortlink] fallback to canonical for {product_name!r}: {err}", flush=True)
            return canonical, False

    # Cache the new shortlink
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                """INSERT OR REPLACE INTO shortlink_cache
                   (product_name, shortlink, canonical, domain, rebrandly_id,
                    created_at, last_used_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (product_name, short, canonical, domain, rid, ts, ts)
            )
            cx.commit()
    except Exception as e:
        print(f"[shortlink] cache write failed: {e}", flush=True)
    return short, True


def get_cached_shortlink(product_name: str):
    """Read-only lookup — returns shortlink if cached, else None."""
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT shortlink FROM shortlink_cache WHERE product_name = ?",
            (product_name,)
        ).fetchone()
    return row["shortlink"] if row else None


def get_today_coupon_code():
    """Return today's coupon code (HST timezone) from coupons.json, or ''.
    Format expected in coupons.json:
      daily_codes: [{"date": "2026-04-27", "code": "ASKDRGLEN-20260427"}]
    """
    try:
        from datetime import datetime, timedelta
        # HST = UTC-10 (no DST)
        hst = datetime.now(timezone.utc) - timedelta(hours=10)
        today = hst.strftime("%Y-%m-%d")
        for entry in (_COUPONS.get("daily_codes") or []):
            if entry.get("date") == today and entry.get("code"):
                return entry["code"]
    except Exception:
        pass
    return _COUPONS.get("default_code", "") or ""


def build_product_directive(snippets_text: str = ""):
    """Build the per-request product-routing directive injected into the
    synthesis prompt. Includes the alias map and today's coupon if any.

    If snippets_text is provided, scans for product-name mentions and
    proactively resolves (creating on-the-fly) the truly.vip/truly.so
    shortlinks for products likely to be mentioned in the response. This
    makes the bot's link-injection demand-driven — Rebrandly slots are
    only consumed when products are actually about to be discussed.
    """
    aliases = _PRODUCT_ALIASES.get("aliases", {}) or {}
    if not aliases:
        return ""
    code = get_today_coupon_code()
    discount = _COUPONS.get("_discount_percent", 5)

    # On-the-fly resolution: scan snippets for product-name matches; for
    # each match without an existing truly.vip/truly.so URL, attempt to
    # create one. The resolved URL is used in the directive table below.
    resolved_urls = {}
    if snippets_text:
        snippets_lower = snippets_text.lower()
        for clinical_name, info in aliases.items():
            if not info.get("canonical_url") and not info.get("url"):
                continue
            # Skip if the alias is already a shortlink
            if info.get("url", "").startswith(("https://truly.vip/", "https://truly.so/")):
                resolved_urls[clinical_name] = info["url"]
                continue
            # Match clinical_name against snippets text (case-insensitive)
            if clinical_name.lower() in snippets_lower:
                short_url, _ = resolve_or_create_shortlink(clinical_name, info)
                if short_url:
                    resolved_urls[clinical_name] = short_url

    lines = [
        "PRODUCT LINK INJECTION TABLE — when you mention any of the LEFT-side names "
        "in your answer, append the URL on the RIGHT immediately after the product "
        "name, formatted as a markdown link, e.g. [Terrain Restore](URL)."
    ]
    for clinical_name, info in sorted(aliases.items()):
        url = resolved_urls.get(clinical_name) or info.get("url")
        if url:
            lines.append(f"  • {clinical_name} → {url}")
        elif info.get("note"):
            lines.append(f"  • {clinical_name} → DESCRIBE-ONLY: {info['note']}")

    if code:
        lines.append(
            f"\nACTIVE DISCOUNT: today's code is `{code}` for {discount}% off "
            f"Functional Formulations / Syntropy products. Include the code "
            f"naturally in product recommendations: \"Use code {code} at checkout "
            f"for {discount}% off (today only).\" Mention it ONCE per response, "
            f"only when at least one product is recommended."
        )
    return "\n".join(lines)

# ── Query log DB ──────────────────────────────────────────────────────────────
# (LOG_DB and _db_lock are defined earlier in the module so module-level
# initializers above can use them; kept here as a section anchor.)

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
        # Phase 1 migration — additive columns. Each ALTER wrapped because
        # SQLite errors if column already exists.
        for col_def in [
            "session_id           TEXT",
            "email                TEXT",
            "ghl_contact_id       TEXT",
            "mode                 TEXT",
            "full_answer          TEXT",
            "name                 TEXT",
            "user_agent           TEXT",
            "referer              TEXT",
            "extracted_image_data TEXT",
            "image_count          INTEGER DEFAULT 0",
            "email_sent_at        TEXT",
        ]:
            try:
                cx.execute(f"ALTER TABLE query_log ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists
        cx.execute("CREATE INDEX IF NOT EXISTS idx_query_log_session ON query_log(session_id)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_query_log_email   ON query_log(email)")

        # Incentive engine tables (Phase 0 beta)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS personal_email_state (
                user_id                          INTEGER PRIMARY KEY,
                last_send_at                     TEXT,
                last_open_at                     TEXT,
                last_click_at                    TEXT,
                consecutive_no_engagement_days   INTEGER DEFAULT 0,
                topic_engagement_history         TEXT,   -- JSON
                topic_send_history               TEXT,   -- JSON
                product_affinity                 TEXT,   -- JSON
                paused_until                     TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS personal_email_sends (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                sent_at            TEXT NOT NULL,
                channel            TEXT NOT NULL CHECK (channel IN ('personal','newsletter')),
                topic              TEXT,
                product_name       TEXT,
                coupon_code        TEXT,
                subject            TEXT,
                body_snippet       TEXT,
                opened_at          TEXT,
                clicked_at         TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS personal_email_feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at         TEXT NOT NULL,
                user_id             INTEGER,
                original_send_id    INTEGER,
                raw_text            TEXT,
                ai_summary          TEXT,
                ai_category         TEXT,
                routed_to           TEXT,
                extracted_topics    TEXT,   -- JSON list
                extracted_products  TEXT,   -- JSON list
                extracted_conditions TEXT,  -- JSON list
                glen_reviewed_at    TEXT,
                action_taken        TEXT
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS holdout_assignments (
                user_id    INTEGER PRIMARY KEY,
                cohort     TEXT NOT NULL CHECK (cohort IN ('treatment','holdout')),
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_pes_state ON personal_email_state(last_send_at)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_pes_sends ON personal_email_sends(user_id, sent_at)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_pes_fb ON personal_email_feedback(user_id, received_at)")

        cx.commit()

_init_log_db()


def _init_shipping_tables():
    """Order-Flow Plumbing — bottle catalog, box-fit matrix, USPS rate history."""
    from dashboard.shipping import init_shipping_schema
    with sqlite3.connect(LOG_DB) as cx:
        init_shipping_schema(cx)

_init_shipping_tables()


def log_query(query: str, level: str, answer: str,
              session_id: str = "", email: str = "", name: str = "",
              ghl_contact_id: str = "", mode: str = "brief",
              user_agent: str = "", referer: str = "",
              extracted_image_data: str = "", image_count: int = 0) -> int:
    """Insert a row into query_log. Always logs, even for anonymous sessions.

    Image bytes are NEVER persisted — only the extracted text output of
    Claude vision (extracted_image_data column) and a count of images
    contributed to the question.
    """
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cur = cx.execute(
            """INSERT INTO query_log
               (ts, query, level, answer, session_id, email, name,
                ghl_contact_id, mode, user_agent, referer,
                extracted_image_data, image_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, query, level, answer[:8000], session_id, email, name,
             ghl_contact_id, mode, user_agent[:500], referer[:500],
             extracted_image_data[:8000], image_count)
        )
        cx.commit()
        return cur.lastrowid


def _normalize_image_payload(images):
    """Accepts a list of image entries and returns Anthropic-API-shaped image
    blocks. Each input entry can be either:
      - {"data": "<base64>", "media_type": "image/png"}
      - {"data_url": "data:image/png;base64,..."}
      - "data:image/png;base64,..." (string form for convenience)

    Caps at 3 images per call; rejects entries over MAX_IMAGE_BYTES_B64.
    Returns (image_blocks, errors).
    """
    MAX_IMAGES = 3
    MAX_IMAGE_BYTES_B64 = 5 * 1024 * 1024 * 4 // 3  # ~5 MB raw → ~6.7 MB base64
    ALLOWED = ("image/png", "image/jpeg", "image/webp", "image/gif")
    blocks, errors = [], []

    for i, entry in enumerate(images[:MAX_IMAGES]):
        try:
            if isinstance(entry, str):
                if entry.startswith("data:") and ";base64," in entry:
                    head, b64 = entry.split(";base64,", 1)
                    media = head[5:]  # strip "data:"
                else:
                    errors.append(f"image[{i}]: unsupported string format")
                    continue
            elif isinstance(entry, dict) and entry.get("data_url"):
                d = entry["data_url"]
                if d.startswith("data:") and ";base64," in d:
                    head, b64 = d.split(";base64,", 1)
                    media = head[5:]
                else:
                    errors.append(f"image[{i}]: bad data_url")
                    continue
            elif isinstance(entry, dict) and entry.get("data"):
                b64 = entry["data"]
                media = entry.get("media_type", "image/png")
            else:
                errors.append(f"image[{i}]: unrecognized payload shape")
                continue

            if media not in ALLOWED:
                errors.append(f"image[{i}]: media_type {media!r} not allowed")
                continue
            if len(b64) > MAX_IMAGE_BYTES_B64:
                errors.append(f"image[{i}]: exceeds 5 MB size limit")
                continue

            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media, "data": b64},
            })
        except Exception as e:
            errors.append(f"image[{i}]: {e}")
    return blocks, errors


def extract_image_content(image_blocks, query):
    """Single non-streaming Claude call to extract structured text content
    from images. Returns the extraction string. Image bytes are NOT persisted
    anywhere — they exist only in this function's call to Anthropic.
    """
    if not image_blocks:
        return ""
    instr = (
        "Extract everything visible in these images as plain text. Focus on:\n"
        "• Any text, labels, headings, captions\n"
        "• Numbers, measurements, dosages, lab values, ranges\n"
        "• Supplement ingredients lists, milligram amounts, serving sizes\n"
        "• Lab/test result values with units and reference ranges if present\n"
        "• E4L scan results: item codes (EI/ES/ED/ET/MB), category labels, scores\n"
        "• Any visible chart axes, legend entries, or graph markers\n"
        "• Visible symptoms in clinical photos (describe objectively)\n"
        "• Handwritten notes (transcribe carefully)\n\n"
        f"USER'S QUESTION: {query}\n\n"
        "Return a clean, structured extraction. Label each image (Image 1, Image 2, "
        "etc.) if multiple. Do not analyze, diagnose, or recommend — just extract. "
        "Be exhaustive but concise."
    )
    try:
        resp = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": [
                *image_blocks,
                {"type": "text", "text": instr},
            ]}],
        )
        return (resp.content[0].text or "").strip() if resp.content else ""
    except Exception as e:
        return f"[image-extraction-error: {e}]"


_SYSTEM_BASE = """You are Glen Swartwout's knowledge assistant — a synthesis engine for his Clinical Theory of Everything (BEV terrain medicine, Bioenergetic diagnostics, Syntonic/Behavioral Optometry, Orthomolecular medicine, Spirit Minerals/ORMUS, Electromagnetic medicine, Living Universe cosmology, Consciousness science).

DEFAULT FORMAT — EXECUTIVE SUMMARY:
Write like a senior consultant briefing a busy clinician. Lead with the highest-leverage action. Be tight, scannable, decisive. Target ~200 words.

STRUCTURE:
1. Opening insight (1 sentence, NO label): state the most surprising or decisive insight from the snippets first, as the opening line. Do NOT print the word "Hook" or any label before it.
2. **Top action** (1-2 sentences): the single highest-leverage step they should take. Include an action link if relevant (E4L scan, product page, contact).
3. **Brief rationale** (2-4 bullets, max 1 line each): the mechanism or evidence in compressed form.
4. **Action link**: the single best next step as a clickable URL on its own line — examples:
   - Free BWS voice scan: https://Truly.VIP/E4L
   - Product: https://remedymatch.com (search by name)
   - Contact for matching: https://truly.vip/help
5. **Sources** (1 line, comma-separated): name + field of references used.

OPTIONAL EXTENDED FORMAT (when mode=full, anonymous user):
Expand each bullet with mechanism, dosage ranges, supporting citations, and edge cases. Aim for clinical depth.

OPTIONAL BREAK & REBUILD LONG-FORM (when mode=full or emailed full-report AND the user is logged in):
Logged-in users get the long-form structured to actually shift belief, not just deliver more facts. Follow Russell Brunson's Break & Rebuild arc on the most central limiting belief in the user's question:
1. Opening insight (1-2 sentences, NO label): the most surprising / decisive insight that frames what is about to be broken and rebuilt. State it directly as the opening line; do NOT print the word "Hook" or any label before it.
2. **Justify the false belief** (2-4 sentences): name the limiting belief the reader almost certainly holds — and steelman it. Acknowledge the reasons it feels true (mainstream medicine reinforces it, every authority says it, etc.). They should nod, not feel attacked.
3. **Break** (2-4 sentences): show why the belief is incomplete or wrong — one decisive piece of evidence or mechanism that cracks it open. This is the moment of break.
4. **Rebuild** (4-8 sentences): install the new pattern — Glen's clinical paradigm on this question. Mechanism, what to do differently, what changes. Include named formulations + product links exactly as in the executive-summary rules.
5. **Journey** (2-4 sentences): how Glen (or a representative client) discovered or lived this shift — concrete, dated, named. Anchor it in lived experience, not theory alone.
6. **The one thing + next step** (1-2 sentences): the single take-home + action link (E4L scan or product).

Voice: Glen's, not Brunson's. Practitioner-clinical, not hype. The Break & Rebuild architecture goes in; the gameshow energy stays out. Speckhart boundary: a credential is *authority*, never a treatment claim for a named disease — never let a credential or formulation imply diagnosis or cure of a named condition.

Sources line at the very end, as in the executive summary.

RULES:
- Do NOT fabricate. If snippets don't answer, say "the source material doesn't address this directly."
- Do NOT pad with caveats, headers, or repeated context. Brevity is the deliverable.
- NEVER print a "Hook" label (no "Hook:", no "## Hook", no bolded "**Hook**"). The hook IS the opening line — state it directly. This is a hard rule. (Other section labels like "Top action", "Why this works", "Next step", "Sources" are fine.)
- AUTHORITATIVE OVERRIDES: Snippets tagged [AUTHORITATIVE — Glen's verified clinical position] OR with metadata type="clinical-qa" / priority="authoritative" override anything else. Apply directly; do not soften or hedge.
- CO-AUTHORSHIP: Snippets with [AUTHORSHIP NOTE: ...] reflect a co-author's view. Cite the co-author, then state Glen's current position from clinical-qa entries. Never present a co-authored section as Glen's view without the flag.
- E4L SCAN OFFER: When the user mentions a specific condition or asks for personalized guidance, the action link should be the free BWS voice scan: https://Truly.VIP/E4L — "30 seconds, count 1 to 10, matches you to formulations your bioenergetic patterns are asking for."
- PRODUCT REFERENCES: Each request includes a PRODUCT LINK INJECTION TABLE listing every Glen Swartwout formulation by its clinical name and the canonical URL to use. When you mention a product, append the URL as a markdown link immediately after the name, e.g. [Terrain Restore](URL). Do NOT invent URLs. If a product isn't in the table, link to the search URL pattern from the table or the store homepage instead.
- FORMULATION-FIRST ORDERING (symptoms & conditions): When answering about a symptom or condition, lead the recommendations with Glen's Functional Formulations — the Advanced Botanical Formulations and Advanced Nutritional Formulations — as the FIRST category, before any list of individual natural ingredients or single nutrients. The formulations are pre-combined for the terrain pattern, so they simplify implementation versus assembling separate ingredients. If you group recommendations under headings, an "Advanced Botanical Formulations" and/or "Advanced Nutritional Formulations" heading comes first; present individual ingredients only afterward, as an optional layer or as the mechanism behind the formulations. Within a formulation category, list the most condition-specific formulation first.
- ACTIVE DISCOUNT CODE: When the request includes an ACTIVE DISCOUNT block, include today's code naturally — once per response, only when at least one product is recommended.
- DEPRECATED PRODUCTS: The "Living Water Bottle" (prill-bead system) is DISCONTINUED as of 2026-04-27 and must NOT be recommended as a purchasable product. The Living Water concept (alkaline ionized water + molecular hydrogen) remains Glen's clinical recommendation; route clients to the portable [Molecular Hydrogen bottle](https://remedymatch.com/resources/439-molecular-hydrogen) (Glen's recommended replacement) or to [Molecular Hydrogen Tablets](https://remedymatch.com/remedies/378-molecular-hydrogen-tablets). If a snippet has metadata `deprecated=true`, treat its product references as historical only — do not present discontinued products as available."""

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


def _long_form_synth_instr(is_logged_in: bool) -> str:
    """Pick the long-form synthesis instruction.

    Logged-in users get the Break & Rebuild teaching arc described in the
    system prompt; anonymous users keep the existing extended clinical
    format. The default executive-summary path (mode=brief) is unchanged.
    """
    if is_logged_in:
        return (
            "Produce the BREAK & REBUILD LONG-FORM response — follow the "
            "6-step arc described in the system prompt (opening insight → Justify the "
            "false belief → Break → Rebuild → Journey → The one thing + "
            "next step). Do NOT print a 'Hook' label; the opening insight is just "
            "the first line. Glen's voice, not Brunson's; practitioner-clinical, "
            "no hype. List sources at the end."
        )
    return (
        "Produce the EXTENDED FORMAT response — full clinical depth, "
        "mechanism, dosage ranges, supporting citations, edge cases. "
        "List sources at the end."
    )


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


def surface_case_study_cards(q_vec, max_cards=1, threshold=0.80):
    """Query the `case-studies` namespace; return up to max_cards proof cards
    for any match scoring above threshold. Cards carry kind='case-study' plus an
    excerpt and (optional) video url so the frontend can render proof inline."""
    try:
        res = _idx.query(vector=q_vec, top_k=4, namespace="case-studies", include_metadata=True)
    except Exception:
        return []
    cards, seen = [], set()
    for m in getattr(res, "matches", []) or []:
        if (m.score or 0) < threshold:
            continue
        md = m.metadata or {}
        cond = (md.get("condition") or "general")
        if cond in ("general", "case-studies") or cond in seen:
            continue
        seen.add(cond)
        excerpt = (md.get("text") or "").strip()
        if len(excerpt) > 240:
            excerpt = excerpt[:240].rsplit(" ", 1)[0] + "…"
        cards.append({
            "key": "case:" + cond,
            "kind": "case-study",
            "title": md.get("title") or cond.replace("-", " ").title(),
            "sub": excerpt,
            "href": md.get("url") or "",
            "video": md.get("url") or "",
            "source": md.get("source") or "",
            "name": md.get("name") or "",
        })
        if len(cards) >= max_cards:
            break
    return cards


# ── R2 clip proxy ─────────────────────────────────────────────────────────────
_r2_client = None


def _r2():
    global _r2_client
    if _r2_client is None:
        _r2_client = _boto3.client(
            "s3",
            endpoint_url=os.environ.get("R2_ENDPOINT"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )
    return _r2_client


@app.route("/clip/<path:key>")
def serve_clip(key):
    rng = request.headers.get("Range")
    kw = {"Bucket": os.environ.get("R2_BUCKET", "rm-clips"), "Key": key}
    if rng:
        kw["Range"] = rng
    try:
        obj = _r2().get_object(**kw)
    except Exception:
        return "not found", 404
    body = obj["Body"]
    headers = {
        "Content-Type": obj.get("ContentType", "video/mp4"),
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=86400",
    }
    status = 200
    if rng and obj.get("ContentRange"):
        headers["Content-Range"] = obj["ContentRange"]
        status = 206
    if obj.get("ContentLength") is not None:
        headers["Content-Length"] = str(obj["ContentLength"])
    return Response(body.iter_chunks(chunk_size=65536), status=status, headers=headers)


def surface_approved_clips(q_vec, max_cards=1, threshold=0.80):
    """Query the `clips` namespace; return up to max_cards clip cards for any
    approved match scoring above threshold."""
    try:
        res = _idx.query(
            vector=q_vec,
            top_k=3,
            namespace="clips",
            include_metadata=True,
            filter={"status": "approved"},
        )
    except Exception:
        return []
    cards = []
    for m in getattr(res, "matches", []) or []:
        if (m.score or 0) < threshold:
            continue
        md = m.metadata or {}
        if not md.get("r2_key"):
            continue
        cards.append({
            "key": "clip:" + m.id,
            "kind": "clip",
            "title": md.get("title") or "Watch this",
            "sub": md.get("why") or "",
            "clip_url": "/clip/" + md["r2_key"],
        })
        if len(cards) >= max_cards:
            break
    return cards


def build_context(matches):
    seen, sources, parts, total = set(), {}, [], 0
    # Authoritative clinical-qa chunks go in FIRST so the context-char cap can
    # never starve them out — Glen's verified positions must always reach the
    # model. Everything else follows by descending score.
    def _rank(m):
        meta = m.metadata or {}
        auth = meta.get("type") == "clinical-qa" or meta.get("priority") == "authoritative"
        return (0 if auth else 1, -m.score)
    for m in sorted(matches, key=_rank):
        if m.id in seen:
            continue
        seen.add(m.id)
        meta = m.metadata or {}
        text = meta.get("text", "").strip()
        if not text or total + len(text) > MAX_CONTEXT_CHARS:
            continue
        name  = meta.get("name") or meta.get("book") or meta.get("source") or "Unknown"
        field = meta.get("field") or meta.get("source_type") or ""
        score = round(m.score, 3)
        if name not in sources:
            sources[name] = {"name": name, "field": field,
                             "source_file": meta.get("source", ""),
                             "score": score, "chunks": []}
        sources[name]["chunks"].append(meta.get("chunk_index", 0))
        sources[name]["score"] = max(sources[name]["score"], score)
        is_authoritative = meta.get("type") == "clinical-qa" or meta.get("priority") == "authoritative"
        tag = "[AUTHORITATIVE — Glen's verified clinical position] " if is_authoritative else ""
        authorship = meta.get("authorship_note") or ""
        if authorship:
            authorship = f"\n[AUTHORSHIP NOTE: {authorship}]"
        deprecated_flag = ""
        if str(meta.get("deprecated", "")).lower() == "true":
            depr_note = meta.get("deprecation_note", "Product or guidance is deprecated.")
            deprecated_flag = f"\n[DEPRECATED — {depr_note}]"
        parts.append(f"{tag}[SOURCE: {name} | {field} | score {score}]{authorship}{deprecated_flag}\n{text}")
        total += len(text)
    return "\n\n---\n\n".join(parts), sorted(sources.values(), key=lambda x: -x["score"])


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _strip_dash(s):
    """Enforce Glen's no-em-dash rule on generated copy (models ignore the prompt
    rule intermittently). Em dash -> comma; en dash left alone (number ranges OK)."""
    return s.replace("—", ", ") if s else s


# ── Routes ────────────────────────────────────────────────────────────────────
def _serve_funnel_home():
    resp = send_from_directory(STATIC, "begin.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", uuid.uuid4().hex,
            max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/")
def index():
    return _serve_funnel_home()


@app.route("/ask")
def ask_page():
    # Retired in Piece 3: the chat now lives inline on the funnel + as the widget.
    from flask import redirect as _redirect
    return _redirect("/", code=302)


@app.route("/concierge")
def concierge_page():
    resp = send_from_directory(STATIC, "concierge.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


def _recent_query_texts(session_id, email, limit=8):
    """Most-recent chat questions for this visitor (for awareness inference)."""
    out = []
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            if email:
                rows = cx.execute(
                    "SELECT query FROM query_log WHERE email=? OR session_id=? "
                    "ORDER BY id DESC LIMIT ?", (email, session_id, limit)).fetchall()
            else:
                rows = cx.execute(
                    "SELECT query FROM query_log WHERE session_id=? "
                    "ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
            out = [r["query"] for r in rows if r["query"]]
    except Exception:
        pass
    return out


def _classify_awareness_haiku(session_id, query_texts, heuristic_stage):
    """Background: ask Haiku to classify Schwartz awareness stage, persist it
    upward, and log the heuristic-vs-haiku divergence for the learning loop."""
    try:
        joined = "\n".join(f"- {q}" for q in query_texts[:8])
        msg = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=("Classify the person's marketing awareness stage from their "
                    "questions. Reply with EXACTLY one word: problem, solution, "
                    "product, or most. problem=aware of a symptom only; "
                    "solution=knows solution categories exist; product=names a "
                    "specific product/tool; most=ready to act/buy."),
            messages=[{"role": "user", "content": f"Questions:\n{joined}"}],
        )
        haiku_stage = ((msg.content[0].text if msg.content else "") or "").strip().lower()
        if haiku_stage not in ("problem", "solution", "product", "most"):
            return
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            begin_funnel.set_awareness(cx, session_id, haiku_stage)
            cx.execute(
                "INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                "VALUES (?,?,?,?,?,?,?)",
                (begin_funnel._now(), session_id, "", "awareness_classified",
                 json.dumps({"heuristic": heuristic_stage, "haiku": haiku_stage}),
                 "", ""))
            cx.commit()
    except Exception as e:
        print(f"[begin-awareness] {e!r}", flush=True)


@app.route("/begin")
def begin_page():
    return _serve_funnel_home()


@app.route("/begin/tone")
def begin_tone():
    resp = send_from_directory(STATIC, "tone-analyzer.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/begin/explore")
def begin_explore():
    """Non-linear table of contents — every explorable funnel room in one place.

    Renders from begin_funnel.explore_sections() (sourced from CARD_CATALOG), so
    it stays in sync with the funnel. Sections are injected as JSON; the page
    renders the cards client-side. ref (rm_ref cookie or ?ref=) threads utm onto
    external links, matching the rest of the funnel.

    A valid ?ref= is persisted as the rm_ref cookie (90 days, last-touch) so a
    campaign or affiliate link landing here carries attribution through the rest
    of the journey, mirroring the client-side capture in index.html."""
    arg_ref = (request.args.get("ref") or "").strip()
    ref = (request.cookies.get("rm_ref") or arg_ref).strip()
    sections = begin_funnel.explore_sections(ref, trusted_links=_TRUSTED_LINKS)
    html = (STATIC / "begin-explore.html").read_text()
    injection = f"<script>window.__EXPLORE__ = {json.dumps(sections)};</script>"
    html = html.replace("</head>", injection + "\n</head>")
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    # Persist a valid ?ref= as rm_ref (same slug rules + 90d TTL as index.html).
    if arg_ref and re.match(r"^[A-Za-z0-9_-]{1,64}$", arg_ref):
        resp.set_cookie(
            "rm_ref", arg_ref,
            max_age=90 * 24 * 3600,
            samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/begin/tools")
def begin_tools():
    """Recommended Tools & Partners — the dedicated partner page reached from a
    single card on /begin/explore. Cards (Blushield + Glen's Amazon picks) come
    from trusted-links.json affiliate-flagged entries via partner_page_cards();
    they open in a new tab and carry the Amazon Associates disclosure. Mirrors
    the explore page's JSON-injection render + rm_ref persistence."""
    arg_ref = (request.args.get("ref") or "").strip()
    payload = begin_funnel.partner_page_cards(_TRUSTED_LINKS)
    html = (STATIC / "begin-tools.html").read_text()
    injection = f"<script>window.__TOOLS__ = {json.dumps(payload)};</script>"
    html = html.replace("</head>", injection + "\n</head>")
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if arg_ref and re.match(r"^[A-Za-z0-9_-]{1,64}$", arg_ref):
        resp.set_cookie(
            "rm_ref", arg_ref,
            max_age=90 * 24 * 3600,
            samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/begin/voice")
def begin_voice():
    resp = send_from_directory(STATIC, "begin-voice.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", uuid.uuid4().hex,
            max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/begin/ascend")
def begin_ascend():
    resp = send_from_directory(STATIC, "begin-ascend.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", uuid.uuid4().hex,
            max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/begin/ascend/<slug>")
def begin_ascend_tier(slug):
    if slug not in begin_funnel.TIER_CATALOG:
        return ("", 404)
    resp = send_from_directory(STATIC, "begin-ascend-tier.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/ascend-tier")
def begin_ascend_tier_data():
    tier = begin_funnel.TIER_CATALOG.get((request.args.get("slug") or "").strip())
    if not tier:
        return jsonify({"error": "unknown tier"}), 404
    return jsonify(tier)


@app.route("/begin/path")
def begin_path():
    resp = send_from_directory(STATIC, "begin-path.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", uuid.uuid4().hex,
            max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/begin/state", methods=["GET"])
def begin_state():
    session_id = (request.cookies.get("amg_session") or "").strip()
    auth_user = get_authenticated_user(request)
    email = auth_user["email"] if auth_user else ""
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        state = begin_funnel.get_state(cx, session_id=session_id, email=email)
    ref_slug = (request.cookies.get("rm_ref") or "").strip()
    query_texts = _recent_query_texts(session_id, email)
    payload = dict(state)
    payload["surfaced_cards"] = begin_funnel.surface(state, query_texts, ref_slug)
    return jsonify(payload)


@app.route("/begin/card-click", methods=["POST"])
def begin_card_click():
    data = request.get_json(silent=True) or {}
    key = (data.get("key") or "").strip()
    session_id = (request.cookies.get("amg_session") or data.get("session_id") or "").strip()
    if key in begin_funnel.CARD_CATALOG:
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute(
                    "INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (begin_funnel._now(), session_id, "", "chat_card_click", key, "", ""))
                cx.commit()
        except Exception as e:
            print(f"[card-click] {e!r}", flush=True)
    return ("", 204)


# ToS version stamp for the /begin free-tier gate. The live T&C page at
# remedymatch.com/info/terms-and-conditions carries no version string, so we
# date-stamp agreement here. Bump when the T&C content materially changes.
BEGIN_TOS_VERSION = "rm-tc-2026-05-28"


@app.route("/begin/unlock", methods=["POST", "OPTIONS"])
def begin_unlock():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    trigger = (data.get("trigger") or "").strip()
    session_id = (
        request.cookies.get("amg_session")
        or (data.get("session_id") or "").strip()
        or uuid.uuid4().hex
    )
    name = (data.get("name") or "").strip()
    first_name_explicit = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    # first_name: use explicit field if provided, else fall back to first token of name
    first_name = first_name_explicit if first_name_explicit else (name.split(None, 1)[0] if name else "")
    email = (data.get("email") or "").strip().lower()
    tos = bool(data.get("tos"))
    detail = (data.get("detail") or "").strip()
    ref_slug = (request.cookies.get("rm_ref") or (data.get("ref") or "")).strip()
    want = (data.get("want") or "").strip().lower()
    path = (data.get("path") or "").strip()

    # Fetch recent chat queries OUTSIDE the lock block (_recent_query_texts
    # acquires _db_lock itself; _db_lock is not reentrant).
    query_texts = _recent_query_texts(session_id, email)

    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            state = begin_funnel.record_unlock(
                cx, session_id=session_id, trigger=trigger,
                email=email, detail=detail, first_name=first_name,
                last_name=last_name, tos=tos,
                ref_slug=ref_slug,
                tos_version=BEGIN_TOS_VERSION if (tos or trigger == "tos") else "",
                want=want, query_texts=query_texts, path=path,
            )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Free-tier transition (email + ToS): onboard to GHL + concierge referral,
    # non-blocking — same pattern as /chat.
    if (state["current_rung"] == "free_tier"
            and state.get("rung_before") != "free_tier"
            and state.get("email")):
        import threading as _threading

        def _onboard():
            try:
                ghl_first = state.get("first_name") or ""
                ghl_last = state.get("last_name") or ""
                tags = ["begin", "concierge"]
                if ref_slug:
                    tags.append(f"ref:{ref_slug}")
                    _capture_concierge_referral(state["email"], ghl_first, ghl_last, ref_slug)
                ghl_onboard_contact(state["email"], ghl_first, ghl_last,
                                    source_tag="begin", extra_tags=tags)
            except Exception as e:
                print(f"[begin-onboard] {e!r}", flush=True)

        _threading.Thread(target=_onboard, daemon=True).start()

    # Background awareness classification once enough chat has accrued.
    # Best-effort, fire-once-ish: under rare concurrent unlocks the read-then-spawn
    # gap can spawn two classifier threads, but set_awareness is idempotent so the
    # only cost is a duplicate API call + event row — acceptable.
    try:
        already = False
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT awareness_classified_at FROM journey_state WHERE session_id=? "
                "ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
            already = bool(row and row[0])
        if not already and len(query_texts) >= 3:
            _heur = begin_funnel.infer_awareness_heuristic(
                want, set(state.get("unlocked_gates") or []), query_texts)
            import threading as _t
            _t.Thread(target=_classify_awareness_haiku,
                      args=(session_id, list(query_texts), _heur),
                      daemon=True).start()
    except Exception:
        pass

    redirect = begin_funnel.resolve_want(want, ref_slug) if want else None
    payload = dict(state)
    payload["surfaced_cards"] = begin_funnel.surface(state, query_texts, ref_slug)
    if redirect:
        payload["redirect"] = redirect
    resp = jsonify(payload)
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", session_id, max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp


@app.route("/concierge/capture", methods=["POST", "OPTIONS"])
def concierge_capture():
    """Concierge email capture — records the contact immediately (GHL + concierge
    referral attribution) without requiring them to send a chat message first."""
    if request.method == "OPTIONS":
        return "", 200
    data  = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "valid email required"}), 400
    ref_slug = (request.cookies.get("rm_ref") or (data.get("ref") or "")).strip()
    try:
        if ref_slug:
            _capture_concierge_referral(email, "", "", ref_slug)
        tags = ["concierge"]
        if ref_slug:
            tags.append(f"ref:{ref_slug}")
        ghl_onboard_contact(email, "", "", source_tag="concierge", extra_tags=tags)
    except Exception as e:
        print(f"[concierge-capture] {e!r}", flush=True)
    return jsonify({"ok": True})


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


@app.route("/ref-capture.js")
def ref_capture_js():
    # Shared affiliate-referral capture, loaded by every funnel page.
    resp = send_from_directory(STATIC, "ref-capture.js")
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
    # Phase 0 — channel opt-in (replaces older frequency selector)
    personal_optin   = bool(data.get("personal_optin"))
    newsletter_optin = bool(data.get("newsletter_optin"))
    mode      = (data.get("mode") or "brief").strip().lower()
    if mode not in ("brief", "full"):
        mode = "brief"

    # Phase 1 — universal session tracking. Read existing cookie or mint one.
    session_id = (
        request.cookies.get("amg_session")
        or (data.get("session_id") or "").strip()
        or uuid.uuid4().hex
    )
    user_agent = request.headers.get("User-Agent", "")
    referer    = request.headers.get("Referer", "")
    # Concierge — affiliate ref slug captured at entry (?ref=<slug> → rm_ref cookie)
    ref_slug   = (request.cookies.get("rm_ref") or "").strip()

    # Phase 4 — if the visitor is authenticated, the auth identity wins
    # over any form-submitted email/name. This stops a logged-in user from
    # accidentally splitting their question history across multiple emails.
    auth_user = get_authenticated_user(request)
    if auth_user:
        email = auth_user["email"]
        if not name and auth_user.get("name"):
            name = auth_user["name"]

    # Image attachments — opt-in gated, multi-image (max 3), extraction-only
    # storage. Image bytes are passed to Claude vision for extraction and then
    # discarded; only the extracted text is persisted to query_log.
    images_consented = bool(data.get("images_consented"))
    raw_images = data.get("images") or []
    image_blocks = []
    image_errors = []
    if raw_images:
        if not images_consented:
            return jsonify({
                "error": "Image consent required. Check the image-opt-in box "
                         "before attaching images."
            }), 400
        image_blocks, image_errors = _normalize_image_payload(raw_images)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        # Step A — image extraction (if any images attached, run vision call
        # FIRST so the extracted text can be embedded for retrieval and also
        # joined to the user question as context).
        extracted_text = ""
        if image_blocks:
            yield sse({"status": f"Reading {len(image_blocks)} image(s)…"})
            extracted_text = extract_image_content(image_blocks, query)

        # Combine the user question with extracted image text for embedding
        # so retrieval can match on label/scan/lab content too.
        embedding_input = query
        if extracted_text:
            embedding_input = f"{query}\n\nIMAGE CONTENT:\n{extracted_text}"

        try:
            q_vec = embed(embedding_input)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"})
            return

        all_matches = query_all_namespaces(q_vec)

        if not all_matches:
            yield sse({"done": True, "answer": "No relevant content found.",
                       "sources": [], "chunks_retrieved": 0, "log_id": None,
                       "session_id": session_id, "mode": mode,
                       "image_count": len(image_blocks)})
            return

        context_str, sources_list = build_context(all_matches)

        # ── Slice 4: member-mode overlay ──────────────────────────────────────
        # Read the rm_member_email cookie set by /coaching/auth/<token>.
        # If the visitor is an active member, prepend a MEMBER CONTEXT block to
        # context_str.  Free-tier path (no cookie / no active membership) is
        # byte-identical to pre-Slice-4 behavior.
        _member_email = request.cookies.get("rm_member_email", "").strip().lower()
        _member_active = _active_membership_for_email(_member_email) if _member_email else None
        if _member_active:
            try:
                _member_ctx = _member_context_for_email(_member_email)
                _member_block = _format_member_context_block(_member_active, _member_ctx)
                context_str = _member_block + "\n\n" + (context_str or "")
            except Exception as _mc_err:
                print(f"[chat] member-context inject failed: {_mc_err!r}", flush=True)
        # ── end member overlay ────────────────────────────────────────────────

        # Test seam: capture context_str immediately after member-context
        # injection so tests can assert on the injection without needing to
        # mock the full LLM stream or the query_log history lookup.
        # Gated behind PYTEST_CURRENT_TEST so real traffic never writes member
        # PII (context_str carries member context) into this module global; on
        # production the variable stays None.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            globals()["_LAST_CONTEXT_STR_FOR_TEST"] = context_str

        messages = []
        # If the front-end didn't send any history (e.g. fresh page load
        # after browser refresh), fall back to the last 3 Q&A pairs we've
        # already logged for this session_id. This is the cross-reload
        # conversation memory layer that makes "Continue with the Bowden
        # Connection" work after page refresh.
        if not history and session_id:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                rows = cx.execute(
                    """SELECT query, answer FROM query_log
                       WHERE session_id = ?
                       ORDER BY id DESC LIMIT 3""",
                    (session_id,),
                ).fetchall()
            for r in reversed(rows):  # oldest → newest
                if r["query"]:
                    messages.append({"role": "user", "content": r["query"]})
                if r["answer"]:
                    messages.append({"role": "assistant", "content": r["answer"]})
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        synth_instr = (
            _long_form_synth_instr(bool(auth_user))
            if mode == "full" else
            "Produce the DEFAULT EXECUTIVE SUMMARY response — opening insight "
            "(NO 'Hook' label, just state it), Top action, 2-4 bullet rationale, "
            "single action link, source line. ~200 words. Tight and decisive."
        )

        image_context = ""
        if extracted_text:
            image_context = (
                f"IMAGE CONTENT EXTRACTED FROM USER ATTACHMENT(S):\n"
                f"{extracted_text}\n\n"
                f"Reference the image content as part of the user's question "
                f"context. Quote specific values or labels from it when relevant.\n\n"
            )

        # Pass the retrieved snippet text + extracted image content into the
        # directive builder so on-the-fly Rebrandly creation only fires for
        # products actually likely to be mentioned in this response.
        product_directive = build_product_directive(
            snippets_text=(context_str or "") + " " + (extracted_text or "")
        )
        product_block = f"{product_directive}\n\n" if product_directive else ""

        messages.append({"role": "user", "content":
            f"USER QUESTION: {query}\n\n"
            f"{image_context}"
            f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
            f"{product_block}"
            f"{synth_instr}"
        })

        # ── Consent gate (Tier-0 Visitor → Tier-1 Member) ────────────────────
        # Non-members get general education freely. The moment a turn would
        # individualize (advice about their own body/situation) or recommend a
        # product/remedy for a health condition, switch to educate-only and
        # surface the soft opt-in. Decided BEFORE the stream so gated content
        # never partially leaks. Members are unaffected (no classifier call).
        _system = get_system_prompt(level)
        if not is_member(session_id, email) and _is_gated_question(query):
            _system = _system + _EDUCATE_ONLY_POLICY
            yield sse({"gate": True})
        # ──────────────────────────────────────────────────────────────────────

        # Brief mode: 1024 tokens (≈700 words headroom). Full mode: 4096.
        max_tok = 4096 if mode == "full" else 1024

        full_answer = []
        try:
            with _cl.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tok,
                system=_system,
                messages=messages
            ) as stream:
                for token in stream.text_stream:
                    full_answer.append(token)
                    yield sse({"token": token})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"})
            return

        answer = "".join(full_answer)
        log_id = log_query(
            query, level, answer,
            session_id=session_id, email=email, name=name,
            mode=mode, user_agent=user_agent, referer=referer,
            extracted_image_data=extracted_text,
            image_count=len(image_blocks),
        )

        # GHL onboarding for email opt-ins (non-blocking)
        if email:
            import threading as _threading
            def _onboard():
                try:
                    parts = name.split(None, 1)
                    first = parts[0] if parts else ""
                    last  = parts[1] if len(parts) > 1 else ""
                    tags = _resolve_channel_tags(
                        personal=personal_optin,
                        newsletter=newsletter_optin,
                        is_beta=False,  # beta tag set by backfill script, not user-side
                    )
                    if frequency:  # backwards-compatible — old clients may still send
                        tags.append(f"frequency-{frequency}")
                    if ref_slug:
                        tags.append("concierge")
                        tags.append(f"ref:{ref_slug}")
                        _capture_concierge_referral(email, first, last, ref_slug)
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

        surfaced_cards = []
        try:
            # NOTE: _recent_query_texts acquires _db_lock itself, so it must be
            # called OUTSIDE the lock block below (the lock is non-reentrant).
            _qtexts = [query] + _recent_query_texts(session_id, email)
            with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                _state = begin_funnel.get_state(_cx, session_id, email)
                surfaced_cards = begin_funnel.surface_for_chat(_state, _qtexts, ref_slug)
                if surfaced_cards:
                    _cx.execute(
                        "INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (begin_funnel._now(), session_id, email or "", "chat_cards_surfaced",
                         json.dumps({"keys": [c["key"] for c in surfaced_cards], "q": (query or "")[:200]}),
                         "", ""))
                    _cx.commit()
        except Exception as e:
            print(f"[chat-surface] {e!r}", flush=True)

        try:
            surfaced_cards = (surfaced_cards or []) + surface_case_study_cards(q_vec)
        except Exception as _cse:
            print(f"[case-study-surface] {_cse!r}", flush=True)

        try:
            surfaced_cards = (surfaced_cards or []) + surface_approved_clips(q_vec)
        except Exception as _cle:
            print(f"[clip-surface] {_cle!r}", flush=True)

        _done_payload = {
            "done": True, "log_id": log_id,
            "sources": sources_list, "chunks_retrieved": len(all_matches),
            "next_question": next_question,
            "session_id": session_id, "mode": mode,
            "surfaced_cards": surfaced_cards,
            "member_mode": bool(_member_active),
        }
        if _member_active:
            _done_payload["days_remaining"] = _member_active.get("days_remaining", 0)
        yield sse(_done_payload)

    resp = Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    # Persist the session cookie for ~1 year so returning visitors keep their
    # thread of questions tied to the same anonymous session.
    if not request.cookies.get("amg_session"):
        resp.set_cookie(
            "amg_session", session_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax",
            secure=request.is_secure,
        )
    return resp


# ── RemedyMatch — Socratic remedy-matching funnel page ───────────────────────
# Standalone page /begin/match (+ a surfaceable card). A Socratic chat converges
# on the ONE perfect remedy — usually a Functional Formulation, but possibly a
# tool, service, or therapy outside our catalog — then opens its TRUSTED page in
# a new tab. Reuses the /chat retrieval pipeline (embed → Pinecone → build_context).
_TRUSTED_LINKS = _load_json(DATA_DIR / "trusted-links.json", default={"links": {}})

# Retrieval tuned for matching: specific-formulations is NOT in the /chat
# NAMESPACES, so include it here alongside the authoritative clinical sources.
MATCH_NAMESPACES = ["specific-formulations", "clinical-qa", "e4l-protocols",
                    "ingredients", "glen-authored-works"]

def _match_query_namespaces(vec):
    out = []
    with ThreadPoolExecutor(max_workers=len(MATCH_NAMESPACES)) as pool:
        futs = {pool.submit(query_ns, vec, ns, TOP_K_PER_NS): ns for ns in MATCH_NAMESPACES}
        for f in as_completed(futs):
            out.extend(f.result())
    return out

def _resolve_remedy_url(name):
    """Resolve a remedy/product NAME to a TRUSTED url. Returns (url, source) or
    (None, None). Never guesses a URL. Sources: 'catalog' (product-aliases.json,
    prefers the remedymatch.com canonical) or 'trusted' (trusted-links.json)."""
    if not name:
        return (None, None)
    nl = name.strip().lower()
    for key, info in (_PRODUCT_ALIASES.get("aliases", {}) or {}).items():
        kl = (key or "").lower()
        cat = (info.get("catalog_name") or "").lower()
        if kl and (nl == kl or nl == cat or (len(nl) > 4 and (nl in kl or kl in nl))):
            url = info.get("canonical_url") or info.get("url")
            if url:
                return (url, "catalog")
    for key, val in (_TRUSTED_LINKS.get("links", {}) or {}).items():
        kl = (key or "").lower()
        url = val if isinstance(val, str) else (val or {}).get("url")
        if url and kl and (nl == kl or (len(nl) > 4 and (nl in kl or kl in nl))):
            return (url, "trusted")
    return (None, None)

def _store_search_url(name):
    from urllib.parse import quote_plus
    return f"https://remedymatch.com/?controller=search&s={quote_plus(name or '')}"

_REMEDY_MATCH_SYSTEM = (
    "You are RemedyMatch, Dr. Glen Swartwout's warm, Socratic remedy-matching guide "
    "(naturopathic physician, Hilo Hawai'i). Goal: through brief back-and-forth, help the "
    "person find the ONE perfect remedy for their need right now.\n\n"
    "How you work:\n"
    "- Ask ONE focused question at a time, warmly and plainly. Gather: their main concern or "
    "goal, who it's for, what they've tried, and current patterns (energy, sleep, stress, terrain).\n"
    "- Prefer Functional Formulations (Advanced Botanical / Nutritional) FIRST — they simplify "
    "implementation — then individual remedies, healing tools, services, or a natural therapy "
    "outside our catalog when that is genuinely the best fit.\n"
    "- The RETRIEVED SNIPPETS are your source of truth. Snippets tagged [AUTHORITATIVE ...] or "
    "type clinical-qa override anything else; apply them directly.\n"
    "- If you do NOT yet have enough to name a single best match, ask the next best question. "
    "Do not guess or list many options.\n"
    "- When you ARE confident in the single best match, name it clearly, say in 1-2 sentences why "
    "it fits THIS person, and invite them to open its page.\n"
    "- You may suggest a quick 10-second E4L voice scan (truly.vip/E4L) to read current stress "
    "patterns when that would sharpen the match.\n"
    "- Never invent product URLs or prices. Keep replies short. Sign off warmly as Dr. Glen."
)

_MATCH_EXTRACT_SYSTEM = (
    "You analyze a remedy-matching conversation. If the assistant has CONFIDENTLY identified the "
    "single perfect remedy to recommend now, return JSON: "
    "{\"matched\": true, \"name\": \"<exact remedy/product name>\", "
    "\"kind\": \"formulation|remedy|tool|service|external\", \"why\": \"<one short sentence>\"}. "
    "If it is still gathering information, or offered several options, return {\"matched\": false}. "
    "Output ONLY the JSON, no prose, no code fences."
)


@app.route("/begin/match")
def begin_match_page():
    resp = send_from_directory(STATIC, "begin-match.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/match/chat", methods=["POST", "OPTIONS"])
def begin_match_chat():
    if request.method == "OPTIONS":
        return "", 200
    data    = request.get_json() or {}
    query   = (data.get("query") or "").strip()
    history = data.get("history") or []
    for_whom = (data.get("for_whom") or "").strip().lower()   # "me" | "someone-else" | ""
    name    = (data.get("name") or "").strip()
    email   = (data.get("email") or "").strip().lower()
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip() or uuid.uuid4().hex)
    auth_user = get_authenticated_user(request)
    if auth_user:
        email = auth_user["email"]
        if not name and auth_user.get("name"):
            name = auth_user["name"]
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        try:
            q_vec = embed(query)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"}); return
        matches = _match_query_namespaces(q_vec)
        context_str, sources_list = build_context(matches) if matches else ("", [])

        # Personal context — only when it's FOR THEM and we know the email.
        personal_block, household_note = "", ""
        if email and for_whom != "someone-else":
            try:
                with sqlite3.connect(LOG_DB) as cx:
                    ppl = cx.execute("SELECT name FROM people WHERE email=?", (email,)).fetchall()
                if len(ppl) > 1:
                    nm = ", ".join([p[0] for p in ppl if p[0]][:5])
                    household_note = (f"HOUSEHOLD NOTE: this email is shared by multiple people "
                                      f"({nm}). Confirm WHICH person this remedy is for before "
                                      f"personalizing.\n")
                mc = _member_context_for_email(email)
                bits = []
                if mc.get("intake_summary"):   bits.append("Intake: " + str(mc["intake_summary"]))
                if mc.get("recent_inquiries"): bits.append("Recent concerns/goals: " + "; ".join(str(x) for x in mc["recent_inquiries"][:3]))
                if mc.get("recent_queries"):   bits.append("Recent questions: " + "; ".join(str(x) for x in mc["recent_queries"][:3]))
                if mc.get("voice_scan_summary"): bits.append("Voice scan: " + str(mc["voice_scan_summary"]))
                if bits:
                    personal_block = "PERSONAL CONTEXT (apply only if this remedy is for THEM):\n" + "\n".join(bits) + "\n\n"
            except Exception as e:
                print(f"[match] personal ctx: {e!r}", flush=True)

        whom_line = {
            "me": "This match is FOR THE PERSON THEMSELVES.",
            "someone-else": "This match is FOR SOMEONE ELSE — do not apply the chatter's personal data.",
        }.get(for_whom, "The person hasn't said who this is for — gently confirm it's for them or someone else.")

        # Glen's recommended off-catalog tools/products (trusted-links.json) — give
        # the model their names so it can recommend them by name when they truly fit
        # (Functional Formulations still come first; these are adjunct tools/devices).
        tools_lines = []
        for k, v in (_TRUSTED_LINKS.get("links", {}) or {}).items():
            note = v.get("note", "") if isinstance(v, dict) else ""
            tools_lines.append(f"- {k}" + (f": {note}" if note else ""))
        tools_block = ("GLEN'S RECOMMENDED ADJUNCT TOOLS/PRODUCTS (name one by its EXACT name only "
                       "when it is genuinely the best fit; Functional Formulations still come first):\n"
                       + "\n".join(tools_lines) + "\n\n") if tools_lines else ""

        messages = []
        for turn in history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content":
            f"USER MESSAGE: {query}\n\n{whom_line}\n{household_note}\n{personal_block}"
            f"{tools_block}"
            f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
            "Continue the Socratic match. If you can now name the ONE best remedy, name it and "
            "invite them to open its page; otherwise ask the single best next question."})

        # ── Consent gate ── the matcher follows the SAME education-vs-recommendation
        # rule as the main chat. A non-member gets educate-only on a gated turn,
        # and the specific product + Buy button (match_evt below) is withheld
        # until they opt in. Decided before the stream so nothing leaks.
        _member = is_member(session_id, email)
        _match_system = _REMEDY_MATCH_SYSTEM
        if not _member and _is_gated_question(query):
            _match_system = _REMEDY_MATCH_SYSTEM + _EDUCATE_ONLY_POLICY
            yield sse({"gate": True})

        full = []
        try:
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=900,
                                     system=_match_system, messages=messages) as stream:
                for tok in stream.text_stream:
                    tok = _strip_dash(tok); full.append(tok); yield sse({"token": tok})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"}); return
        answer = "".join(full)

        try:
            log_query(query, "self-healing", answer, session_id=session_id,
                      email=email, name=name, mode="brief")
        except Exception:
            pass

        # Confirm a single confident match (separate call so it never pollutes the stream).
        match_evt = None
        try:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-3:]) + f"\nassistant: {answer}"
            mx = _cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=200,
                                     system=_MATCH_EXTRACT_SYSTEM,
                                     messages=[{"role": "user", "content": convo[:4000]}])
            txt = mx.content[0].text.strip()
            if txt.startswith("```"):
                txt = txt.split("```", 2)[1]
                if txt.startswith("json\n"): txt = txt[5:]
            obj = json.loads(txt)
            if obj.get("matched") and obj.get("name"):
                url, src = _resolve_remedy_url(obj["name"])
                buy_slug = _resolve_buy_slug(obj["name"])
                match_evt = {"name": obj["name"], "kind": obj.get("kind", ""),
                             "why": obj.get("why", ""), "url": url, "url_source": src,
                             "buy_url": (f"/begin/buy/{buy_slug}" if buy_slug else ""),
                             "search_url": "" if url else _store_search_url(obj["name"])}
        except Exception as e:
            print(f"[match] extract: {e!r}", flush=True)
        # Withhold the named product + Buy button until the visitor is a Member
        # (recommending a remedy for their condition requires ToS agreement).
        if match_evt and _member:
            yield sse({"match": match_evt})

        yield sse({"done": True, "session_id": session_id,
                   "sources": sources_list, "chunks_retrieved": len(matches)})

    resp = Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/match/voice-signal", methods=["POST"])
def begin_match_voice_signal():
    """Exploratory: log a voice-derived signal from the match page's mic dictation.
    v1 stores the transcript + optional client metrics; acoustic tone analysis is Phase 2."""
    data = request.get_json(silent=True) or {}
    session_id = (request.cookies.get("amg_session") or (data.get("session_id") or "").strip())
    email      = (data.get("email") or "").strip().lower()
    transcript = (data.get("transcript") or "").strip()[:2000]
    source     = (data.get("source") or "match-dictation").strip()[:40]
    metrics    = data.get("metrics") or {}
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS voice_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                session_id TEXT, email TEXT, source TEXT,
                transcript TEXT, metrics_json TEXT)""")
            cx.execute("INSERT INTO voice_signals (ts, session_id, email, source, transcript, metrics_json) "
                       "VALUES (?,?,?,?,?,?)",
                       (datetime.now(timezone.utc).isoformat(), session_id, email, source,
                        transcript, json.dumps(metrics)[:4000]))
            cx.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── QuickBooks invoicing — write-layer diagnostics + test (console-key gated) ──
def _qbo_auth_ok():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return False
    return True


@app.route("/api/qbo/diagnostics", methods=["GET"])
def qbo_diagnostics():
    """Read-only: confirm QBO write-layer prerequisites (items + income accounts)."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        from dashboard import qbo_billing as qb
        items = qb.list_items()
        inc = qb._query("SELECT * FROM Account WHERE AccountType = 'Income'") \
                .get("QueryResponse", {}).get("Account", [])
        return jsonify({
            "ok": True,
            "item_count": len(items),
            "items": [{"id": i.get("Id"), "name": i.get("Name"),
                       "type": i.get("Type"), "price": i.get("UnitPrice")} for i in items[:25]],
            "income_accounts": [{"id": a.get("Id"), "name": a.get("Name")} for a in inc[:10]],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/qbo/test-invoice", methods=["POST"])
def qbo_test_invoice():
    """Create ONE test invoice (no online pay) for a clearly-named test customer to
    verify write capability. Void it afterward via /api/qbo/void-invoice."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer("zztest+remedymatch@example.com", "ZZ Test DeleteMe")
        inv = qb.create_invoice(cust,
                                [{"name": "TEST RemedyMatch Product", "amount": 69.97, "qty": 1,
                                  "description": "TEST — verifying QBO write layer, please void"}],
                                allow_online_pay=False)
        return jsonify({"ok": True,
                        "invoice_id": inv.get("Id"), "doc_number": inv.get("DocNumber"),
                        "total": inv.get("TotalAmt"), "sync_token": inv.get("SyncToken"),
                        "pay_link": qb.get_invoice_pay_link(inv),
                        "customer": cust.get("DisplayName")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/qbo/void-invoice", methods=["POST"])
def qbo_void_invoice():
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    try:
        from dashboard import qbo_billing as qb
        out = qb.void_invoice(data.get("id"), data.get("sync_token"))
        return jsonify({"ok": True, "voided": out.get("Invoice", {}).get("Id")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Recurring membership subscriptions (Group Coaching) ───────────────────────
@app.route("/admin/membership")
def admin_membership_page():
    """Admin form to start a Group Coaching subscription. The page collects the console
    key and the /api/qbo/membership/start action it calls is console-gated."""
    resp = send_from_directory(STATIC, "membership-admin.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _membership_schedule(data):
    """Resolve (tier, start_date, day_of_month, tier_key) from a request.

    Defaults: rebill the same day each month (no proration). `first_month_free`
    (the studio.com $99-annual promo) shifts the FIRST billed cycle one month out
    while keeping the original day-of-month for the recurrence, so the member rides
    this month free and the first invoice lands next month at the founders rate."""
    import datetime as _dt
    import calendar as _cal
    tier_key = (data.get("tier") or "standard").strip().lower()
    tier = _MEMBERSHIP_TIERS.get(tier_key)
    if not tier:
        raise ValueError(f"unknown tier '{tier_key}' (use standard|founders)")
    today = _dt.date.today()
    explicit = (data.get("start_date") or "").strip()
    if explicit:
        try:
            sd = _dt.date.fromisoformat(explicit)
        except Exception:
            raise ValueError("start_date must be YYYY-MM-DD")
    else:
        sd = today
    dom = int(data.get("day_of_month") or sd.day)
    dom = max(1, min(dom, 28))  # QBO recurring supports DayOfMonth 1-28 only
    if data.get("first_month_free") and not explicit:
        y, m = (sd.year + 1, 1) if sd.month == 12 else (sd.year, sd.month + 1)
        sd = _dt.date(y, m, min(dom, _cal.monthrange(y, m)[1]))
    return tier, sd.isoformat(), dom, tier_key


@app.route("/api/qbo/membership/start", methods=["POST"])
def qbo_membership_start():
    """Console-gated: start a recurring Group Coaching subscription for a customer."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    try:
        tier, start_date, dom, tier_key = _membership_schedule(data)
    except ValueError as ve:
        return jsonify({"ok": False, "error": str(ve)}), 400
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer(email, name)
        allow_online = bool(_QBO_PAYMENTS_ACTIVE)  # card autopay only once Payments is live
        rt = qb.create_recurring_invoice(
            cust, item_name=_MEMBERSHIP_ITEM, amount=tier["amount"],
            day_of_month=dom, start_date=start_date,
            template_name=f"{tier['label']} - {cust.get('DisplayName', email)}",
            email_to=email, allow_online_pay=allow_online, description=tier["label"])
        return jsonify({"ok": True, "tier": tier_key, "amount": tier["amount"],
                        "customer": cust.get("DisplayName"),
                        "recurring_id": (rt or {}).get("Id"),
                        "sync_token": (rt or {}).get("SyncToken"),
                        "day_of_month": dom, "start_date": start_date,
                        "first_month_free": bool(data.get("first_month_free")),
                        "auto_charge": allow_online})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/qbo/membership/cancel", methods=["POST"])
def qbo_membership_cancel():
    """Console-gated: deactivate (default) or delete a recurring template."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    rid, stok = data.get("id"), data.get("sync_token")
    if not rid:
        return jsonify({"ok": False, "error": "id required"}), 400
    try:
        from dashboard import qbo_billing as qb
        if data.get("delete"):
            qb.delete_recurring(rid, stok)
            return jsonify({"ok": True, "deleted": str(rid)})
        qb.set_recurring_active(rid, stok, False)
        return jsonify({"ok": True, "deactivated": str(rid)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/qbo/membership/test", methods=["POST"])
def qbo_membership_test():
    """Console-gated guarded test: create a recurring template for a ZZ test customer,
    confirm the mechanism, then delete it. Proves the RecurringTransaction shape on the
    real books without leaving anything behind."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    import datetime as _dt
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer("zztest+groupcoaching@example.com", "ZZ Test Member DeleteMe")
        today = _dt.date.today()
        rt = qb.create_recurring_invoice(
            cust, item_name=_MEMBERSHIP_ITEM, amount=149.00,
            day_of_month=today.day, start_date=today.isoformat(),
            template_name="ZZ TEST Group Coaching DeleteMe",
            email_to="zztest+groupcoaching@example.com", description="TEST recurring, auto-deleted")
        rid, stok = (rt or {}).get("Id"), (rt or {}).get("SyncToken")
        deleted = None
        if rid:
            try:
                qb.delete_recurring(rid, stok)
                deleted = rid
            except Exception as de:
                return jsonify({"ok": True, "created_id": rid, "delete_error": str(de),
                                "note": "template created but NOT deleted — clean up manually"}), 200
        return jsonify({"ok": True, "created_id": rid, "deleted": deleted,
                        "item": _MEMBERSHIP_ITEM})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Funnel product checkout — product page → Buy → QBO invoice ────────────────
_PRODUCTS = _load_json(DATA_DIR / "products.json",
                       default={"default_price_cents": 6997, "products": {}})
# Card/ACH online payment is gated until QuickBooks Payments is activated.
_QBO_PAYMENTS_ACTIVE = os.environ.get("QBO_PAYMENTS_ACTIVE", "").strip().lower() in ("1", "true", "yes", "on")
_STRIPE_ACTIVE = os.environ.get("STRIPE_ACTIVE", "").strip().lower() in ("1", "true", "yes", "on")
# Shown to the customer when Stripe was active but no checkout URL came back
# (create_checkout_session failed) — so the Pay button surfaces a clear message
# instead of silently no-opping. _alert_stripe() already notifies Glen.
_CARD_UNAVAILABLE = ("Card payment is temporarily unavailable — please try again "
                     "shortly, use another payment method, or contact us.")
# Phase 3: customer-facing emailed invoice pay-link. OFF until tested in prod —
# gates the OWNER "Send invoice" action and the public /invoice pay endpoints.
INVOICE_PAYLINK_ENABLED = os.environ.get("INVOICE_PAYLINK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# Module-level qb alias so tests can monkeypatch appmod.qb.* and the engine path references it.
try:
    from dashboard import qbo_billing as qb
except Exception:
    qb = None  # type: ignore[assignment]
# Module-level stripe_pay alias so tests can monkeypatch appmod.stripe_pay.*
try:
    from dashboard import stripe_pay
except Exception:
    stripe_pay = None  # type: ignore[assignment]
_ALT_PAY = {
    "zelle": {"label": "Zelle (US)",
              "to": os.environ.get("ZELLE_PAY_TO", "(set ZELLE_PAY_TO)"),
              "note": "Send the invoice total via Zelle, using the invoice number as the memo. "
                      "You earn extra loyalty points for choosing a fee-free method."},
    "wise":  {"label": "Wise (International)",
              "to": os.environ.get("WISE_PAY_TO", "(set WISE_PAY_TO)"),
              "note": "Send the invoice total via Wise, using the invoice number as the reference. "
                      "You earn extra loyalty points for choosing a fee-free method."},
}


# Quantity pricing for capsule Functional Formulations ($69.97 base). Per-unit price
# drops by quantity (Glen 2026-05-30): 3+ $59.97, 6+ $49.97, 12+ $39.97. Applies only
# to products flagged qty_pricing=true in products.json (capsule + $69.97).
_QTY_TIERS = [(12, 3997), (6, 4997), (3, 5997), (1, 6997)]   # (min_qty, unit_cents) desc
_FORMATS = [
    {"id": "bottle", "label": "Standard bottles", "note": "30 capsules per bottle"},
    {"id": "larger", "label": "Larger bottle", "note": "90, 180, or 360 capsules in one bottle (quantity 3, 6, or 12)"},
    {"id": "refill", "label": "Cellophane refill packs", "note": "Capsules only, no bottle"},
]


def _qty_eligible(p):
    return bool(p.get("qty_pricing")) and p.get("price_cents") == 6997 and not p.get("info_only")


def _qty_unit_cents(p, qty):
    """Per-unit price honoring the capsule quantity tiers for eligible products."""
    if not _qty_eligible(p):
        return p.get("price_cents", 6997)
    for min_q, unit in _QTY_TIERS:
        if qty >= min_q:
            return unit
    return p.get("price_cents", 6997)


def _inhouse_ff_unit_cents(p, total_ff_qty, settings):
    """Effective in-house unit price (cents) for a $69.97 functional-formulation
    capsule (_qty_eligible): the order-wide volume rate driven by the TOTAL FF
    capsule quantity (1 bottle = 1 month). The linear volume curve is monotonic,
    so the total-quantity rate is the best price each FF line can get. Clamped at
    the wholesale discount floor so a console max above 43% can't breach it.
    Non-eligible products return their list price (callers handle overrides)."""
    if not _qty_eligible(p):
        return int(p.get("price_cents") or 0)
    from dashboard import pricing as _pricing
    pct = _pricing.volume_pct(int(total_ff_qty or 0), settings)
    eff = int(round(6997 * (1 - (pct or 0) / 100.0)))
    floor = int(round(6997 * float(settings.get("discount_floor_pct", 0.57))))
    return max(eff, floor)


def _inhouse_total_ff_qty(lines_in):
    """Sum of qty across $69.97 functional-formulation lines in the cart."""
    tot = 0
    for ln in lines_in:
        p = _get_product((ln.get("slug") or "").strip())
        if p and _qty_eligible(p):
            tot += max(1, min(int(ln.get("qty") or 1), 99))
    return tot


def _inhouse_line_unit_cents(p, override, total_ff_qty, settings):
    """Per-line unit price for the in-house form: an explicit owner override wins;
    else FF capsules get the volume rate, everything else its list price."""
    if override not in (None, ""):
        return int(override)
    return _inhouse_ff_unit_cents(p, total_ff_qty, settings)


# Recurring membership tiers (Group Coaching). One QBO Item, price set per tier.
_MEMBERSHIP_ITEM = "Group Coaching Membership"
_MEMBERSHIP_TIERS = {
    "standard": {"label": "Group Coaching Membership", "amount": 149.00},
    "founders": {"label": "Group Coaching Membership (Founders)", "amount": 99.00},
}


def _get_product(slug):
    p = (_PRODUCTS.get("products") or {}).get(slug)
    if not p or p.get("inactive"):
        return None  # inactive products are not sellable/visible on the funnel
    out = dict(p)
    out["slug"] = slug
    out.setdefault("price_cents", _PRODUCTS.get("default_price_cents", 6997))
    return out


def _is_pure_powder(p):
    return "pure powder" in (p.get("name") or "").lower() or "pure-powder" in (p.get("slug") or "")


def _engine_item(p, qty):
    """Build a dashboard.pricing.compute() item from a product dict + quantity.
    Derives months (months_per_unit*qty, default 1/unit), volume eligibility
    (all Functional Formulations; Pure Powders + info_only excluded), and the
    per-SKU floor override for Pure Powders (0.75 -> ~$30)."""
    qty = max(1, int(qty))
    months_per_unit = int(p.get("months_per_unit", 1))
    if "volume_eligible" in p:
        eligible = bool(p["volume_eligible"])
    else:
        eligible = not (_is_pure_powder(p) or p.get("info_only"))
    prod = dict(p)
    if _is_pure_powder(p) and "sku_discount_floor_pct" not in prod:
        prod["sku_discount_floor_pct"] = 0.75
        prod["sku_points_floor_pct"] = 0.75
    return {
        "slug": p.get("slug"), "name": p.get("name", p.get("slug")), "qty": qty,
        "product": prod, "unit_cents": int(p.get("price_cents") or 0),
        "months": months_per_unit * qty, "volume_eligible": eligible,
    }


class CheckoutError(Exception):
    """Raised for a checkout the customer must fix (e.g. non-US ship-to)."""


# USPS flat-rate box thresholds by total bottle count (Glen 2026-06-14): <=4 S, <=12 M, else L.
# Used as a fail-safe when the box-fit catalog can't pick a box; the catalog (when Rae fills it
# at /admin/shipping) takes precedence via shipping.quote().
_SHIP_FALLBACK_HARD = {"S": 1300, "M": 2300, "L": 3200}   # rounded rates if even the rates table is down


def _fallback_shipping_cents(total_bottles):
    """Qty-based USPS box fallback (rounded charged rate). Never raises."""
    n = int(total_bottles or 0)
    if n <= 0:
        return 0
    size = "S" if n <= 4 else ("M" if n <= 12 else "L")
    try:
        rates = _shipping.get_current_rates() or {}
        r = rates.get(size) or {}
        cents = int(r.get("charged_cents") or r.get("usps_retail_cents") or 0)
        return cents or _SHIP_FALLBACK_HARD[size]
    except Exception:
        return _SHIP_FALLBACK_HARD[size]


def _shipping_for_cart(box_counts, total_bottles):
    """Real USA shipping for a cart, always charged, never breaking checkout.
    Tries the box-fit catalog (shipping.quote keyed by bottle type); if the catalog
    is empty / a type is unknown / no box fits, falls back to the qty->box rule."""
    if not box_counts:
        return 0
    try:
        q = _shipping.quote(box_counts) or {}
        cents = int(q.get("shipping_cents") or 0)
        if cents > 0:
            return cents
    except Exception as e:
        print(f"[shipping] quote fell back ({e!r}); using qty rule for {total_bottles} bottles", flush=True)
    return _fallback_shipping_cents(total_bottles)


def _price_cart(cart, *, ship, coupon_pct=None, subscriber_tier_pct=None,
                points_to_redeem_cents=0, channel="retail"):
    """Price a reorder/checkout cart through the pricing engine + shipping.
    Returns {priced, qbo_lines, discount_cents, points_redeemed_cents, shipping_cents,
    items_rec, subtotal_list_cents}. Raises CheckoutError for non-US ship-to."""
    from dashboard import pricing as _pricing, tax as _tax
    country = (ship.get("country") or "US").strip().upper()
    if country not in ("US", "USA", ""):
        raise CheckoutError("We ship to US addresses only — please use a US forwarding address.")
    settings = _pricing.load_settings(_pricing_settings())
    items, qbo_lines, items_rec, box_counts, subtotal_list, total_bottles = [], [], [], {}, 0, 0
    for c in (cart or []):
        p = _get_product((c.get("slug") or "").strip())
        if not p:
            continue
        qty = max(1, min(int(c.get("qty", 1) or 1), 99))
        it = _engine_item(p, qty)
        items.append(it)
        subtotal_list += it["unit_cents"] * qty
        qbo_lines.append({"name": p["name"], "amount": round(it["unit_cents"] / 100.0, 2),
                          "qty": qty, "item_id": p.get("qbo_item_id"), "description": p["name"]})
        items_rec.append({"name": p["name"], "qty": qty, "desc": p["name"]})
        # shipping.pick_box keys by BOTTLE TYPE (not product name); default-typed if unset.
        bt = p.get("bottle_type") or "default"
        box_counts[bt] = box_counts.get(bt, 0) + qty
        total_bottles += qty
    priced = _pricing.compute(items, settings=settings, coupon_pct=coupon_pct,
                              subscriber_tier_pct=subscriber_tier_pct, channel=channel,
                              points_to_redeem_cents=int(points_to_redeem_cents or 0),
                              ship_to_state=ship.get("state", ""),
                              tax_fn=_tax.compute_get_cents)
    shipping_cents = _shipping_for_cart(box_counts, total_bottles)
    return {
        "priced": priced, "qbo_lines": qbo_lines, "items_rec": items_rec,
        "subtotal_list_cents": subtotal_list,
        "discount_cents": priced["discount_cents"],
        "points_redeemed_cents": priced["points_redeemed_cents"],
        "shipping_cents": shipping_cents,
    }


def _rewards_enabled() -> bool:
    """True when the rewards/referral system is switched on."""
    return bool(os.environ.get("REWARDS_TIERS_ENABLED", ""))


def _referrer_slug_for_email(cx, email: str):
    """Return the slug of the most-recent approved affiliate who referred this email, or None."""
    email = (email or "").strip().lower()
    if not email:
        return None
    row = cx.execute(
        """SELECT re.utm_source FROM referral_events re
           JOIN affiliate_signups a ON a.slug = re.utm_source AND a.status = 'approved'
           WHERE LOWER(re.email) = ?
           ORDER BY re.received_at DESC LIMIT 1""",
        (email,)
    ).fetchone()
    return row[0] if row else None


def _referral_pct_for_referrer(referrer_email, settings):
    """Cert-scaled referral FRACTION. If the referrer is a practitioner, interpolate their
    modules_completed through referral_cert_anchors; otherwise (or on any lookup error) the
    base referral_reward_pct. Best-effort: never raises."""
    from dashboard import rewards as _rewards
    base = float(settings.get("referral_reward_pct", 0.05))
    try:
        modules = _pp.modules_completed_for_email(referrer_email)
        if modules is None:
            return base
        return _rewards.referral_pct_for_modules(modules, settings)
    except Exception as _e:
        print(f"[rewards] cert lookup failed for {referrer_email}: {_e!r}", flush=True)
        return base


def _settle_referral(order, *, order_ref: str) -> None:
    """Credit the referrer (points or cash) on a full-price referred sale.
    Idempotent per order_ref. Best-effort -- never raises."""
    if not _rewards_enabled():
        return
    try:
        from dashboard import points as _points
        from dashboard import rewards as _rewards
        buyer_email = (order.get("email") or "").strip().lower()
        if not buyer_email:
            return
        product_cents = (int(order.get("total_cents") or 0)
                         - int(order.get("shipping_cents") or 0)
                         - int(order.get("get_cents") or 0))
        # Only credit on full-price sales
        if int(order.get("discount_cents") or 0) != 0:
            return
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _points.init_points_table(cx)
            _rewards.init_affiliate_earnings_table(cx)
            slug = _referrer_slug_for_email(cx, buyer_email)
            if not slug:
                return
            referrer_email = _rewards.referrer_email_for_slug(cx, slug)
            if not referrer_email:
                return
            # Exclude self-referral
            if referrer_email.strip().lower() == buyer_email:
                return
            settings = _rewards.load_settings(_rewards_settings())
            referral_reward_pct = _referral_pct_for_referrer(referrer_email, settings)
            reward = round(product_cents * referral_reward_pct)
            if reward <= 0:
                return
            mode = _rewards.reward_mode_for_slug(cx, slug)
            if mode == "points":
                _points.credit(cx, referrer_email,
                               value_cents=reward, reason="referral", order_ref=order_ref)
            else:
                _rewards.accrue_cash(cx, slug=slug, email=referrer_email,
                                     order_ref=order_ref, amount_cents=reward)
            _maybe_raise_cashout_review(cx, slug, mode)
    except Exception as _e:
        print(f"[rewards] _settle_referral failed inv={order_ref}: {_e!r}", flush=True)


def _maybe_raise_cashout_review(cx, slug: str, mode: str) -> None:
    """Insert a review todo if the referrer's balance has crossed a threshold band.
    Idempotent: dedup_key prevents duplicates within the same threshold band."""
    try:
        from dashboard import rewards as _rewards
        from dashboard import points as _points
        settings = _rewards.load_settings(_rewards_settings())
        threshold = int(settings["cash_out_threshold_cents"])
        if mode == "cash":
            amount = _rewards.pending_cash_total(cx, slug)
        else:
            referrer_email = _rewards.referrer_email_for_slug(cx, slug)
            if not referrer_email:
                return
            amount = _points.balance(cx, referrer_email)
        if amount < threshold:
            return
        band = amount // threshold
        dedup_key = f"cashout:{slug}:{band}"
        cash_value = round(amount * float(settings["cash_out_face_pct"]))
        if mode == "cash":
            body = (f"Affiliate '{slug}' has ${amount/100:.2f} pending cash earnings "
                    f"(threshold ${threshold/100:.2f}). Approve payout via rewards.process_payout.")
        else:
            body = (f"Affiliate '{slug}' has {amount} points pending "
                    f"(threshold {threshold} pts). Cash value at "
                    f"{int(settings['cash_out_face_pct']*100)}% face: ${cash_value/100:.2f}. "
                    f"Approve via rewards.process_payout.")
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc).isoformat()
        cx.execute(
            """INSERT INTO todos (created_at, owner, category, title, body, priority, source, dedup_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dedup_key) DO NOTHING""",
            (now, "glen", "Finance",
             f"Review ${amount/100:.2f} cash-out for {slug}",
             body, "high", "affiliate-cashout", dedup_key)
        )
        cx.commit()
    except Exception as _e:
        print(f"[rewards] _maybe_raise_cashout_review failed slug={slug}: {_e!r}", flush=True)


def _settle_order_points(order, *, order_ref):
    """On a PAID order: deduct redeemed points, and earn 5% if it was a full-price order.
    Idempotent per order_ref. Best-effort -- never raises into the return handler.
    Buyer earn is suppressed on an affiliate-acquired first order (REWARDS_TIERS_ENABLED)."""
    from dashboard import points as _points
    email = (order.get("email") or "").strip().lower()
    if not email:
        return
    earn_pct = float(_pricing_settings().get("points_earn_pct", 0.05))
    redeemed = int(order.get("points_redeemed_cents") or 0)
    discount = int(order.get("discount_cents") or 0)
    # Earn on PRODUCT spend only: total includes GET (absorbed) when TAX_ENABLED, so net out
    # both shipping and get_cents to avoid earning points on tax/shipping.
    product_cents = (int(order.get("total_cents") or 0)
                     - int(order.get("shipping_cents") or 0)
                     - int(order.get("get_cents") or 0))
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _points.init_points_table(cx)
        if redeemed > 0 and not _points.has_entry(cx, order_ref=order_ref, reason="redeem"):
            try:
                _points.redeem(cx, email, value_cents=redeemed, order_ref=order_ref)
            except ValueError:
                pass   # balance already spent elsewhere; don't block the order
        # Earn only on a full-price order (no discount AND no points used) -- the "full-price only" rule.
        # Additional suppression: skip buyer earn on an affiliate-acquired FIRST order.
        if discount == 0 and redeemed == 0 and product_cents > 0 \
                and not _points.has_entry(cx, order_ref=order_ref, reason="earn"):
            # Suppression gate: if rewards are enabled AND this is buyer's first order
            # AND there is an attributed referrer, don't give the buyer earn points.
            suppress = False
            if _rewards_enabled():
                try:
                    order_count = cx.execute(
                        "SELECT COUNT(*) FROM orders WHERE lower(email)=?",
                        (email,)
                    ).fetchone()[0]
                    if order_count <= 1 and _referrer_slug_for_email(cx, email) is not None:
                        suppress = True
                except Exception as _se:
                    print(f"[rewards] suppression check failed: {_se!r}", flush=True)
            if not suppress:
                _points.earn(cx, email, full_price_cents=product_cents, earn_pct=earn_pct,
                             order_ref=order_ref)


# Generated/cached product content (ingredients + benefits + learn-more research).
# Source = Pinecone specific-formulations (page copy) + ingredients (study citations).
try:
    from dashboard import product_content as _product_content
    with sqlite3.connect(LOG_DB) as _cx:
        _product_content.init_product_content_table(_cx)
except Exception as _pce:
    _product_content = None
    print(f"[product_content] init failed: {_pce}", flush=True)


def _product_card(product):
    """Cached {description, ingredients[], benefits[]} for a product (best-effort)."""
    if not _product_content:
        return {"description": "", "ingredients": [], "benefits": []}
    try:
        return _product_content.get_or_generate(product, "card")["content"]
    except Exception as e:
        print(f"[product_content] card failed {product.get('slug')}: {e}", flush=True)
        return {"description": "", "ingredients": [], "benefits": []}


def _product_how(product):
    """Cached 'How it works' mechanism text for a product (best-effort)."""
    if not _product_content:
        return ""
    try:
        return _product_content.get_or_generate(product, "how_it_works")["content"].get("text", "")
    except Exception as e:
        print(f"[product_content] how_it_works failed {product.get('slug')}: {e}", flush=True)
        return ""


# Section preferences: remember which detail panels a client opens (What's inside /
# How it works / The research), keyed by session + email, so future formulations
# default-open those AND email campaigns can focus on what the client engages with.
with sqlite3.connect(LOG_DB) as _cx:
    _cx.execute("""CREATE TABLE IF NOT EXISTS section_prefs (
        session_id TEXT PRIMARY KEY, email TEXT, opened TEXT, updated_at TEXT)""")
    _cx.execute("CREATE INDEX IF NOT EXISTS idx_section_prefs_email ON section_prefs(email)")
_SECTIONS = ("ingredients", "how", "research")


def _read_open_sections(session_id, email=""):
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = None
            if session_id:
                row = cx.execute("SELECT opened FROM section_prefs WHERE session_id=?", (session_id,)).fetchone()
            if not row and email:
                row = cx.execute("SELECT opened FROM section_prefs WHERE email=? ORDER BY updated_at DESC LIMIT 1",
                                 (email,)).fetchone()
        return json.loads(row[0]) if row and row[0] else []
    except Exception:
        return []


@app.route("/begin/section-pref", methods=["POST"])
def begin_section_pref():
    """Record that a client opened a detail panel (remembered for defaults + email focus)."""
    data = request.get_json(silent=True) or {}
    section = (data.get("section") or "").strip()
    if section not in _SECTIONS:
        return jsonify({"ok": False}), 400
    session_id = request.cookies.get("amg_session", "")
    if not session_id:
        return jsonify({"ok": True})
    au = get_authenticated_user(request)
    email = (au or {}).get("email", "") if au else ""
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            row = cx.execute("SELECT opened FROM section_prefs WHERE session_id=?", (session_id,)).fetchone()
            opened = set(json.loads(row[0])) if row and row[0] else set()
            opened.add(section)
            cx.execute("INSERT INTO section_prefs (session_id, email, opened, updated_at) VALUES (?,?,?,?) "
                       "ON CONFLICT(session_id) DO UPDATE SET opened=excluded.opened, "
                       "email=COALESCE(NULLIF(excluded.email,''), section_prefs.email), updated_at=excluded.updated_at",
                       (session_id, email, json.dumps(sorted(opened)), begin_funnel._now()))
    except Exception as e:
        print(f"[section-pref] {e}", flush=True)
    return jsonify({"ok": True})


def _resolve_buy_slug(name):
    """Map a remedy NAME to a products.json slug (our QBO checkout catalog) so
    RemedyMatch can offer a Buy button. Returns slug or None."""
    if not name:
        return None
    nl = name.strip().lower()
    for slug, p in (_PRODUCTS.get("products") or {}).items():
        pn = (p.get("name") or "").lower()
        if pn and (nl == pn or (len(nl) > 4 and (nl in pn or pn in nl))):
            return slug
    return None


@app.route("/begin/buy/<slug>")
def begin_buy_page(slug):
    if not _get_product(slug):
        return ("", 404)
    resp = send_from_directory(STATIC, "begin-buy.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/product-data/<slug>")
def begin_product_data(slug):
    p = _get_product(slug)
    if not p:
        return jsonify({"error": "not found"}), 404
    # Generated content (ingredients + benefits + short description). Static JSON
    # values override the generated card when present (lets Glen pin copy).
    card = _product_card(p) if not p.get("info_only") else {}
    how = "" if p.get("info_only") else _product_how(p)
    ingredients = p.get("ingredients") or card.get("ingredients", [])
    qty_tiers, formats = None, None
    if _qty_eligible(p):
        qty_tiers = [{"min": m, "unit_cents": u, "unit": f"${u/100:.2f}",
                      "save": ((6997 - u) // 100) if u < 6997 else 0}
                     for m, u in [(1, 6997), (3, 5997), (6, 4997), (12, 3997)]]
        formats = _FORMATS
    return jsonify({
        "slug": slug, "name": p["name"],
        "price_cents": p["price_cents"], "price": f"${p['price_cents']/100:.2f}",
        "description": p.get("description") or card.get("description", ""),
        "ingredients": ingredients,
        "benefits": p.get("benefits") or card.get("benefits", []),
        "how_it_works": how,
        "info_only": bool(p.get("info_only")), "affiliate_url": p.get("affiliate_url", ""),
        "payments_active": _QBO_PAYMENTS_ACTIVE or _STRIPE_ACTIVE,
        "group_bundle_enabled": bool(os.environ.get("GROUP_BUNDLE_ENABLED")),
        "learn_url": f"/begin/learn/{slug}",
        "qty_pricing": qty_tiers, "formats": formats,
        "open_sections": _read_open_sections(
            request.cookies.get("amg_session", ""),
            (get_authenticated_user(request) or {}).get("email", "")),
    })


@app.route("/begin/learn/<slug>")
def begin_learn_page(slug):
    """Research page — the 3rd Buy-button surface."""
    if not _get_product(slug):
        return ("", 404)
    resp = send_from_directory(STATIC, "begin-learn.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/begin/learn-data/<slug>")
def begin_learn_data(slug):
    p = _get_product(slug)
    if not p:
        return jsonify({"error": "not found"}), 404
    markdown, sources = "", []
    if _product_content and not p.get("info_only"):
        try:
            lm = _product_content.get_or_generate(p, "learn_more")
            markdown = lm["content"].get("markdown", "")
            sources = lm.get("sources", [])
        except Exception as e:
            print(f"[product_content] learn_more failed {slug}: {e}", flush=True)
    return jsonify({
        "slug": slug, "name": p["name"],
        "price": f"${p['price_cents']/100:.2f}",
        "markdown": markdown, "sources": sources,
        "buy_url": f"/begin/buy/{slug}",
        "info_only": bool(p.get("info_only")), "affiliate_url": p.get("affiliate_url", ""),
    })


@app.route("/api/qbo/content-refresh/<slug>", methods=["POST"])
def qbo_content_refresh(slug):
    """Console-gated: force-regenerate the cached card + learn_more for a product."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    p = _get_product(slug)
    if not p:
        return jsonify({"ok": False, "error": "unknown product"}), 404
    if not _product_content:
        return jsonify({"ok": False, "error": "product_content unavailable"}), 503
    try:
        card = _product_content.get_or_generate(p, "card", force=True)["content"]
        how = _product_content.get_or_generate(p, "how_it_works", force=True)["content"]
        lm = _product_content.get_or_generate(p, "learn_more", force=True)
        return jsonify({"ok": True, "slug": slug,
                        "ingredients": len(card.get("ingredients", [])),
                        "benefits": len(card.get("benefits", [])),
                        "how_it_works_chars": len(how.get("text", "")),
                        "learn_more_chars": len(lm["content"].get("markdown", "")),
                        "sources": len(lm.get("sources", []))})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Biofield ($300 points-redeemable service checkout) ───────────────────────
# ── Studio.com bridge (Flow B) ────────────────────────────────────────────────
# A customer who joined studio.com/drglen claims a free first month of live group
# coaching. We vault their card (Stripe mode=setup, $0 charge) and create a
# membership subscription whose first $99/mo charge lands one month out. The
# charge cron then bills it off-session. Ships dark behind STUDIO_BRIDGE_ENABLED.
def _studio_enabled() -> bool:
    return os.environ.get("STUDIO_BRIDGE_ENABLED", "").strip().lower() \
        in ("1", "true", "yes", "on")


@app.route("/api/studio/claim", methods=["POST"])
def api_studio_claim():
    """Record a pending studio.com/drglen claim and start a $0 card-vault setup
    session. The free month is granted on /studio/claim-return."""
    if not _studio_enabled():
        return jsonify({"ok": False, "error": "Studio claim is not available."}), 404
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400

    # ── Consent gate ── claiming requires Tier-1 Membership (ToS agreed +
    # identified). Enforced at the API so it can't be bypassed. The front-end
    # shows the soft opt-in on need_optin.
    _sid = request.cookies.get("amg_session", "")
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please add your name and agree to our Terms "
                                 "to claim your free month."}), 403

    # Optional proof of purchase (best-effort — ignored if absent).
    has_receipt = bool((data.get("receipt") or "").strip())

    from dashboard import studio_bridge as _sbr
    try:
        with sqlite3.connect(LOG_DB) as _cx:
            _sbr.init_table(_cx)
            _sbr.record_pending(
                _cx, email,
                signup_via=("receipt" if has_receipt else "self-attest"))
    except Exception as e:
        print(f"[studio] record_pending failed: {e!r}", flush=True)

    try:
        from dashboard import stripe_pay
        sess = stripe_pay.create_setup_session(
            customer_email=email,
            metadata={"kind": "studio_bridge", "email": email},
            success_url=(f"{PUBLIC_BASE_URL}/studio/claim-return"
                         f"?session_id={{CHECKOUT_SESSION_ID}}"),
            cancel_url=f"{PUBLIC_BASE_URL}/studio/claim")
    except Exception as e:
        print(f"[studio] setup session create failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Could not start the claim."}), 502
    return jsonify({"ok": True, "stripe_url": sess.get("url", "")})


@app.route("/studio/claim-return")
def studio_claim_return():
    """Stripe setup return: read the vaulted card off the SetupIntent and create
    the membership subscription (free month now, first $99 charge one month out).
    Idempotent via studio_bridge.already_granted. Best-effort — never 500s."""
    sid = (request.args.get("session_id") or "").strip()
    if _studio_enabled() and sid:
        try:
            from dashboard import (stripe_pay, studio_bridge as _sbr,
                                    subscriptions as _subs, group_bundle as _gb)
            import datetime as _dt
            sess  = stripe_pay.get_session(sid)
            email = ((sess.get("metadata") or {}).get("email") or "").strip().lower()
            si    = stripe_pay.get_setup_intent(sess.get("setup_intent"))
            cus   = si.get("customer")
            pm    = si.get("payment_method")
            with sqlite3.connect(LOG_DB) as _cx:
                _cx.row_factory = sqlite3.Row
                _sbr.init_table(_cx)
                _subs.init_subscriptions_table(_cx)
                _subs.migrate_add_membership_columns(_cx)
                if email and cus and pm and not _sbr.already_granted(_cx, email):
                    next_date = _subs.add_months(
                        _dt.date.today().isoformat(), 1)
                    sub_id = _subs.create_membership(
                        _cx, email=email, stripe_customer_id=cus,
                        stripe_payment_method_id=pm,
                        amount_cents=_gb.MEMBERSHIP_AMOUNT_CENTS,
                        next_charge_date=next_date)
                    _sbr.mark_granted(_cx, email, sub_id)
                    print(f"[studio] free month granted for {email} sub={sub_id}",
                          flush=True)
        except Exception as e:
            print(f"[studio] claim-return grant failed: {e!r}", flush=True)
    return redirect("/studio/claim?done=1")


@app.route("/studio/claim")
def studio_claim_page():
    resp = send_from_directory(STATIC, "studio-claim.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/studio")
def studio_welcome():
    resp = send_from_directory(STATIC, "studio-welcome.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    # last-touch attribution: only stamp source=studio when no real referral is set
    if not (request.cookies.get("rm_ref") or "").strip():
        resp.set_cookie("rm_ref", "studio", max_age=90 * 24 * 3600,
                        samesite="Lax", secure=request.is_secure)
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=365 * 24 * 3600,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


BIOFIELD_PRICE_CENTS = 30000   # $300 Causal Biofield Analysis (service, no shipping)
_BIOFIELD_ITEM_NAME = "Causal Biofield Analysis"


def _biofield_enabled() -> bool:
    return os.environ.get("BIOFIELD_CHECKOUT_ENABLED", "").strip().lower() \
        in ("1", "true", "yes", "on")


def _price_biofield(points_to_redeem_cents=0):
    """Price the single $300 Biofield service line through the engine.

    A service item has no bottle_type and is volume-ineligible, so it yields NO
    shipping. We build the engine line directly (not via _price_cart, which keys
    shipping off bottle counts) so shipping_cents is always 0. Points are applied
    by the engine and clamped at the per-SKU points floor (43% of list)."""
    from dashboard import pricing as _pricing, tax as _tax
    settings = _pricing.load_settings(_pricing_settings())
    product = {"slug": "biofield-analysis", "name": _BIOFIELD_ITEM_NAME,
               "volume_eligible": False}
    item = {"slug": "biofield-analysis", "name": _BIOFIELD_ITEM_NAME, "qty": 1,
            "product": product, "unit_cents": BIOFIELD_PRICE_CENTS,
            "months": 0, "volume_eligible": False}
    priced = _pricing.compute([item], settings=settings,
                              points_to_redeem_cents=int(points_to_redeem_cents or 0),
                              channel="retail", ship_to_state=None,
                              tax_fn=_tax.compute_get_cents)
    qbo_lines = [{"name": _BIOFIELD_ITEM_NAME,
                  "amount": round(BIOFIELD_PRICE_CENTS / 100.0, 2), "qty": 1,
                  "description": _BIOFIELD_ITEM_NAME}]
    items_rec = [{"name": _BIOFIELD_ITEM_NAME, "qty": 1, "desc": _BIOFIELD_ITEM_NAME}]
    return {"priced": priced, "qbo_lines": qbo_lines, "items_rec": items_rec,
            "discount_cents": priced["discount_cents"],
            "points_redeemed_cents": priced["points_redeemed_cents"],
            "shipping_cents": 0}


@app.route("/biofield/checkout", methods=["POST"])
def biofield_checkout():
    """Sell the $300 Biofield as a points-redeemable service. Ships dark behind
    BIOFIELD_CHECKOUT_ENABLED. On payment, /begin/checkout-return (kind=biofield)
    seeds the readiness record and routes the customer to the readiness gate."""
    if not _biofield_enabled():
        return jsonify({"ok": False, "error": "Biofield checkout is not available."}), 404
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    if not _STRIPE_ACTIVE:
        return jsonify({"ok": False, "error": "card payment not active"}), 400
    try:
        redeem = int(data.get("points_to_redeem_cents") or 0)
    except (TypeError, ValueError):
        redeem = 0
    if redeem > 0:
        from dashboard import points as _points
        with sqlite3.connect(LOG_DB) as _bcx:
            _points.init_points_table(_bcx)
            redeem = min(redeem, _points.balance(_bcx, email))
    pc = _price_biofield(points_to_redeem_cents=redeem)
    charged_cents = pc["priced"]["total_cents"]
    redeemed = pc["points_redeemed_cents"]

    from dashboard import qbo_billing as qb
    cust = qb.find_or_create_customer(email, name)
    inv = qb.create_invoice(cust, pc["qbo_lines"], allow_online_pay=_QBO_PAYMENTS_ACTIVE,
                            email_to=email,
                            discount_cents=pc["discount_cents"] + redeemed)
    out = {"ok": True, "invoice_id": inv.get("Id"), "doc_number": inv.get("DocNumber"),
           "customer_id": cust.get("Id"), "total": inv.get("TotalAmt")}

    _ingest_order(source="biofield", external_ref=inv.get("Id"), email=email, name=name,
                  items=pc["items_rec"], total_cents=charged_cents, channel="retail",
                  get_cents=pc["priced"]["get_cents"], discount_cents=pc["discount_cents"],
                  points_redeemed_cents=redeemed, shipping_cents=0)
    try:
        from dashboard import stripe_pay
        success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}")
        metadata = {"kind": "biofield", "email": email,
                    "invoice_id": inv.get("Id") or "",
                    "customer_id": cust.get("Id") or "",
                    "points_redeemed_cents": str(int(redeemed))}
        sess = stripe_pay.create_checkout_session(
            charged_cents, customer_email=email,
            description=f"{_BIOFIELD_ITEM_NAME} #{inv.get('DocNumber') or ''}",
            metadata=metadata, success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/begin")
        out["stripe_url"] = sess.get("url") or ""
    except Exception as e:
        print(f"[biofield] session create failed: {e!r}", flush=True)
        out["stripe_url"] = ""
    return jsonify(out)


@app.route("/begin/checkout/<slug>", methods=["POST"])
def begin_checkout(slug):
    p = _get_product(slug)
    if not p:
        return jsonify({"ok": False, "error": "unknown product"}), 404
    if p.get("info_only"):
        return jsonify({"ok": True, "info_only": True, "affiliate_url": p.get("affiliate_url", "")})
    data   = request.get_json(silent=True) or {}
    email  = (data.get("email") or "").strip().lower()
    name   = (data.get("name") or "").strip()
    method = (data.get("method") or "").strip().lower()   # zelle | wise | card
    fmt    = (data.get("format") or "").strip().lower()    # bottle | larger | refill
    try:
        qty = max(1, min(int(data.get("qty", 1) or 1), 99))
    except Exception:
        qty = 1
    # Ship-to address — drives GET tax (HI vs out-of-state) and auto-fills the
    # EasyPost label. Normalized to the shape easypost.build_shipment reads.
    ship = _normalize_ship_address(data.get("address") or {}, fallback_name=name)
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400

    # ── Consent gate ── ordering requires Tier-1 Membership (ToS agreed +
    # identified). Enforced here at the API so it can't be bypassed by hitting
    # the endpoint directly. The front-end shows the soft opt-in on need_optin.
    _sid = request.cookies.get("amg_session", "")
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please add your name and agree to our Terms "
                                 "to place an order."}), 403
    session_id = request.cookies.get("amg_session", "")
    if os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1", "true", "yes", "on"):
        from dashboard import qbo_billing as qb
        try:
            redeem = int((data.get("points_to_redeem_cents") or 0))
        except (TypeError, ValueError):
            redeem = 0
        if redeem > 0:
            from dashboard import points as _points
            with sqlite3.connect(LOG_DB) as _bcx:
                _points.init_points_table(_bcx)
                redeem = min(redeem, _points.balance(_bcx, email))
        try:
            pc = _price_cart([{"slug": slug, "qty": qty}], ship=ship,
                             coupon_pct=_active_coupon_pct(),
                             points_to_redeem_cents=redeem)
        except CheckoutError as ce:
            return jsonify({"ok": False, "error": str(ce)}), 400
        cust = qb.find_or_create_customer(email, name)
        allow_online = (method == "card") and _QBO_PAYMENTS_ACTIVE
        inv = qb.create_invoice(cust, pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
                                allow_online_pay=allow_online, email_to=email,
                                discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
        out = {"ok": True, "invoice_id": inv.get("Id"), "sync_token": inv.get("SyncToken"),
               "doc_number": inv.get("DocNumber"), "total": inv.get("TotalAmt"),
               "method": method, "customer_id": cust.get("Id"),
               "pay_link": qb.get_invoice_pay_link(inv)}
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute("INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                           "VALUES (?,?,?,?,?,?,?)",
                           (begin_funnel._now(), session_id, email, "purchase",
                            f"buy-{slug}-{method}", "", ""))
                cx.commit()
        except Exception:
            pass
        _ingest_order(source="funnel", external_ref=inv.get("Id"), email=email, name=name,
                      items=pc["items_rec"],
                      total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                      address=ship, channel="retail", get_cents=pc["priced"]["get_cents"],
                      discount_cents=pc["discount_cents"],
                      points_redeemed_cents=pc["points_redeemed_cents"],
                      shipping_cents=pc["shipping_cents"])
        if method in ("zelle", "wise"):
            out["pay_instructions"] = _ALT_PAY.get(method, {})
        elif method == "card" and _STRIPE_ACTIVE:
            # Group-bundle opt-in: when the flag is on AND the buyer opted in AND the buyer
            # is a PAID Biofield Analysis client (designed-program gate; the free E4L scan
            # does not qualify) AND the cart carries >=1 program month, grant 1 free
            # live-group month per program month (capped at 3). Vaults the card so the
            # membership can auto-continue.
            _gb_months = 0
            if (os.environ.get("GROUP_BUNDLE_ENABLED") and bool(data.get("group_bundle"))
                    and _has_paid_biofield(email)):
                from dashboard import group_bundle
                _gb_months = group_bundle.included_group_months(
                    pc["priced"].get("volume_months", 0))
            out["stripe_url"] = _stripe_checkout_url_for_retail(
                out, email, slug, group_bundle_months=_gb_months)
            if _STRIPE_ACTIVE and not out.get("stripe_url"):
                out["payment_error"] = _CARD_UNAVAILABLE
        try:
            disp = (request.cookies.get("rm_dispensary") or "").strip()
            if disp:
                _record_dispensary_sale(disp, email, qty, inv.get("Id"))
        except Exception as e:
            print(f"[dispensary] hook: {e!r}", flush=True)
        return jsonify(out)
    # else: legacy body unchanged
    else:
        try:
            from dashboard import qbo_billing as qb
            cust = qb.find_or_create_customer(email, name)
            unit = round(_qty_unit_cents(p, qty) / 100.0, 2)   # capsule quantity tiers
            desc = p["name"]
            fmt_label = next((f["label"] for f in _FORMATS if f["id"] == fmt), "")
            if fmt and fmt != "bottle" and fmt_label:
                desc = f"{p['name']} ({fmt_label})"
            allow_online = (method == "card") and _QBO_PAYMENTS_ACTIVE
            from dashboard import tax as _tax
            subtotal_cents = int(_qty_unit_cents(p, qty)) * qty
            # Absorb-and-track: GET is recorded on the order (below), NOT added to the
            # invoice — the customer pays the all-in price.
            get_cents = _tax.compute_get_cents(
                subtotal_cents, channel="retail", ship_to_state=ship.get("state", ""))
            inv = qb.create_invoice(
                cust,
                [{"name": p["name"], "amount": unit, "qty": qty,
                  "item_id": p.get("qbo_item_id"), "description": desc}],
                allow_online_pay=allow_online, email_to=email)
            # best-effort journey log (never break checkout)
            try:
                with _db_lock, sqlite3.connect(LOG_DB) as cx:
                    cx.execute("INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                               "VALUES (?,?,?,?,?,?,?)",
                               (begin_funnel._now(), session_id, email, "purchase",
                                f"buy-{slug}-{method}", "", ""))
                    cx.commit()
            except Exception:
                pass
            out = {"ok": True, "invoice_id": inv.get("Id"), "sync_token": inv.get("SyncToken"),
                   "doc_number": inv.get("DocNumber"),
                   "total": inv.get("TotalAmt"), "method": method,
                   "pay_link": qb.get_invoice_pay_link(inv)}
            out["customer_id"] = cust.get("Id")
            _ingest_order(source="funnel", external_ref=inv.get("Id"), email=email, name=name,
                          items=[{"name": p["name"], "qty": qty, "desc": desc}],
                          total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                          address=ship, channel="retail", get_cents=get_cents)
            if method in ("zelle", "wise"):
                out["pay_instructions"] = _ALT_PAY.get(method, {})
                out["earns_points"] = True   # awarded on confirmed payment (reconciliation, Phase 2)
            elif method == "card" and _STRIPE_ACTIVE:
                out["stripe_url"] = _stripe_checkout_url_for_retail(out, email, slug)
                if not out.get("stripe_url"):
                    out["payment_error"] = _CARD_UNAVAILABLE
            # dispensary attribution: credit the referring practitioner $20/bottle (best-effort,
            # idempotent on the invoice id; never break a customer checkout)
            try:
                disp = (request.cookies.get("rm_dispensary") or "").strip()
                if disp:
                    _record_dispensary_sale(disp, email, qty, inv.get("Id"))
            except Exception as e:
                print(f"[dispensary] hook: {e!r}", flush=True)
            return jsonify(out)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/begin/checkout-return")
def begin_checkout_return():
    """Stripe retail return: verify the session, record the QBO payment, capture
    the PaymentIntent on the order, then back to the buy page (paid)."""
    from flask import redirect as _redir
    sid = (request.args.get("session_id") or "").strip()
    slug = ""
    paid = "0"
    _kind = ""
    if sid:
        try:
            from dashboard import stripe_pay as _sp
            sess = _sp.get_session(sid)
            md = sess.get("metadata") or {}
            slug = md.get("slug", "")
            _kind = md.get("kind", "")
            if sess.get("payment_status") == "paid":
                paid = "1"
                pi_id = sess.get("payment_intent")
                inv, cid = md.get("invoice_id"), md.get("customer_id")
                # In-house pay-link (Phase 3): a card payment on an in-house proposed/
                # confirmed invoice auto-records as paid and drops into fulfillment.
                if _kind == "in-house" and inv:
                    try:
                        _cxih = _sqlite3.connect(LOG_DB); _cxih.row_factory = _sqlite3.Row
                        try:
                            _oih = _bos_orders.find_order_by_external_ref(_cxih, inv)
                            if _oih:
                                _bos_orders.set_order_payment(
                                    _cxih, _oih["id"], method="card",
                                    amount_cents=int(sess.get("amount_total") or 0))
                                if pi_id:
                                    _bos_orders.set_order_stripe_pi(_cxih, _oih["id"], pi_id)
                                # Settle points (redeem applied + earn), idempotent.
                                _bos_orders.settle_order_points(
                                    _cxih, _bos_orders.get_order(_cxih, _oih["id"]))
                        finally:
                            _cxih.close()
                    except Exception as _e:
                        print(f"[begin-return] in-house pay record: {_e!r}", flush=True)
                if inv and cid:
                    try:
                        from dashboard import qbo_billing as _qb_ret
                        _qb_ret.record_payment(cid, int(sess.get("amount_total") or 0), inv)
                    except Exception as e:
                        print(f"[begin-return] qbo payment failed: {e!r}", flush=True)
                    if pi_id:
                        _o_for_points = None
                        try:
                            _cxo = _sqlite3.connect(LOG_DB); _cxo.row_factory = _sqlite3.Row
                            try:
                                _o = _bos_orders.find_order_by_external_ref(_cxo, inv)
                                if _o:
                                    _bos_orders.set_order_stripe_pi(_cxo, _o["id"], pi_id)
                                    _o_for_points = _o  # plain dict; safe after close
                            finally:
                                _cxo.close()
                        except Exception as _e:
                            print(f"[begin-return] pi capture: {_e!r}", flush=True)
                        # ── Points settlement (earn + deduct redeemed) ──
                        if _o_for_points:
                            try:
                                _settle_order_points(_o_for_points, order_ref=inv)
                            except Exception as _pe:
                                print(f"[points] settle failed inv={inv}: {_pe!r}", flush=True)
                            # ── Referral crediting (points or cash to referrer) ──
                            try:
                                _settle_referral(_o_for_points, order_ref=inv)
                            except Exception as _re:
                                print(f"[rewards] referral settle failed inv={inv}: {_re!r}", flush=True)

                # ── Subscribe kind: vault the card and create the subscription row ──
                if md.get("kind") == "subscribe" and pi_id:
                    try:
                        from dashboard import subscriptions as _subs_ret
                        pi_data = _sp.get_payment_intent(pi_id)
                        stripe_cus = pi_data.get("customer") or ""
                        stripe_pm  = pi_data.get("payment_method") or ""
                        cadence    = int(md.get("cadence_months") or 1)
                        sub_email  = md.get("email") or ""

                        # Recover items + ship from metadata or stash
                        stash_key = md.get("stash_key")
                        if stash_key:
                            with sqlite3.connect(LOG_DB) as _scx:
                                _scx.row_factory = sqlite3.Row
                                _sr = _scx.execute(
                                    "SELECT items_json, ship_json FROM pending_subscriptions WHERE key=?",
                                    (stash_key,)).fetchone()
                            if not _sr:
                                app.logger.warning(
                                    "subscribe stash miss for key=%s — subscription will have "
                                    "empty items/address", stash_key)
                            items_list = json.loads(_sr["items_json"]) if _sr else []
                            ship_dict  = json.loads(_sr["ship_json"])  if _sr else {}
                        else:
                            items_list = json.loads(md.get("items") or "[]")
                            ship_dict  = json.loads(md.get("ship")  or "{}")

                        today = _now_utc().strftime("%Y-%m-%d")
                        next_date = _subs_ret.add_months(today, cadence)

                        with sqlite3.connect(LOG_DB) as _scx2:
                            _scx2.row_factory = sqlite3.Row
                            _subs_ret.init_subscriptions_table(_scx2)
                            _subs_ret.create(
                                _scx2,
                                email=sub_email,
                                stripe_customer_id=stripe_cus,
                                stripe_payment_method_id=stripe_pm,
                                items=items_list,
                                cadence_months=cadence,
                                ship_address=ship_dict,
                                next_charge_date=next_date,
                                order_count=1)   # the setup checkout already placed order #1 (5%)
                        print(f"[subscribe-return] subscription created for {sub_email}", flush=True)
                    except Exception as _se:
                        print(f"[subscribe-return] failed to create subscription: {_se!r}", flush=True)

                # ── Group bundle: grant the free live-group window (flag-gated) ──
                # Best-effort, never raises. A per-invoice marker makes a literal
                # re-run of the SAME return a no-op; a genuinely new program order
                # (new invoice) extends an existing membership's window instead of
                # creating a second row.
                if os.environ.get("GROUP_BUNDLE_ENABLED") and int(md.get("grant_group_months") or 0) > 0 and pi_id:
                    try:
                        from dashboard import subscriptions as _subs_gb, group_bundle as _gb
                        from datetime import date as _date_gb
                        n = int(md.get("grant_group_months"))
                        g_email = (md.get("email") or "").strip().lower()
                        g_invoice = (md.get("invoice_id") or pi_id or "").strip()
                        if g_email:
                            pi_d = _sp.get_payment_intent(pi_id)
                            g_cus = pi_d.get("customer") or ""
                            g_pm  = pi_d.get("payment_method") or ""
                            with sqlite3.connect(LOG_DB) as _gcx:
                                _gcx.row_factory = sqlite3.Row
                                _subs_gb.init_subscriptions_table(_gcx)
                                _subs_gb.migrate_add_membership_columns(_gcx)
                                _gcx.execute(
                                    "CREATE TABLE IF NOT EXISTS group_bundle_grants "
                                    "(invoice_id TEXT PRIMARY KEY, created_at TEXT)")
                                _gcx.commit()
                                # Idempotency: skip if this invoice/pi was already granted.
                                already = _gcx.execute(
                                    "SELECT 1 FROM group_bundle_grants WHERE invoice_id=?",
                                    (g_invoice,)).fetchone()
                                if not already:
                                    existing = _subs_gb.active_memberships_by_email(_gcx, g_email)
                                    if existing:
                                        cur = existing[0]
                                        _subs_gb.set_next_charge_date(
                                            _gcx, cur["id"],
                                            _subs_gb.add_months(cur["next_charge_date"], n))
                                    elif g_cus and g_pm:
                                        start = _subs_gb.add_months(_date_gb.today().isoformat(), n)
                                        _subs_gb.create_membership(
                                            _gcx, email=g_email, stripe_customer_id=g_cus,
                                            stripe_payment_method_id=g_pm,
                                            amount_cents=_gb.MEMBERSHIP_AMOUNT_CENTS,
                                            next_charge_date=start)
                                    _gcx.execute(
                                        "INSERT INTO group_bundle_grants (invoice_id, created_at) "
                                        "VALUES (?,?)", (g_invoice, _now_utc().isoformat()))
                                    _gcx.commit()
                                    print(f"[group-bundle] granted {n}mo to {g_email} inv={g_invoice}",
                                          flush=True)
                    except Exception as _ge:
                        app.logger.exception("group bundle grant failed: %s", _ge)

                # ── Client kind: credit practitioner margin to wallet ──
                try:
                    if md.get("kind") == "client":
                        from dashboard import wallet as _wallet_ret
                        _pid   = md.get("practitioner_id")
                        _marg  = int(md.get("margin_cents") or 0)
                        _inv   = md.get("invoice_id")
                        _wallet_ret.earn_dropship_margin(
                            _pid, _marg, qbo_invoice_id=_inv)
                        if hasattr(_pp, "record_dispensary_order"):
                            _pp.record_dispensary_order(
                                _pid,
                                invoice_id=_inv,
                                credit_earned_cents=_marg,
                            )
                        if os.environ.get("CLIENT_POINTS_ENABLED"):
                            try:
                                from dashboard import points as _points_ret
                                p_email = (md.get("patient_email") or "").strip().lower()
                                redeemed = int(md.get("points_redeemed_cents") or 0)
                                subtotal = int(md.get("subtotal_cents") or 0)
                                scope = f"dispensary:{_pid}"
                                if p_email:
                                    with sqlite3.connect(LOG_DB) as _pcx:
                                        _pcx.row_factory = sqlite3.Row
                                        _points_ret.init_points_table(_pcx)
                                        if redeemed > 0:
                                            # redeem() writes reason "redeem"; guard on that
                                            # so a re-run of the same invoice does not double-redeem.
                                            if not _points_ret.has_entry(_pcx, order_ref=_inv,
                                                    reason="redeem", scope=scope):
                                                take = min(redeemed,
                                                           _points_ret.balance(_pcx, p_email, scope=scope))
                                                if take > 0:
                                                    _points_ret.redeem(_pcx, p_email, value_cents=take,
                                                                       order_ref=_inv, scope=scope)
                                        elif subtotal > 0:
                                            earn_pct = float(_pricing_settings().get("points_earn_pct", 0.05))
                                            _points_ret.credit(_pcx, p_email,
                                                value_cents=round(subtotal * earn_pct),
                                                reason="earn:dispensary", order_ref=_inv, scope=scope)
                            except Exception as _pe:
                                app.logger.exception("client points settle failed: %s", _pe)
                except Exception as _ce:
                    app.logger.exception(
                        "client margin credit failed: %s", _ce)

                # ── Biofield kind: seed readiness + settle points (best-effort) ──
                if md.get("kind") == "biofield":
                    try:
                        from dashboard import biofield_store as _bf
                        bf_email = (md.get("email") or "").strip().lower()
                        bf_inv = md.get("invoice_id") or ""
                        if bf_email:
                            _bcx = _sqlite3.connect(LOG_DB)
                            try:
                                _bf.init_table(_bcx)
                                _bf.seed_paid(_bcx, bf_email, via="stripe", order_ref=bf_inv)
                            finally:
                                _bcx.close()
                            # Settle loyalty points on the recorded order (idempotent).
                            try:
                                _bcxo = _sqlite3.connect(LOG_DB); _bcxo.row_factory = _sqlite3.Row
                                try:
                                    _bo = _bos_orders.find_order_by_external_ref(_bcxo, bf_inv)
                                finally:
                                    _bcxo.close()
                                if _bo:
                                    _settle_order_points(_bo, order_ref=bf_inv)
                            except Exception as _bpe:
                                print(f"[biofield] points settle failed inv={bf_inv}: {_bpe!r}",
                                      flush=True)
                    except Exception as _be:
                        print(f"[biofield] return seed failed: {_be!r}", flush=True)

        except Exception as e:
            print(f"[begin-return] {e!r}", flush=True)
    # Biofield checkouts land on the readiness gate; everything else returns to the funnel.
    if _kind == "biofield":
        return _redir(f"/biofield/ready?paid={paid}")
    dest = (f"/begin/buy/{slug}?paid={paid}" if slug else f"/begin?paid={paid}")
    return _redir(dest)


# ── Post-buy concierge (consultative upsell) ─────────────────────────────────
_PAIRINGS = _load_json(DATA_DIR / "upsell-pairings.json", default={"pairings": {}})
# pinecone_title -> catalog slug (deterministic in-catalog resolution; avoids the
# "Stress Release" vs "Emotional Stress Release" false match).
_TITLE_TO_SLUG = {(p.get("pinecone_title") or p.get("name")): s
                  for s, p in (_PRODUCTS.get("products") or {}).items()}
_COMPLEMENT_CACHE = {}


def _resolve_complement(name):
    """Resolve a complement product NAME to {name, title, url, price, slug, in_catalog}.
    in_catalog (slug present) -> addable to the invoice; else -> open on the store."""
    if not name:
        return None
    key = name.strip().lower()
    if key in _COMPLEMENT_CACHE:
        return _COMPLEMENT_CACHE[key]
    title = url = price = None
    try:
        vec = embed(name)
        res = _idx.query(vector=vec, top_k=1, namespace="specific-formulations", include_metadata=True)
        if res.matches and res.matches[0].score >= 0.83:
            md = res.matches[0].metadata or {}
            title, url, price = md.get("title"), md.get("url"), md.get("price")
    except Exception as e:
        print(f"[concierge] resolve {name}: {e}", flush=True)
    slug = _TITLE_TO_SLUG.get(title) if title else None
    out = {"name": name, "title": title, "url": url, "price": price,
           "slug": slug, "in_catalog": bool(slug)}
    _COMPLEMENT_CACHE[key] = out
    return out


_CONCIERGE_SYSTEM = (
    "You are Dr. Glen Swartwout's warm post-purchase concierge (naturopathic physician, Hilo "
    "Hawai'i). The person just ordered a remedy. Your job is to help them complete their protocol "
    "in a calm, consultative, concierge way: they should feel served and in control, because they "
    "are.\n\n"
    "How you work:\n"
    "- Open by affirming their choice and what it supports. Then ask ONE gentle question at a time "
    "to understand their fuller goal or terrain (energy, sleep, stress, digestion, what else they "
    "are working on).\n"
    "- When it fits, suggest ONE complementary remedy at a time, with a short plain reason it pairs "
    "well with what they bought. Prefer the SUGGESTED COMPLEMENTS provided; you may go beyond them "
    "if the person's needs point elsewhere. Functional Formulations first.\n"
    "- Never pressure. After a suggestion, invite them to add it or keep exploring, and make clear "
    "they can stop anytime and they are all set whenever they choose.\n"
    "- Keep replies short and warm. Do not invent prices or URLs. Sign off warmly as Dr. Glen only "
    "when concluding.\n"
    "- Style: do NOT use em dashes (use commas, colons, or periods). Do NOT use ALL CAPS. Never "
    "prefix anything with the word 'Hook:'."
)

_CONCIERGE_EXTRACT_SYSTEM = (
    "You read a concierge turn. If the assistant is suggesting ONE specific product to add now, "
    "return JSON {\"suggest\": true, \"name\": \"<exact product name>\"}. If it is just asking a "
    "question or chatting with no single specific product offered, return {\"suggest\": false}. "
    "Output ONLY the JSON, no prose, no code fences."
)


@app.route("/begin/concierge/chat", methods=["POST", "OPTIONS"])
def begin_concierge_chat():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    history = data.get("history") or []
    bought_slug = (data.get("bought_slug") or "").strip()
    bought = _get_product(bought_slug) if bought_slug else None
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip() or uuid.uuid4().hex)
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Pairing priors for what they bought + a little RAG for rationale/benefits.
    priors = (_PAIRINGS.get("pairings", {}) or {}).get(bought_slug, []) if bought_slug else []
    priors_block = (f"SUGGESTED COMPLEMENTS for {bought['name'] if bought else 'their purchase'} "
                    f"(offer these first, one at a time): {', '.join(priors)}\n\n") if priors else ""
    context_str = ""
    try:
        matches = _match_query_namespaces(embed(query + " " + (bought["name"] if bought else "")))
        context_str, _ = build_context(matches) if matches else ("", [])
    except Exception as e:
        print(f"[concierge] retrieval: {e}", flush=True)

    def generate():
        messages = []
        for turn in history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content":
            f"THEY JUST BOUGHT: {bought['name'] if bought else 'a remedy'}.\n"
            f"{priors_block}"
            f"RETRIEVED SNIPPETS (for rationale/benefits):\n{context_str}\n\n"
            f"MEMBER MESSAGE: {query}\n\n"
            "Continue as the concierge: affirm, ask the single best next question, or suggest ONE "
            "complement with a short why. Keep it warm and brief."})
        full = []
        try:
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=700,
                                     system=_CONCIERGE_SYSTEM, messages=messages) as stream:
                for tok in stream.text_stream:
                    tok = _strip_dash(tok); full.append(tok); yield sse({"token": tok})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"}); return
        answer = "".join(full)

        # Extract a single suggested complement (separate call) and resolve it.
        try:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-2:]) + f"\nassistant: {answer}"
            mx = _cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=120,
                                     system=_CONCIERGE_EXTRACT_SYSTEM,
                                     messages=[{"role": "user", "content": convo[:3500]}])
            txt = mx.content[0].text.strip()
            if txt.startswith("```"):
                txt = txt.split("```", 2)[1]
                if txt.startswith("json\n"): txt = txt[5:]
            obj = json.loads(txt)
            if obj.get("suggest") and obj.get("name"):
                c = _resolve_complement(obj["name"])
                if c and (c["in_catalog"] or c["url"]):
                    yield sse({"suggestion": c})
        except Exception as e:
            print(f"[concierge] extract: {e!r}", flush=True)
        yield sse({"done": True, "session_id": session_id})

    resp = Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return resp


@app.route("/begin/concierge/add", methods=["POST"])
def begin_concierge_add():
    """Add a catalog complement to the member's existing (unpaid) invoice."""
    data = request.get_json(silent=True) or {}
    slug = (data.get("slug") or "").strip()
    invoice_id = (data.get("invoice_id") or "").strip()
    p = _get_product(slug)
    if not p or p.get("info_only"):
        return jsonify({"ok": False, "error": "not an addable catalog product"}), 400
    if not invoice_id:
        return jsonify({"ok": False, "error": "invoice_id required"}), 400
    try:
        qty = max(1, min(int(data.get("qty", 1) or 1), 99))
    except Exception:
        qty = 1
    try:
        from dashboard import qbo_billing as qb
        unit = round(p["price_cents"] / 100.0, 2)
        inv = qb.add_invoice_line(invoice_id, name=p["name"], amount=unit, qty=qty,
                                  item_id=p.get("qbo_item_id"), description=p["name"])
        return jsonify({"ok": True, "added": p["name"], "qty": qty,
                        "invoice_id": inv.get("Id"), "sync_token": inv.get("SyncToken"),
                        "total": inv.get("TotalAmt")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/qbo/price-coverage", methods=["GET"])
def qbo_price_coverage():
    """Diagnostic: for each products.json product, query specific-formulations for a
    price (the last scrape stored price_your/price_retail). Shows match title + score
    so we can judge coverage + match quality before populating products.json."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    out = []
    for slug, p in (_PRODUCTS.get("products") or {}).items():
        name = p["name"]
        try:
            vec = embed(name)
            res = _idx.query(vector=vec, top_k=4, namespace="specific-formulations",
                             include_metadata=True)
            best = None
            for m in res.matches:
                md = m.metadata or {}
                price = md.get("price") or md.get("price_your") or md.get("price_retail")
                if price:
                    best = {"match_title": md.get("title", ""), "price": str(price),
                            "score": round(float(m.score), 3)}
                    break
            row = {"slug": slug, "name": name, "found": bool(best)}
            if best:
                row.update(best)
            out.append(row)
        except Exception as e:
            out.append({"slug": slug, "name": name, "error": str(e)[:120]})
    return jsonify({"ok": True, "covered": sum(1 for o in out if o.get("found")),
                    "total": len(out), "products": out})


@app.route("/api/qbo/pay-methods", methods=["GET"])
def qbo_pay_methods():
    """Read-only: confirm the Zelle/Wise pay-to text (from env) renders correctly,
    without creating an invoice."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"payments_active": _QBO_PAYMENTS_ACTIVE, "alt_pay": _ALT_PAY})


# ── Phase 2B — full-report endpoint (View full / Email full) ─────────────────
@app.route("/full-report", methods=["POST", "OPTIONS"])
def full_report():
    """Regenerate the original query in mode=full. If email is provided,
    send the result via SMTP and tag the GHL contact. Otherwise stream
    the full content back inline (same SSE protocol as /chat).
    """
    if request.method == "OPTIONS":
        return "", 200

    data    = request.get_json() or {}
    log_id  = data.get("log_id")
    email   = (data.get("email") or "").strip().lower()
    name    = (data.get("name") or "").strip()
    if not log_id:
        return jsonify({"error": "log_id required"}), 400

    # Look up the original query
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT query, level, session_id FROM query_log WHERE id = ?",
            (log_id,)
        ).fetchone()
    if not row:
        return jsonify({"error": "log_id not found"}), 404

    original_query = row["query"]
    level          = row["level"] or "self-healing"
    session_id     = row["session_id"]

    # If user is authenticated, prefer their identity
    auth_user = get_authenticated_user(request)
    if auth_user:
        email = email or auth_user["email"]
        name  = name  or (auth_user.get("name") or "")

    # Gate Break & Rebuild long-form on a real logged-in identity. Anonymous
    # full-reports (rare — they require an email but no auth) keep the
    # existing extended clinical format.
    is_logged_in = bool(auth_user)

    # If email provided, send via SMTP/console (synchronous since user is waiting).
    # If no email, stream the response back as SSE for inline rendering.
    if email:
        return _full_report_send_email(
            log_id, original_query, level, email, name, session_id, is_logged_in
        )
    return _full_report_stream(log_id, original_query, level, session_id, is_logged_in)


def _generate_full_answer(query: str, level: str, is_logged_in: bool = False):
    """Run the same retrieval + synthesis pipeline as /chat but in
    mode=full (synchronous; used when the email path needs the body
    before sending).

    When is_logged_in is True the synthesis follows the Break & Rebuild
    teaching arc; otherwise it uses the existing extended clinical format.

    Returns (answer_str, sources_list, chunks_retrieved_int).
    """
    q_vec = embed(query)
    matches = query_all_namespaces(q_vec)
    if not matches:
        return ("No relevant content found.", [], 0)

    context_str, sources_list = build_context(matches)
    product_directive = build_product_directive(
        snippets_text=(context_str or "")
    )
    product_block = f"{product_directive}\n\n" if product_directive else ""

    synth_instr = _long_form_synth_instr(is_logged_in)
    user_msg = (
        f"USER QUESTION: {query}\n\n"
        f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
        f"{product_block}"
        f"{synth_instr}"
    )

    answer = ""
    msg = _cl.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=get_system_prompt(level),
        messages=[{"role": "user", "content": user_msg}],
    )
    if msg.content:
        answer = msg.content[0].text
    return (answer, sources_list, len(matches))


def _send_full_report_email(to_email: str, name: str,
                            subject: str, body: str):
    """Send the full report — tries Gmail API → SMTP → console log.
    Returns (sent_via, error_or_none).

    Gmail API path is preferred because it reuses the same OAuth token as
    the inbox feature (no extra SMTP_USER/PASS to manage).
    """
    # Path 1: Gmail API (preferred — reuses inbox auth)
    try:
        from dashboard.inbox import send_email as _gmail_send
        _gmail_send(to_email, subject, body)
        return ("gmail-api", None)
    except Exception as e:
        print(f"[full-report] Gmail API send failed: {e}", flush=True)

    # Path 2: SMTP (fallback — only if env vars set)
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    if smtp_host and smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"]    = smtp_from
            msg["To"]      = to_email
            port = int(os.environ.get("SMTP_PORT", "587"))
            with smtplib.SMTP(smtp_host, port, timeout=15) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_from, [to_email], msg.as_string())
            return ("smtp", None)
        except Exception as e:
            print(f"[full-report] SMTP fallback also failed: {e}", flush=True)

    # Path 3: console log (last resort — dev / nothing configured)
    print(f"\n[full-report] TO: {to_email}\nSUBJECT: {subject}\n\n{body}\n",
          flush=True)
    return ("console-log", "no email-send mechanism configured")


def _full_report_send_email(log_id, query, level, email, name, session_id,
                            is_logged_in: bool = False):
    """Synchronous full-mode regeneration → email send → GHL tag + log."""
    full_answer, sources, chunks = _generate_full_answer(query, level, is_logged_in)

    subject = f"Your full report from Dr. Glen: {query[:60]}"
    body = (
        f"Hi {name or ''},\n\n"
        f"Here's the full clinical breakdown for your question.\n\n"
        f"YOUR QUESTION: {query}\n\n"
        f"DR. GLEN'S RESPONSE:\n\n"
        f"{full_answer}\n\n"
        f"— Dr. Glen Swartwout\n"
        f"{PUBLIC_BASE_URL}/"
    )

    sent_via, _err = _send_full_report_email(email, name, subject, body)

    # Tag the GHL contact (best-effort, non-blocking failure)
    try:
        topic_slug = re.sub(r'[^a-z0-9]+', '-', query.lower()[:30]).strip('-')
        parts = name.split(None, 1) if name else []
        first = parts[0] if parts else ""
        last  = parts[1] if len(parts) > 1 else ""
        ghl_onboard_contact(
            email, first, last,
            source_tag="chatbot-fullreport",
            extra_tags=["chatbot-fullreport", f"topic-{topic_slug}"]
        )
    except Exception as e:
        print(f"[full-report] GHL onboard failed: {e}", flush=True)

    # Update email_sent_at in query_log (best-effort)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "UPDATE query_log SET email_sent_at = ? WHERE id = ?",
                (now_iso, log_id)
            )
            cx.commit()
    except Exception as e:
        print(f"[full-report] log update failed: {e}", flush=True)

    return jsonify({
        "ok":       True,
        "sent_via": sent_via,
        "to":       email,
        "preview":  (full_answer[:200] + "...") if full_answer else "",
    })


def _full_report_stream(log_id, query, level, session_id,
                        is_logged_in: bool = False):
    """Stream the full-mode regeneration via SSE, like /chat. The
    front-end replaces the brief inline body with this stream.
    When is_logged_in is True the synthesis follows the Break & Rebuild
    teaching arc; otherwise it uses the existing extended clinical format.
    """
    def generate():
        try:
            q_vec = embed(query)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"})
            return

        matches = query_all_namespaces(q_vec)
        if not matches:
            yield sse({"done": True, "answer": "No relevant content found.",
                       "sources": [], "chunks_retrieved": 0,
                       "log_id": log_id, "mode": "full"})
            return

        context_str, sources_list = build_context(matches)
        product_directive = build_product_directive(
            snippets_text=(context_str or "")
        )
        product_block = f"{product_directive}\n\n" if product_directive else ""

        synth_instr = _long_form_synth_instr(is_logged_in)
        user_msg = (
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
            f"{product_block}"
            f"{synth_instr}"
        )

        try:
            with _cl.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=get_system_prompt(level),
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for tok in stream.text_stream:
                    yield sse({"token": tok})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"})
            return

        yield sse({"done": True, "sources": sources_list,
                   "chunks_retrieved": len(matches), "log_id": log_id,
                   "mode": "full"})

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/auth/magic-link/request", methods=["POST", "OPTIONS"])
def auth_magic_link_request():
    """Generate a magic-link token, email it, and stash the hash in
    auth_tokens table. Returns 200 always (no email enumeration leak).
    """
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    if not email or "@" not in email:
        return jsonify({"error": "Valid email required"}), 400

    token = secrets.token_urlsafe(32)
    th    = _hash_token(token)
    now   = _now_utc()
    expires = now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            """INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at)
               VALUES (?,?,?,?,?)""",
            (th, email, "magic_link", now.isoformat(), expires.isoformat())
        )
        cx.commit()

    magic_url = f"{PUBLIC_BASE_URL}/auth/magic-link/verify?token={token}"
    sent_via, err = send_magic_link_email(email, name, magic_url)

    return jsonify({
        "ok":       True,
        "sent_via": sent_via,
        "expires_in_minutes": AUTH_TOKEN_TTL_MIN,
        "note":     ("Check your email for the sign-in link."
                     if sent_via in ("ghl-workflow", "smtp")
                     else "Email sending not yet configured. Check Render logs for the magic link.")
    })


@app.route("/auth/magic-link/verify", methods=["GET"])
def auth_magic_link_verify():
    """Click target for the magic link. Validates token, creates user (if
    new) + session, sets HttpOnly auth cookie, redirects to /.
    """
    token = (request.args.get("token") or "").strip()
    if not token:
        return jsonify({"error": "Missing token"}), 400
    th = _hash_token(token)

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens WHERE token_hash = ?",
            (th,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Invalid or expired token"}), 400
        if row["consumed_at"]:
            return jsonify({"error": "Token already used"}), 400
        try:
            expires = datetime.fromisoformat(row["expires_at"])
            if expires < _now_utc():
                return jsonify({"error": "Token expired"}), 400
        except Exception:
            return jsonify({"error": "Token corrupted"}), 400

        email = row["email"]
        # Mark the token consumed
        cx.execute("UPDATE auth_tokens SET consumed_at = ? WHERE token_hash = ?",
                   (_now_utc().isoformat(), th))

        # Find or create user
        u = cx.execute("SELECT id, name, ghl_contact_id FROM users WHERE email = ?",
                        (email,)).fetchone()
        if u:
            user_id = u["id"]
            cx.execute("UPDATE users SET last_login_at = ? WHERE id = ?",
                       (_now_utc().isoformat(), user_id))
        else:
            cur = cx.execute(
                """INSERT INTO users (email, auth_method, created_at, last_login_at)
                   VALUES (?,?,?,?)""",
                (email, "magic_link", _now_utc().isoformat(), _now_utc().isoformat())
            )
            user_id = cur.lastrowid

        # Create session
        sess = secrets.token_urlsafe(32)
        sh   = _hash_token(sess)
        sess_expires = _now_utc() + timedelta(days=SESSION_TTL_DAYS)
        cx.execute(
            """INSERT INTO sessions (token_hash, user_id, created_at, expires_at, ip, user_agent)
               VALUES (?,?,?,?,?,?)""",
            (sh, user_id, _now_utc().isoformat(), sess_expires.isoformat(),
             request.headers.get("X-Forwarded-For", request.remote_addr or "")[:64],
             request.headers.get("User-Agent", "")[:500])
        )
        cx.commit()

    # GHL reconciliation in background — non-blocking
    def _reconcile():
        try:
            cid, _, err = ghl_upsert_contact(email, "", "", source_tag="chatbot-login")
            if cid:
                with _db_lock, sqlite3.connect(LOG_DB) as cx2:
                    cx2.execute("UPDATE users SET ghl_contact_id = ? WHERE id = ?",
                                (cid, user_id))
                    cx2.commit()
        except Exception as e:
            print(f"[auth] GHL reconciliation failed: {e}", flush=True)
    threading.Thread(target=_reconcile, daemon=True).start()

    # Set auth cookie + redirect to chat home
    resp = Response(
        f"""<!DOCTYPE html><html><head><title>Signed in</title>
        <meta http-equiv=\"refresh\" content=\"1;url=/\"/>
        <style>body{{font-family:sans-serif;background:#0a150d;color:#fdf4d8;
        text-align:center;padding-top:80px}}</style></head>
        <body><h2>You're signed in.</h2>
        <p>Redirecting to chat… or <a href=\"/\" style=\"color:#d4a843\">click here</a>.</p>
        </body></html>""",
        mimetype="text/html",
    )
    resp.set_cookie(
        "amg_auth", sess,
        max_age=60 * 60 * 24 * SESSION_TTL_DAYS,
        httponly=True, samesite="Lax",
        secure=request.is_secure,
    )
    return resp


@app.route("/auth/logout", methods=["POST", "OPTIONS"])
def auth_logout():
    if request.method == "OPTIONS":
        return "", 200
    tok = request.cookies.get("amg_auth", "")
    if tok:
        th = _hash_token(tok)
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("DELETE FROM sessions WHERE token_hash = ?", (th,))
            cx.commit()
    resp = jsonify({"ok": True})
    resp.set_cookie("amg_auth", "", max_age=0, httponly=True, samesite="Lax",
                    secure=request.is_secure)
    return resp


@app.route("/history", methods=["GET"])
def history():
    """Return up to N (default 30) most recent Q&A pairs logged against the
    current session cookie. Drives both:
      (a) cross-page-reload conversation memory — the front-end re-renders
          past turns so multi-turn questions work after browser refresh.
      (b) debugging/development — visibility into what the bot has been
          answering for any visitor identified by their session cookie.

    Order: oldest → newest (chronological — natural for chat re-render).
    Image content (extracted_image_data) is included as a separate field
    when present; image bytes were never stored.
    """
    sid = request.cookies.get("amg_session", "").strip()
    if not sid:
        return jsonify({"turns": []})
    try:
        limit = int(request.args.get("limit", "30"))
    except (TypeError, ValueError):
        limit = 30
    limit = max(1, min(limit, 100))

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            """SELECT id, ts, query, answer, mode, level, image_count,
                      extracted_image_data
               FROM query_log
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (sid, limit),
        ).fetchall()
    rows = list(reversed(rows))  # oldest → newest
    turns = [{
        "log_id":           r["id"],
        "ts":               r["ts"],
        "query":            r["query"] or "",
        "answer":           r["answer"] or "",
        "mode":             r["mode"] or "brief",
        "level":            r["level"] or "self-healing",
        "image_count":      r["image_count"] or 0,
        "extracted_image_data": r["extracted_image_data"] or "",
    } for r in rows]
    return jsonify({"turns": turns})


@app.route("/me", methods=["GET"])
def me():
    """Return the current visitor's contact info.
    Priority:
      1. Authenticated user (auth_token cookie → users table)
      2. Anonymous session cookie → most recent query_log row with contact

    Returns:
      {email, name, authenticated: bool, user_id: int|null}
    """
    auth_user = get_authenticated_user(request)
    if auth_user:
        return jsonify({
            "email":         auth_user["email"],
            "name":          auth_user["name"] or "",
            "authenticated": True,
            "user_id":       auth_user["id"],
        })

    sid = request.cookies.get("amg_session", "").strip()
    if not sid:
        return jsonify({"authenticated": False})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            """SELECT email, name FROM query_log
               WHERE session_id = ? AND (email != '' OR name != '')
               ORDER BY id DESC LIMIT 1""",
            (sid,)
        ).fetchone()
    if not row:
        return jsonify({"authenticated": False})
    return jsonify({
        "email":         row["email"] or "",
        "name":          row["name"] or "",
        "authenticated": False,
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


def ghl_upsert_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None, custom_fields=None):
    """Find or create a GHL contact. Returns (contact_id, created_bool, error).

    custom_fields: optional dict of {field_key: value} pairs sent as GHL
    customField array. Applied after the tag-merge PUT for existing contacts,
    or via a separate PUT after creation for new contacts."""
    if not GHL_API_KEY:
        return None, False, "GHL_API_KEY not set"

    all_new_tags = set()
    if source_tag:
        all_new_tags.add(source_tag)
    if extra_tags:
        all_new_tags.update(extra_tags)

    custom_field_payload = None
    if custom_fields:
        custom_field_payload = [
            {"key": k, "field_value": v} for k, v in custom_fields.items()
        ]

    # Try to find existing contact via GHL v1 /contacts/lookup?email= (the
    # /contacts/?email= query param does NOT actually filter — it returns
    # random contacts and silently caused duplicates). lookup returns ALL
    # contacts matching the email; we prefer the oldest one when multiple
    # historical entries exist for the same address.
    data, err = _ghl_get("/contacts/lookup", {"email": email})
    if not err:
        contacts = data.get("contacts", [])
        if contacts:
            match = min(contacts, key=lambda c: c.get("dateAdded") or "9999")
            contact_id = match["id"]
            if all_new_tags:
                existing_tags = set(match.get("tags", []))
                existing_tags.update(all_new_tags)
                _ghl_put(f"/contacts/{contact_id}", {"tags": list(existing_tags)})
            if custom_field_payload:
                _ghl_put(f"/contacts/{contact_id}", {"customField": custom_field_payload})
            return contact_id, False, None

    # Create new contact
    payload = {"email": email, "firstName": first_name, "lastName": last_name}
    if phone:
        payload["phone"] = phone
    if all_new_tags:
        payload["tags"] = list(all_new_tags)
    if custom_field_payload:
        payload["customField"] = custom_field_payload

    data, err = _ghl_post("/contacts/", payload)
    if err:
        return None, False, err

    contact_id = data.get("contact", {}).get("id") or data.get("id")
    return contact_id, True, None


def ghl_update_tags(email, add=None, remove=None):
    """Find GHL contact via /contacts/lookup (the correct exact-email endpoint
    per the 2026-05-26 fix), add and/or remove tags as set operations, PUT
    the merged result. Returns (contact_id, error).

    If no contact exists for that email, falls through to ghl_upsert_contact
    so the contact gets created with the `add` tags. `remove` on a non-existent
    contact is a no-op (returns (None, None))."""
    add    = set(add or [])
    remove = set(remove or [])
    if not (add or remove):
        return None, "no tags specified"
    if not GHL_API_KEY:
        return None, "GHL_API_KEY not set"

    data, err = _ghl_get("/contacts/lookup", {"email": email})
    if err:
        return None, err
    contacts = data.get("contacts", []) if isinstance(data, dict) else []
    if not contacts:
        if not add:
            return None, None   # nothing to do — no contact, nothing to remove from
        # Fall through to create — preserves first/last so the new contact has names
        contact_id, _created, err = ghl_upsert_contact(email, extra_tags=list(add))
        return contact_id, err

    # Prefer oldest contact when multiple match (matches ghl_upsert_contact's behavior)
    match = min(contacts, key=lambda c: c.get("dateAdded") or "9999")
    existing = set(match.get("tags", []) or [])
    new_tags = (existing | add) - remove
    if new_tags == existing:
        return match["id"], None   # nothing changed; skip the PUT
    _, err = _ghl_put(f"/contacts/{match['id']}", {"tags": sorted(new_tags)})
    return match["id"], err


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


def _resolve_channel_tags(personal: bool = False,
                           newsletter: bool = False,
                           is_beta: bool = False) -> list:
    """Map the front-end's channel-opt-in booleans to GHL tags.
    Replaces the older frequency-* tags. Both old and new tags can
    coexist during transition; the engine reads the new tags only."""
    tags = ["chatbot-lead"]
    if personal:
        tags.append("personal-email-opt-in")
    if newsletter:
        tags.append("newsletter-opt-in")
    if is_beta:
        tags.append("beta-personal-email")
    return tags


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
        for col in ["instructions TEXT DEFAULT ''"]:
            try:
                cx.execute(f"ALTER TABLE affiliate_offers ADD COLUMN {col}")
            except Exception:
                pass
        # Conversions credited to an affiliate (store purchases + course enrollments),
        # attributed by email-match against referral_events.
        cx.execute("""
            CREATE TABLE IF NOT EXISTS affiliate_conversions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at     TEXT NOT NULL,
                email           TEXT,
                affiliate_slug  TEXT,
                conversion_type TEXT,
                detail          TEXT,
                order_value     REAL,
                source          TEXT,
                raw_json        TEXT
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_affiliate_conversions_slug ON affiliate_conversions(affiliate_slug)")
        # Seed quiz as first offer
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Accelerate Self-Healing Quiz'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, active)
                VALUES (1, 'Accelerate Self-Healing Quiz',
                    'Free quiz — discover your top healing opportunities. Share with anyone curious about natural healing.',
                    'https://healing.scoreapp.com?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz',
                    1)
            """)
        # Seed E4L bioenergetic wellness scan
        E4L_INSTRUCTIONS = (
            "New here? Get a free account at https://truly.vip/E4L\n"
            "Already have an account? Log in at https://portal.E4L.com\n"
            "\n"
            "Once logged in:\n"
            "1. Click the Scans tab\n"
            '2. Click "Voice Scan"\n'
            "3. Follow the prompts\n"
            "4. Count out loud: 1 to 10\n"
            "5. View your scan\n"
            "\n"
            "If you don't see 'Scans' anywhere, send us a note at support@RemedyMatch.com"
        )
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Free Bioenergetic Wellness Scan'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, instructions, active)
                VALUES (2, 'Free Bioenergetic Wellness Scan',
                    'A free voice-based bioenergetic scan from E4L — reveals the body''s current wellness priorities in minutes.',
                    'https://truly.vip/E4L?utm_source={slug}&utm_medium=affiliate&utm_campaign=e4l-scan',
                    ?,
                    1)
            """, (E4L_INSTRUCTIONS,))
        else:
            # Keep existing prod row's instructions in sync with the canonical seed.
            cx.execute(
                "UPDATE affiliate_offers SET instructions=? "
                "WHERE name='Free Bioenergetic Wellness Scan' AND instructions != ?",
                (E4L_INSTRUCTIONS, E4L_INSTRUCTIONS),
            )
        # Seed ASH MasterClass (free evergreen intro on Practice Better)
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Free ASH MasterClass'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, instructions, active)
                VALUES (3, 'Free ASH MasterClass',
                    'Dr. Glen''s evergreen introduction to the Accelerated Self Healing™ method. Free MasterClass on Practice Better — always available.',
                    'https://truly.vip/Intro?utm_source={slug}&utm_medium=affiliate&utm_campaign=ash-masterclass',
                    'Free on Practice Better — students create a free account to access. Affiliates can share the link as-is.',
                    1)
            """)
        # Seed DIY ASH Course — Heal Yourself (free self-paced on Practice Better)
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Free DIY Accelerated Self Healing Course — Heal Yourself'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, instructions, active)
                VALUES (4, 'Free DIY Accelerated Self Healing Course — Heal Yourself',
                    'The full DIY protocol for Accelerated Self Healing™. Free self-paced course on Practice Better — work through the modules at your own pace.',
                    'https://truly.vip/GetWell?utm_source={slug}&utm_medium=affiliate&utm_campaign=ash-diy-course',
                    'Free on Practice Better — students create a free account to access. Affiliates can share the link as-is.',
                    1)
            """)
        # Seed Shop for Remedies (the GrooveKart store)
        if not cx.execute("SELECT id FROM affiliate_offers WHERE name='Shop for Remedies'").fetchone():
            cx.execute("""
                INSERT INTO affiliate_offers (sort_order, name, description, url_template, instructions, active)
                VALUES (5, 'Shop for Remedies',
                    'Dr. Glen''s full line of remedies and formulations at RemedyMatch.com. Share with anyone ready to start their protocol.',
                    'https://remedymatch.com?utm_source={slug}&utm_medium=affiliate&utm_campaign=store',
                    'Direct link to the store. Purchases are credited to you automatically when the buyer first came through one of your free-offer links (same email). Cold store visitors aren''t tracked yet — coupon codes for that are a future add-on.',
                    1)
            """)
        # ── Funnel offers (funnel-first hybrid) ──────────────────────────────
        # The funnel front door is now the primary share link: it captures the
        # ref into journey_state once and carries it through every step + any
        # later conversion. The ?want= deep-links route THROUGH the funnel (so
        # the referral sticks) and then redirect outward with utm_source threaded.
        # Direct external offers below are demoted to a secondary "advanced" tier.
        _FUNNEL_OFFERS = [
            (1, 'Your Front Door — start here',
             "Your main link. Send anyone here first. The page meets each visitor where "
             "they are and walks them to the right next step, and every result is credited "
             "to you. Best for cold or curious people who don't know Dr. Glen yet.",
             f"{PUBLIC_BASE_URL}/?ref={{slug}}", ''),
            (2, 'Free Wellness Scan (E4L)',
             "Sends people to the free voice-based bioenergetic scan, routed through your "
             "front door so the referral always sticks. Best for anyone open to a quick, "
             "free assessment.",
             f"{PUBLIC_BASE_URL}/?ref={{slug}}&want=e4l", ''),
            (3, 'Self-Healing Quiz',
             "Sends people to the free Accelerate Self-Healing quiz through your front door. "
             "Best for a broad audience curious about natural healing.",
             f"{PUBLIC_BASE_URL}/?ref={{slug}}&want=quiz", ''),
            (4, 'Explore the Path & Tiers',
             "Opens the full ladder of ways to work with Dr. Glen, from the free course up "
             "to the consultant package. Best for warm people ready to see their options.",
             f"{PUBLIC_BASE_URL}/?ref={{slug}}&want=ascend", ''),
            (5, 'Talk to Dr. Glen (Join)',
             "Routes to the consultative intake to start working with Dr. Glen directly. "
             "Best for the most ready, high-intent people.",
             f"{PUBLIC_BASE_URL}/?ref={{slug}}&want=join", ''),
            (6, 'Explore Everything: the full map',
             "Opens the full table of contents of everything available, every room on "
             "one page, so people can wander and pick what draws them. Best for "
             "self-directed people who would rather browse than be guided step by step. "
             "The link sets your referral, so every step and any later purchase is credited to you.",
             f"{PUBLIC_BASE_URL}/begin/explore?ref={{slug}}", ''),
        ]
        for sort_order, oname, odesc, ourl, oinstr in _FUNNEL_OFFERS:
            existing_offer = cx.execute(
                "SELECT id FROM affiliate_offers WHERE name=?", (oname,)).fetchone()
            if not existing_offer:
                cx.execute(
                    "INSERT INTO affiliate_offers (sort_order, name, description, url_template, instructions, active) "
                    "VALUES (?,?,?,?,?,1)",
                    (sort_order, oname, odesc, ourl, oinstr))
            else:
                # Keep the live row's ordering, copy, and URL in sync with the seed.
                cx.execute(
                    "UPDATE affiliate_offers SET sort_order=?, description=?, url_template=? WHERE name=?",
                    (sort_order, odesc, ourl, oname))
        # Demote the direct external offers below the funnel links + flag them as
        # advanced. Idempotent: only bump sort_order while it's still in the
        # funnel-link range, and only prepend the note once.
        _DIRECT_OFFER_NAMES = (
            'Accelerate Self-Healing Quiz',
            'Free Bioenergetic Wellness Scan',
            'Free ASH MasterClass',
            'Free DIY Accelerated Self Healing Course — Heal Yourself',
            'Shop for Remedies',
        )
        _ADVANCED_NOTE = ("Advanced direct link. The funnel links above preserve full "
                          "attribution across every step. ")
        for oname in _DIRECT_OFFER_NAMES:
            cx.execute(
                "UPDATE affiliate_offers SET sort_order = sort_order + 20 "
                "WHERE name=? AND sort_order < 20", (oname,))
            cx.execute(
                "UPDATE affiliate_offers SET description = ? || description "
                "WHERE name=? AND description NOT LIKE ?",
                (_ADVANCED_NOTE, oname, _ADVANCED_NOTE + '%'))
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


# ── /begin funnel journey-state engine ───────────────────────────────────────

def _init_journey_tables():
    with sqlite3.connect(LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)

_init_journey_tables()


# ── Practitioner inquiry bridge — SQLite tables ──────────────────────────────

def init_inquiry_tables(cx):
    """Idempotent CREATE TABLE IF NOT EXISTS for the inquiry bridge.
    Called on app startup (alongside init_journey_tables) and wherever
    init_journey_tables is called defensively inside request handlers."""
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS inquiries (
          id            TEXT PRIMARY KEY,
          created_at    TEXT NOT NULL,
          session_id    TEXT NOT NULL,
          client_email  TEXT NOT NULL,
          client_name   TEXT,
          client_phone  TEXT,
          ref_slug      TEXT,
          main_challenge TEXT NOT NULL,
          main_goal      TEXT NOT NULL,
          practitioner_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inquiry_practitioners (
          id              TEXT PRIMARY KEY,
          inquiry_id      TEXT NOT NULL,
          practitioner_id TEXT NOT NULL,
          practitioner_email TEXT NOT NULL,
          status          TEXT NOT NULL,
          email_sent_at   TEXT,
          UNIQUE(inquiry_id, practitioner_id)
        );

        CREATE TABLE IF NOT EXISTS inquiry_reply_tokens (
          token_hash      TEXT PRIMARY KEY,
          inquiry_id      TEXT NOT NULL,
          practitioner_id TEXT NOT NULL,
          created_at      TEXT NOT NULL,
          expires_at      TEXT NOT NULL,
          UNIQUE(inquiry_id, practitioner_id)
        );

        CREATE TABLE IF NOT EXISTS inquiry_replies (
          id              TEXT PRIMARY KEY,
          inquiry_id      TEXT NOT NULL,
          practitioner_id TEXT NOT NULL,
          body            TEXT NOT NULL,
          reply_method    TEXT NOT NULL,
          received_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inquiry_reply_impressions (
          ts              TEXT NOT NULL,
          inquiry_id      TEXT NOT NULL,
          practitioner_id TEXT NOT NULL,
          ip              TEXT,
          user_agent      TEXT
        );

        CREATE TABLE IF NOT EXISTS practitioner_inquiry_opt_outs (
          email           TEXT PRIMARY KEY,
          ts              TEXT NOT NULL,
          practitioner_id TEXT
        );
    """)
    # Additive migration: ip column for per-IP rate limiting (mirrors schema-evolution pattern)
    try:
        cx.execute("ALTER TABLE inquiries ADD COLUMN ip TEXT")
    except Exception:
        pass  # already exists
    # Additive migration: shared_at for Phase 2b share-with-practitioner tracking
    try:
        cx.execute("ALTER TABLE inquiry_practitioners ADD COLUMN shared_at TEXT")
    except sqlite3.OperationalError:
        pass  # already exists


def _init_inquiry_tables():
    with sqlite3.connect(LOG_DB) as cx:
        init_inquiry_tables(cx)

_init_inquiry_tables()


def init_membership_tables(cx):
    """Idempotent CREATE TABLE IF NOT EXISTS for the Pay-It-Forward membership layer.
    Called on app startup alongside init_inquiry_tables and wherever
    init_inquiry_tables is called defensively inside request handlers."""
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS memberships (
          id              TEXT PRIMARY KEY,
          email           TEXT NOT NULL,
          granted_at      TEXT NOT NULL,
          expires_at      TEXT,
          granted_by      TEXT,
          source          TEXT,
          truly_vip_ref   TEXT,
          notes           TEXT,
          last_reminder_at TEXT
        );

        CREATE TABLE IF NOT EXISTS escalation_queue (
          id              TEXT PRIMARY KEY,
          created_at      TEXT NOT NULL,
          email           TEXT NOT NULL,
          query_text      TEXT NOT NULL,
          ai_response     TEXT,
          ai_confidence   REAL,
          flag_reason     TEXT,
          status          TEXT NOT NULL DEFAULT 'pending',
          glen_reply_url  TEXT,
          glen_reply_text TEXT,
          replied_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS studio_credit_intents (
          id          TEXT PRIMARY KEY,
          created_at  TEXT NOT NULL,
          email       TEXT NOT NULL,
          studio_ref  TEXT,
          notes       TEXT
        );
    """)


def _init_membership_tables():
    with sqlite3.connect(LOG_DB) as cx:
        init_membership_tables(cx)

_init_membership_tables()


def _mint_membership_magic_link(email, ttl_min=15):
    """Mint a single-use magic-link token for a membership grant or return-flow.
    Returns the plaintext token; caller is responsible for emailing it."""
    import secrets, json
    plain = secrets.token_urlsafe(32)
    th = _hash_token(plain)
    now_iso = datetime.utcnow().isoformat() + "Z"
    exp_iso = (datetime.utcnow() + timedelta(minutes=int(ttl_min))).isoformat() + "Z"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (th, email, "membership_magic_link", json.dumps({}), now_iso, exp_iso)
        )
    return plain


def _validate_membership_magic_link(token):
    """Return the email if the token is a valid (purpose, not consumed, not expired)
    membership_magic_link, else None. Does NOT mark consumed_at; caller decides."""
    if not token:
        return None
    th = _hash_token(token)
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='membership_magic_link'",
            (th,)
        ).fetchone()
    if not row:
        return None
    email, expires_at, consumed_at = row
    if consumed_at:
        return None
    try:
        exp_dt = datetime.fromisoformat(expires_at.rstrip("Z"))
        if exp_dt < datetime.utcnow():
            return None
    except Exception:
        return None
    return email


def _active_membership_for_email(email):
    """Return the active membership row as a dict (with derived days_remaining), or None."""
    if not email:
        return None
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT * FROM memberships WHERE email=? AND expires_at > ? "
            "ORDER BY expires_at DESC LIMIT 1",
            (email, datetime.utcnow().isoformat() + "Z")
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        exp_dt = datetime.fromisoformat(d["expires_at"].rstrip("Z"))
        d["days_remaining"] = max(0, (exp_dt - datetime.utcnow()).days)
    except Exception:
        d["days_remaining"] = 0
    return d


def _member_context_for_email(email, *, query_log_n=5):
    """Aggregate the member's recent context for the /chat agent overlay.

    Returns a dict with keys: intake_summary, recent_inquiries, recent_queries,
    voice_scan_summary.  All keys are always present; empty when no data found.
    """
    out = {
        "intake_summary": "",
        "recent_inquiries": [],
        "recent_queries": [],
        "voice_scan_summary": "",
    }
    if not email:
        return out

    # Intake from inbound_leads (most recent scoreapp / practice-better / concierge)
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT first_name, raw_json FROM inbound_leads "
                "WHERE email=? AND source IN ('scoreapp','practice-better','concierge') "
                "ORDER BY id DESC LIMIT 1",
                (email,)
            ).fetchone()
            if row:
                first_name, raw_json_str = row
                parts = []
                if first_name:
                    parts.append(f"first name: {first_name}")
                try:
                    payload = json.loads(raw_json_str or "{}")
                    data = payload.get("data", payload) or {}
                    score = (data.get("total_score") or {}).get("percent") or data.get("score")
                    if score:
                        parts.append(f"assessment score: {score}%")
                    qs = data.get("quiz_questions") or []
                    for q in qs[:6]:
                        qt = (q.get("question") or "").strip()
                        ans = ", ".join(
                            (a.get("answer") or "").strip()
                            for a in (q.get("answers") or [])
                            if a.get("answer")
                        )
                        if qt and ans:
                            parts.append(f"  {qt}: {ans}")
                except Exception:
                    pass
                out["intake_summary"] = "\n".join(parts)
    except Exception as e:
        print(f"[member-context] intake fetch failed: {e!r}", flush=True)

    # Recent inquiries
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            for r in cx.execute(
                "SELECT main_challenge, main_goal, created_at FROM inquiries "
                "WHERE client_email=? ORDER BY created_at DESC LIMIT 3",
                (email,)
            ).fetchall():
                out["recent_inquiries"].append(dict(r))
    except Exception as e:
        print(f"[member-context] inquiries fetch failed: {e!r}", flush=True)

    # Recent queries. Production table uses 'query' column; test table uses
    # 'question'.  Try 'question' first; fall back to 'query' on OperationalError.
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            try:
                rows = cx.execute(
                    "SELECT question, ts FROM query_log WHERE email=? "
                    "ORDER BY id DESC LIMIT ?",
                    (email, int(query_log_n))
                ).fetchall()
                out["recent_queries"] = [{"question": r["question"], "ts": r["ts"]}
                                         for r in rows]
            except Exception:
                # Production table uses 'query' column; normalise to 'question' key
                rows = cx.execute(
                    "SELECT query, ts FROM query_log WHERE email=? "
                    "ORDER BY id DESC LIMIT ?",
                    (email, int(query_log_n))
                ).fetchall()
                out["recent_queries"] = [{"question": r["query"], "ts": r["ts"]}
                                         for r in rows]
    except Exception as e:
        print(f"[member-context] query_log fetch failed: {e!r}", flush=True)

    # Voice scan from Pinecone (best-effort; skipped if client not available)
    try:
        import pinecone  # noqa: F401
        # Locate the existing Pinecone client and e4l-scans namespace.
        # If unavailable (e.g. test environment), silently skip.
        pass
    except Exception:
        pass

    return out


def _format_member_context_block(member, ctx):
    """Format the member context dict into a compact human-readable block for
    injection into the /chat system prompt."""
    lines = ["===== MEMBER CONTEXT (active coaching member) ====="]
    if member and member.get("days_remaining") is not None:
        lines.append(f"Days remaining in current 30-day window: {member['days_remaining']}.")
    if ctx.get("intake_summary"):
        lines.append("Intake:")
        lines.append(ctx["intake_summary"])
    if ctx.get("recent_inquiries"):
        lines.append("Recent inquiries:")
        for r in ctx["recent_inquiries"]:
            lines.append(
                f"  challenge: {r.get('main_challenge', '')}; "
                f"goal: {r.get('main_goal', '')}"
            )
    if ctx.get("recent_queries"):
        lines.append("Recent questions:")
        for r in ctx["recent_queries"]:
            q = (r.get("question") or "").strip()
            if q:
                lines.append(f"  - {q}")
    if ctx.get("voice_scan_summary"):
        lines.append(f"Voice scan summary: {ctx['voice_scan_summary']}")
    lines.append("===== END MEMBER CONTEXT =====")
    return "\n".join(lines)


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
RM_INBOUND_INQUIRY_EMAIL = "this.elf+rm-inquiry@gmail.com"
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
            "SELECT name, description, url_template, COALESCE(instructions, '') "
            "FROM affiliate_offers WHERE active=1 ORDER BY sort_order ASC"
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
                "instructions": o[3],
            }
            for o in offers
        ]
    })


@app.route("/affiliate/portal")
def affiliate_portal_page():
    # Token-only access. Email-based instant-redirect was removed in favor of
    # /affiliate/login-request → email magic-link → /affiliate/login-verify.
    resp = send_from_directory(STATIC, "affiliate-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _send_affiliate_magic_link(to_email: str, name: str, magic_url: str) -> tuple:
    """Send the affiliate-portal magic-link email. SMTP first, console-log fallback.
    Returns (sent_via, error_or_none). Mirrors send_magic_link_email but with
    affiliate-specific copy and without the chat-auth GHL workflow path.
    """
    subject = "Your affiliate-portal sign-in link"
    body = (
        f"Hi {name or 'there'},\n\n"
        f"Click the link below to access your affiliate portal. The link is "
        f"single-use and expires in {AUTH_TOKEN_TTL_MIN} minutes.\n\n"
        f"{magic_url}\n\n"
        f"If you didn't request this, you can ignore this email.\n\n"
        f"— Remedy Match Affiliate Team\n"
    )
    html_body = (
        f"<p>Hi {name or 'there'},</p>"
        f"<p>Click the link below to access your affiliate portal. The link is "
        f"single-use and expires in {AUTH_TOKEN_TTL_MIN} minutes.</p>"
        f"<p><a href=\"{magic_url}\">Open my affiliate portal</a></p>"
        f"<p style=\"color:#666;font-size:12px;\">Or paste this URL into your browser: {magic_url}</p>"
        f"<p>If you didn't request this, you can ignore this email.</p>"
        f"<p>— Remedy Match Affiliate Team</p>"
    )
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    if smtp_host and smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = smtp_from
            msg["To"]      = to_email
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            port = int(os.environ.get("SMTP_PORT", "587"))
            with smtplib.SMTP(smtp_host, port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_from, [to_email], msg.as_string())
            print(f"[affiliate-magic] SMTP send OK: from={smtp_from} to={to_email}", flush=True)
            return "smtp", None
        except Exception as e:
            print(f"[affiliate-magic] SMTP send failed: from={smtp_from} to={to_email} err={e!r}", flush=True)
            return "smtp-failed", str(e)
    print(f"\n[affiliate-magic] MAGIC LINK for {to_email}: {magic_url}\n", flush=True)
    return "console-log", "no SMTP configured"


def _send_inquiry_email(to_email, subject, body, reply_to=None):
    """Per-recipient SMTP send for the inquiry fan-out. Returns True on success,
    False on SMTP failure (never raises). Falls back to print() when SMTP env
    is unset (dev)."""
    import os, smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@remedymatch.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    if not (smtp_host and smtp_user and smtp_pass):
        print(f"[inquiry-email] (no SMTP env) would send to={to_email} subject={subject!r} reply_to={reply_to}", flush=True)
        return True
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        if reply_to:
            msg["Reply-To"] = reply_to
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_from, [to_email], msg.as_bytes())
        return True
    except Exception as e:
        print(f"[inquiry-email] FAIL to={to_email} err={e!r}", flush=True)
        return False


def _register_finance_email_actions():
    from dashboard.actions import action as _act, get_action as _get, LOW_WRITE
    from dashboard import rbac as _r
    if _get("finance.send_payment_reminder"):
        return

    def _reminder(params, ctx):
        email = (params.get("email") or "").strip()
        if not email:
            raise ValueError("email required")
        doc = params.get("doc") or params.get("invoice_id") or ""
        amount = params.get("amount")
        amt = f" of ${float(amount):.2f}" if amount not in (None, "") else ""
        subject = "A quick note about your invoice"
        body = (f"Aloha,\n\nThis is a friendly reminder that invoice {doc} "
                f"{('with a balance' + amt) if amt else ''} is still open. "
                f"You can reply here with any questions.\n\nIn wellness,\nDr. Glen")
        ok = _send_inquiry_email(to_email=email, subject=subject, body=body,
                                 reply_to=RM_INBOUND_INQUIRY_EMAIL)
        return {"email": email, "doc": doc, "sent": bool(ok),
                "message": f"Payment reminder {'sent to' if ok else 'failed for'} {email}."}

    _act(key="finance.send_payment_reminder", module="money",
         title="Send payment reminder", description="Email a customer about an open invoice.",
         risk_tier=LOW_WRITE, permission=(_r.OWNER, _r.OPS, _r.VA))(_reminder)


_register_finance_email_actions()


def _send_client_receipt(client_email, client_name, sent_records, base_url):
    """Send a one-time receipt to the client after a successful inquiry POST.
    sent_records is the list of practitioner dicts we actually emailed (NOT the
    full requested set; skipped recipients are not in the receipt).
    Reply-To is the inbound RM inquiry mailbox so a client reply lands somewhere
    we can automate against."""
    client_first = (client_name.split(None, 1)[0] if client_name else "there")
    n = len(sent_records)
    if n == 0:
        return False  # nothing to confirm
    lines = [
        f"Hi {client_first},",
        "",
        f"Your inquiry just went out to {n} practitioner{'s' if n != 1 else ''} on RemedyMatch:",
        "",
    ]
    for rec in sent_records:
        name = (rec.get("name") or "").strip() or "(name unavailable)"
        city = (rec.get("city") or "").strip()
        state = (rec.get("state") or "").strip()
        loc = ", ".join(p for p in [city, state] if p)
        lines.append(f"  - {name}" + (f" ({loc})" if loc else ""))
    lines += [
        "",
        f"Replies will arrive at this address ({client_email}), usually within a few days.",
        "You can reply directly to each practitioner just by hitting Reply on their email.",
        "",
        "While you wait, here's a 60-second self-assessment that helps you understand your health context:",
        "https://healing.scoreapp.com",
        "",
        "---",
        "Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA",
        "This is a one-time receipt for the inquiry you just sent.",
    ]
    subject = f"Your inquiry was sent to {n} practitioner{'s' if n != 1 else ''}"
    return _send_inquiry_email(
        to_email=client_email,
        subject=subject,
        body="\n".join(lines),
        reply_to=RM_INBOUND_INQUIRY_EMAIL,
    )


def _scoreapp_payload_for(email):
    """Fetch the most recent ScoreApp payload (dict) for this email, or None.
    Reads from inbound_leads where source='scoreapp'."""
    if not email:
        return None
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT raw_json FROM inbound_leads "
                "WHERE source='scoreapp' AND email=? "
                "ORDER BY id DESC LIMIT 1",
                (email,)
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def _recent_inquiry_practitioner_ids(client_email, days=30):
    """Return the list of (inquiry_id, practitioner_id, practitioner_email,
    shared_at) tuples for inquiries the client sent in the last `days` days.
    Only includes rows where status='sent' (not skipped, not failed)."""
    if not client_email:
        return []
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute(
            "SELECT ip.inquiry_id, ip.practitioner_id, ip.practitioner_email, ip.shared_at "
            "FROM inquiry_practitioners ip "
            "JOIN inquiries i ON i.id = ip.inquiry_id "
            "WHERE i.client_email=? "
            "  AND i.created_at > datetime('now', ?) "
            "  AND ip.status='sent'",
            (client_email, f"-{int(days)} days")
        ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def _compose_share_email(client_first, client_email, main_challenge, main_goal,
                          scoreapp_payload):
    """Compose the body of the share-with-practitioner email."""
    data = scoreapp_payload.get("data", scoreapp_payload)
    score = (data.get("total_score") or {}).get("percent") or data.get("score") or ""
    questions = data.get("quiz_questions") or []
    lines = [
        f"Hi,",
        "",
        f"{client_first} reached out to you on RemedyMatch and has now completed "
        f"our self-assessment. They asked us to share the full results with you "
        f"so you have additional context.",
        "",
        f"Their original inquiry:",
        f"  What they are working through: {main_challenge}",
        f"  What success looks like for them: {main_goal}",
        "",
        f"Self-assessment results:",
    ]
    if score:
        lines.append(f"  Overall score: {score}%")
    if questions:
        lines.append("")
        lines.append("  Full Q&A:")
        for q in questions:
            qt = (q.get("question") or "").strip()
            answers = q.get("answers") or []
            atext = ", ".join((a.get("answer") or "").strip() for a in answers if a.get("answer"))
            if qt and atext:
                lines.append(f"    - {qt}")
                lines.append(f"      -> {atext}")
    lines += [
        "",
        f"You can reply by hitting Reply (this email is set to send your response "
        f"directly to {client_email}).",
        "",
        f"---",
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA",
        f"This is a one-time follow-up tied to {client_first}'s recent inquiry.",
    ]
    return "\n".join(lines)


@app.route("/affiliate/login-request", methods=["POST"])
def affiliate_login_request():
    """Email-based sign-in. Always redirects to /affiliate?info=... — never
    leaks whether the email is registered (prevents enumeration). If the
    email matches an approved affiliate, a single-use magic-link is emailed.
    """
    from flask import redirect as _redir
    import urllib.parse as _up
    email = (request.form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return _redir("/affiliate?error=" + _up.quote("Valid email required"))

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT name FROM affiliate_signups WHERE LOWER(email)=? AND status='approved'",
            (email,)
        ).fetchone()
        if row:
            magic = secrets.token_urlsafe(32)
            th    = _hash_token(magic)
            now   = _now_utc()
            expires = now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)
            cx.execute(
                """INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at)
                   VALUES (?,?,?,?,?)""",
                (th, email, "affiliate_magic_link", now.isoformat(), expires.isoformat())
            )
            cx.commit()
            magic_url = f"{PUBLIC_BASE_URL}/affiliate/login-verify?token={magic}"
            _send_affiliate_magic_link(email, row[0] or "", magic_url)

    return _redir("/affiliate?info=" + _up.quote(
        f"If that email matches an approved affiliate, we just sent a sign-in link. "
        f"Check your inbox — the link expires in {AUTH_TOKEN_TTL_MIN} minutes."
    ))


@app.route("/affiliate/login-verify", methods=["GET"])
def affiliate_login_verify():
    """Consume the magic-link token and 302 to the affiliate's portal."""
    from flask import redirect as _redir
    import urllib.parse as _up
    token = (request.args.get("token") or "").strip()
    if not token:
        return _redir("/affiliate?error=" + _up.quote("Missing sign-in token"))
    th = _hash_token(token)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            """SELECT email, expires_at, consumed_at FROM auth_tokens
               WHERE token_hash=? AND purpose='affiliate_magic_link'""",
            (th,)
        ).fetchone()
        if not row:
            return _redir("/affiliate?error=" + _up.quote("Sign-in link invalid or already used. Request a new one."))
        if row["consumed_at"]:
            return _redir("/affiliate?error=" + _up.quote("Sign-in link already used. Request a new one."))
        try:
            if datetime.fromisoformat(row["expires_at"]) < _now_utc():
                return _redir("/affiliate?error=" + _up.quote("Sign-in link expired. Request a new one."))
        except Exception:
            return _redir("/affiliate?error=" + _up.quote("Sign-in link corrupted. Request a new one."))
        cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                   (_now_utc().isoformat(), th))
        aff = cx.execute(
            "SELECT token FROM affiliate_signups WHERE LOWER(email)=? AND status='approved'",
            (row["email"],)
        ).fetchone()
        cx.commit()
    if not aff:
        return _redir("/affiliate?error=" + _up.quote("Affiliate account no longer active. Apply below."))
    return _redir(f"/affiliate/portal?token={aff[0]}")


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

    # Split name into first/last (same logic as /begin/unlock)
    _name_parts = name.split(None, 1)
    _first = _name_parts[0] if _name_parts else ""
    _last  = _name_parts[1] if len(_name_parts) > 1 else ""

    # Session and recruiter for journey wiring
    _session_id = (request.cookies.get("amg_session") or "").strip()
    _minted_session = not _session_id
    if _minted_session:
        _session_id = uuid.uuid4().hex
    _recruiter_slug = (request.cookies.get("rm_ref") or referred_by or "").strip()

    base = re.sub(r"[^a-z0-9]+", "-", (org or name).lower()).strip("-")[:30]
    import secrets as _sec
    token = _sec.token_urlsafe(24)
    slug  = base
    ts    = datetime.now(timezone.utc).isoformat()

    # Return existing portal if email already registered
    with sqlite3.connect(LOG_DB) as cx:
        existing = cx.execute("SELECT token FROM affiliate_signups WHERE email=?", (email,)).fetchone()
    if existing:
        resp = _redirect(f"/affiliate/portal?token={existing[0]}")
        _stamp_affiliate_journey(_session_id, email, _first, _last, _recruiter_slug)
        if _minted_session:
            resp.set_cookie("amg_session", _session_id, max_age=60 * 60 * 24 * 365,
                            httponly=True, samesite="Lax", secure=request.is_secure)
        return resp

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

    resp = _redirect(f"/affiliate/portal?token={token}")
    _stamp_affiliate_journey(_session_id, email, _first, _last, _recruiter_slug)
    if _minted_session:
        resp.set_cookie("amg_session", _session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
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

    # Split name into first/last (same logic as /begin/unlock)
    _name_parts = name.split(None, 1)
    _first = _name_parts[0] if _name_parts else ""
    _last  = _name_parts[1] if len(_name_parts) > 1 else ""

    # Session and recruiter for journey wiring
    _session_id = (request.cookies.get("amg_session") or "").strip()
    _minted_session = not _session_id
    if _minted_session:
        _session_id = uuid.uuid4().hex
    _recruiter_slug = (request.cookies.get("rm_ref") or data.get("referred_by") or "").strip()

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
    resp = jsonify({
        "ok": True,
        "portal_url": portal_url,
        "tracking_url": tracking_url,
        "slug": slug,
    })
    resp.status_code = 201
    _stamp_affiliate_journey(_session_id, email, _first, _last, _recruiter_slug)
    if _minted_session:
        resp.set_cookie("amg_session", _session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


def _mask_lead_name(first: str, last: str) -> str:
    """'Mary', 'Johnson' -> 'Mary J.' — preserves first name, masks last to initial.
    Both fields are tolerant of None and whitespace.
    """
    fn = (first or "").strip()
    ln = (last or "").strip()
    if ln:
        return f"{fn} {ln[0]}.".strip()
    return fn


@app.route("/affiliate/social-links", methods=["POST", "OPTIONS"])
def affiliate_social_links_submit():
    if request.method == "OPTIONS":
        return "", 200
    data  = request.get_json() or {}
    token = (data.get("token") or "").strip()
    urls  = data.get("urls") or []
    if not token:
        return jsonify({"error": "token required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT slug, email, status FROM affiliate_signups WHERE token=?",
                         (token,)).fetchone()
        if not row:
            return jsonify({"error": "invalid token"}), 404
        slug, email, status = row
        if status != "approved":
            return jsonify({"error": "application pending review"}), 403
        ts = datetime.now(timezone.utc).isoformat()
        count = 0
        for u in (urls or [])[:10]:
            u = (u or "").strip()[:500]
            if not u.startswith(("http://", "https://")):
                continue
            cx.execute("INSERT INTO affiliate_social_links (ts, slug, email, url) VALUES (?,?,?,?)",
                       (ts, slug, email, u))
            count += 1
        cx.commit()
    return jsonify({"ok": True, "count": count})


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
    recruit_url    = f"{PUBLIC_BASE_URL}/affiliate?ref={slug}"

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
        conversions_count = cx.execute("""
            SELECT COUNT(*) FROM affiliate_conversions WHERE affiliate_slug=?
        """, (slug,)).fetchone()[0]
        offers = cx.execute(
            "SELECT name, description, url_template, COALESCE(instructions, '') "
            "FROM affiliate_offers WHERE active=1 ORDER BY sort_order ASC"
        ).fetchall()
        social = cx.execute(
            "SELECT url, points, views, likes, shares, ts FROM affiliate_social_links "
            "WHERE slug=? ORDER BY id DESC", (slug,)).fetchall()

    return jsonify({
        "name": name,
        "organization": org,
        "slug": slug,
        "tracking_url": tracking_url,
        "recruit_url": recruit_url,
        "total_leads": stats[0] if stats else 0,
        "last_lead": stats[1] if stats else None,
        "recruited_count": recruited_count,
        "conversions_count": conversions_count,
        "recent": [{"received_at": r[0],
                    "name": _mask_lead_name(r[1], r[2]),
                    "score": r[3]} for r in recent],
        "offers": [
            {
                "name": o[0],
                "description": o[1],
                "url": o[2].replace("{slug}", slug),
                "instructions": o[3],
            }
            for o in offers
        ],
        "social_links": [
            {"url": s[0], "points": s[1], "views": s[2], "likes": s[3], "shares": s[4], "ts": s[5]}
            for s in social
        ],
        "member_since": created_at,
    })


@app.route("/affiliate/payload-peek", methods=["GET"])
def affiliate_payload_peek():
    """Console-gated diagnostic: return the most recent stored webhook payloads
    so we can confirm field shapes (GrooveKart order total key, whether Practice
    Better forwards utm). Payloads contain customer PII — console key required."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    source = (request.args.get("source") or "").strip().lower()
    try:
        limit = max(1, min(int(request.args.get("limit", "3")), 20))
    except ValueError:
        limit = 3
    if source in ("groovekart", "gk", "store"):
        sql = "SELECT received_at, raw_json FROM inbound_leads WHERE source='groovekart' ORDER BY id DESC LIMIT ?"
    elif source in ("practice-better", "pb", "practicebetter"):
        sql = "SELECT received_at, raw_json FROM pb_events ORDER BY id DESC LIMIT ?"
    else:
        return jsonify({"error": "source must be 'groovekart' or 'practice-better'"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute(sql, (limit,)).fetchall()
    out = []
    for received_at, raw in rows:
        try:
            parsed = json.loads(raw) if raw else None
            top_keys = sorted(parsed.keys()) if isinstance(parsed, dict) else None
        except Exception:
            parsed, top_keys = None, None
        out.append({"received_at": received_at, "top_keys": top_keys, "payload": parsed})
    return jsonify({"source": source, "count": len(out), "rows": out})


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


@app.route("/practitioner")
def practitioner_page():
    resp = send_from_directory(STATIC, "practitioner.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/practitioner-application", methods=["POST", "OPTIONS"])
def practitioner_application():
    """Practitioner Panel application.
    Upserts GHL contact with practitioner-application tags + emails Rae.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(force=True) or {}

    full_name      = (data.get("full_name") or "").strip()
    email          = (data.get("email") or "").strip().lower()
    phone          = (data.get("phone") or "").strip()
    practice_name  = (data.get("practice_name") or "").strip()
    practice_type  = (data.get("practice_type") or "").strip()
    license_info   = (data.get("license_info") or "").strip()
    website        = (data.get("website") or "").strip()
    monthly_volume = (data.get("monthly_volume") or "").strip()
    cert_interest  = (data.get("cert_interest") or "").strip()
    tools_interest = data.get("tools_interest") or []
    notes          = (data.get("notes") or "").strip()

    if not full_name or not email or not phone or not practice_name or not practice_type:
        return jsonify({
            "error": "Name, email, phone, practice name, and practice type are required"
        }), 400

    if website and not website.startswith(("http://", "https://")):
        website = "https://" + website

    parts = full_name.split(None, 1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else ""

    tags = ["practitioner-application", f"practice-type-{re.sub(r'[^a-z0-9]+', '-', practice_type.lower()).strip('-')}"]
    if cert_interest.lower() == "yes":
        tags.append("practitioner-cert-interested")
    if tools_interest:
        tags.append("practitioner-tools-interested")

    contact_id, created, err = ghl_upsert_contact(
        email, first_name, last_name, phone,
        source_tag="practitioner-application",
        extra_tags=tags,
    )
    if err:
        print(f"[practitioner-application] GHL upsert failed: {err}", flush=True)

    tools_str = ", ".join(tools_interest) if tools_interest else "(none selected)"
    body = (
        f"New Practitioner Panel application — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'=' * 60}\n\n"
        f"Name:            {full_name}\n"
        f"Email:           {email}\n"
        f"Phone:           {phone}\n"
        f"Practice:        {practice_name}\n"
        f"Practice type:   {practice_type}\n"
        f"License / state: {license_info or '(not provided)'}\n"
        f"Website:         {website or '(not provided)'}\n"
        f"Monthly volume:  {monthly_volume or '(not provided)'}\n"
        f"Cert interest:   {cert_interest or '(not provided)'}\n"
        f"Tools interest:  {tools_str}\n\n"
        f"Notes:\n{notes or '(none)'}\n\n"
        f"{'=' * 60}\n"
        f"GHL contact: {contact_id or '(upsert failed — check logs)'}\n"
        f"Tags applied: {', '.join(tags)}\n"
    )

    rae_email = os.environ.get("RAE_EMAIL", "suerae1111@gmail.com")
    glen_email = os.environ.get("GLEN_EMAIL", "drglenswartwout@gmail.com")
    subject = f"Practitioner Panel application: {full_name} — {practice_name}"

    sent_via_rae, rae_err = _send_full_report_email(rae_email, "Rae", subject, body)
    sent_via_glen, _ = _send_full_report_email(glen_email, "Glen", f"[cc] {subject}", body)
    if rae_err and sent_via_rae == "console-log":
        print(f"[practitioner-application] Rae email fell back to console: {rae_err}", flush=True)

    return jsonify({
        "ok": True,
        "contact_id": contact_id,
        "contact_created": created,
        "rae_notified_via": sent_via_rae,
    }), 201


# ── Practitioner wholesale portal (Phase 3: auth + registration + cart) ───────
from dashboard import practitioner_portal as _pp
from dashboard import wholesale_checkout as _wc
from dashboard import dropship_checkout as _dropship
from dashboard import wallet as _wallet
import dashboard.practitioner_chat as _chat


def _send_practitioner_magic_link(to_email, name, magic_url):
    """Practitioner-portal sign-in email (reuses the report-email transport)."""
    subject = "Your Remedy Match practitioner sign-in link"
    body = (
        f"Hi {name or 'there'},\n\n"
        f"Click the link below to open your practitioner portal. It is single-use and "
        f"expires in {_pp.MAGIC_TTL_MIN} minutes.\n\n{magic_url}\n\n"
        f"If you didn't request this, you can ignore this email.\n\n— Remedy Match\n"
    )
    return _send_full_report_email(to_email, name or "there", subject, body)


def _practitioner_session_pid():
    token = (request.args.get("token") or "").strip()
    if not token:
        token = ((request.get_json(silent=True) or {}).get("token") or "").strip()
    return _pp.practitioner_id_from_session(token) if token else None


@app.route("/practitioner/portal")
def practitioner_portal_page():
    return send_from_directory(STATIC, "practitioner-portal.html")


@app.route("/practitioner/dropship")
def practitioner_dropship_page():
    resp = send_from_directory(STATIC, "practitioner-dropship.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/practitioner/register", methods=["GET"])
def practitioner_register_page():
    return send_from_directory(STATIC, "practitioner-register.html")


@app.route("/api/practitioner/register", methods=["POST"])
def api_practitioner_register():
    clean, err = _pp.validate_registration(request.get_json(silent=True) or {})
    if err:
        return jsonify({"ok": False, "error": err}), 400
    try:
        pid, unlocked = _pp.register_practitioner(clean)
    except Exception as e:
        print(f"[practitioner-register] insert failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Could not create your account. Please try again."}), 500
    try:
        parts = clean["name"].split(None, 1)
        ghl_upsert_contact(clean["email"], parts[0], (parts[1] if len(parts) > 1 else ""),
                           clean.get("phone") or "", source_tag="practitioner-portal",
                           extra_tags=[f"portal-{clean['portal_role']}"])
    except Exception as e:
        print(f"[practitioner-register] GHL upsert failed: {e!r}", flush=True)
    module_pay = None
    if clean["portal_role"] == "coach":
        try:
            mo = _wc.build_module_order(
                {"id": pid, "email": clean["email"], "name": clean["name"]},
                "module-1", today=datetime.now(timezone.utc))
            module_pay = {
                "invoice_id": mo.get("invoice_id"), "total": mo.get("total"),
                "doc_number": mo.get("doc_number"),
                "pay_instructions": [
                    {"label": _ALT_PAY["zelle"]["label"], "to": _ALT_PAY["zelle"]["to"]},
                    {"label": _ALT_PAY["wise"]["label"], "to": _ALT_PAY["wise"]["to"]},
                ],
            }
            try:
                _pp.record_order(pid, invoice_id=mo.get("invoice_id"),
                                 doc_number=mo.get("doc_number"),
                                 total_cents=int(round((mo.get("total") or 0) * 100)),
                                 credit_cents=mo.get("credit_redeemed_cents", 0))
            except Exception:
                pass
            # NOTE: coaches no longer auto-unlock reselling here. Reselling
            # (drop-ship + wholesale ordering) is now gated behind a resale
            # license: a coach submits one via /api/practitioner/resale-apply,
            # then Glen/Rae approve via /admin/wholesale. `unlocked` stays as
            # register_practitioner returned it (False for coaches; licensed
            # registrants are unlocked there, independent of this branch).
        except Exception as e:
            print(f"[practitioner-register] module invoice failed: {e!r}", flush=True)
    try:
        magic = _pp.create_magic_link_token(pid, clean["email"])
        _send_practitioner_magic_link(
            clean["email"], clean["name"],
            f"{PUBLIC_BASE_URL}/practitioner/login-verify?token={magic}")
    except Exception as e:
        print(f"[practitioner-register] magic link failed: {e!r}", flush=True)
    try:
        _send_full_report_email(
            os.environ.get("RAE_EMAIL", "suerae1111@gmail.com"), "Rae",
            f"New practitioner portal registration: {clean['name']}",
            f"{clean['name']} ({clean['email']}) joined as {clean['portal_role']}. "
            f"Wholesale unlocked: {unlocked}.")
    except Exception:
        pass
    return jsonify({"ok": True, "wholesale_unlocked": unlocked, "module_pay": module_pay,
                    "message": "Check your email for a sign-in link."}), 201


@app.route("/api/practitioner/resale-apply", methods=["POST"])
def api_practitioner_resale_apply():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    rln = (body.get("resale_license_number") or "").strip()
    state = (body.get("license_state") or "").strip()
    if not rln:
        return jsonify({"ok": False, "error": "resale license number required"}), 400
    try:
        _pp.submit_resale_for_pid(pid, rln, state)
    except Exception as e:
        print(f"[resale-apply] failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "submission failed"}), 500
    # notify Rae (best-effort; mirror the existing apply route's admin email)
    try:
        _send_full_report_email(
            os.environ.get("RAE_EMAIL", "suerae1111@gmail.com"), "Rae",
            "Resale license submitted for reselling activation",
            f"Practitioner {pid} submitted resale license {rln} ({state}). "
            f"Approve at {PUBLIC_BASE_URL}/admin/wholesale.")
    except Exception:
        pass
    return jsonify({"ok": True, "status": "pending"})


@app.route("/practitioner/login-request", methods=["POST"])
def practitioner_login_request():
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if "@" in email:
        try:
            pid = _pp.find_practitioner_id_by_email(email)
        except Exception:
            pid = None
        if pid:
            magic = _pp.create_magic_link_token(pid, email)
            _send_practitioner_magic_link(
                email, "", f"{PUBLIC_BASE_URL}/practitioner/login-verify?token={magic}")
    return jsonify({"ok": True,
                    "message": "If that email has a portal account, a sign-in link is on its way."})


@app.route("/practitioner/login-verify", methods=["GET"])
def practitioner_login_verify():
    from flask import redirect as _redir
    token = (request.args.get("token") or "").strip()
    pid = _pp.consume_magic_link(token) if token else None
    if not pid:
        return _redir("/practitioner/register?error=link")
    return _redir(f"/practitioner/portal?token={_pp.create_session_token(pid)}")


@app.route("/api/practitioner/portal-data", methods=["GET"])
def api_practitioner_portal_data():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid, include_orders=True)
    if not data:
        return jsonify({"ok": False, "error": "account not found"}), 404
    if data.get("dispensary_code"):
        data["dispensary_link"] = f"{PUBLIC_BASE_URL}/dispensary/{data['dispensary_code']}"
    data["stripe_active"] = _STRIPE_ACTIVE
    # Include practitioner's branding (best-effort — never crash portal-data)
    branding = {}
    try:
        from dashboard import practitioner_settings as _ps
        with sqlite3.connect(LOG_DB) as _cx:
            _cx.row_factory = sqlite3.Row
            _ps.init_settings_table(_cx)
            _s = _ps.get_settings(_cx, pid)
            branding = _s.get("branding") or {}
    except Exception:
        pass
    data["branding"] = branding
    return jsonify({"ok": True, **data})


@app.route("/api/practitioner/cart", methods=["POST"])
def api_practitioner_cart():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = request.get_json(silent=True) or {}
    slug = (data.get("slug") or "").strip()
    if not slug:
        return jsonify({"ok": False, "error": "slug required"}), 400
    try:
        qty = int(data.get("qty", 0))
    except Exception:
        qty = 0
    if qty > 0 and not _pp.is_orderable(slug):
        return jsonify({"ok": False,
                        "error": "That item is ordered separately (e.g. on the Centropix store), "
                                 "not through wholesale."}), 400
    _pp.cart_set(pid, slug, qty)
    return jsonify({"ok": True, **(_pp.portal_data(pid) or {})})


@app.route("/api/practitioner/quote", methods=["POST"])
def api_practitioner_quote():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid) or {}
    return jsonify({"ok": True, "quote": data.get("quote"),
                    "wallet_balance_cents": data.get("wallet_balance_cents")})


@app.route("/api/practitioner/checkout", methods=["POST"])
def api_practitioner_checkout():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid)
    if not data:
        return jsonify({"ok": False, "error": "account not found"}), 404
    if not data.get("wholesale_unlocked"):
        return jsonify({"ok": False,
                        "error": "Finish your first certification module to unlock ordering."}), 403
    items = data.get("cart") or []
    if not items:
        return jsonify({"ok": False, "error": "Your cart is empty."}), 400
    prac = {"id": pid, "modules_completed": data.get("modules_completed", 0),
            "email": data.get("email"), "name": data.get("name") or ""}
    _body = request.get_json(silent=True) or {}
    method = (_body.get("method") or "zelle").strip().lower()
    _session_token = (_body.get("token") or "").strip()
    if method == "card" and not _STRIPE_ACTIVE:
        method = "zelle"   # card not enabled yet
    if method not in ("zelle", "wise", "card"):
        method = "zelle"
    ship = _normalize_ship_address(_body.get("address") or {}, fallback_name=prac["name"])
    resale_ok = bool(data.get("resale_license_number"))   # resale cert on file → 0.5% GET
    try:
        out = _wc.build_order(items, prac, method=method,
                              ship_to_state=ship.get("state", ""), resale_ok=resale_ok)
    except Exception as e:
        print(f"[practitioner-checkout] failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Checkout failed. Please try again."}), 500
    if out.get("ok"):
        _pp.cart_clear(pid)
        try:
            _pp.record_order(pid, invoice_id=out.get("invoice_id"),
                             doc_number=out.get("doc_number"),
                             total_cents=int(round((out.get("total") or 0) * 100)),
                             credit_cents=out.get("credit_redeemed_cents", 0))
        except Exception as e:
            print(f"[practitioner-checkout] record_order failed: {e!r}", flush=True)
        _ingest_order(source="wholesale",
                      external_ref=str(out.get("invoice_id") or out.get("Id") or ""),
                      email=(prac.get("email") if isinstance(prac, dict) else "") or "",
                      name=(prac.get("name") if isinstance(prac, dict) else "") or "",
                      total_cents=int(round((out.get("total") or 0) * 100)),
                      address=ship, channel="wholesale", get_cents=out.get("get_cents", 0))
        if method in ("zelle", "wise"):
            out["pay_instructions"] = _ALT_PAY.get(method, {})
        elif method == "card":
            out["stripe_url"] = _stripe_checkout_url_for_order(out, prac["email"], _session_token)
    return jsonify(out), (200 if out.get("ok") else 422)


# ── Personal ordering for cert participants ────────────────────────────────────
# A near-clone of the wholesale checkout above, with TWO crucial differences:
#   (a) NO wholesale_unlocked gate — any valid practitioner session may order for
#       themselves at cert-level pricing; and
#   (b) NEVER resale-exempt (resale_ok=False) — personal purchases are taxed.
# Fee-free (zelle/wise) orders earn 3.5% Wellness Credit. To avoid double-crediting,
# build_order is called with method=None (which suppresses its OWN internal 3%
# fee-free earn), and the route credits the full 3.5% itself.

@app.route("/api/practitioner/personal/quote", methods=["POST"])
def api_practitioner_personal_quote():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid) or {}
    # portal_data already prices the cart at the participant's cert level.
    return jsonify({"ok": True, "quote": data.get("quote"),
                    "wallet_balance_cents": data.get("wallet_balance_cents")})


@app.route("/api/practitioner/personal/checkout", methods=["POST"])
def api_practitioner_personal_checkout():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid)
    if not data:
        return jsonify({"ok": False, "error": "account not found"}), 404
    # NOTE: no wholesale_unlocked gate here — personal ordering is always allowed.
    items = data.get("cart") or []
    if not items:
        return jsonify({"ok": False, "error": "Your cart is empty."}), 400
    prac = {"id": pid, "modules_completed": data.get("modules_completed", 0),
            "email": data.get("email"), "name": data.get("name") or ""}
    _body = request.get_json(silent=True) or {}
    method = (_body.get("method") or "zelle").strip().lower()
    _session_token = (_body.get("token") or "").strip()
    if method == "card" and not _STRIPE_ACTIVE:
        method = "zelle"   # card not enabled yet
    if method not in ("zelle", "wise", "card"):
        method = "zelle"
    ship = _normalize_ship_address(_body.get("address") or {}, fallback_name=prac["name"])
    resale_ok = False   # personal purchases are taxed; never resale-exempt
    try:
        # method=None suppresses build_order's internal 3% fee-free earn so we can
        # credit the 3.5% PERSONAL rate below without double-crediting. The real
        # payment method is kept in `method` for the pay-instructions/Stripe tail.
        out = _wc.build_order(items, prac, method=None,
                              ship_to_state=ship.get("state", ""), resale_ok=resale_ok)
    except Exception as e:
        print(f"[practitioner-personal-checkout] failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Checkout failed. Please try again."}), 500
    if out.get("ok"):
        _pp.cart_clear(pid)
        try:
            _pp.record_order(pid, invoice_id=out.get("invoice_id"),
                             doc_number=out.get("doc_number"),
                             total_cents=int(round((out.get("total") or 0) * 100)),
                             credit_cents=out.get("credit_redeemed_cents", 0))
        except Exception as e:
            print(f"[practitioner-personal-checkout] record_order failed: {e!r}", flush=True)
        _ingest_order(source="personal",
                      external_ref=str(out.get("invoice_id") or out.get("Id") or ""),
                      email=(prac.get("email") if isinstance(prac, dict) else "") or "",
                      name=(prac.get("name") if isinstance(prac, dict) else "") or "",
                      total_cents=int(round((out.get("total") or 0) * 100)),
                      address=ship, channel="personal", get_cents=out.get("get_cents", 0))
        # Personal fee-free earn: 3.5% of the charged amount (zelle/wise), else 0.
        # build_order was called with method=None above, so it did NOT credit its
        # own 3% — this is the only earn for this order. Credit the explicit 3.5%
        # via earn_personal (its own ledger entry_type, so reconciliation never
        # conflates it with drop-ship margin). Idempotent per invoice.
        charged_cents = int(round((out.get("total") or 0) * 100))
        earn = _wallet.personal_earn_cents(charged_cents, method)
        if earn > 0:
            try:
                _wallet.earn_personal(
                    pid, earn,
                    qbo_invoice_id=str(out.get("invoice_id") or ""),
                    note="personal fee-free earn 3.5%")
            except Exception as e:
                print(f"[practitioner-personal-checkout] earn credit failed: {e!r}",
                      flush=True)
        if method in ("zelle", "wise"):
            out["pay_instructions"] = _ALT_PAY.get(method, {})
        elif method == "card":
            out["stripe_url"] = _stripe_checkout_url_for_order(out, prac["email"], _session_token)
    return jsonify(out), (200 if out.get("ok") else 422)


@app.route("/api/practitioner/dropship/quote", methods=["GET", "POST"])
def api_practitioner_dropship_quote():
    """Return per-line drop-ship wholesale pricing for the practitioner's cart."""
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid) or {}
    items = data.get("cart") or []
    modules = int(data.get("modules_completed", 0) or 0)
    settings = _dropship._settings()
    total_bottles = sum(int(it.get("qty", 0)) for it in items)
    lines = []
    subtotal_cents = 0
    for it in items:
        slug = it["slug"]
        qty = int(it.get("qty", 1))
        try:
            from app import _get_product
            retail = _get_product(slug)["price_cents"]
        except Exception:
            retail = 0
        if retail and total_bottles > 0:
            dl = _dropship.dropship_line_cents(
                retail_cents=retail, qty=total_bottles,
                modules=modules, settings=settings)
        else:
            dl = {"base_cents": 0, "fee_cents": 0, "unit_cents": 0, "line_cents": 0}
        line_total = dl["unit_cents"] * qty
        subtotal_cents += line_total
        lines.append({"slug": slug, "qty": qty, "unit_cents": dl["unit_cents"],
                      "base_cents": dl["base_cents"], "fee_cents": dl["fee_cents"],
                      "line_cents": line_total})
    return jsonify({"ok": True, "lines": lines, "subtotal_cents": subtotal_cents})


@app.route("/api/practitioner/dropship/checkout", methods=["POST"])
def api_practitioner_dropship_checkout():
    """Practitioner pays drop-ship wholesale; we ship to the patient."""
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    data = _pp.portal_data(pid)
    if not data:
        return jsonify({"ok": False, "error": "account not found"}), 404
    if not data.get("wholesale_unlocked"):
        return jsonify({"ok": False,
                        "error": "Finish your first certification module to unlock ordering."}), 403
    items = data.get("cart") or []
    if not items:
        return jsonify({"ok": False, "error": "Your cart is empty."}), 400
    prac = {"id": pid, "modules_completed": data.get("modules_completed", 0),
            "email": data.get("email"), "name": data.get("name") or ""}
    _body = request.get_json(silent=True) or {}
    method = (_body.get("method") or "zelle").strip().lower()
    _session_token = (_body.get("token") or "").strip()
    if method == "card" and not _STRIPE_ACTIVE:
        method = "zelle"   # card not enabled yet
    if method not in ("zelle", "wise", "card"):
        method = "zelle"
    # Patient shipping address is required for drop-ship.
    patient_addr_raw = _body.get("patient_address") or {}
    ship = _normalize_ship_address(patient_addr_raw, fallback_name="")
    if not ship:
        return jsonify({"ok": False, "error": "patient_address is required"}), 400
    try:
        out = _dropship.build_dropship_order(items, prac, patient_ship=ship, method=method)
    except Exception as e:
        print(f"[dropship-checkout] failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Checkout failed. Please try again."}), 500
    if out.get("ok"):
        _pp.cart_clear(pid)
        _ingest_order(source="dropship",
                      external_ref=str(out.get("invoice_id") or ""),
                      email=(prac.get("email") or ""),
                      name=(prac.get("name") or ""),
                      total_cents=int(round((out.get("total") or 0) * 100)),
                      address=ship, channel="wholesale",
                      get_cents=out.get("get_cents", 0))
        if method in ("zelle", "wise"):
            out["pay_instructions"] = _ALT_PAY.get(method, {})
        elif method == "card":
            out["stripe_url"] = _stripe_checkout_url_for_order(out, prac["email"], _session_token)
    return jsonify(out), (200 if out.get("ok") else 422)


_PRACTITIONER_ASSIST_SYSTEM = (
    "You are Dr. Glen Swartwout's clinical formulation assistant, helping a licensed "
    "practitioner or certified coach build a wholesale order for a patient (naturopathic "
    "physician, Hilo Hawai'i). Help them choose the right Functional Formulations for the "
    "patient's terrain and condition.\n\n"
    "How you work:\n"
    "- Write at a clinical practitioner level: anatomical and physiological terms, meridian "
    "names, mechanism of action, dosage ranges. Be precise and protocol-oriented.\n"
    "- A practitioner often gives enough up front; ask ONE focused clinical question only when "
    "you genuinely need more (presentation, terrain, what they have tried). Don't over-question.\n"
    "- Prefer Functional Formulations (Advanced Botanical / Nutritional) FIRST, since they "
    "simplify implementation, then individual remedies, healing tools, or adjunct therapies.\n"
    "- The RETRIEVED SNIPPETS are your source of truth; snippets tagged [AUTHORITATIVE ...] or "
    "type clinical-qa override anything else.\n"
    "- When you can name the single best PRIMARY formulation for this case, name it clearly with "
    "1-2 sentences of clinical rationale, and you may name 1-2 adjacent formulations that complete "
    "the protocol. Name products by their EXACT catalog name so they can be added to the order.\n"
    "- Never invent product names, URLs, prices, or product codes. Many products are NES-style "
    "infoceuticals identified by codes; there is no 'EN' series. The families and what they target: "
    "ES (Energetic Stars) = body SYSTEMS (lead with ES9 for ear/hearing, ES3 for the nervous "
    "system); ED (Energetic Drivers) = organs/tissues, except ED1 'Source' which affects all cells; "
    "EI (Energetic Integrators) = meridian-related; ET (Energetic Terrains) = terrains, mostly "
    "Phase 1 low-energy viral terrains except those specifically Fungal or Bacterial; MB = "
    "mind-body. Use ONLY the exact code/name that appears in the retrieved snippets; if unsure of "
    "the exact code, describe what's needed rather than guessing.\n"
    "- EMF/Kloud and other Centropix devices are ordered on the Centropix store, not through "
    "wholesale; mention them if relevant, but do not present them as add-to-order items.\n"
    "- Keep replies concise and clinical. Sign off as Dr. Glen."
)

_ASSIST_EXTRACT_SYSTEM = (
    "You read a clinical formulation recommendation written for a practitioner. Return JSON "
    "{\"products\": [{\"name\": \"<exact product/formulation name as written>\", "
    "\"why\": \"<short reason it was recommended>\"}]} listing EVERY specific product or "
    "formulation the assistant told the practitioner they could add to their order (primary and "
    "adjuncts). Use the exact product names. If none were named yet, return {\"products\": []}. "
    "Output ONLY the JSON, no prose, no code fences."
)


def _assist_resolve_products(items):
    """Resolve assistant-named products to addable cart slugs: fuzzy name/title
    match first, then a semantic fallback (embed + specific-formulations) for
    descriptively-named formulations. Excludes external/info_only items."""
    cat = _PRODUCTS.get("products") or {}
    out, seen = [], set()
    for it in (items or []):
        nm = (it.get("name") or "").strip()
        if not nm:
            continue
        slug = _pp.name_to_slug(nm, cat)
        if not slug:
            try:
                res = _idx.query(vector=embed(nm), top_k=1,
                                 namespace="specific-formulations", include_metadata=True)
                if res.matches and res.matches[0].score >= 0.83:
                    title = (res.matches[0].metadata or {}).get("title")
                    slug = _TITLE_TO_SLUG.get(title)
            except Exception as e:
                print(f"[assist] semantic resolve {nm!r}: {e!r}", flush=True)
        if not slug or slug in seen or (cat.get(slug) or {}).get("info_only"):
            continue
        seen.add(slug)
        out.append({"name": nm, "why": (it.get("why") or "").strip(), "slug": slug})
    return out


@app.route("/api/practitioner/assist", methods=["POST", "OPTIONS"])
def api_practitioner_assist():
    if request.method == "OPTIONS":
        return "", 200
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"error": "not signed in"}), 401
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    history = data.get("history") or []
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        try:
            q_vec = embed(query)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"}); return
        matches = _match_query_namespaces(q_vec)
        context_str, sources_list = build_context(matches) if matches else ("", [])
        messages = []
        for turn in history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content":
            f"PRACTITIONER MESSAGE: {query}\n\n"
            f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
            "Continue the clinical formulation match. If you can name the best primary Functional "
            "Formulation (and any adjuncts), name them with brief clinical rationale; otherwise ask "
            "the single best clinical question."})
        full = []
        try:
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=900,
                                     system=_PRACTITIONER_ASSIST_SYSTEM, messages=messages) as stream:
                for tok in stream.text_stream:
                    tok = _strip_dash(tok); full.append(tok); yield sse({"token": tok})
        except Exception as e:
            yield sse({"error": f"Claude error: {e}"}); return
        answer = "".join(full)

        products = []
        try:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-3:]) + f"\nassistant: {answer}"
            mx = _cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=400,
                                     system=_ASSIST_EXTRACT_SYSTEM,
                                     messages=[{"role": "user", "content": convo[:4000]}])
            txt = mx.content[0].text.strip()
            if txt.startswith("```"):
                txt = txt.split("```", 2)[1]
                if txt.startswith("json\n"): txt = txt[5:]
            obj = json.loads(txt)
            products = _assist_resolve_products(obj.get("products") or [])
        except Exception as e:
            print(f"[assist] extract: {e!r}", flush=True)
        if products:
            yield sse({"products": products})
        yield sse({"done": True, "sources": sources_list, "chunks_retrieved": len(matches)})

    return Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/practitioner/settings")
def practitioner_settings_page():
    resp = send_from_directory(STATIC, "practitioner-settings.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/console/biofield-portal")
def console_biofield_portal_page():
    resp = send_from_directory(STATIC, "console-biofield-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/console/pricing-settings")
def console_pricing_settings_page():
    resp = send_from_directory(STATIC, "console-pricing-settings.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/practitioner/settings", methods=["GET"])
def api_practitioner_settings_get():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    from dashboard import practitioner_settings as _ps
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ps.init_settings_table(cx)
        settings = _ps.get_settings(cx, pid)

    # show_contact lives on the Supabase practitioners row, not the SQLite
    # settings table. Read it directly; never let a failure 500 the page.
    show_contact = False
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("SELECT show_contact FROM practitioners WHERE id=%s", (pid,))
            row = cur.fetchone()
            show_contact = bool(row and row["show_contact"])
    except Exception as e:
        print(f"[practitioner-settings] show_contact read failed: {e!r}", flush=True)

    return jsonify({"ok": True, "branding": settings["branding"], "pricing": settings["pricing"],
                    "chat_enabled": settings.get("chat_enabled", False),
                    "show_contact": show_contact})


@app.route("/api/practitioner/settings", methods=["POST"])
def api_practitioner_settings_post():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    from dashboard import practitioner_settings as _ps
    from dashboard import practitioner_pricing as _ppr
    body = request.get_json(silent=True) or {}
    branding_in = body.get("branding") or {}
    pricing_in = body.get("pricing") or {}

    # Validate markup_pct is numeric
    markup_pct = pricing_in.get("default_markup_pct", 0)
    try:
        markup_pct = float(markup_pct)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "default_markup_pct must be numeric"}), 400

    # Validate and clamp per-SKU overrides to MAP
    map_cents = _ppr.DEFAULTS["map_default_cents"]
    overrides_in = pricing_in.get("overrides") or {}
    clamped = []
    overrides_out = {}
    for slug, price_val in overrides_in.items():
        try:
            price_val = int(price_val)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": f"override price for {slug!r} must be an integer (cents)"}), 400
        if price_val < map_cents:
            clamped.append({"slug": slug, "requested_cents": price_val, "clamped_to_cents": map_cents})
            price_val = map_cents
        overrides_out[slug] = price_val

    # Validate branding fields are strings
    for field in ("practice_name", "contact_email", "web_link", "logo_url", "photo_url",
                  "brand_color_1", "brand_color_2"):
        val = branding_in.get(field)
        if val is not None and not isinstance(val, str):
            return jsonify({"ok": False, "error": f"branding field {field!r} must be a string"}), 400

    branding_clean = {k: v for k, v in branding_in.items()
                      if k in ("practice_name", "contact_email", "web_link",
                               "logo_url", "photo_url", "brand_color_1", "brand_color_2")}
    pricing_clean = {"default_markup_pct": markup_pct, "overrides": overrides_out}

    # chat_enabled is a top-level bool field (not nested under branding or pricing)
    chat_enabled_in = body.get("chat_enabled")
    chat_enabled = bool(chat_enabled_in) if chat_enabled_in is not None else None

    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ps.init_settings_table(cx)
        _ps.set_branding(cx, pid, branding_clean, chat_enabled=chat_enabled)
        _ps.set_pricing(cx, pid, pricing_clean)
        settings = _ps.get_settings(cx, pid)

    # show_contact lives on the Supabase practitioners row. Only touch it when
    # the key is present, so saving branding/pricing alone never resets it.
    # Never let a Supabase failure 500 the whole settings save (log + continue).
    show_contact_out = None
    if "show_contact" in body:
        show_contact_out = bool(body.get("show_contact"))
        try:
            from db_supabase import supabase_cursor
            with supabase_cursor() as cur:
                cur.execute("UPDATE practitioners SET show_contact=%s, updated_at=now() "
                            "WHERE id=%s", (show_contact_out, pid))
        except Exception as e:
            print(f"[practitioner-settings] show_contact write failed: {e!r}", flush=True)

    resp = {"ok": True, "branding": settings["branding"],
            "pricing": settings["pricing"],
            "chat_enabled": settings.get("chat_enabled", False),
            "clamped": clamped}
    if show_contact_out is not None:
        resp["show_contact"] = show_contact_out
    return jsonify(resp)


def _record_dispensary_sale(code, customer_email, bottles, invoice_id):
    """Attribute a drop-ship sale to a practitioner by dispensary code and credit
    $20/bottle Wellness Credit. Idempotent on the client invoice id."""
    if not invoice_id:
        return
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return
    from dashboard import wallet as _wallet
    _wallet.earn_dropship(pid, bottles, qbo_invoice_id=str(invoice_id))
    _pp.record_dispensary_order(
        pid, invoice_id=str(invoice_id), customer_email=customer_email,
        bottles=int(bottles or 0),
        credit_earned_cents=int(bottles or 0) * _wallet.DROPSHIP_CREDIT_PER_BOTTLE_CENTS)
    if invoice_id:
        _ingest_order(source="dispensary", external_ref=str(invoice_id),
                      email=customer_email or "",
                      items=[{"name": "Dispensary", "qty": bottles}],
                      channel="retail")


@app.route("/dispensary/<code>")
def dispensary_landing(code):
    """Set the dispensary-attribution cookie and serve the branded client page.
    Invalid codes (no matching practitioner) return 404."""
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return jsonify({"ok": False, "error": "unknown dispensary code"}), 404
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", code or ""):
        return jsonify({"ok": False, "error": "invalid dispensary code"}), 404
    resp = send_from_directory(STATIC, "practitioner-client.html")
    resp.set_cookie("rm_dispensary", code, max_age=90 * 24 * 3600,
                    samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/api/client/<code>/catalog")
def api_client_catalog(code):
    """Return sellable Functional Formulations at the practitioner's price (>= MAP).
    Used by practitioner-client.html to populate the product list at page load.
    Also returns the practitioner's branding (may be empty dict if not set)."""
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return jsonify({"ok": False, "error": "unknown dispensary code"}), 404

    data = _pp.portal_data(pid) or {}
    practice_name = (data.get("practice_name") or data.get("name") or "Your Practitioner")

    # Load practitioner branding and settings (best-effort — never crash the catalog)
    branding = {}
    chat_enabled = False
    try:
        from dashboard import practitioner_settings as _ps
        with sqlite3.connect(LOG_DB) as _cx:
            _cx.row_factory = sqlite3.Row
            _ps.init_settings_table(_cx)
            _settings = _ps.get_settings(_cx, pid)
            branding = _settings.get("branding") or {}
            chat_enabled = bool(_settings.get("chat_enabled", False))
    except Exception:
        pass

    items = []
    for slug, p in (_PRODUCTS.get("products") or {}).items():
        if not p or p.get("inactive") or p.get("info_only"):
            continue
        if _is_pure_powder(p):
            continue
        price_cents = _dropship.practitioner_price_for(pid, slug)
        name = p.get("name") or slug
        items.append({"slug": slug, "name": name, "price_cents": price_cents})

    items.sort(key=lambda x: x["name"])
    return jsonify({"ok": True, "practice_name": practice_name, "branding": branding,
                    "chat_enabled": chat_enabled,
                    "client_points_enabled": bool(os.environ.get("CLIENT_POINTS_ENABLED")),
                    "items": items})


@app.route("/api/client/<code>/checkout", methods=["POST"])
def api_client_checkout(code):
    """Patient-paid dispensary checkout: patient buys at the practitioner's price,
    we ship to the patient, and the practitioner's margin is credited to their wallet
    on payment (Task 3 / checkout-return). Consent-gated: patient must be a Tier-1
    Member (ToS agreed + identified) before placing an order."""
    # 1. Resolve practitioner by dispensary code.
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return jsonify({"ok": False, "error": "unknown dispensary code"}), 404

    data = _pp.portal_data(pid) or {}

    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    method = (body.get("method") or "zelle").strip().lower()
    items = body.get("items") or []

    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400

    # 2. Consent gate — mirrors begin_checkout exactly: session cookie + email.
    _sid = request.cookies.get("amg_session", "")
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please add your name and agree to our Terms "
                                 "to place an order."}), 403

    # 3. Normalize and validate shipping address.
    ship = _normalize_ship_address(body.get("address") or {}, fallback_name=name)
    country = (ship.get("country") or "US").strip().upper()
    if country not in ("US", "USA", ""):
        return jsonify({"ok": False,
                        "error": "We ship to US addresses only — please use a US forwarding address."}), 400

    # 4. Build order structs.
    patient = {"email": email, "name": name, "ship": ship}
    prac = {
        "id": pid,
        "modules_completed": data.get("modules_completed", 0),
        "dispensary_code": code,
    }

    if method not in ("zelle", "wise", "card"):
        method = "zelle"
    if method == "card" and not _STRIPE_ACTIVE:
        method = "zelle"

    # 5. Build the patient-paid order (invoice billed to the patient at S).
    # Patient channel-locked points (flag-gated, card only): resolve scoped
    # balance + requested redemption so build_client_order can fee-cap-discount.
    redeem_req, bal_cents = 0, 0
    _scope = f"dispensary:{pid}"
    if os.environ.get("CLIENT_POINTS_ENABLED") and method == "card":
        try:
            redeem_req = max(0, int(body.get("points_to_redeem_cents") or 0))
        except (TypeError, ValueError):
            redeem_req = 0
        try:
            from dashboard import points as _points
            with sqlite3.connect(LOG_DB) as _bcx:
                _bcx.row_factory = sqlite3.Row
                _points.init_points_table(_bcx)
                bal_cents = _points.balance(_bcx, email, scope=_scope)
        except Exception:
            bal_cents = 0
    try:
        out = _dropship.build_client_order(items, prac, patient=patient, method=method,
                                           points_to_redeem_cents=redeem_req,
                                           points_balance_cents=bal_cents)
    except Exception as e:
        print(f"[client-checkout] build failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Checkout failed. Please try again."}), 500

    if not out.get("ok"):
        return jsonify(out), 422

    # 6. Record order in BOS.
    _ingest_order(source="dispensary",
                  external_ref=str(out.get("invoice_id") or ""),
                  email=email,
                  name=name,
                  total_cents=int(round((out.get("total") or 0) * 100)),
                  address=ship,
                  channel="retail",
                  get_cents=out.get("get_cents", 0))

    # 7. Payment method: alt-pay or Stripe.
    # Stripe metadata carries kind/practitioner_id/margin_cents/invoice_id so the
    # checkout-return handler (Task 3) can credit the margin to the wallet.
    if method in ("zelle", "wise"):
        out["pay_instructions"] = _ALT_PAY.get(method, {})
    elif method == "card" and _STRIPE_ACTIVE:
        try:
            from dashboard import stripe_pay as _sp
            total_cents = int(round((out.get("total") or 0) * 100))
            if total_cents > 0:
                success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                           f"?session_id={{CHECKOUT_SESSION_ID}}")
                sess = _sp.create_checkout_session(
                    total_cents, customer_email=email,
                    description=f"Remedy Match dispensary order #{out.get('invoice_id')}",
                    metadata={
                        "kind": "client",
                        "practitioner_id": pid,
                        "margin_cents": str(out.get("margin_cents", 0)),
                        "invoice_id": str(out.get("invoice_id") or ""),
                        "customer_id": str(out.get("customer_id") or ""),
                        "patient_email": email,
                        "subtotal_cents": str(out.get("subtotal_cents", 0)),
                        "points_redeemed_cents": str(out.get("points_redeemed_cents", 0)),
                    },
                    success_url=success,
                    cancel_url=f"{PUBLIC_BASE_URL}/dispensary/{code}")
                out["stripe_url"] = sess.get("url") or ""
        except Exception as e:
            print(f"[client-checkout] stripe failed: {e!r}", flush=True)
            _alert_stripe("client-checkout", e)
            out["stripe_url"] = ""
            out["payment_error"] = _CARD_UNAVAILABLE

    return jsonify(out)


@app.route("/api/client/<code>/points", methods=["POST"])
def api_client_points(code):
    """Patient's practitioner-scoped loyalty balance (consent-gated, flag-aware)."""
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return jsonify({"ok": False, "error": "unknown dispensary code"}), 404
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    _sid = request.cookies.get("amg_session", "")
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True}), 403
    enabled = bool(os.environ.get("CLIENT_POINTS_ENABLED"))
    bal = 0
    if enabled:
        from dashboard import points as _points
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _points.init_points_table(cx)
            bal = _points.balance(cx, email, scope=f"dispensary:{pid}")
    return jsonify({"ok": True, "balance_cents": bal, "client_points_enabled": enabled})


def _build_ff_catalog():
    """Return [{slug, name, description}] for all active Functional Formulations.
    Excludes Pure Powders and info_only items — mirrors /api/client/<code>/catalog."""
    catalog = []
    for slug, p in (_PRODUCTS.get("products") or {}).items():
        if not p or p.get("inactive") or p.get("info_only"):
            continue
        if _is_pure_powder(p):
            continue
        catalog.append({
            "slug": slug,
            "name": p.get("name") or slug,
            "description": p.get("description") or "",
        })
    return catalog


@app.route("/api/practitioner/chat", methods=["POST"])
def api_practitioner_chat():
    """Practitioner-authenticated product-selection chat scoped to FF catalog.
    Auth via session token (401 if none). Returns {ok, reply, suggestions}."""
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "authentication required"}), 401

    body = request.get_json(silent=True) or {}
    message = body.get("message") or ""
    history = body.get("history") or []

    catalog = _build_ff_catalog()
    result = _chat.scoped_reply(message, history, catalog)

    suggestions = []
    for slug in (result.get("suggested_slugs") or []):
        p = _get_product(slug)
        if not p:
            continue
        suggestions.append({
            "slug": slug,
            "name": p.get("name") or slug,
            "price_cents": _dropship.practitioner_price_for(pid, slug),
        })

    return jsonify({"ok": True, "reply": result["reply"], "suggestions": suggestions})


@app.route("/api/client/<code>/chat", methods=["POST"])
def api_client_chat(code):
    """Patient-facing product-selection chat scoped to the practitioner's FF catalog.
    Resolves the practitioner by dispensary code (404 if unknown).
    Consent-gated: patient must be a Tier-1 Member (403 need_optin if not).
    Returns {ok, reply, suggestions} with prices at the practitioner's price."""
    pid = _pp.practitioner_id_by_dispensary_code(code)
    if not pid:
        return jsonify({"ok": False, "error": "unknown dispensary code"}), 404

    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    _sid = request.cookies.get("amg_session", "")
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please add your name and agree to our Terms "
                                 "to use the assistant."}), 403

    message = body.get("message") or ""
    history = body.get("history") or []

    catalog = _build_ff_catalog()
    result = _chat.scoped_reply(message, history, catalog)

    suggestions = []
    for slug in (result.get("suggested_slugs") or []):
        p = _get_product(slug)
        if not p:
            continue
        suggestions.append({
            "slug": slug,
            "name": p.get("name") or slug,
            "price_cents": _dropship.practitioner_price_for(pid, slug),
        })

    return jsonify({"ok": True, "reply": result["reply"], "suggestions": suggestions})


def _stripe_checkout_url_for_retail(out, email, slug, group_bundle_months=0):
    """Create a Stripe Checkout Session for a retail funnel order; returns its URL.
    Captures invoice_id/customer_id/slug in metadata so the return handler records
    the QBO payment and the PaymentIntent (for refunds).

    When group_bundle_months > 0 (caller decided the buyer opted in and the flag is
    on), vault the card (save_card=True) and stamp grant_group_months + email so the
    return handler grants the free live-group window and starts the $99/mo membership."""
    try:
        from dashboard import stripe_pay
        total_cents = int(round(float(out.get("total") or 0) * 100))
        if total_cents <= 0:
            return ""
        success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}")
        metadata = {"invoice_id": out.get("invoice_id"),
                    "customer_id": out.get("customer_id"),
                    "kind": "retail", "slug": slug}
        save_card = False
        if group_bundle_months and int(group_bundle_months) > 0:
            metadata["grant_group_months"] = str(int(group_bundle_months))
            metadata["email"] = email
            save_card = True
        sess = stripe_pay.create_checkout_session(
            total_cents, customer_email=email,
            description=f"Remedy Match order #{out.get('doc_number')}",
            metadata=metadata,
            success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/begin/buy/{slug}",
            save_card=save_card)
        return sess.get("url") or ""
    except Exception as e:
        print(f"[stripe-retail] session create failed: {e!r}", flush=True)
        _alert_stripe("retail", e)
        return ""


# ── Reorder cart (logged-in client reorder → existing Stripe/QBO flow) ─────────
def _shipping_line(shipping_cents):
    if not shipping_cents:
        return []
    return [{"name": "Shipping (USPS)", "amount": round(int(shipping_cents) / 100.0, 2),
             "qty": 1, "description": "USPS shipping"}]


def _active_coupon_pct():
    """Return the daily coupon discount_percent if a coupon is active today, else None."""
    return _COUPONS.get("_discount_percent", 0) if get_today_coupon_code() else None


def _stripe_checkout_url_for_reorder(out, email):
    """Stripe Checkout for a reorder cart; mirrors _stripe_checkout_url_for_retail
    but the cancel returns to /reorder. Records via /begin/checkout-return (invoice-based)."""
    try:
        from dashboard import stripe_pay
        total_cents = int(round(float(out.get("total") or 0) * 100))
        if total_cents <= 0:
            return ""
        success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}")
        sess = stripe_pay.create_checkout_session(
            total_cents, customer_email=email,
            description=f"Remedy Match reorder #{out.get('doc_number')}",
            metadata={"invoice_id": out.get("invoice_id"),
                      "customer_id": out.get("customer_id"), "kind": "reorder"},
            success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/reorder")
        return sess.get("url") or ""
    except Exception as e:
        print(f"[stripe-reorder] session create failed: {e!r}", flush=True)
        _alert_stripe("reorder", e)
        return ""


def _reorder_email_from_cookie():
    return (request.cookies.get("rm_reorder_email", "") or "").strip().lower()


@app.route("/reorder")
def reorder_page():
    return send_from_directory(STATIC, "reorder.html")


@app.route("/reorder/request", methods=["POST"])
def reorder_request():
    """Email a magic link to start a reorder session. Always 200 (no enumeration)."""
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if email and "@" in email:
        token = secrets.token_urlsafe(32)
        now = _now_utc()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                "VALUES (?,?,?,?,?)",
                (_hash_token(token), email, "reorder", now.isoformat(),
                 (now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)).isoformat()))
            cx.commit()
        try:
            send_magic_link_email(email, "", f"{PUBLIC_BASE_URL}/reorder/auth/{token}")
        except Exception as e:
            print(f"[reorder] magic link send failed: {e!r}", flush=True)
    return jsonify({"ok": True})


@app.route("/reorder/auth/<token>", methods=["GET"])
def reorder_auth(token):
    """Validate the reorder magic link, set the rm_reorder_email cookie, → /reorder."""
    from flask import redirect as _redirect
    th = _hash_token((token or "").strip())
    email = None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='reorder'", (th,)).fetchone()
        if row and not row["consumed_at"]:
            try:
                if datetime.fromisoformat(row["expires_at"]) >= _now_utc():
                    email = row["email"]
            except Exception:
                email = None
        if email:
            cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                       (_now_utc().isoformat(), th))
            cx.commit()
    if not email:
        return ("<p style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
                "This reorder link is invalid or has expired. Please request a new one.</p>"), 400
    resp = _redirect("/reorder", code=302)
    resp.set_cookie("rm_reorder_email", email, max_age=60 * 60 * 24 * 30,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


# ── Tokenized per-client portal ("Create Your Own Healing Adventure") ────────
# One durable row per client (dashboard/client_portal.py); the token is the only
# auth — no login, mirroring /invoice/<token>. Inherently dark: no route is
# reachable until a token is minted via /admin/portal/upsert (CONSOLE_SECRET) and
# shared with the client. First client: Brooke Webb.

def _portal_console_ok():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return False
    return True


def _client_login_enabled() -> bool:
    """Real client login (magic-link → session cookie) is dark by default. While
    off, the emailed /portal/<token> link is the only auth and resolve_identity
    ignores any session cookie. Flip this flag (next slice) to light up login."""
    return os.environ.get("CLIENT_LOGIN_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _portal_offers_enabled() -> bool:
    """Master flag for the portal 'What's next' offer surface. Dark by default."""
    return os.environ.get("PORTAL_OFFERS_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _enabled_offer_keys() -> set:
    """Which ladder rungs are purchasable right now (master + per-rung flags)."""
    if not _portal_offers_enabled():
        return set()
    keys = set()
    if os.environ.get("SUBSCRIPTIONS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        keys.add("live_group")
    if os.environ.get("BIOFIELD_CHECKOUT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        keys.add("biofield")
    return keys


def _portal_priced_lines(items):
    """Build QBO invoice lines from a portal's reorder items, honoring an optional
    per-item ``price_cents`` override (the client's practitioner-special price);
    falls back to catalog/volume pricing when no override is present.
    Returns ``(lines, items_rec, subtotal_cents)``."""
    lines, items_rec, subtotal_cents = [], [], 0
    for it in (items or []):
        slug = (it.get("slug") or "").strip()
        p = _get_product(slug) if slug else None
        if not p:
            continue
        try:
            qty = max(1, min(int(it.get("qty", 1) or 1), 99))
        except Exception:
            qty = 1
        override = it.get("price_cents")
        unit_cents = int(override) if override is not None else int(_qty_unit_cents(p, qty))
        subtotal_cents += unit_cents * qty
        lines.append({"name": p["name"], "amount": round(unit_cents / 100.0, 2),
                      "qty": qty, "item_id": p.get("qbo_item_id"), "description": p["name"]})
        items_rec.append({"name": p["name"], "qty": qty, "desc": p["name"]})
    return lines, items_rec, subtotal_cents


def _portal_record_for(cx, token):
    """A portal content record by path token, OR — when client login is live —
    by the rm_portal_session cookie (the tokenless /portal/me logged-in home).
    Returns {email, name, content} or None. The token path is unchanged, so
    emailed /portal/<token> links keep working exactly as before."""
    from dashboard import client_portal as _cp
    portal = _cp.get_portal_by_token(cx, token)
    if portal:
        return portal
    if _client_login_enabled():
        from dashboard import portal_identity as _pi
        sess = request.cookies.get("rm_portal_session", "")
        ident = _pi.identity_from_session(cx, sess) if sess else None
        if ident:
            rec = _cp.get_portal_content_by_email(cx, ident.email)
            if rec:
                return {"email": ident.email, "name": rec.get("name"),
                        "content": rec.get("content")}
    return None


@app.route("/portal/<token>")
def client_portal_page(token):
    return send_from_directory(STATIC, "client-portal.html")


@app.route("/api/portal/<token>")
def api_client_portal(token):
    from dashboard import client_portal as _cp
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return jsonify({"error": "not found"}), 404
    try:
        from dashboard import notify_state as _ns
        with _db_lock, sqlite3.connect(LOG_DB) as _cxe:
            _ns.mark_engaged(_cxe, (portal.get("email") or ""))
    except Exception as e:
        print(f"[engaged] {e!r}", flush=True)
    content = dict(portal.get("content") or {})
    from dashboard import portal_biofield_reports as _pbr
    import datetime as _dt
    email_for_reports = (portal.get("email") or "").strip().lower()
    cx_r = sqlite3.connect(LOG_DB)
    _pbr.init_table(cx_r)
    dates = _pbr.list_report_dates(cx_r, email_for_reports) if email_for_reports else []
    req_date = (request.args.get("scan_date") or "").strip()
    if dates:
        picked = req_date if req_date in dates else dates[0]
        rep = _pbr.get_report(cx_r, email_for_reports, picked) or {}
        bf_content = rep.get("content") or {}
        bf_status = rep.get("status") or "confirmed"
        bf_scan_date, bf_scan_dates = picked, dates
        bf_actionable = (bf_status != "confirmed") and _pbr.is_actionable(
            picked, _dt.date.today().isoformat())
    else:
        bf_content = content
        bf_status = content.get("biofield_status") or "confirmed"
        bf_scan_date, bf_scan_dates, bf_actionable = None, [], False
    cx_r.close()
    bf_confirmed = bf_status == "confirmed"
    bf_layers = []
    for L in (bf_content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        if bf_confirmed:
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
        bf_layers.append(item)
    reorder_src = bf_content.get("reorder_items") if dates else content.get("reorder_items")
    display = []
    for it in (reorder_src or []):
        slug = (it.get("slug") or "").strip()
        p = _get_product(slug) if slug else None
        regular = (p or {}).get("price_cents")
        override = it.get("price_cents")
        special = int(override) if override is not None else regular
        display.append({
            "slug": slug, "qty": int(it.get("qty", 1) or 1),
            "name": (p or {}).get("name", slug), "price_cents": special,
            "regular_price_cents": regular,
            "is_special": bool(override is not None and regular is not None and int(override) < int(regular)),
            "available": bool(p)})
    return jsonify({
        "name": portal.get("name"),
        "biofield_status": bf_status, "blurred": not bf_confirmed,
        "actionable": bf_actionable, "scan_date": bf_scan_date, "scan_dates": bf_scan_dates,
        "greeting": bf_content.get("greeting", ""),
        "video": bf_content.get("video") or {},
        "layers": bf_layers,
        "pricing_note": bf_content.get("pricing_note", "") if bf_confirmed else "",
        "reorder_items": display,
    })


@app.route("/api/portal/<token>/notify-pref", methods=["POST"])
def api_portal_notify_pref(token):
    from dashboard import client_portal as _cp, notify_state as _ns
    pref = ((request.get_json(silent=True) or {}).get("pref") or "").strip().lower()
    if pref not in ("in", "out"):
        return jsonify({"error": "pref must be in|out"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        portal = _cp.get_portal_by_token(cx, token)
        if not portal:
            return jsonify({"error": "not found"}), 404
        _ns.set_opt(cx, portal["email"], pref)
    return jsonify({"ok": True, "pref": pref})


@app.route("/sms/inbound", methods=["POST"])
def sms_inbound():
    from dashboard import notify_state as _ns
    frm = (request.form.get("From") or request.values.get("From") or "").strip()
    body = (request.form.get("Body") or request.values.get("Body") or "").strip().upper()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        email = _ns.email_by_phone(cx, frm)
        if email:
            if body in ("STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "QUIT"):
                _ns.set_opt(cx, email, "out")
            elif body in ("START", "YES", "UNSTOP"):
                _ns.set_opt(cx, email, "in")
    return ("", 204)


@app.route("/api/portal/<token>/checkout", methods=["POST"])
def api_client_portal_checkout(token):
    """Build a real live Stripe checkout for the client's pre-loaded remedies at
    their portal price (per-item practitioner-special override honored). Mirrors
    the legacy reorder path but with custom pricing, so the live /reorder/checkout
    stays untouched."""
    from dashboard import client_portal as _cp
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return jsonify({"error": "not found"}), 404
    email = (portal.get("email") or "").strip().lower()
    items = (portal.get("content") or {}).get("reorder_items") or []
    lines, items_rec, _subtotal = _portal_priced_lines(items)
    if not lines:
        return jsonify({"error": "Your remedies are no longer available — please reach out and we'll help."}), 400
    if not _STRIPE_ACTIVE:
        return jsonify({"error": "Card checkout is temporarily unavailable. Please reach out and we'll help."}), 503
    try:
        from dashboard import qbo_billing as _qb_local
        ship = {}
        try:
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                prior = _bos_orders.list_orders_by_email(cx, email, limit=1)
            if prior:
                ship = prior[0].get("address") or {}
        except Exception:
            ship = {}  # no prior order / address on file — proceed; collected at checkout
        cust = _qb_local.find_or_create_customer(email, ship.get("name", ""))
        inv = _qb_local.create_invoice(cust, lines, allow_online_pay=True, email_to=email)
        out = {"invoice_id": inv.get("Id"), "customer_id": cust.get("Id"),
               "doc_number": inv.get("DocNumber"), "total": inv.get("TotalAmt")}
        _ingest_order(source="portal-reorder", external_ref=inv.get("Id"), email=email,
                      name=ship.get("name", ""), items=items_rec,
                      total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                      address=ship, channel="retail")
        stripe_url = _stripe_checkout_url_for_reorder(out, email)
        if not stripe_url:
            return jsonify({"error": _CARD_UNAVAILABLE}), 502
        return jsonify({"ok": True, "stripe_url": stripe_url})
    except Exception as e:
        app.logger.exception("portal checkout failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


def _biofield_content_clean(content):
    """Drop blank layers (no title); return (clean_content, has_content)."""
    content = dict(content or {})
    layers = [L for L in (content.get("layers") or []) if (L.get("title") or "").strip()]
    content["layers"] = layers
    has = bool(layers or (content.get("video") or {}).get("url") or (content.get("greeting") or "").strip())
    return content, has


@app.route("/api/console/biofield-portal", methods=["POST"])
def api_console_biofield_publish():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    scan_date = (body.get("scan_date") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    content, has = _biofield_content_clean(body.get("content") or {})
    if not has:
        return jsonify({"error": "Add some content — at least one layer, a video, or a greeting."}), 400
    from dashboard import client_portal as _cp
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, pid = _cp.upsert_portal(cx, email, name, content)
        _cp.set_biofield_status(cx, email, "confirmed")
        try:
            from dashboard import ghl_queue as _gq
            _gq.init_ghl_queue_table(cx)
            _gq.enqueue(cx, op="tag_add", email=email,
                        payload={"tags": ["e4l:confirmed"]}, actor="console")
        except Exception as e:
            print(f"[biofield-publish] confirm tag failed: {e!r}", flush=True)
        if scan_date:
            from dashboard import portal_biofield_reports as _pbr
            _pbr.init_table(cx)
            _pbr.upsert_report(cx, email, scan_date,
                               (body.get("scan_id") or ""), content, "confirmed")
        _log_biofield_correction(cx, email, scan_date, content)
    url = f"{PUBLIC_BASE_URL}/portal/{token}" if token else None
    emailed = False
    if token and body.get("send"):
        try:
            _send_full_report_email(
                email, name, "Your personal healing home is ready 🌺",
                f"Aloha {name or ''},\n\nYour personal healing home is ready:\n\n{url}\n\n"
                f"With aloha,\nDr. Glen & Rae")
            emailed = True
        except Exception as e:
            print(f"[biofield-publish] send failed: {e!r}", flush=True)
    return jsonify({"ok": True, "token": token, "url": url, "portal_id": pid,
                    "updated": token is None, "emailed": emailed})


@app.route("/api/console/biofield-portal", methods=["GET"])
def api_console_biofield_load():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    email = (request.args.get("email") or "").strip().lower()
    if not email:
        return jsonify({"found": False, "content": {}})
    from dashboard import client_portal as _cp
    from dashboard import portal_biofield_reports as _pbr
    req_date = (request.args.get("scan_date") or "").strip()
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pbr.init_table(cx)
        dates = _pbr.list_report_dates(cx, email)
        rec = _cp.get_portal_content_by_email(cx, email)
        if dates:
            picked = req_date if req_date in dates else dates[0]
            rep = _pbr.get_report(cx, email, picked) or {}
            return jsonify({"found": True, "name": (rec or {}).get("name") or "",
                            "content": rep.get("content") or {}, "status": rep.get("status") or "",
                            "scan_date": picked, "scan_dates": dates, "has_token": bool(rec)})
    if not rec:
        return jsonify({"found": False, "name": "", "content": {}, "has_token": False,
                        "scan_dates": [], "scan_date": None})
    return jsonify({"found": True, "name": rec.get("name") or "",
                    "content": rec.get("content") or {}, "has_token": True,
                    "scan_dates": [], "scan_date": None})


@app.route("/api/console/biofield/review-queue", methods=["GET"])
def api_console_biofield_review_queue():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_portal as _cp
    from dashboard import portal_biofield_reports as _pbr
    queue = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cp.init_client_portal_table(cx)
        _pbr.init_table(cx)
        names = {r["email"]: r["name"] for r in
                 cx.execute("SELECT email, name FROM client_portals").fetchall()}
        rows = cx.execute("SELECT email, scan_date, updated_at FROM portal_biofield_reports "
                          "WHERE status='requested' ORDER BY updated_at DESC").fetchall()
        for r in rows:
            queue.append({"email": r["email"], "name": names.get(r["email"], ""),
                          "scan_date": r["scan_date"], "requested_at": r["updated_at"]})
        for r in cx.execute("SELECT email, name, updated_at FROM client_portals").fetchall():
            if _cp.get_biofield_status(cx, r["email"]) == "requested" \
               and not _pbr.list_report_dates(cx, r["email"]):
                queue.append({"email": r["email"], "name": r["name"],
                              "scan_date": None, "requested_at": r["updated_at"]})
    return jsonify({"queue": queue})


def _init_biofield_corrections(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT,
        content_json TEXT, created_at TEXT)""")
    cx.commit()


def _log_biofield_correction(cx, email, scan_date, content):
    _init_biofield_corrections(cx)
    cx.execute("INSERT INTO biofield_corrections (email, scan_date, content_json, created_at) "
               "VALUES (?,?,?,?)",
               ((email or "").strip().lower(), scan_date or "",
                json.dumps(content or {}), datetime.utcnow().isoformat() + "Z"))
    cx.commit()


@app.route("/api/console/biofield/corrections", methods=["GET"])
def api_console_biofield_corrections():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    since = (request.args.get("since") or "").strip()
    out = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _init_biofield_corrections(cx)
        q = "SELECT email, scan_date, content_json, created_at FROM biofield_corrections"
        args = ()
        if since:
            q += " WHERE created_at >= ?"; args = (since,)
        q += " ORDER BY created_at ASC"
        for r in cx.execute(q, args).fetchall():
            try:
                content = json.loads(r["content_json"] or "{}")
            except Exception:
                content = {}
            out.append({"email": r["email"], "scan_date": r["scan_date"],
                        "content": content, "created_at": r["created_at"]})
    return jsonify({"corrections": out})


@app.route("/api/console/biofield-portal/import-fmp", methods=["POST"])
def api_console_biofield_import_fmp():
    """Pull a client's FMP Causal Chain Report, build portal layers + AI-drafted
    prose, and return content to pre-fill the editor (review-then-publish)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    from dashboard import fmp_biofield as _fb
    content = _fb.import_content(email, (body.get("name") or "").strip())
    if content is None:
        return jsonify({"found": False})
    return jsonify({"found": True, "content": content})


@app.route("/api/console/biofield-portal/catalog", methods=["GET"])
def api_console_biofield_catalog():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    items = []
    for p in _bos_products.catalog(with_ingredients_only=False):
        items.append({"slug": p.get("slug"), "name": p.get("name"),
                      "price_cents": p.get("price_cents")})
    return jsonify({"products": items})


@app.route("/api/portal/<token>/view")
def api_client_portal_view(token):
    """Role-aware portal payload: account + orders + biofield + a reserved
    sales/upgrade stub, composed through the identity seam. The page renders
    whichever blocks come back. Today identity comes from the path token; the
    (dark) session cookie is honored only when CLIENT_LOGIN_ENABLED is on."""
    from dashboard import client_portal as _cp
    from dashboard import portal_identity as _pi
    from dashboard import portal_view as _pv
    sess = request.cookies.get("rm_portal_session", "")
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        ident = _pi.resolve_identity(
            cx, token=token, session_token=sess,
            client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not found"}), 404
        view = _pv.get_portal_view(cx, ident.person_id,
                                   offers_enabled_keys=_enabled_offer_keys())
    if view is None:
        return jsonify({"error": "not found"}), 404
    view["auth_method"] = ident.auth_method
    return jsonify(view)


def _biofield_transition(token, new_status, tag):
    from dashboard import client_portal as _cp
    from dashboard import portal_identity as _pi
    from dashboard import portal_biofield_reports as _pbr
    from dashboard import ghl_queue as _gq
    import datetime as _dt
    sess = request.cookies.get("rm_portal_session", "")
    body = request.get_json(silent=True) or {}
    req_date = (body.get("scan_date") or request.args.get("scan_date") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pi._ensure_people_table(cx)
        _pbr.init_table(cx)
        ident = _pi.resolve_identity(cx, token=token, session_token=sess,
                                     client_login_enabled=_client_login_enabled())
        if ident is None:
            return jsonify({"error": "not found"}), 404
        dates = _pbr.list_report_dates(cx, ident.email)
        if dates:
            picked = req_date if req_date in dates else dates[0]
            if not _pbr.is_actionable(picked, _dt.date.today().isoformat()):
                return jsonify({"error": "scan no longer actionable"}), 409
            if not _pbr.set_report_status(cx, ident.email, picked, new_status):
                return jsonify({"error": "not found"}), 404
        else:
            if not _cp.set_biofield_status(cx, ident.email, new_status):
                return jsonify({"error": "not found"}), 404
        try:
            _gq.init_ghl_queue_table(cx)
            _gq.enqueue(cx, op="tag_add", email=ident.email,
                        payload={"tags": [tag]}, actor="portal")
        except Exception as e:
            print(f"[biofield-transition] ghl enqueue failed: {e!r}", flush=True)
    return jsonify({"ok": True, "status": new_status})


@app.route("/api/portal/<token>/biofield/interest", methods=["POST"])
def api_portal_biofield_interest(token):
    return _biofield_transition(token, "interested", "e4l:interested")


@app.route("/api/portal/<token>/biofield/request", methods=["POST"])
def api_portal_biofield_request(token):
    return _biofield_transition(token, "requested", "e4l:requested")


# ── Scaffolded client login (DARK behind CLIENT_LOGIN_ENABLED) ────────────────
# The real-login drop-in: magic-link email -> one-time token -> client session
# cookie, targeting portal_identity's session branch. Every route 404s while the
# flag is off, so the emailed /portal/<token> link stays the only client auth.
# These static routes outrank the dynamic /portal/<token> rule when registered.

@app.route("/portal/login", methods=["GET"])
def client_login_page():
    if not _client_login_enabled():
        return ("Not found", 404)
    return send_from_directory(STATIC, "client-login.html")


@app.route("/portal/login-request", methods=["POST"])
def client_login_request():
    if not _client_login_enabled():
        return jsonify({"error": "not found"}), 404
    from dashboard import portal_identity as _pi
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if "@" in email:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            row = cx.execute("SELECT id, name FROM people WHERE email=?", (email,)).fetchone()
            if row:
                magic = _pi.create_client_magic_link(cx, row[0], email)
                try:
                    _send_full_report_email(
                        email, row[1] or "", "Your Remedy Match sign-in link",
                        "Aloha,\n\nClick to sign in to your healing home:\n"
                        f"{PUBLIC_BASE_URL}/portal/login-verify?token={magic}\n\n"
                        "This link expires in 15 minutes.")
                except Exception as e:
                    print(f"[client-login] email failed: {e!r}", flush=True)
    # No account enumeration: same response whether or not the email exists.
    return jsonify({"ok": True,
                    "message": "If that email has a portal, a sign-in link is on its way."})


@app.route("/portal/login-verify", methods=["GET"])
def client_login_verify():
    from flask import redirect as _redir, make_response as _mkresp
    if not _client_login_enabled():
        return ("Not found", 404)
    from dashboard import portal_identity as _pi
    token = (request.args.get("token") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        pid = _pi.consume_client_magic_link(cx, token) if token else None
        if not pid:
            return _redir("/portal/login?error=link")
        sess = _pi.create_client_session(cx, pid, "")
    resp = _mkresp(_redir("/portal/me"))
    resp.set_cookie("rm_portal_session", sess, max_age=30 * 86400,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/portal/me", methods=["GET"])
def client_portal_me():
    # Tokenless portal home for a logged-in client; the page reads the session
    # cookie via /api/portal/me/view. Dark until login is enabled.
    if not _client_login_enabled():
        return ("Not found", 404)
    return send_from_directory(STATIC, "client-portal.html")


# ── Portal "What's next" offer checkouts (dark behind PORTAL_OFFERS_ENABLED) ──
# The Live Group rung has no standalone checkout elsewhere, so we add one here:
# a $0 Stripe setup session vaults the card; the membership row is created on
# return and billed monthly by the subscriptions scheduler. Mirrors /api/studio.

@app.route("/portal/offer/live-group/checkout", methods=["POST"])
def portal_group_join_checkout():
    if not _portal_offers_enabled():
        return jsonify({"error": "not found"}), 404
    token = request.args.get("token", "") or (request.get_json(silent=True) or {}).get("token", "")
    sess_cookie = request.cookies.get("rm_portal_session", "")
    from dashboard import portal_identity as _pi
    with sqlite3.connect(LOG_DB) as cx:
        ident = _pi.resolve_identity(cx, token=token, session_token=sess_cookie,
                                     client_login_enabled=_client_login_enabled())
    if ident is None:
        return jsonify({"error": "not found"}), 404
    try:
        from dashboard import stripe_pay
        sess = stripe_pay.create_setup_session(
            customer_email=ident.email,
            metadata={"kind": "group_join", "email": ident.email},
            success_url=(f"{PUBLIC_BASE_URL}/portal/offer/live-group/return"
                         f"?session_id={{CHECKOUT_SESSION_ID}}"),
            cancel_url=f"{PUBLIC_BASE_URL}/portal/me")
    except Exception:
        app.logger.exception("group-join setup session failed")
        return jsonify({"error": "Could not start checkout. Please reach out and we'll help."}), 502
    return jsonify({"ok": True, "stripe_url": sess.get("url", "")})


@app.route("/portal/offer/live-group/return")
def portal_group_join_return():
    """Stripe setup return: vault the card and create the membership (first charge
    one cycle out, billed by the subscriptions scheduler). Idempotent; never 500s."""
    from flask import redirect as _redir
    sid = (request.args.get("session_id") or "").strip()
    if _portal_offers_enabled() and sid:
        try:
            from dashboard import stripe_pay, subscriptions as _subs, portal_offers as _po
            import datetime as _dt
            sess = stripe_pay.get_session(sid)
            email = ((sess.get("metadata") or {}).get("email") or "").strip().lower()
            si = stripe_pay.get_setup_intent(sess.get("setup_intent"))
            cus, pm = si.get("customer"), si.get("payment_method")
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                _subs.init_subscriptions_table(cx)
                _subs.migrate_add_membership_columns(cx)
                if email and cus and pm and not _subs.active_memberships_by_email(cx, email):
                    next_date = _subs.add_months(_dt.date.today().isoformat(), 1)
                    _subs.create_membership(
                        cx, email=email, stripe_customer_id=cus,
                        stripe_payment_method_id=pm,
                        amount_cents=_po.MEMBERSHIP_PRICE_CENTS, next_charge_date=next_date)
        except Exception as e:
            print(f"[group-join] return failed: {e!r}", flush=True)
    return _redir("/portal/me?joined=1")


@app.route("/admin/portal/upsert", methods=["POST"])
def admin_client_portal_upsert():
    """Seed/update a client portal on the live DB (the bridge, since Render can't
    read the local vault). Console-secret gated; reusable for every future client."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    content = body.get("content") or {}
    if not email:
        return jsonify({"error": "email required"}), 400
    from dashboard import client_portal as _cp
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, pid = _cp.upsert_portal(cx, email, name, content)
        scan_date = (body.get("scan_date") or "").strip()
        if scan_date:
            from dashboard import portal_biofield_reports as _pbr
            _pbr.init_table(cx)
            _pbr.upsert_report(cx, email, scan_date, (body.get("scan_id") or ""),
                               content, content.get("biofield_status") or "ai_draft")
    if token is None:
        return jsonify({"ok": True, "updated": True, "portal_id": pid,
                        "note": "existing portal updated; prior link unchanged"})
    url = f"{PUBLIC_BASE_URL}/portal/{token}"
    emailed = False
    if body.get("send"):
        try:
            _send_full_report_email(
                email, name, "Your personal healing home is ready 🌺",
                f"Aloha {name or ''},\n\nYour personal healing home is ready — your "
                f"biofield walkthrough, your healing path, and your remedies, all in "
                f"one place:\n\n{url}\n\nWith aloha,\nDr. Glen & Rae")
            emailed = True
        except Exception as e:
            print(f"[portal-upsert] send failed: {e!r}", flush=True)
    return jsonify({"ok": True, "token": token, "url": url,
                    "portal_id": pid, "emailed": emailed})


@app.route("/admin/portal/delete", methods=["POST"])
def admin_client_portal_delete():
    """Remove a client portal and ALL its biofield traces (reports + corrections),
    keyed by email. Console-secret gated. For clearing test/erroneous portals."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    from dashboard import client_portal as _cp
    from dashboard import portal_biofield_reports as _pbr
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pbr.init_table(cx)
        _init_biofield_corrections(cx)
        n1 = cx.execute("DELETE FROM client_portals WHERE email=?", (email,)).rowcount
        n2 = cx.execute("DELETE FROM portal_biofield_reports WHERE email=?", (email,)).rowcount
        n3 = cx.execute("DELETE FROM biofield_corrections WHERE email=?", (email,)).rowcount
        cx.commit()
    return jsonify({"ok": True, "deleted": {"portals": n1, "reports": n2, "corrections": n3}})


@app.route("/admin/portal/reissue-link", methods=["POST"])
def admin_portal_reissue_link():
    """Mint a fresh token for an existing portal and return its URL (the original
    token is stored only as a one-way hash and can't be recovered). The old link
    stops working. Console-secret gated. Optional send=true emails the new link."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    from dashboard import client_portal as _cp
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token = _cp.reissue_token(cx, email)
        rec = _cp.get_portal_content_by_email(cx, email)
    if not token:
        return jsonify({"error": "no portal for that email"}), 404
    url = f"{PUBLIC_BASE_URL}/portal/{token}"
    emailed = False
    if body.get("send"):
        try:
            name = (rec or {}).get("name") or ""
            _send_full_report_email(
                email, name, "Your Healing Oasis portal link",
                f"Aloha {name or ''},\n\nHere's your personal Healing Oasis portal:\n\n{url}\n\n"
                f"With aloha,\nDr. Glen & Rae")
            emailed = True
        except Exception as e:
            print(f"[reissue-link] send failed: {e!r}", flush=True)
    return jsonify({"ok": True, "url": url, "emailed": emailed})


@app.route("/api/admin/notify-state", methods=["POST"])
def api_admin_notify_state():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_portal as _cp, notify_state as _ns
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        if body.get("phone"):
            _ns.set_phone(cx, email, body["phone"])
        token = _cp.ensure_token(cx, email, body.get("name") or "")
        d = _ns.decide(_ns.get_state(cx, email))
    return jsonify({**d, "url": f"{PUBLIC_BASE_URL}/portal/{token}",
                    "unsubscribe": f"{PUBLIC_BASE_URL}/unsubscribe?token={token}"})


@app.route("/api/admin/notify-sent", methods=["POST"])
def api_admin_notify_sent():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import notify_state as _ns
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _ns.incr_notify(cx, email)
    return jsonify({"ok": True})


# ── Certification work-product portal ───────────────────────────────────────
# Cert students submit published work here; Glen reviews behind two gates
# (approve -> publish). Student-facing surface gates behind CERT_PORTAL_ENABLED.
# The auth_tokens table is created at startup by _init_auth_tables().


def _cert_portal_enabled() -> bool:
    return os.environ.get("CERT_PORTAL_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _cert_data_dir():
    """Runtime data root (same root LOG_DB uses). Uploaded files live here,
    never under the static/web-served dir."""
    return Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))


def _cert_email_from_cookie():
    return (request.cookies.get("rm_cert_email", "") or "").strip().lower()


@app.route("/cert")
def cert_portal_page():
    if not _cert_portal_enabled():
        return ("Not found", 404)
    resp = send_from_directory(STATIC, "cert-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/cert/login", methods=["POST"])
def cert_login():
    """Email a magic link to the cert portal. Always 200 (no enumeration)."""
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if email and "@" in email:
        token = secrets.token_urlsafe(32)
        now = _now_utc()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                "VALUES (?,?,?,?,?)",
                (_hash_token(token), email, "cert_portal", now.isoformat(),
                 (now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)).isoformat()))
            cx.commit()
        try:
            send_magic_link_email(email, "", f"{PUBLIC_BASE_URL}/cert/auth/{token}")
        except Exception as e:
            print(f"[cert] magic link send failed: {e!r}", flush=True)
    return jsonify({"ok": True})


@app.route("/cert/auth/<token>", methods=["GET"])
def cert_auth(token):
    """Validate the cert magic link, set rm_cert_email cookie, -> /cert."""
    from flask import redirect as _redirect
    if not _cert_portal_enabled():
        return ("Not found", 404)
    th = _hash_token((token or "").strip())
    email = None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='cert_portal'", (th,)).fetchone()
        if row and not row["consumed_at"]:
            try:
                if datetime.fromisoformat(row["expires_at"]) >= _now_utc():
                    email = row["email"]
            except Exception:
                email = None
        if email:
            cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                       (_now_utc().isoformat(), th))
            cx.commit()
    if not email:
        return ("<p style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
                "This certification portal link is invalid or has expired. "
                "Please request a new one.</p>"), 400
    resp = _redirect("/cert", code=302)
    resp.set_cookie("rm_cert_email", email, max_age=60 * 60 * 24 * 30,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/api/cert/submit", methods=["POST"])
def api_cert_submit():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    url = (body.get("url") or "").strip()
    file_token = (body.get("file_token") or "").strip()
    formats = [str(x) for x in (body.get("formats") or [])]
    modules = []
    for m in (body.get("modules") or []):
        try:
            modules.append(int(m))
        except (TypeError, ValueError):
            pass
    permission = bool(body.get("permission"))

    if not title:
        return jsonify({"ok": False, "error": "title required"}), 400
    if not permission:
        return jsonify({"ok": False, "error":
                        "permission to publish is required to submit"}), 400
    if not modules:
        return jsonify({"ok": False, "error":
                        "select at least one module topic"}), 400
    if url and not url.lower().startswith(("http://", "https://")):
        return jsonify({"ok": False, "error":
                        "link must start with http:// or https://"}), 400
    # resolve a previously-uploaded file (path-guarded inside the cert dir)
    file_path = ""
    if file_token:
        cand = (_cert_data_dir() / "cert-files" / file_token).resolve()
        root = (_cert_data_dir() / "cert-files").resolve()
        if str(cand).startswith(str(root)) and cand.is_file():
            file_path = str(cand)
    if not url and not file_path:
        return jsonify({"ok": False, "error":
                        "provide a link to the published work or upload a file"}), 400

    from dashboard import practitioner_portal as _pp, cert_submissions as _cs
    try:
        pid = _pp.id_for_email(email) or ""
    except Exception:
        pid = ""
    sid = uuid.uuid4().hex
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        _cs.create(
            cx, sid=sid, email=email, practitioner_id=pid, title=title,
            description=(body.get("description") or "").strip(), url=url,
            file_path=file_path, formats=formats,
            format_other=(body.get("format_other") or "").strip(),
            modules=modules, module_other=(body.get("module_other") or "").strip(),
            topic_angle=(body.get("topic_angle") or "").strip(),
            permission=permission)
        sub = _cs.get(cx, sid)
    # best-effort notifications (never block)
    try:
        _send_inquiry_email(email, "We received your certification submission",
                            f"Thanks — '{title}' is in the review queue.")
        notify = os.environ.get("CERT_NOTIFY_EMAIL", "drglenswartwout@gmail.com")
        _send_inquiry_email(notify, "New certification submission",
                            f"{email} submitted '{title}'. Review in /console/cert.")
    except Exception as e:
        print(f"[cert] submit notify failed: {e!r}", flush=True)
    return jsonify({"ok": True, "submission": sub})


@app.route("/api/cert/upload", methods=["POST"])
def api_cert_upload():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    f = request.files.get("file")
    if not f or not (f.filename or "").strip():
        return jsonify({"ok": False, "error": "file required"}), 400
    ctype = (f.mimetype or "").lower()
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
           "image/webp": "webp", "image/gif": "gif",
           "application/pdf": "pdf"}.get(ctype)
    if not ext:
        return jsonify({"ok": False, "error": "image or PDF only"}), 400
    blob = f.read()
    if len(blob) > 10 * 1024 * 1024:
        return jsonify({"ok": False, "error": "file too large (max 10MB)"}), 400
    cert_dir = _cert_data_dir() / "cert-files"
    cert_dir.mkdir(parents=True, exist_ok=True)
    # Unique per upload so a re-upload never silently overwrites a prior file.
    token = (f"{hashlib.sha256((email + f.filename).encode()).hexdigest()[:16]}"
             f"-{uuid.uuid4().hex[:8]}.{ext}")
    (cert_dir / token).write_bytes(blob)
    return jsonify({"ok": True, "file_token": token})


@app.route("/api/cert/mine", methods=["GET"])
def api_cert_mine():
    if not _cert_portal_enabled():
        return jsonify({"ok": False, "error": "not available"}), 404
    email = _cert_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        subs = _cs.list_for_email(cx, email)
    counted = [s for s in subs if s["status"] in ("approved", "published")]
    prog = _cr.evaluate(counted)
    prog["modules_covered"] = sorted(prog["modules_covered"])  # JSON-safe
    return jsonify({"ok": True, "submissions": subs, "progress": prog,
                    "modules": _cr.MODULES, "formats": _cr.FORMATS})


def _cert_console_ok():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return False
    return True


def _cert_sync_modules(cx, email):
    """Recompute covered modules from this email's approved+published submissions
    and sync modules_completed on the practitioner record. Best-effort."""
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    subs = [s for s in _cs.list_for_email(cx, email)
            if s["status"] in ("approved", "published")]
    covered = _cr.evaluate(subs)["modules_covered"]
    try:
        from dashboard import practitioner_portal as _pp
        _pp.upsert_cert_student(email, modules_completed=len(covered))
    except Exception as e:
        print(f"[cert] modules sync failed for {email}: {e!r}", flush=True)
    return len(covered)


def _cert_publish_to_proof(sub, name):
    """Embed + upsert one vector into the case-studies namespace. Returns the id."""
    title = sub.get("title") or "Practitioner case"
    desc = (sub.get("description") or "").strip()
    text = (title + ". " + desc).strip()
    vec = embed(text)
    vid = "cert-" + sub["id"]
    cond = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or ("cert-" + sub["id"][:8])
    excerpt = desc[:240]
    _idx.upsert(vectors=[{
        "id": vid,
        "values": vec,
        "metadata": {
            "condition": cond,
            "text": excerpt or title,
            "title": title,
            "url": sub.get("url") or "",
            "source": "cert-submission",
            "name": name or "",
        },
    }], namespace="case-studies")
    return vid


@app.route("/api/cert/review/list", methods=["GET"])
def api_cert_review_list():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    from dashboard import cert_submissions as _cs, cert_rules as _cr
    status = (request.args.get("status") or "").strip() or None
    email = (request.args.get("email") or "").strip() or None
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        if email:
            subs = _cs.list_for_email(cx, email)
        else:
            subs = _cs.list_by_status(cx, status)
        # per-student rollups
        emails = sorted({s["email"] for s in subs})
        rollups = {}
        for em in emails:
            counted = [s for s in _cs.list_for_email(cx, em)
                       if s["status"] in ("approved", "published")]
            p = _cr.evaluate(counted)
            p["modules_covered"] = sorted(p["modules_covered"])
            rollups[em] = p
    return jsonify({"ok": True, "submissions": subs, "rollups": rollups,
                    "modules": _cr.MODULES, "formats": _cr.FORMATS})


@app.route("/api/cert/review/approve", methods=["POST"])
def api_cert_review_approve():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    credited = [int(m) for m in (body.get("credited_modules") or [])
                if str(m).strip().isdigit()]
    note = (body.get("note") or "").strip() or None
    from dashboard import cert_submissions as _cs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        sub = _cs.get(cx, sid)
        if not sub:
            return jsonify({"ok": False, "error": "not found"}), 404
        if not credited:
            credited = list(sub.get("modules") or [])
        _cs.set_status(cx, sid, "approved", credited_modules=credited,
                       review_note=note)
        covered = _cert_sync_modules(cx, sub["email"])
    try:
        _send_inquiry_email(sub["email"], "Your certification submission was approved",
                            f"'{sub.get('title')}' is approved. Modules covered so far: {covered}/12.")
    except Exception as e:
        print(f"[cert] approve notify failed: {e!r}", flush=True)
    return jsonify({"ok": True, "id": sid, "modules_covered": covered})


@app.route("/api/cert/review/return", methods=["POST"])
def api_cert_review_return():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    note = (body.get("note") or "").strip() or "Please revise and resubmit."
    from dashboard import cert_submissions as _cs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        if not _cs.get(cx, sid):
            return jsonify({"ok": False, "error": "not found"}), 404
        _cs.set_status(cx, sid, "returned", review_note=note)
    return jsonify({"ok": True, "id": sid})


@app.route("/api/cert/review/publish", methods=["POST"])
def api_cert_review_publish():
    if not _cert_console_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("id") or "").strip()
    from dashboard import cert_submissions as _cs
    from dashboard import practitioner_portal as _pp
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_tables(cx)
        sub = _cs.get(cx, sid)
        if not sub:
            return jsonify({"ok": False, "error": "not found"}), 404
        if sub["status"] not in ("approved", "published"):
            return jsonify({"ok": False, "error": "approve before publishing"}), 400
        if not sub.get("permission"):
            return jsonify({"ok": False, "error": "no publish permission on file"}), 400
        try:
            name = _pp.name_for_email(sub["email"])
        except Exception:
            name = ""
        try:
            vid = _cert_publish_to_proof(sub, name)
        except Exception as e:
            print(f"[cert] publish upsert failed: {e!r}", flush=True)
            return jsonify({"ok": False, "error": "publish failed; retry"}), 502
        _cs.set_status(cx, sid, "published", case_study_id=vid)
    return jsonify({"ok": True, "id": sid, "case_study_id": vid})


# ── Biofield readiness gate ─────────────────────────────────────────────────
# After paying for the $300 Causal Biofield Analysis, the customer lands on a
# readiness gate. It must show 3 items (photo on file / intake / fresh voice
# scan) and unlock a booking link only when all are satisfied. PHI: client
# photos are stored under DATA_DIR (NOT static / never web-served).

def _biofield_data_dir():
    """Runtime data root (same root LOG_DB uses). PHI photos live here, never
    under the static/web-served dir."""
    return Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))


def _biofield_email():
    """Resolve the gate's email from the biofield magic-link cookie, falling
    back to an authenticated member session if one exists."""
    em = (request.cookies.get("rm_biofield_email", "") or "").strip().lower()
    if em:
        return em
    user = get_authenticated_user(request)
    if user and user.get("email"):
        return user["email"].strip().lower()
    return ""


def _has_paid_biofield(email):
    """True when this email has a PAID Biofield Analysis on record — the group-bundle
    qualifying gate. The free E4L voice scan does NOT qualify; only a paid Biofield-designed
    program (a biofield_readiness payment: our checkout, a confirmed PB receipt, or a cert
    comp). Forward-looking: existing pre-gate Biofield clients qualify once they have a
    recorded paid Biofield. Best-effort, never raises."""
    if not email:
        return False
    try:
        from dashboard import biofield_store as _bf
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _bf.init_table(cx)
            row = cx.execute(
                "SELECT paid_at FROM biofield_readiness WHERE lower(email)=lower(?)",
                (str(email).strip(),)).fetchone()
            return bool(row and row["paid_at"])
    except Exception:
        return False


def _biofield_has_intake(email):
    """True when an inbound_leads intake row exists for this email (mirrors the
    intake query in the client-profile builder)."""
    if not email:
        return False
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT 1 FROM inbound_leads "
                "WHERE email=? AND source IN ('scoreapp','practice-better','concierge') "
                "LIMIT 1", (email,)).fetchone()
            return bool(row)
    except Exception:
        return False


def _biofield_has_fresh_scan(email):
    """True when the server-side scan-freshness index (pushed by the local e4l
    ingestion) has a scan within the gate window for this email."""
    try:
        from dashboard import scan_freshness as _sf
        import datetime as _dt
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            _sf.init_table(cx)
            return _sf.is_fresh(cx, email, today=_dt.date.today().isoformat(),
                                window_days=7)
    except Exception:
        return False


@app.route("/api/e4l/scan-freshness", methods=["POST"])
def api_e4l_scan_freshness():
    """Ingest the latest voice-scan dates from the local e4l pipeline so the
    Biofield gate can auto-verify a fresh scan. Auth: X-Cron-Secret (== CRON_SECRET,
    falls back to CONSOLE_SECRET). Body: {"rows":[{email,last_scan_date}, ...]}."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    rows = ((request.get_json(silent=True) or {}).get("rows") or [])[:5000]
    from dashboard import scan_freshness as _sf
    with sqlite3.connect(LOG_DB) as cx:
        _sf.init_table(cx)
        _sf.upsert(cx, rows)
    return jsonify({"ok": True, "upserted": len(rows)})


@app.route("/biofield/ready")
def biofield_ready_page():
    """Serve the readiness-gate page (no-store; PHI-adjacent)."""
    resp = send_from_directory(STATIC, "biofield-ready.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/biofield/request", methods=["POST"])
def biofield_request():
    """Email a magic link to start a biofield readiness session. Always 200
    (no email enumeration leak). Mirrors /reorder/request."""
    email = ((request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    if email and "@" in email:
        token = secrets.token_urlsafe(32)
        now = _now_utc()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                "VALUES (?,?,?,?,?)",
                (_hash_token(token), email, "biofield", now.isoformat(),
                 (now + timedelta(minutes=AUTH_TOKEN_TTL_MIN)).isoformat()))
            cx.commit()
        try:
            send_magic_link_email(email, "",
                                  f"{PUBLIC_BASE_URL}/biofield/auth/{token}")
        except Exception as e:
            print(f"[biofield] magic link send failed: {e!r}", flush=True)
    return jsonify({"ok": True})


@app.route("/biofield/auth/<token>", methods=["GET"])
def biofield_auth(token):
    """Validate the biofield magic link, set the rm_biofield_email cookie,
    redirect to /biofield/ready. Mirrors /reorder/auth/<token>."""
    from flask import redirect as _redirect
    th = _hash_token((token or "").strip())
    email = None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='biofield'", (th,)).fetchone()
        if row and not row["consumed_at"]:
            try:
                if datetime.fromisoformat(row["expires_at"]) >= _now_utc():
                    email = row["email"]
            except Exception:
                email = None
        if email:
            cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                       (_now_utc().isoformat(), th))
            cx.commit()
    if not email:
        return ("<p style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
                "This sign-in link is invalid or has expired. Please request a new one.</p>"), 400
    resp = _redirect("/biofield/ready", code=302)
    resp.set_cookie("rm_biofield_email", email, max_age=60 * 60 * 24 * 30,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/api/biofield/ready", methods=["GET"])
def api_biofield_ready():
    if not _biofield_enabled():
        return jsonify({"error": "not available"}), 404
    email = _biofield_email()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    from dashboard import biofield_store as _bf, biofield_gate as _gate
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _bf.init_table(cx)
        st = _gate.gate_state(cx, email, has_intake=_biofield_has_intake,
                              has_fresh_scan=_biofield_has_fresh_scan)
    if st.get("booking_unlocked"):
        st["booking_url"] = os.environ.get("BIOFIELD_BOOKING_URL", "")
    return jsonify(st)


@app.route("/api/biofield/photo", methods=["POST"])
def api_biofield_photo():
    """Accept a client photo (PHI). Stored to a PRIVATE path under DATA_DIR,
    never under the static/web-served dir."""
    if not _biofield_enabled():
        return jsonify({"error": "not available"}), 404
    email = _biofield_email()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    f = request.files.get("file")
    if not f or not (f.filename or "").strip():
        return jsonify({"ok": False, "error": "file required"}), 400
    ctype = (f.mimetype or "").lower()
    if not ctype.startswith("image/"):
        return jsonify({"ok": False, "error": "image file required"}), 400
    blob = f.read()
    if len(blob) > 10 * 1024 * 1024:
        return jsonify({"ok": False, "error": "file too large"}), 400
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
           "image/webp": "webp", "image/gif": "gif", "image/heic": "heic"}.get(
        ctype, "img")
    photo_dir = _biofield_data_dir() / "biofield-photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()
    path = photo_dir / f"{digest}.{ext}"
    path.write_bytes(blob)
    from dashboard import biofield_store as _bf
    with sqlite3.connect(LOG_DB) as cx:
        _bf.init_table(cx)
        _bf.set_photo_on_file(cx, email, str(path))
    return jsonify({"ok": True})


@app.route("/admin/biofield/photo", methods=["GET"])
def admin_biofield_photo():
    """Console-gated PHI photo viewer. Returns the raw image bytes for the team
    to review. Hard path guard: only ever serves a file that resolves INSIDE the
    private biofield-photos dir (never a symlink/abs path pointing elsewhere)."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import biofield_store as _bf
    email = (request.args.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _bf.init_table(cx)
        row = _bf.get(cx, email)
    path_str = (row or {}).get("photo_path")
    if not path_str:
        return jsonify({"error": "no photo on file"}), 404
    photo_dir = (_biofield_data_dir() / "biofield-photos").resolve()
    fp = Path(path_str).resolve()
    if photo_dir not in fp.parents or not fp.is_file():
        return jsonify({"error": "no photo on file"}), 404
    mt = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
    return Response(fp.read_bytes(), mimetype=mt)


@app.route("/admin/biofield/photo/delete", methods=["POST"])
def admin_biofield_photo_delete():
    """Console-gated ON-REQUEST photo deletion (only when a patient asks). Removes the
    file AND clears the DB flag, path-guarded to the private photo dir. Photos are
    otherwise kept on file indefinitely (a face photo is reused across Biofield cycles);
    there is no time-based purge."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import biofield_store as _bf
    email = (request.args.get("email")
             or (request.get_json(silent=True) or {}).get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    deleted_file = False
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _bf.init_table(cx)
        row = _bf.get(cx, email)
        path_str = (row or {}).get("photo_path")
        if path_str:
            photo_dir = (_biofield_data_dir() / "biofield-photos").resolve()
            fp = Path(path_str).resolve()
            if photo_dir in fp.parents and fp.is_file():
                try:
                    fp.unlink()
                    deleted_file = True
                except OSError:
                    pass
        _bf.clear_photo(cx, email)
    return jsonify({"ok": True, "deleted_file": deleted_file})


@app.route("/api/biofield/confirm", methods=["POST"])
def api_biofield_confirm():
    if not _biofield_enabled():
        return jsonify({"error": "not available"}), 404
    email = _biofield_email()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    data = request.get_json(silent=True) or {}
    item = (data.get("item") or "").strip().lower()
    from dashboard import biofield_store as _bf, biofield_gate as _gate
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _bf.init_table(cx)
        if item == "scan":
            _bf.set_scan_confirmed(cx, email, True)
        elif item == "intake":
            _bf.set_intake_confirmed(cx, email, True)
        elif item == "payment":
            ref = (data.get("receipt") or "").strip() or "pb-selfattest"
            _bf.seed_paid(cx, email, via="pb", order_ref=ref)
        else:
            return jsonify({"ok": False, "error": "unknown item"}), 400
        st = _gate.gate_state(cx, email, has_intake=_biofield_has_intake,
                              has_fresh_scan=_biofield_has_fresh_scan)
    if st.get("booking_unlocked"):
        st["booking_url"] = os.environ.get("BIOFIELD_BOOKING_URL", "")
    return jsonify(st)


@app.route("/api/biofield/book", methods=["POST"])
def api_biofield_book():
    """Once the gate is unlocked, mark the session booked and drop a 48-hour
    prep task for the team, returning the booking URL. Idempotent: the todo's
    dedup_key makes a repeat call a no-op."""
    if not _biofield_enabled():
        return jsonify({"error": "not available"}), 404
    email = _biofield_email()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    from dashboard import biofield_store as _bf, biofield_gate as _gate
    _init_todos_table()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _bf.init_table(cx)
        st = _gate.gate_state(cx, email, has_intake=_biofield_has_intake,
                              has_fresh_scan=_biofield_has_fresh_scan)
        if not st.get("booking_unlocked"):
            return jsonify(st), 409
        _bf.set_booked(cx, email)
        row = _bf.get(cx, email)
        order_ref = (row["order_ref"] if row and row["order_ref"] else "") or email
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc).isoformat()
        cx.execute(
            """INSERT INTO todos (created_at, owner, category, title, body, priority, source, dedup_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dedup_key) DO NOTHING""",
            (now, "glen", "biofield",
             f"Biofield prep due 48h — {email}",
             (f"Biofield analysis booked for {email} (order_ref {order_ref}). "
              "Prep due within 48 hours. On file and confirmed: payment, photo, "
              "intake, scan. Run the biofield analysis, design the program, and "
              "prepare the client report ahead of the session."),
             "high", "biofield", f"biofield-prep-{email}-{order_ref}"))
        cx.commit()
    return jsonify({"ok": True,
                    "booking_url": os.environ.get("BIOFIELD_BOOKING_URL", "")})


# ── Certification bonus Biofields (ships dark behind CERT_BONUS_ENABLED) ───────

def _cert_bonus_enabled() -> bool:
    return os.environ.get("CERT_BONUS_ENABLED", "").strip().lower() \
        in ("1", "true", "yes", "on")


@app.route("/api/cert/commitment", methods=["POST"])
def api_cert_commitment():
    """Admin: set or clear a certification commitment. Console-key gated.

    Body: {email, kind, started_at, clear?}. Returns the commitment dict."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    from dashboard import cert_bonus
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        cert_bonus.init_tables(cx)
        if body.get("clear"):
            cert_bonus.clear_commitment(cx, email)
        else:
            cert_bonus.set_commitment(
                cx, email,
                kind=(body.get("kind") or "").strip(),
                started_at=(body.get("started_at") or "").strip(),
            )
        commitment = cert_bonus.get_commitment(cx, email)
    return jsonify({"ok": True, "commitment": commitment})


@app.route("/api/cert/student", methods=["POST"])
def api_cert_student():
    """Console-gated: create/update a certification student's practitioner record
    (portal_role coach + modules_completed 0-12). Feeds the cert bonus + cert referral."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    name = (body.get("name") or "").strip()
    try:
        modules = int(body.get("modules_completed", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "modules_completed must be an integer"}), 400
    from dashboard import practitioner_portal as _pp
    pid, mc = _pp.upsert_cert_student(email, name=name, modules_completed=modules)
    return jsonify({"ok": True, "practitioner_id": pid, "email": email,
                    "modules_completed": mc})


@app.route("/api/cert/show-contact", methods=["POST"])
def api_cert_show_contact():
    """Console-gated: set a practitioner's public contact visibility in the finder
    (show_contact). Default is hidden; flip true to publish their email/phone."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    show = bool(body.get("show", False))
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("UPDATE practitioners SET show_contact=%s, updated_at=now() "
                        "WHERE lower(email)=lower(%s)", (show, email))
            updated = cur.rowcount
    except Exception as e:
        return jsonify({"error": f"update failed: {e}"}), 500
    return jsonify({"ok": True, "email": email, "show_contact": show, "updated": updated})


@app.route("/api/cert/portal-invite", methods=["POST"])
def api_cert_portal_invite():
    """Console-gated: email a cert participant their practitioner-portal magic link."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    pid = _pp.id_for_email(email)
    if not pid:
        return jsonify({"ok": True, "sent": False, "reason": "no practitioner record"})
    try:
        name = _pp.name_for_email(email) or ""
        # Invite link is emailed (not clicked within 15 min), so give it a 7-day TTL.
        magic = _pp.create_magic_link_token(pid, email, ttl_min=7 * 24 * 60)
        _send_practitioner_magic_link(
            email, name, f"{PUBLIC_BASE_URL}/practitioner/login-verify?token={magic}")
    except Exception as e:
        print(f"[cert-portal-invite] failed for {email}: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "send failed"}), 500
    return jsonify({"ok": True, "sent": True, "email": email})


def _run_biofield_bonuses(dry_run=False):
    """Sweep active certification commitments and grant due bonus Biofields concierge-style
    (one todos task + idempotent ledger row per grant). Flag-gated (CERT_BONUS_ENABLED) — a
    no-op when off. Returns {ok, granted, ...}. Called by both the cron route and the daily
    in-process scheduler job."""
    if not _cert_bonus_enabled():
        return {"ok": True, "granted": 0, "message": "disabled"}

    from dashboard import cert_bonus
    _init_todos_table()
    from datetime import date as _date, datetime as _dt, timezone as _tz
    today = _date.today().isoformat()

    granted_total = 0
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        cert_bonus.init_tables(cx)
        for c in cert_bonus.list_active(cx):
            email = c["email"]
            try:
                modules = _pp.modules_completed_for_email(email) or 0
            except Exception:
                modules = 0
            due = cert_bonus.due_bonuses(
                started_at=c["started_at"],
                modules_completed=modules,
                granted=cert_bonus.granted_pairs(cx, email),
                today=today,
            )
            for kind, idx in due:
                if dry_run:
                    granted_total += 1
                    continue
                now = _dt.now(_tz.utc).isoformat()
                cur = cx.execute(
                    """INSERT INTO todos (created_at, owner, category, title, body,
                                          priority, source, dedup_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(dedup_key) DO NOTHING""",
                    (now, "glen", "biofield-bonus",
                     f"Biofield bonus due — {email} ({kind} {idx})",
                     (f"Certification bonus Biofield earned by {email} "
                      f"({kind} {idx}). Schedule and run the Biofield analysis "
                      "for the client as part of their certification benefit."),
                     "normal", "cert-bonus",
                     f"cert-bonus-{email}-{kind}-{idx}"))
                todo_id = cur.lastrowid
                cert_bonus.record_grant(cx, email, kind=kind, idx=idx, todo_id=todo_id)
                granted_total += 1
        cx.commit()
    return {"ok": True, "granted": granted_total, "dry_run": bool(dry_run)}


@app.route("/api/cron/biofield-bonuses", methods=["POST"])
def cron_biofield_bonuses():
    """Cron sweep for certification bonus Biofields (also runs daily in-process via the
    scheduler). Auth: X-Cron-Secret (== CRON_SECRET, falls back to CONSOLE_SECRET).
    Monthly (1..12, one per elapsed month) + one per completed module; idempotent ledger
    + todos task per grant. ?dry_run=1 counts only. Flag-gated by CERT_BONUS_ENABLED."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    return jsonify(_run_biofield_bonuses(dry_run=dry_run))


@app.route("/api/reorder/items", methods=["GET"])
def api_reorder_items():
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    scope = (request.args.get("scope") or "last").strip().lower()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        orders = _bos_orders.list_orders_by_email(cx, email)
    src = orders[:1] if scope == "last" else orders
    seen, items = set(), []
    for o in src:
        for it in (o.get("items") or []):
            nm = (it.get("name") or "").strip()
            if not nm or nm.lower() in seen:
                continue
            seen.add(nm.lower())
            slug = _resolve_buy_slug(nm)
            p = _get_product(slug) if slug else None
            items.append({
                "name": nm,
                "slug": (p.get("slug") if p else None),
                "current_price_cents": (p.get("price_cents") if p else None),
                "current_price": (f"${p['price_cents'] / 100:.2f}" if p else None),
                "qty": int(it.get("qty") or 1),
                "last_ordered": (o.get("created_at") or "")[:10],
                "available": bool(p),
            })
    return jsonify({"email": email, "scope": scope, "items": items})


@app.route("/api/points/balance", methods=["GET"])
def api_points_balance():
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    from dashboard import points as _points
    with sqlite3.connect(LOG_DB) as cx:
        _points.init_points_table(cx)
        bal = _points.balance(cx, email)
    return jsonify({"balance_cents": bal, "balance_dollars": f"{bal/100:.2f}"})


@app.route("/reorder/checkout", methods=["POST"])
def reorder_checkout():
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    body = request.get_json(silent=True) or []
    # body may be a list (legacy) or a dict {items, address}
    if isinstance(body, dict):
        cart = body.get("items", [])
        body_address = body.get("address") or {}
    else:
        cart = body
        body_address = {}

    # ── Pricing-engine path (feature flag) ────────────────────────────────────
    if os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            # Prefer address from request body; fall back to last order on file
            ship = body_address or {}
            if not ship:
                with sqlite3.connect(LOG_DB) as cx:
                    cx.row_factory = sqlite3.Row
                    prior = _bos_orders.list_orders_by_email(cx, email, limit=1)
                if prior:
                    ship = prior[0].get("address") or {}
            # Cap requested redemption to the caller's current balance
            requested_redeem = int((body.get("points_to_redeem_cents") if isinstance(body, dict) else 0) or 0)
            if requested_redeem > 0:
                from dashboard import points as _pts_co
                with sqlite3.connect(LOG_DB) as _cx_pts:
                    _pts_co.init_points_table(_cx_pts)
                    _bal = _pts_co.balance(_cx_pts, email)
                requested_redeem = min(requested_redeem, _bal)
            try:
                pc = _price_cart(cart, ship=ship, coupon_pct=_active_coupon_pct(),
                                 points_to_redeem_cents=requested_redeem)
            except CheckoutError as e:
                return jsonify({"ok": False, "error": str(e)}), 400
            if not pc["qbo_lines"]:
                return jsonify({"ok": False,
                                "error": "Your cart is empty or those items are no longer available."}), 400
            cust = qb.find_or_create_customer(email, ship.get("name", ""))
            inv = qb.create_invoice(
                cust,
                pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
                allow_online_pay=True,
                email_to=email,
                discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
            _ingest_order(source="reorder", external_ref=inv.get("Id"), email=email,
                          name=ship.get("name", ""), items=pc["items_rec"],
                          total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                          address=ship, channel="retail",
                          get_cents=pc["priced"].get("get_cents", 0),
                          discount_cents=pc["discount_cents"],
                          points_redeemed_cents=pc["points_redeemed_cents"],
                          shipping_cents=pc["shipping_cents"])
            out = {"invoice_id": inv.get("Id"), "doc_number": inv.get("DocNumber"),
                   "customer_id": cust.get("Id"),   # needed by /begin/checkout-return to record the QBO payment
                   "total": inv.get("TotalAmt")}
            stripe_url = _stripe_checkout_url_for_reorder(out, email) if _STRIPE_ACTIVE else ""
            _pe = {"payment_error": _CARD_UNAVAILABLE} if (_STRIPE_ACTIVE and not stripe_url) else {}
            return jsonify({"ok": True, "stripe_url": stripe_url, **out, **_pe})
        except Exception as e:
            app.logger.exception("reorder checkout (engine) failed")
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
    # ── Legacy path (existing code, unchanged) ────────────────────────────────
    lines, items_rec, subtotal_cents = [], [], 0
    for c in cart:
        slug = (c.get("slug") or "").strip()
        try:
            qty = max(1, min(int(c.get("qty", 1) or 1), 99))
        except Exception:
            qty = 1
        p = _get_product(slug) if slug else None
        if not p:
            continue
        unit_cents = int(_qty_unit_cents(p, qty))
        subtotal_cents += unit_cents * qty
        lines.append({"name": p["name"], "amount": round(unit_cents / 100.0, 2),
                      "qty": qty, "item_id": p.get("qbo_item_id"), "description": p["name"]})
        items_rec.append({"name": p["name"], "qty": qty, "desc": p["name"]})
    if not lines:
        return jsonify({"ok": False,
                        "error": "Your cart is empty or those items are no longer available."}), 400
    try:
        ship = {}
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            prior = _bos_orders.list_orders_by_email(cx, email, limit=1)
        if prior:
            ship = prior[0].get("address") or {}
        from dashboard import qbo_billing as _qb_local
        from dashboard import tax as _tax
        cust = _qb_local.find_or_create_customer(email, ship.get("name", ""))
        get_cents = _tax.compute_get_cents(subtotal_cents, channel="retail",
                                           ship_to_state=ship.get("state", ""))
        inv = _qb_local.create_invoice(cust, lines, allow_online_pay=True, email_to=email)
        out = {"invoice_id": inv.get("Id"), "customer_id": cust.get("Id"),
               "doc_number": inv.get("DocNumber"), "total": inv.get("TotalAmt")}
        _ingest_order(source="reorder", external_ref=inv.get("Id"), email=email,
                      name=ship.get("name", ""), items=items_rec,
                      total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                      address=ship, channel="retail", get_cents=get_cents)
        stripe_url = _stripe_checkout_url_for_reorder(out, email) if _STRIPE_ACTIVE else ""
        _pe = {"payment_error": _CARD_UNAVAILABLE} if (_STRIPE_ACTIVE and not stripe_url) else {}
        return jsonify({"ok": True, "stripe_url": stripe_url,
                        "invoice_id": out["invoice_id"], "total": out["total"], **_pe})
    except Exception as e:
        app.logger.exception("reorder checkout failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ── Subscription setup flow ────────────────────────────────────────────────────

def _subscriptions_enabled() -> bool:
    """Return True if SUBSCRIPTIONS_ENABLED is set to a truthy value."""
    return os.environ.get("SUBSCRIPTIONS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


@app.route("/reorder/subscribe", methods=["POST"])
def reorder_subscribe():
    """POST /reorder/subscribe — price the first subscription order, create a QBO invoice,
    then create a Stripe checkout session with save_card=True so the card is vaulted.
    On return, /begin/checkout-return writes the subscription row."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    if not _subscriptions_enabled():
        return jsonify({"error": "subscriptions not enabled"}), 400
    if not _STRIPE_ACTIVE:
        # Guard BEFORE creating the QBO invoice, else a Stripe outage leaves a dangling invoice.
        return jsonify({"error": "card payment not active"}), 400

    body = request.get_json(silent=True) or {}
    cadence = body.get("cadence_months")
    try:
        cadence = int(cadence)
        if cadence not in (1, 2, 3):
            raise ValueError("cadence out of range")
    except (TypeError, ValueError):
        return jsonify({"error": "cadence_months must be 1, 2, or 3"}), 400

    cart = body.get("items") or []
    ship = body.get("address") or {}

    # ── Pricing-engine path (required for subscribe) ──────────────────────────
    if not os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1", "true", "yes", "on"):
        return jsonify({"error": "PRICING_ENGINE_CHECKOUT must be enabled for subscriptions"}), 400

    try:
        from dashboard import subscriptions as _subs
        try:
            pc = _price_cart(cart, ship=ship, subscriber_tier_pct=_subs.tier_for(0))
        except CheckoutError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not pc["qbo_lines"]:
            return jsonify({"ok": False,
                            "error": "Your cart is empty or those items are no longer available."}), 400

        cust = qb.find_or_create_customer(email, ship.get("name", ""))
        inv = qb.create_invoice(
            cust,
            pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
            allow_online_pay=True,
            email_to=email,
            discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])

        # Build metadata; stash items+ship in pending_subscriptions if too long
        items_json = json.dumps(cart)
        ship_json = json.dumps(ship)
        metadata = {
            "kind": "subscribe",
            "cadence_months": str(cadence),
            "email": email,
            "invoice_id": inv.get("Id") or "",
            "customer_id": cust.get("Id") or "",
        }
        if len(items_json) + len(ship_json) <= 450:
            metadata["items"] = items_json
            metadata["ship"] = ship_json
        else:
            # Stash in DB and put the key in metadata
            stash_key = uuid.uuid4().hex
            with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                _cx.execute(
                    "CREATE TABLE IF NOT EXISTS pending_subscriptions "
                    "(key TEXT PRIMARY KEY, items_json TEXT, ship_json TEXT, created_at TEXT)")
                _cx.execute(
                    "INSERT INTO pending_subscriptions (key, items_json, ship_json, created_at) "
                    "VALUES (?,?,?,?)",
                    (stash_key, items_json, ship_json, _now_utc().isoformat()))
                _cx.commit()
            metadata["stash_key"] = stash_key

        total_cents = int(round(float(inv.get("TotalAmt") or 0) * 100))
        success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}")
        sess = stripe_pay.create_checkout_session(
            total_cents,
            customer_email=email,
            description=f"Remedy Match subscription setup #{inv.get('DocNumber') or ''}",
            metadata=metadata,
            success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/reorder",
            save_card=True)

        return jsonify({"ok": True, "stripe_url": sess.get("url") or ""})
    except Exception as e:
        app.logger.exception("reorder subscribe failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ── Subscription manage-plan portal ──────────────────────────────────────────

@app.route("/subscription")
def subscription_page():
    """Serve the self-serve subscription management page."""
    return send_from_directory(STATIC, "subscription.html")


def _get_sub_authed(sub_id, cookie_email):
    """Return (sub_dict, error_response) for the given sub_id.
    Verifies the subscription belongs to cookie_email.
    Returns (sub, None) on success; (None, response) on failure.
    """
    from dashboard import subscriptions as _subs
    try:
        sub_id = int(sub_id)
    except (TypeError, ValueError):
        return None, (jsonify({"error": "invalid id"}), 400)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        sub = _subs.get(cx, sub_id)
    if sub is None:
        return None, (jsonify({"error": "not found"}), 404)
    if sub["email"].strip().lower() != cookie_email.strip().lower():
        return None, (jsonify({"error": "forbidden"}), 403)
    return sub, None


@app.route("/api/subscription/details", methods=["GET"])
def api_subscription_details():
    """Return the caller's active subscriptions with tier/status info."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _subs.init_subscriptions_table(cx)
        rows = _subs.get_manageable_by_email(cx, email)   # active + paused, so paused can be resumed
    result = []
    for s in rows:
        result.append({
            "id": s["id"],
            "email": s["email"],
            "status": s["status"],
            "cadence_months": s["cadence_months"],
            "next_charge_date": s["next_charge_date"],
            "current_tier_pct": _subs.tier_for(s["order_count"]),
            "order_count": s["order_count"],
            "items": s.get("items") or [],
            "skip_next": bool(s["skip_next"]),
        })
    return jsonify({"subscriptions": result})


@app.route("/api/subscription/skip", methods=["POST"])
def api_subscription_skip():
    """Skip the next charge cycle for the caller's subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_skip_next(cx, sub_id, True)
    return jsonify({"ok": True})


@app.route("/api/subscription/resume-skip", methods=["POST"])
def api_subscription_resume_skip():
    """Clear the skip-next flag for the caller's subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_skip_next(cx, sub_id, False)
    return jsonify({"ok": True})


@app.route("/api/subscription/pause", methods=["POST"])
def api_subscription_pause():
    """Pause the caller's subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_status(cx, sub_id, "paused")
    return jsonify({"ok": True})


@app.route("/api/subscription/resume", methods=["POST"])
def api_subscription_resume():
    """Resume a paused subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_status(cx, sub_id, "active")
    return jsonify({"ok": True})


@app.route("/api/subscription/cancel", methods=["POST"])
def api_subscription_cancel():
    """Cancel the caller's subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_status(cx, sub_id, "cancelled")
    return jsonify({"ok": True})


@app.route("/api/subscription/cadence", methods=["POST"])
def api_subscription_cadence():
    """Change the cadence (1, 2, or 3 months) for the caller's subscription."""
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    body = request.get_json(silent=True) or {}
    sub_id = body.get("id")
    try:
        cadence = int(body.get("cadence_months") or 0)
        if cadence not in (1, 2, 3):
            raise ValueError("out of range")
    except (TypeError, ValueError):
        return jsonify({"error": "cadence_months must be 1, 2, or 3"}), 400
    sub, err = _get_sub_authed(sub_id, email)
    if err:
        return err
    from dashboard import subscriptions as _subs
    with sqlite3.connect(LOG_DB) as cx:
        _subs.set_cadence(cx, sub_id, cadence)
    return jsonify({"ok": True})


def _stripe_checkout_url_for_order(out, email, session_token):
    """Create a Stripe Checkout Session for a wholesale invoice; returns its URL."""
    try:
        from dashboard import stripe_pay
        import urllib.parse as _up
        total_cents = int(round((out.get("total") or 0) * 100))
        if total_cents <= 0:
            return ""
        success = (f"{PUBLIC_BASE_URL}/practitioner/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}&t={_up.quote(session_token)}")
        sess = stripe_pay.create_checkout_session(
            total_cents, customer_email=email,
            description=f"Remedy Match wholesale order #{out.get('doc_number')}",
            metadata={"invoice_id": out.get("invoice_id"),
                      "customer_id": out.get("customer_id"), "kind": "wholesale"},
            success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/practitioner/portal?token={_up.quote(session_token)}")
        return sess.get("url") or ""
    except Exception as e:
        print(f"[stripe] session create failed: {e!r}", flush=True)
        _alert_stripe("stripe-generic", e)
        return ""


@app.route("/practitioner/checkout-return")
def practitioner_checkout_return():
    """Stripe return: verify the session and record the QBO payment, then back to the portal."""
    from flask import redirect as _redir
    import urllib.parse as _up
    sid = (request.args.get("session_id") or "").strip()
    token = (request.args.get("t") or "").strip()
    paid = "0"
    if sid:
        try:
            from dashboard import stripe_pay
            sess = stripe_pay.get_session(sid)
            if sess.get("payment_status") == "paid":
                paid = "1"
                md = sess.get("metadata") or {}
                inv, cid = md.get("invoice_id"), md.get("customer_id")
                if inv and cid:
                    try:
                        from dashboard import qbo_billing as qb
                        qb.record_payment(cid, int(sess.get("amount_total") or 0), inv)
                    except Exception as e:
                        print(f"[stripe-return] qbo payment failed: {e!r}", flush=True)
                    pi = sess.get("payment_intent")
                    if pi:
                        try:
                            _cxo = _sqlite3.connect(LOG_DB); _cxo.row_factory = _sqlite3.Row
                            try:
                                _o = _bos_orders.find_order_by_external_ref(_cxo, inv)
                                if _o:
                                    _bos_orders.set_order_stripe_pi(_cxo, _o["id"], pi)
                            finally:
                                _cxo.close()
                        except Exception as _e:
                            print(f"[stripe-return] pi capture: {_e!r}", flush=True)
        except Exception as e:
            print(f"[stripe-return] {e!r}", flush=True)
    dest = "/practitioner/portal?paid=" + paid
    if token:
        dest += "&token=" + _up.quote(token)
    return _redir(dest)


@app.route("/api/practitioner/admin/clear-orders", methods=["POST"])
def api_practitioner_admin_clear_orders():
    """Console-gated: clear a practitioner's local wholesale + dispensary order
    history (e.g. to tidy test data). Does not touch QBO or the wallet."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
    pid = ((request.get_json(silent=True) or {}).get("practitioner_id") or "").strip()
    if not pid:
        return jsonify({"ok": False, "error": "practitioner_id required"}), 400
    counts = {}
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            for tbl in ("wholesale_orders", "dispensary_orders", "wholesale_cart"):
                try:
                    counts[tbl] = cx.execute(
                        f"DELETE FROM {tbl} WHERE practitioner_id=?", (pid,)).rowcount
                except Exception:
                    counts[tbl] = 0
            cx.commit()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "deleted": counts})


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


def _capture_concierge_referral(email, first_name, last_name, ref_slug):
    """Concierge entry attribution: when a visitor who arrived via ?ref=<slug>
    (rm_ref cookie) identifies themselves with an email in the chat, log a
    referral_event crediting that approved affiliate. This is what lets a
    concierge-originated journey attribute a later purchase/enrollment via
    _attribute_conversion_by_email. Idempotent per (email, slug)."""
    email    = (email or "").strip().lower()
    ref_slug = (ref_slug or "").strip()
    if not email or not ref_slug:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        if not cx.execute(
            "SELECT 1 FROM affiliate_signups WHERE slug=? AND status='approved'",
            (ref_slug,)
        ).fetchone():
            return  # not a real approved affiliate slug
        if cx.execute(
            "SELECT 1 FROM referral_events WHERE LOWER(email)=? AND utm_source=? "
            "AND utm_medium='concierge' LIMIT 1",
            (email, ref_slug)
        ).fetchone():
            return  # already captured — idempotent
        cx.execute("""
            INSERT INTO referral_events
              (received_at, lead_id, email, first_name, last_name,
               utm_source, utm_medium, utm_campaign, utm_content, utm_term, quiz_score, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, None, email, first_name, last_name,
              ref_slug, "concierge", "concierge-entry", "", "", "", ""))
        cx.commit()


def _stamp_affiliate_journey(session_id, email, first_name, last_name, recruiter_slug):
    """Additive, defensive: record an affiliate signup into the journey engine.
    Stamps paid_fork + path=pay_forward (-> choose_path). Idempotent per session
    (skips if a become_affiliate event already exists). Credits the recruiter if a
    real approved slug is present. NEVER raises into the caller."""
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            begin_funnel.init_journey_tables(cx)
            init_inquiry_tables(cx)
            init_membership_tables(cx)
            cur = cx.execute(
                "SELECT 1 FROM journey_events WHERE session_id=? AND trigger='paid_fork' AND detail='become_affiliate' LIMIT 1",
                (session_id,))
            if cur.fetchone():
                return  # already stamped
            begin_funnel.record_unlock(
                cx, session_id=session_id, trigger="paid_fork", email=email,
                first_name=first_name, last_name=last_name, path="pay_forward",
                detail="become_affiliate", ref_slug=(recruiter_slug or ""))
        if recruiter_slug:
            _capture_concierge_referral(email, first_name, last_name, recruiter_slug)
    except Exception as e:
        print(f"[affiliate-journey] {e!r}", flush=True)


def _attribute_conversion_by_email(email, conversion_type, detail="", order_value=None,
                                   source="", raw_json=""):
    """Credit a conversion (store purchase / course enrollment) to the most-recent
    affiliate who referred this email via a free offer (referral_events.utm_source).
    Records an affiliate_conversions row. Returns the credited slug, or None when the
    email was never referred by an approved affiliate (cold / unattributable)."""
    email = (email or "").strip().lower()
    if not email:
        return None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("""
            SELECT re.utm_source FROM referral_events re
            JOIN affiliate_signups a ON a.slug = re.utm_source AND a.status = 'approved'
            WHERE LOWER(re.email) = ?
            ORDER BY re.received_at DESC LIMIT 1
        """, (email,)).fetchone()
        if not row:
            return None
        cx.execute("""
            INSERT INTO affiliate_conversions
              (received_at, email, affiliate_slug, conversion_type, detail, order_value, source, raw_json)
            VALUES (?,?,?,?,?,?,?,?)
        """, (datetime.now(timezone.utc).isoformat(), email, row[0],
              conversion_type, detail, order_value, source, raw_json))
        cx.commit()
        return row[0]


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
        # Per-item action columns (interactive dashboard) — idempotent migration
        for col, ddl in [("tags",             "TEXT DEFAULT '[]'"),
                         ("status",           "TEXT DEFAULT 'pending'"),
                         ("last_outbound_at", "TEXT DEFAULT ''")]:
            try:
                cx.execute(f"ALTER TABLE inbound_leads ADD COLUMN {col} {ddl}")
            except Exception:
                pass
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
    ws = os.environ.get("WEBHOOK_SECRET", "")
    cs = os.environ.get("CONSOLE_SECRET", "")
    given = request.headers.get("X-Webhook-Secret", "") or request.headers.get("X-Console-Key", "")
    if not ((ws and given == ws) or (cs and given == cs)):
        return jsonify({"error": "unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("""
            SELECT id, received_at, source, email, first_name, last_name, phone, raw_json, ghl_error
            FROM inbound_leads
            WHERE ghl_contact_id IS NULL
              AND email IS NOT NULL AND email != ''
              AND (status IS NULL OR status != 'dismissed')
            ORDER BY received_at ASC LIMIT 100
        """).fetchall()
    return jsonify({"leads": [dict(r) for r in rows], "count": len(rows)})


@app.route("/leads/mark-ghl-synced", methods=["POST"])
def leads_mark_ghl_synced():
    ws = os.environ.get("WEBHOOK_SECRET", "")
    cs = os.environ.get("CONSOLE_SECRET", "")
    given = request.headers.get("X-Webhook-Secret", "") or request.headers.get("X-Console-Key", "")
    if not ((ws and given == ws) or (cs and given == cs)):
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


# ── GHL write-queue drain endpoints (for local-machine drain when WAF blocks Render→GHL) ──

def _ghl_queue_auth():
    ws = os.environ.get("WEBHOOK_SECRET", "")
    cs = os.environ.get("CONSOLE_SECRET", "")
    given = request.headers.get("X-Webhook-Secret", "") or request.headers.get("X-Console-Key", "")
    return (ws and given == ws) or (cs and given == cs)


@app.route("/api/ghl/queue/pending", methods=["GET"])
def ghl_queue_pending():
    if not _ghl_queue_auth():
        return jsonify({"error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        rows = _bos_ghl_queue.list_pending(cx, limit=int(request.args.get("limit", 100) or 100))
    except (TypeError, ValueError):
        rows = _bos_ghl_queue.list_pending(cx)
    finally:
        cx.close()
    return jsonify({"queue": rows, "count": len(rows)})


@app.route("/api/ghl/queue/result", methods=["POST"])
def ghl_queue_result():
    if not _ghl_queue_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    qid = data.get("id")
    status = data.get("status", "done")
    if not qid:
        return jsonify({"ok": False, "error": "id required"}), 400
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_ghl_queue.mark_result(cx, int(qid), status, data.get("result", ""))
    finally:
        cx.close()
    return jsonify({"ok": True})


# ── Per-lead actions (interactive dashboard) ──────────────────────────────────
# These mirror the todo action surface so leads + scoreapp signups can be
# replied to, tagged, or dismissed without leaving /dashboard.

def _lead_auth_or_401():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    return None


@app.route("/api/leads/<int:lead_id>/draft-reply", methods=["POST"])
def api_lead_draft_reply(lead_id):
    err = _lead_auth_or_401()
    if err: return err
    data     = request.get_json(force=True) or {}
    guidance = (data.get("guidance") or "").strip()
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT first_name, last_name, email, source, raw_json FROM inbound_leads WHERE id=?",
            (lead_id,)
        ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    first, last, email, source, raw = row
    name = (first or "").strip() or "there"
    source_note = {
        "scoreapp":         "They just completed your ScoreApp self-healing quiz.",
        "scoreapp-webhook": "They just completed your ScoreApp self-healing quiz.",
        "pb":               "They booked through Practice Better.",
        "practice-better":  "They booked through Practice Better.",
    }.get((source or "").lower(), f"They came in via {source or 'an inbound channel'}.")
    guidance_block = f"\n\nGlen's guidance for this reply: {guidance}" if guidance else ""
    prompt = (
        "You are drafting a warm first-contact email on behalf of Dr. Glen Swartwout, naturopathic "
        "physician and biofield scientist in Hilo, Hawaiʻi. Be warm, brief (3–5 short paragraphs), "
        "and human — not salesy. Sign off naturally as Dr. Glen.\n\n"
        f"Recipient: {name} {last or ''}  <{email}>\n"
        f"Context: {source_note}{guidance_block}\n\n"
        "Draft the reply now (no subject line, just the body):"
    )
    try:
        msg = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        subject = "A note from Dr. Glen" if not (source or "").startswith("score") \
            else "Following up on your self-healing scan"
        return jsonify({"draft": msg.content[0].text, "subject": subject, "to": email})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads/<int:lead_id>/send-reply", methods=["POST"])
def api_lead_send_reply(lead_id):
    err = _lead_auth_or_401()
    if err: return err
    data    = request.get_json(force=True) or {}
    subject = (data.get("subject") or "").strip()
    body    = (data.get("body") or "").strip()
    if not subject or not body:
        return jsonify({"error": "subject and body required"}), 400
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT email FROM inbound_leads WHERE id=?", (lead_id,)).fetchone()
    if not row or not row[0]:
        return jsonify({"error": "lead not found or has no email"}), 404
    to_email = row[0]
    try:
        from dashboard.inbox import send_email as _gmail_send
        result = _gmail_send(to_email, subject, body, from_name="Dr. Glen Swartwout")
    except Exception as e:
        return jsonify({"error": f"send failed: {e}"}), 502
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE inbound_leads SET last_outbound_at=? WHERE id=?", (ts, lead_id))
        cx.commit()
    return jsonify({"ok": True, "to": to_email, "gmail_id": result.get("id"),
                    "thread_id": result.get("threadId"), "sent_at": ts})


@app.route("/api/leads/<int:lead_id>/tag", methods=["POST"])
def api_lead_tag(lead_id):
    err = _lead_auth_or_401()
    if err: return err
    data = request.get_json(force=True) or {}
    tag  = (data.get("tag") or "").strip()
    if not tag:
        return jsonify({"error": "tag required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT tags FROM inbound_leads WHERE id=?", (lead_id,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        try:
            tags = json.loads(row[0] or "[]")
            if not isinstance(tags, list): tags = []
        except Exception:
            tags = []
        if tag not in tags:
            tags.append(tag)
        cx.execute("UPDATE inbound_leads SET tags=? WHERE id=?",
                   (json.dumps(tags), lead_id))
        cx.commit()
    return jsonify({"ok": True, "tags": tags})


@app.route("/api/leads/<int:lead_id>/dismiss", methods=["POST"])
def api_lead_dismiss(lead_id):
    err = _lead_auth_or_401()
    if err: return err
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("UPDATE inbound_leads SET status='dismissed' WHERE id=?", (lead_id,))
        cx.commit()
    return jsonify({"ok": True})


# ── Practice Better API helpers ───────────────────────────────────────────────
import time as _pb_time
import requests as _pb_requests

PB_BASE_URL       = "https://api.practicebetter.io"
_PB_TOKEN_CACHE   = {"token": None, "expires_at": 0.0}
_PB_TOKEN_TTL_SEC = 3000   # 50 minutes (PB tokens last 1 hour)


def _pb_get_token():
    """OAuth2 client-credentials grant with in-process 50-min TTL cache."""
    now = _pb_time.time()
    if _PB_TOKEN_CACHE["token"] and now < _PB_TOKEN_CACHE["expires_at"]:
        return _PB_TOKEN_CACHE["token"]
    client_id     = os.environ.get("PRACTICE_BETTER_CLIENT_ID", "")
    client_secret = os.environ.get("PRACTICE_BETTER_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        raise RuntimeError("PRACTICE_BETTER_CLIENT_ID/SECRET not configured")
    r = _pb_requests.post(
        f"{PB_BASE_URL}/oauth2/token",
        data={"grant_type":"client_credentials",
              "client_id":client_id, "client_secret":client_secret},
        headers={"Content-Type":"application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    tok = r.json()["access_token"]
    _PB_TOKEN_CACHE["token"]      = tok
    _PB_TOKEN_CACHE["expires_at"] = now + _PB_TOKEN_TTL_SEC
    return tok


def _pb_get(path, params=None):
    """Bearer-authed GET. Retries once on 401 (token may have expired mid-cache)."""
    for attempt in (0, 1):
        tok = _pb_get_token()
        r = _pb_requests.get(f"{PB_BASE_URL}{path}",
                             headers={"Authorization": f"Bearer {tok}"},
                             params=params or {}, timeout=30)
        if r.status_code == 401 and attempt == 0:
            _PB_TOKEN_CACHE["token"] = None
            continue
        r.raise_for_status()
        return r.json()


def _pb_fetch_tag_definitions():
    """Paginate /tags → return {tag_id: tag_name}.
    PB uses 'skip' (not 'offset') for pagination; max page size is 100."""
    out, skip, page_size = {}, 0, 100
    while True:
        body = _pb_get("/tags", params={"limit": page_size, "skip": skip})
        items = body.get("items", []) if isinstance(body, dict) else []
        for it in items:
            tid, name = it.get("id"), (it.get("name") or "").strip()
            if tid and name:
                out[tid] = name
        if not body.get("hasMore"):
            break
        skip += page_size
        if skip > 10000:   # sanity guard
            break
    return out


def _pb_fetch_all_records():
    """Paginate /consultant/records → return list of full record dicts.
    PB uses 'skip' (not 'offset') for pagination; max page size is 100."""
    out, skip, page_size = [], 0, 100
    while True:
        body = _pb_get("/consultant/records", params={"limit": page_size, "skip": skip})
        items = body.get("items", []) if isinstance(body, dict) else []
        out.extend(items)
        if not body.get("hasMore"):
            break
        skip += page_size
        if skip > 100000:   # sanity guard
            break
    return out


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
        _attribute_conversion_by_email(
            pb_email, "course-enrollment", event_type, None, "practice-better", raw)

    return jsonify({"ok": True, "event": event_type}), 200


# ── PB → People DB + GHL tag sync ─────────────────────────────────────────────
def _pb_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def sync_pb_to_people_and_ghl(dry_run=False, limit=None):
    """Walk every PB client record, resolve relatedTags → names, upsert into
    the people table (matching by email, additive tag merge), then push the
    same tags to GHL via ghl_upsert_contact in the `pb:` namespace.

    Phased to avoid holding _db_lock during HTTP work:
      Phase 1: fetch PB tag dict + records (read-only, fast).
      Phase 2: SQLite upsert in one transaction (holds lock briefly).
      Phase 3: GHL upserts outside the lock, throttled.

    Additive-only: tags in the pb: namespace are added but never removed.
    """
    started = _pb_time.time()
    summary = {
        "records_fetched": 0,
        "records_skipped_no_email": 0,
        "people_upserted": 0,
        "ghl_synced": 0,
        "ghl_errors": 0,
        "total_tags_attached": 0,
        "dry_run": bool(dry_run),
        "elapsed_sec": 0,
    }

    # Phase 1 — read PB
    tag_dict = _pb_fetch_tag_definitions()
    records  = _pb_fetch_all_records()
    summary["records_fetched"] = len(records)
    if limit:
        records = records[:int(limit)]

    # Build normalized tuples
    norm = []
    for rec in records:
        profile = rec.get("profile") or {}
        email   = (profile.get("emailAddress") or "").strip().lower()
        if not email:
            summary["records_skipped_no_email"] += 1
            continue
        first = (profile.get("firstName") or "").strip()
        last  = (profile.get("lastName")  or "").strip()
        phone = (profile.get("mobilePhone") or "").strip()
        pb_id = rec.get("id") or ""
        tag_names = []
        for ref in (rec.get("relatedTags") or []):
            name = tag_dict.get(ref.get("id"))
            if name:
                tag_names.append(name)
        norm.append({
            "email": email, "first": first, "last": last, "phone": phone,
            "pb_id": pb_id, "tag_names": tag_names,
            "pb_tags": [f"pb:{_pb_slug(n)}" for n in tag_names if _pb_slug(n)],
        })

    now_iso = datetime.now(timezone.utc).isoformat()

    # Phase 2 — SQLite upsert (one transaction)
    if not dry_run:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            for n in norm:
                row = cx.execute(
                    "SELECT tags FROM people WHERE email=?", (n["email"],)
                ).fetchone()
                if row:
                    try:
                        existing = set(json.loads(row["tags"] or "[]"))
                    except Exception:
                        existing = set()
                    merged = sorted(existing | set(n["pb_tags"]))
                    cx.execute("""
                        UPDATE people SET
                          pb_id      = CASE WHEN pb_id='' THEN ? ELSE pb_id END,
                          first_name = CASE WHEN first_name='' THEN ? ELSE first_name END,
                          last_name  = CASE WHEN last_name='' THEN ? ELSE last_name END,
                          phone      = CASE WHEN phone='' THEN ? ELSE phone END,
                          tags       = ?,
                          source     = CASE WHEN source='' THEN 'practice-better' ELSE source END,
                          updated_at = ?,
                          synced_at  = ?
                        WHERE email=?
                    """, (n["pb_id"], n["first"], n["last"], n["phone"],
                          json.dumps(merged), now_iso, now_iso, n["email"]))
                else:
                    cx.execute("""
                        INSERT INTO people
                          (email, first_name, last_name, name, phone, pb_id,
                           tags, source, created_at, updated_at, synced_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (n["email"], n["first"], n["last"],
                          f"{n['first']} {n['last']}".strip(),
                          n["phone"], n["pb_id"],
                          json.dumps(sorted(set(n["pb_tags"]))),
                          "practice-better", now_iso, now_iso, now_iso))
                summary["people_upserted"] += 1
            cx.commit()
    else:
        summary["people_upserted"] = len(norm)   # would-have count

    # Phase 3 — GHL upsert (outside lock, throttled)
    if not dry_run:
        for n in norm:
            if not n["pb_tags"]:
                continue
            try:
                contact_id, created, err = ghl_upsert_contact(
                    email=n["email"], first_name=n["first"], last_name=n["last"],
                    phone=n["phone"], source_tag="", extra_tags=n["pb_tags"],
                )
                if err:
                    summary["ghl_errors"] += 1
                    app.logger.warning("PB sync GHL error for %s: %s", n["email"], err)
                else:
                    summary["ghl_synced"] += 1
                    summary["total_tags_attached"] += len(n["pb_tags"])
            except Exception as e:
                summary["ghl_errors"] += 1
                app.logger.exception("PB sync GHL exception for %s: %s", n["email"], e)
            _pb_time.sleep(0.15)   # ~7 req/sec, under GHL's ~100/10s limit
    else:
        summary["total_tags_attached"] = sum(len(n["pb_tags"]) for n in norm)

    summary["elapsed_sec"] = round(_pb_time.time() - started, 2)
    return summary


@app.route("/admin/sync-pb-tags", methods=["POST"])
def admin_sync_pb_tags():
    """Trigger PB → People DB + GHL tag sync. Auth: X-Cron-Secret header
    (or ?key=) matching CRON_SECRET env (falls back to CONSOLE_SECRET).
    Query params: dry_run=1 (counts only), limit=N (process first N records)."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    limit   = request.args.get("limit")
    try:
        summary = sync_pb_to_people_and_ghl(
            dry_run=dry_run,
            limit=int(limit) if limit else None,
        )
        # After successful PB sync (not dry-run), run household-side steps:
        # 1. Detect new candidates (new PB contacts may match existing patterns)
        # 2. Resync household tags to GHL (drift recovery)
        if not dry_run:
            try:
                summary["households"] = {
                    "detection": detect_household_candidates(),
                    "resync":    resync_all_households_to_ghl(),
                }
            except Exception as e:
                app.logger.exception("post-pb household steps failed")
                summary["households"] = {"error": f"{type(e).__name__}: {e}"}
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("PB sync failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ── Unified People hub: additive upsert + source feeders ──────────────────────
# Column groups for the additive upsert (mirror the /api/people POST handler).
_PERSON_UPSERT_SCALARS = [
    "first_name", "last_name", "name", "phone", "dob", "birth_time", "birthplace",
    "gender", "city", "state", "country", "island", "profession", "title",
    "ghl_id", "pb_id", "source", "challenges", "goals", "personal_history",
    "family_history", "medications", "surgeries", "budget", "investment",
    "resources", "issue_duration", "form_completed_by",
    "last_order_date", "last_session_date", "last_contact_date", "notes",
]
_PERSON_UPSERT_JSON = [
    "organizations", "tags", "roles", "terrain_concerns", "body_systems",
    "conditions", "healing_response", "interests", "request",
]


def _upsert_person_additive(cx, person, ts=None):
    """Idempotent, additive upsert of one person into the people table, matching
    by email. JSON-array fields (tags, roles, …) are UNIONED with existing;
    scalar fields only overwrite when the incoming value is non-empty (never
    clobbered with blanks); counts take the max. This is what feeders use so
    re-running is safe and a person who appears in several sources keeps every
    tag. Returns "inserted" | "updated" | None (no email). Caller holds the
    transaction (commits)."""
    ts = ts or datetime.now(timezone.utc).isoformat()
    email = (person.get("email") or "").strip().lower()
    if not email:
        return None
    cx.row_factory = sqlite3.Row
    existing = cx.execute("SELECT * FROM people WHERE email=?", (email,)).fetchone()

    scalars = {k: person.get(k, "") for k in _PERSON_UPSERT_SCALARS}
    arrays = {}
    for jf in _PERSON_UPSERT_JSON:
        v = person.get(jf, [])
        if isinstance(v, str):
            try:
                v = json.loads(v or "[]")
            except Exception:
                v = []
        arrays[jf] = list(v or [])
    order_count = int(person.get("order_count", 0) or 0)
    session_count = int(person.get("session_count", 0) or 0)

    # Closed-loop consent: a GoHighLevel email unsubscribe / bounce / DND (pulled
    # in by console_push) revokes opt-in here, so the hub stays the single consent
    # truth and the Phase-2 mirror stops re-pushing opted-out people. Email-channel
    # signals only — SMS-only unsubscribes do not gate email.
    dnd = bool(person.get("dnd")) or any(
        any(s in t.lower() for s in ("email bounced", "email unsubscrib",
                                     "do not email", "spam complain"))
        for t in arrays.get("tags", []))

    def _apply_dnd(tagset):
        return ((set(tagset) - {"consent:opted-in"}) | {"consent:unsubscribed"}) if dnd else set(tagset)

    if existing:
        cols = set(existing.keys())
        upd = {}
        for k in _PERSON_UPSERT_SCALARS:
            if str(scalars[k]).strip():
                upd[k] = scalars[k]
        for jf in _PERSON_UPSERT_JSON:
            if jf not in cols:
                continue
            try:
                ex = set(json.loads(existing[jf] or "[]"))
            except Exception:
                ex = set()
            merged = sorted(ex | set(arrays[jf]))
            if merged != sorted(ex):
                upd[jf] = json.dumps(merged)
        if dnd:  # force the consent flip on the tags column (a removal, so the union check above misses it)
            try:
                ex_tags = set(json.loads(existing["tags"] or "[]"))
            except Exception:
                ex_tags = set()
            upd["tags"] = json.dumps(sorted(_apply_dnd(ex_tags | set(arrays.get("tags", [])))))
        if order_count > int(existing["order_count"] or 0):
            upd["order_count"] = order_count
        if session_count > int(existing["session_count"] or 0):
            upd["session_count"] = session_count
        upd["updated_at"] = ts
        upd["synced_at"] = ts
        upd = {k: v for k, v in upd.items() if k in cols}
        set_clause = ", ".join(f"{k}=?" for k in upd)
        cx.execute(f"UPDATE people SET {set_clause} WHERE email=?",
                   list(upd.values()) + [email])
        return "updated"

    ins = dict(scalars)
    for jf in _PERSON_UPSERT_JSON:
        ins[jf] = json.dumps(sorted(set(arrays[jf])))
    ins["tags"] = json.dumps(sorted(_apply_dnd(set(arrays.get("tags", [])))))
    ins["email"] = email
    ins["order_count"] = order_count
    ins["session_count"] = session_count
    ins["created_at"] = ts
    ins["updated_at"] = ts
    ins["synced_at"] = ts
    col_sql = ", ".join(ins.keys())
    val_sql = ", ".join("?" * len(ins))
    cx.execute(f"INSERT INTO people ({col_sql}) VALUES ({val_sql})", list(ins.values()))
    return "inserted"


def _practitioner_to_person(row):
    """Map a Supabase `practitioners` row (dict) to a people upsert dict with
    contact-type + consent tags. Engaged = has a portal_role or wholesale unlock;
    everything else (scraped directory rows) is cold and suppressed from
    automation. Returns None if the row has no usable email."""
    email = (row.get("email") or "").strip().lower()
    if not email:
        return None
    engaged = bool(row.get("portal_role")) or bool(row.get("wholesale_unlocked_at"))
    type_tag = "type:practitioner" if engaged else "type:practitioner-cold"
    consent_tag = "consent:opted-in" if engaged else "consent:cold-no-consent"
    tags = sorted({type_tag, consent_tag, "source:practitioner-finder",
                   f"pract:{row.get('id')}"})
    name = (row.get("name") or "").strip()
    first, last = "", ""
    if name:
        parts = name.split(" ", 1)
        first = parts[0]
        last = parts[1] if len(parts) > 1 else ""
    orgs = [o for o in [(row.get("practice_name") or "").strip(),
                        (row.get("source_org") or "").strip()] if o]
    return {
        "email": email, "name": name, "first_name": first, "last_name": last,
        "phone": (row.get("phone") or "").strip(),
        "city": (row.get("city") or "").strip(),
        "state": (row.get("state") or "").strip(),
        "country": (row.get("country") or "").strip(),
        "profession": (row.get("credentials") or "").strip(),
        "organizations": orgs,
        "source": "practitioner-finder",
        "tags": tags,
    }


def sync_practitioners_to_people(dry_run=False, limit=None):
    """Feed the Supabase practitioners directory into the people hub, tagged
    type:practitioner / type:practitioner-cold + consent. Reads the base table
    directly (the public view predates the portal columns) and skips rows with
    no email (can't dedup or automate). Additive + idempotent."""
    started = _pb_time.time()
    summary = {"rows_fetched": 0, "skipped_no_email": 0, "people_upserted": 0,
               "engaged": 0, "cold": 0, "dry_run": bool(dry_run), "elapsed_sec": 0}
    try:
        from db_supabase import supabase_cursor
    except Exception as e:
        return {"error": f"supabase unavailable: {type(e).__name__}: {e}"}
    sql = ("SELECT id, name, practice_name, credentials, source_org, phone, "
           "email, city, state, country, tier, portal_role, wholesale_unlocked_at "
           "FROM practitioners WHERE email IS NOT NULL AND email <> ''")
    if limit:
        sql += f" LIMIT {int(limit)}"
    with supabase_cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    summary["rows_fetched"] = len(rows)
    persons = []
    for row in rows:
        person = _practitioner_to_person(dict(row))
        if not person:
            summary["skipped_no_email"] += 1
            continue
        if "type:practitioner" in person["tags"]:
            summary["engaged"] += 1
        else:
            summary["cold"] += 1
        persons.append(person)
    if dry_run:
        summary["people_upserted"] = len(persons)
    else:
        ts = datetime.now(timezone.utc).isoformat()
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            for person in persons:
                if _upsert_person_additive(cx, person, ts):
                    summary["people_upserted"] += 1
            cx.commit()
    summary["elapsed_sec"] = round(_pb_time.time() - started, 2)
    return summary


@app.route("/admin/sync-practitioner-tags", methods=["POST"])
def admin_sync_practitioner_tags():
    """Trigger Supabase practitioners → People hub feeder. Auth + params mirror
    /admin/sync-pb-tags: X-Cron-Secret/X-Console-Key/?key=, dry_run=1, limit=N."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    limit = request.args.get("limit")
    try:
        summary = sync_practitioners_to_people(
            dry_run=dry_run, limit=int(limit) if limit else None)
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("practitioner sync failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ── People hub Phase 2: classify existing people + mirror opted-in to GHL ──────
# Substring patterns (matched lowercased against a person's existing tags).
_OPTIN_TAG_PATTERNS = ("opted in", "opt-in", "email list opted")
# Hard-suppression: email-negative signals override any opt-in signal
# (CAN-SPAM / deliverability). SMS-only unsubscribes are intentionally excluded.
_SUPPRESS_TAG_PATTERNS = ("email bounced", "reengagement:bounced",
                          "do not email", "spam complaint", "email unsubscribed")


def _is_client_tag(t):
    """A tag marking an actual client relationship. The hub's order_count isn't
    populated, so the real client signal lives in tags (e.g. 'client',
    'nes client'). Matched precisely to avoid 'client-intake'-style false hits."""
    t = (t or "").lower()
    return t == "client" or "nes client" in t or t.endswith(" client")


def _classify_person(row, tos_agreed=False):
    """Pure rule. Given a people row (tags/order_count/pb_id) + a tos_agreed flag,
    return the set of type:/consent: tags to ADD. Additive only (never removes,
    never downgrades an existing consent tag). Honors 'explicit opt-in signal'
    with hard-suppression for bounced/unsubscribed."""
    tags = row.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags or "[]")
        except Exception:
            tags = []
    tagset = set(tags)
    low = [t.lower() for t in tagset]
    add = set()

    has_commerce = (int(row.get("order_count") or 0) > 0
                    or bool((row.get("pb_id") or "").strip())
                    or any(_is_client_tag(t) for t in tagset))
    has_type = any(t.startswith("type:") for t in tagset)
    has_consent = any(t.startswith("consent:") for t in tagset)

    if has_commerce:
        add.add("type:client")
    elif not has_type:
        add.add("type:prospect")

    if not has_consent:
        suppressed = any(any(p in t for p in _SUPPRESS_TAG_PATTERNS) for t in low)
        opted = (not suppressed) and (
            has_commerce or tos_agreed
            or any(any(p in t for p in _OPTIN_TAG_PATTERNS) for t in low))
        add.add("consent:opted-in" if opted else "consent:cold-no-consent")
    return add


def _tos_agreed_emails(cx):
    """Lowercased emails that agreed to the soft ToS (journey_state)."""
    try:
        rows = cx.execute(
            "SELECT DISTINCT lower(email) FROM journey_state "
            "WHERE tos_agreed_at IS NOT NULL AND tos_agreed_at<>'' AND email<>''"
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def classify_people(dry_run=False, limit=None):
    """Tag the existing hub population with type:client/prospect + consent from
    explicit opt-in signals. Additive + idempotent."""
    started = _pb_time.time()
    summary = {"scanned": 0, "updated": 0, "added_client": 0, "added_prospect": 0,
               "added_opted_in": 0, "added_cold": 0, "dry_run": bool(dry_run), "elapsed_sec": 0}
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        tos = _tos_agreed_emails(cx)
        q = "SELECT id,email,tags,order_count,pb_id FROM people"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = cx.execute(q).fetchall()
        summary["scanned"] = len(rows)
        for r in rows:
            add = _classify_person(dict(r), tos_agreed=(r["email"] or "").lower() in tos)
            if not add:
                continue
            summary["updated"] += 1
            for t in add:
                summary["added_client"]   += (t == "type:client")
                summary["added_prospect"] += (t == "type:prospect")
                summary["added_opted_in"] += (t == "consent:opted-in")
                summary["added_cold"]     += (t == "consent:cold-no-consent")
            if not dry_run:
                _upsert_person_additive(cx, {"email": r["email"], "tags": sorted(add)}, ts)
        if not dry_run:
            cx.commit()
    summary["elapsed_sec"] = round(_pb_time.time() - started, 2)
    return summary


def sync_people_to_ghl(dry_run=False, limit=None):
    """Mirror opted-in hub people into GHL by enqueuing tag_add ops (their type:*
    tags) onto the WAF-safe write-queue (drained from the Mac). Cold contacts are
    never enqueued, so they never reach GHL and cannot be automated."""
    started = _pb_time.time()
    summary = {"opted_in": 0, "enqueued": 0, "skipped_already_pending": 0,
               "skipped_no_type": 0, "dry_run": bool(dry_run), "elapsed_sec": 0}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        q = 'SELECT email,tags FROM people WHERE tags LIKE ?'
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = cx.execute(q, ['%"consent:opted-in"%']).fetchall()
        summary["opted_in"] = len(rows)
        # Dedup against tags already queued (pending) per email.
        pending = {}
        for it in _bos_ghl_queue.list_pending(cx, limit=1000000):
            if it["op"] == "tag_add":
                pending.setdefault((it["email"] or "").lower(), set()).update(
                    json.loads(it.get("payload_json") or "{}").get("tags", []))
        for r in rows:
            email = (r["email"] or "").strip().lower()
            try:
                tags = set(json.loads(r["tags"] or "[]"))
            except Exception:
                tags = set()
            type_tags = sorted(t for t in tags if t.startswith("type:"))
            if not type_tags:
                summary["skipped_no_type"] += 1
                continue
            todo = [t for t in type_tags if t not in pending.get(email, set())]
            if not todo:
                summary["skipped_already_pending"] += 1
                continue
            if not dry_run:
                _bos_ghl_queue.enqueue(cx, op="tag_add", email=email,
                                       payload={"tags": todo}, actor="people-ghl-sync")
                pending.setdefault(email, set()).update(todo)
            summary["enqueued"] += 1
    summary["elapsed_sec"] = round(_pb_time.time() - started, 2)
    return summary


@app.route("/admin/sync-people-to-ghl", methods=["POST"])
def admin_sync_people_to_ghl():
    """Phase 2: classify existing people, then enqueue opted-in → GHL tag mirrors.
    Auth/params mirror /admin/sync-pb-tags."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    limit = request.args.get("limit")
    lim = int(limit) if limit else None
    try:
        classified = classify_people(dry_run=dry_run, limit=lim)
        mirrored = sync_people_to_ghl(dry_run=dry_run, limit=lim)
        return jsonify({"ok": True, "classified": classified, "mirrored": mirrored})
    except Exception as e:
        app.logger.exception("people→GHL sync failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


def enroll_segment_in_workflow(workflow_id, segment_tags=("type:client", "consent:opted-in"),
                               cap=None, dry_run=False):
    """One-time backfill: enroll an opted-in People-hub segment into an existing
    GHL workflow by enqueuing `workflow` ops (drained from the Mac). GHL "tag
    added" triggers only fire forward, so contacts already carrying the segment
    tag need this to enter the workflow. Deduped by an `enrolled:wf:<id8>` tag so
    re-runs never double-enqueue; GHL also won't re-enroll a contact already in
    the workflow. Skips unsubscribed/no-email."""
    wf = (workflow_id or "").strip()
    if not wf:
        return {"error": "workflow_id required"}
    wf8 = wf[:8]
    enrolled_tag = f"enrolled:wf:{wf8}"
    started = _pb_time.time()
    summary = {"workflow_id": wf, "matched": 0, "enqueued": 0,
               "skipped_already": 0, "dry_run": bool(dry_run), "elapsed_sec": 0}
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        clauses = " AND ".join(["tags LIKE ?"] * len(segment_tags))
        rows = cx.execute(
            f"SELECT email, tags FROM people WHERE email<>'' AND {clauses}",
            [f'%"{t}"%' for t in segment_tags]).fetchall()
        for r in rows:
            email = (r["email"] or "").strip().lower()
            try:
                tags = set(json.loads(r["tags"] or "[]"))
            except Exception:
                tags = set()
            if "consent:unsubscribed" in tags or any(
                    s in " ".join(t.lower() for t in tags)
                    for s in ("email bounced", "do not email", "unsubscribed")):
                continue
            summary["matched"] += 1
            if enrolled_tag in tags:
                summary["skipped_already"] += 1
                continue
            if cap and summary["enqueued"] >= int(cap):
                continue
            if not dry_run:
                _bos_ghl_queue.enqueue(cx, op="workflow", email=email,
                                       payload={"workflow_id": wf}, actor="segment-enroll")
                _upsert_person_additive(cx, {"email": email, "tags": [enrolled_tag]}, ts)
            summary["enqueued"] += 1
        if not dry_run:
            cx.commit()
    summary["elapsed_sec"] = round(_pb_time.time() - started, 2)
    return summary


@app.route("/admin/enroll-segment-workflow", methods=["POST"])
def admin_enroll_segment_workflow():
    """Backfill: enroll an opted-in segment into an existing GHL workflow.
    ?workflow_id=... &segment=type:client,consent:opted-in &cap=N &dry_run=1.
    Auth mirrors /admin/sync-pb-tags."""
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    wf = request.args.get("workflow_id", "")
    seg = request.args.get("segment", "type:client,consent:opted-in")
    tags = tuple(t.strip() for t in seg.split(",") if t.strip())
    cap = request.args.get("cap")
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    try:
        summary = enroll_segment_in_workflow(
            wf, segment_tags=tags, cap=int(cap) if cap else None, dry_run=dry_run)
        code = 400 if summary.get("error") else 200
        return jsonify({"ok": not summary.get("error"), "summary": summary}), code
    except Exception as e:
        app.logger.exception("segment enroll failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


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

    # Phase 2b: if this email has recent practitioner inquiries, offer to
    # share the assessment with those practitioners. Fire-and-forget.
    try:
        recent = _recent_inquiry_practitioner_ids(email)
        if recent:
            plain = secrets.token_urlsafe(32)
            th = _hash_token(plain)
            now_iso = datetime.utcnow().isoformat() + "Z"
            exp_iso = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute(
                    "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (th, email, "practitioner_share",
                     json.dumps({"days": 30}), now_iso, exp_iso)
                )
            n_practitioners = len({pid for (_, pid, _, _) in recent})
            base = request.host_url.rstrip("/")
            share_url = f"{base}/share-with-practitioner/{plain}"
            first_safe = first or "there"
            offer_body = (
                f"Hi {first_safe},\n\n"
                f"Thanks for completing the self-assessment.\n\n"
                f"You recently reached out to {n_practitioners} practitioner"
                f"{'s' if n_practitioners != 1 else ''} on RemedyMatch. "
                f"Would you like to share your assessment results with them so they "
                f"have full context for their reply?\n\n"
                f"Share with one click:\n{share_url}\n\n"
                f"You can review what gets shared on that page before confirming.\n\n"
                f"---\n"
                f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
            )
            _send_inquiry_email(
                to_email=email,
                subject="Share your assessment with the practitioners you contacted?",
                body=offer_body,
                reply_to=RM_INBOUND_INQUIRY_EMAIL,
            )
    except Exception as e:
        print(f"[scoreapp] share-offer send failed: {e!r}", flush=True)

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

    # Best-effort order total — GrooveKart/PrestaShop payloads vary
    order_total = None
    for k in ("total_paid", "total_price", "total", "total_paid_tax_incl", "amount"):
        v = data.get(k)
        if v not in (None, ""):
            try:
                order_total = float(v)
                break
            except (TypeError, ValueError):
                pass

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
    credited = _attribute_conversion_by_email(
        email, "store-purchase", product, order_total, "groovekart", raw)
    _ingest_order(source="groovekart",
                  external_ref=str(data.get("id") or data.get("order_id") or email or _bos_orders._now()),
                  email=email, name=(first + " " + last).strip(),
                  items=[{"name": product}] if product else [],
                  total_cents=int(round(float(order_total or 0) * 100)), channel="retail")
    return jsonify({"ok": True, "ghl": ghl_result, "affiliate_credited": credited}), 200


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
                core_message    TEXT DEFAULT '',
                received_at     TEXT DEFAULT ''
            )
        """)
        # Migrate existing tables that predate these columns
        for col, ddl in [("ai_summary", "TEXT DEFAULT ''"), ("suggested_reply", "TEXT DEFAULT ''"),
                         ("action_note", "TEXT DEFAULT ''"), ("core_message", "TEXT DEFAULT ''"),
                         ("received_at", "TEXT DEFAULT ''"),
                         ("phase", "TEXT NOT NULL DEFAULT 'plan'"),
                         ("first_started_at", "TEXT DEFAULT ''")] :
            try:
                cx.execute(f"ALTER TABLE todos ADD COLUMN {col} {ddl}")
            except Exception:
                pass
        # Backfill phase from existing status (one-time; idempotent)
        try:
            cx.execute("UPDATE todos SET phase='complete' WHERE status='done' AND (phase IS NULL OR phase='plan')")
        except Exception:
            pass
        cx.commit()

_init_todos_table()


def _init_workspace_schema():
    """Tables backing the per-owner Workspace page (focused item, threads, time, steps)."""
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS owner_state (
                owner            TEXT PRIMARY KEY,
                focused_todo_id  INTEGER,
                updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS todo_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                todo_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (todo_id) REFERENCES todos(id)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_todo_messages_todo ON todo_messages(todo_id, created_at)")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS todo_time_sessions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                todo_id           INTEGER NOT NULL,
                owner             TEXT NOT NULL,
                started_at        TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at          TEXT,
                duration_seconds  INTEGER,
                FOREIGN KEY (todo_id) REFERENCES todos(id)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_todo_time_todo ON todo_time_sessions(todo_id)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_todo_time_open ON todo_time_sessions(owner, ended_at)")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS todo_steps (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                todo_id   INTEGER NOT NULL,
                sequence  INTEGER NOT NULL,
                text      TEXT NOT NULL,
                done      INTEGER NOT NULL DEFAULT 0,
                done_at   TEXT,
                FOREIGN KEY (todo_id) REFERENCES todos(id)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_todo_steps_todo ON todo_steps(todo_id, sequence)")
        # Per-user access tokens (Phase 2): supersede the shared CONSOLE_SECRET for
        # offshore / per-user access (Shaira). Admin (CONSOLE_SECRET) keeps full access.
        cx.execute("""
            CREATE TABLE IF NOT EXISTS workspace_users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                display_name  TEXT DEFAULT '',
                scope         TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS access_tokens (
                token         TEXT PRIMARY KEY,
                user_id       INTEGER NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                last_used_at  TEXT,
                revoked_at    TEXT,
                note          TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES workspace_users(id)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_access_tokens_user ON access_tokens(user_id)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_access_tokens_active ON access_tokens(token, revoked_at)")
        # Daily monitoring reports (Phase 4) — one row per owner per day, upserted on re-run.
        cx.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                owner         TEXT NOT NULL,
                report_date   TEXT NOT NULL,
                report_md     TEXT NOT NULL DEFAULT '',
                metrics_json  TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(owner, report_date)
            )
        """)
        cx.commit()

_init_workspace_schema()


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

    data = request.get_json(force=True) or {}
    action = data.get("action", "hide")

    # Delegate: copy the event onto another owner's console calendar. Console-only
    # (synthetic google id, google_cal_id='delegated' so the Google sync never
    # touches it); non-destructive (original stays on the delegator's calendar).
    if action == "delegate":
        to = (data.get("to") or "").lower()
        if to not in ("glen", "rae", "shaira"):
            return jsonify({"error": "Invalid delegate target"}), 400
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT summary, start, end, location, owner FROM calendar_events WHERE id=?",
                (event_id,)).fetchone()
            if not row:
                return jsonify({"error": "not found"}), 404
            summary, start, end, location, from_owner = row
            cx.execute("""
                INSERT INTO calendar_events
                  (pushed_at, google_cal_id, google_event_id, calendar_name,
                   summary, start, end, location, owner, status, cal_alert)
                VALUES (?, 'delegated', ?, ?, ?, ?, ?, ?, ?, 'visible', 0)
                ON CONFLICT(google_cal_id, google_event_id) DO UPDATE SET
                  status='visible', summary=excluded.summary, start=excluded.start,
                  end=excluded.end, location=excluded.location,
                  calendar_name=excluded.calendar_name, pushed_at=excluded.pushed_at
            """, (datetime.now(timezone.utc).isoformat(), f"deleg-{event_id}-{to}",
                  f"Delegated by {(from_owner or 'glen').title()}",
                  summary, start, end, location, to))
            cx.commit()
        return jsonify({"ok": True, "delegated_to": to})

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


@app.route("/funnel")
def funnel_page():
    # Internal ops view: op-nav top row + a second row of funnel sub-tabs that
    # load each funnel page (/begin, /ask, …) in an iframe.
    resp = send_from_directory(STATIC, "funnel.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/todos", methods=["GET"])
def get_todos():
    # Auth: when CONSOLE_SECRET is set, require auth. Scoped tokens get their
    # owner param force-rewritten (you can't enumerate someone else's data).
    requested_owner = request.args.get("owner", "glen").lower()
    if CONSOLE_SECRET:
        ok, ctx, code = _auth()
        if not ok:
            return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
        owner = _scoped_owner(ctx, requested_owner)
    else:
        owner = requested_owner
    status = request.args.get("status", "open")
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute("""
            SELECT id, created_at, owner, category, title, body, priority,
                   status, delegated_to, delegated_at, done_at, source, dedup_key,
                   ai_summary, suggested_reply, action_note, core_message, received_at
            FROM todos
            WHERE owner=? AND status=?
            ORDER BY
                CASE priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                created_at DESC
        """, (owner, status)).fetchall()
    cols = ["id","created_at","owner","category","title","body","priority",
            "status","delegated_to","delegated_at","done_at","source","dedup_key",
            "ai_summary","suggested_reply","action_note","core_message","received_at"]
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
            core_message    = item.get("core_message") or ""
            received_at     = item.get("received_at") or ""
            if not title:
                continue
            try:
                cx.execute("""
                    INSERT INTO todos
                      (created_at, owner, category, title, body, priority, source, dedup_key,
                       ai_summary, suggested_reply, action_note, core_message, received_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(dedup_key) DO UPDATE SET
                      ai_summary=excluded.ai_summary,
                      suggested_reply=excluded.suggested_reply,
                      action_note=excluded.action_note,
                      core_message=excluded.core_message,
                      received_at=CASE WHEN excluded.received_at != '' THEN excluded.received_at ELSE received_at END
                """, (ts, owner, category, title, body, priority, source, dedup,
                      ai_summary, suggested_reply, action_note, core_message, received_at))
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
        "You are drafting a reply on behalf of Dr. Glen Swartwout, naturopathic "
        "physician in Hilo, Hawaiʻi. Be warm, concise, and professional.\n"
        "Sign-off — choose by who the email is from:\n"
        "- Client or patient: sign off informally as:\n    In wellness,\n    Dr. Glen\n"
        "- Doctor, vendor, or professional contact: sign off formally as:\n"
        "    Dr. Glen Swartwout, Naturopathic Optometrist, Hilo, Hawai'i\n"
        "- Warm, long-standing colleague or friend (even if a doctor): relationship\n"
        "  warmth wins over the formal default, so sign off as:\n    Much Aloha,\n    Dr. Glen\n\n"
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

def _el_tts(script: str) -> Tuple[Optional[bytes], Optional[str]]:
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


# ── Chat audio output (per-message "Listen") ─────────────────────────────────
# Public endpoint that speaks a single chat reply via ElevenLabs (Glen's voice).
# The front-end (static/tts-output.js) falls back to the browser voice on any
# non-200 response, so this stays simple: cache to avoid re-charging on replays,
# rate-limit per IP to protect the paid API, and 503 when EL isn't configured.
import time as _tts_time
from collections import OrderedDict as _TTSOrderedDict

_TTS_CACHE       = _TTSOrderedDict()   # sha256(text) -> audio bytes (LRU)
_TTS_CACHE_MAX   = 200
_TTS_RATE        = {}                  # ip -> [timestamps within window]
_TTS_RATE_MAX    = 40                  # requests
_TTS_RATE_WINDOW = 300                 # seconds (5 min)
_TTS_MAX_CHARS   = 2500                # one reply; caps cost + latency
_tts_lock        = threading.Lock()


@app.route("/chat/tts", methods=["POST", "OPTIONS"])
def chat_tts():
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    if not _EL_API_KEY or not _EL_VOICE_ID:
        return jsonify({"error": "tts not configured"}), 503

    text = text[:_TTS_MAX_CHARS]

    # Per-IP rate limit (protects the paid ElevenLabs API on a public route).
    ip  = (request.headers.get("X-Forwarded-For", request.remote_addr or "")
           .split(",")[0].strip()) or "anon"
    now = _tts_time.time()
    with _tts_lock:
        hits = [t for t in _TTS_RATE.get(ip, []) if now - t < _TTS_RATE_WINDOW]
        if len(hits) >= _TTS_RATE_MAX:
            _TTS_RATE[ip] = hits
            return jsonify({"error": "rate limited"}), 429
        hits.append(now)
        _TTS_RATE[ip] = hits

    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with _tts_lock:
        audio = _TTS_CACHE.get(key)
        if audio is not None:
            _TTS_CACHE.move_to_end(key)

    if audio is None:
        audio, err = _el_tts(text)
        if err or not audio:
            return jsonify({"error": f"tts upstream error: {err or 'empty'}"}), 502
        with _tts_lock:
            _TTS_CACHE[key] = audio
            _TTS_CACHE.move_to_end(key)
            while len(_TTS_CACHE) > _TTS_CACHE_MAX:
                _TTS_CACHE.popitem(last=False)

    resp = Response(audio, mimetype="audio/mpeg")
    resp.headers["Cache-Control"]  = "private, max-age=86400"
    resp.headers["Content-Length"] = str(len(audio))
    return resp


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
        # Per-item action columns (interactive dashboard) — idempotent migration
        for col, ddl in [("tag",         "TEXT DEFAULT ''"),       # helpful|not_helpful|noise
                         ("reviewed_at", "TEXT DEFAULT ''")]:
            try:
                cx.execute(f"ALTER TABLE rae_feedback ADD COLUMN {col} {ddl}")
            except Exception:
                pass
        cx.commit()

_init_rae_feedback_table()


def _init_heygen_reviewed_table():
    """Local 'mark reviewed' flag for HeyGen renders (HeyGen has no such concept)."""
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS heygen_reviewed (
                video_id    TEXT PRIMARY KEY,
                reviewed_at TEXT NOT NULL
            )
        """)
        cx.commit()

_init_heygen_reviewed_table()


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
            SELECT id, ts, event_type, greeting_style, transcript, amplitude_peak,
                   tag, reviewed_at
            FROM   rae_feedback
            ORDER  BY ts DESC
            LIMIT  20
        """).fetchall()

    return jsonify({
        "laugh_by_style": [dict(r) for r in by_style],
        "recent_events":  [dict(r) for r in recent],
    })


# ── Per-feedback actions (interactive dashboard) ─────────────────────────────

@app.route("/api/rae-feedback/<int:fb_id>/tag", methods=["POST"])
def api_rae_feedback_tag(fb_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True) or {}
    tag  = (data.get("tag") or "").strip().lower()
    if tag not in ("helpful", "not_helpful", "noise"):
        return jsonify({"error": "tag must be one of: helpful, not_helpful, noise"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        n = cx.execute("UPDATE rae_feedback SET tag=? WHERE id=?", (tag, fb_id)).rowcount
        cx.commit()
    if not n:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True, "tag": tag})


@app.route("/api/rae-feedback/<int:fb_id>/mark-reviewed", methods=["POST"])
def api_rae_feedback_mark_reviewed(fb_id):
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        n = cx.execute("UPDATE rae_feedback SET reviewed_at=? WHERE id=?", (ts, fb_id)).rowcount
        cx.commit()
    if not n:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True, "reviewed_at": ts})


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
        # In-house customer records (order-entry Phase 1): full shipping address.
        for _col in ("address1", "address2", "zip"):
            try:
                cx.execute(f"ALTER TABLE people ADD COLUMN {_col} TEXT DEFAULT ''")
            except Exception:
                pass  # already present
        cx.commit()

_init_people_table()


# ── Households ────────────────────────────────────────────────────────────────
def _init_households_tables():
    """Two tables: `households` for metadata, `household_candidates` for the
    detection-and-suggest workflow. Run at import time alongside other
    schema initializers."""
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS households (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                slug            TEXT UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                head_person_id  INTEGER,
                address         TEXT DEFAULT '',
                notes           TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                created_by      TEXT NOT NULL
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS household_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at     TEXT NOT NULL,
                signal          TEXT NOT NULL,
                person_ids      TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                resolved_at     TEXT DEFAULT '',
                resolved_by     TEXT DEFAULT '',
                household_id    INTEGER
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_household_candidates_status ON household_candidates(status)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_households_head ON households(head_person_id)")
        cx.commit()

_init_households_tables()


def _init_pending_merges_table():
    """Queued people-row merges from the candidate review flow. Apply is a
    separate operator action — never auto-executed."""
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS pending_merges (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id      INTEGER,
                keeper_person_id  INTEGER NOT NULL,
                dupe_person_id    INTEGER NOT NULL,
                queued_at         TEXT NOT NULL,
                queued_by         TEXT NOT NULL,
                status            TEXT NOT NULL DEFAULT 'pending',
                applied_at        TEXT DEFAULT '',
                notes             TEXT DEFAULT ''
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_pending_merges_status ON pending_merges(status)")
        cx.commit()

_init_pending_merges_table()


_PEOPLE_SCALAR_COALESCE = [
    "phone", "dob", "birth_time", "birthplace", "gender", "city", "state",
    "country", "island", "profession", "title", "ghl_id", "pb_id",
    "challenges", "goals", "personal_history", "family_history",
    "medications", "surgeries", "budget", "investment", "resources",
    "issue_duration", "form_completed_by", "source", "notes",
]
_PEOPLE_JSON_UNION = [
    "organizations", "tags", "roles", "terrain_concerns", "body_systems",
    "conditions", "healing_response", "interests", "request",
]
_PEOPLE_COUNT_SUM = ["order_count", "session_count"]
_PEOPLE_DATE_MAX = ["last_order_date", "last_session_date", "last_contact_date"]


def _merge_two_people(cx, keeper_id, dupe_id):
    """Merge dupe person row into keeper, then DELETE the dupe.
    Returns {keeper_id, deleted_dupe_id, fields_filled, tags_added, counts_summed}.
    Caller is responsible for the transaction (holds _db_lock + calls commit)."""
    cx.row_factory = sqlite3.Row
    keeper = cx.execute("SELECT * FROM people WHERE id=?", (keeper_id,)).fetchone()
    dupe   = cx.execute("SELECT * FROM people WHERE id=?", (dupe_id,)).fetchone()
    if not keeper or not dupe:
        raise ValueError(f"merge failed: keeper={keeper_id} or dupe={dupe_id} not found")
    if keeper_id == dupe_id:
        raise ValueError("keeper and dupe are the same row")

    updates = {}
    fields_filled = []
    # Coalesce empty scalar fields
    for col in _PEOPLE_SCALAR_COALESCE:
        try:
            k_val = keeper[col]
        except IndexError:
            continue
        try:
            d_val = dupe[col]
        except IndexError:
            continue
        if not (k_val or "").strip() and (d_val or "").strip():
            updates[col] = d_val
            fields_filled.append(col)
    # Union JSON arrays
    tags_added_count = 0
    for col in _PEOPLE_JSON_UNION:
        try:
            k_raw = keeper[col]
        except IndexError:
            continue
        try:
            d_raw = dupe[col]
        except IndexError:
            continue
        try:
            k = set(json.loads(k_raw or "[]"))
        except Exception:
            k = set()
        try:
            d = set(json.loads(d_raw or "[]"))
        except Exception:
            d = set()
        union = sorted(k | d)
        if union != sorted(k):
            updates[col] = json.dumps(union)
            if col == "tags":
                tags_added_count = len(union) - len(k)
    # Sum counts
    counts_summed = {}
    for col in _PEOPLE_COUNT_SUM:
        try:
            ks = int(keeper[col] or 0)
            ds = int(dupe[col] or 0)
        except (IndexError, TypeError, ValueError):
            continue
        if ds > 0:
            updates[col] = ks + ds
            counts_summed[col] = ks + ds
    # Max date fields (string compare works for YYYY-MM-DD)
    for col in _PEOPLE_DATE_MAX:
        try:
            k_val = keeper[col] or ""
            d_val = dupe[col] or ""
        except IndexError:
            continue
        if d_val > k_val:
            updates[col] = d_val
    # Min created_at
    try:
        if dupe["created_at"] and keeper["created_at"]:
            if dupe["created_at"] < keeper["created_at"]:
                updates["created_at"] = dupe["created_at"]
    except IndexError:
        pass
    # Touch updated/synced (only if the columns exist on this schema)
    now = datetime.now(timezone.utc).isoformat()
    updates["updated_at"] = now
    updates["synced_at"]  = now

    if updates:
        # Filter to columns that actually exist on the people table
        existing_cols = {row[1] for row in cx.execute("PRAGMA table_info(people)").fetchall()}
        safe_updates = {c: v for c, v in updates.items() if c in existing_cols}
        if safe_updates:
            set_clause = ", ".join(f"{c}=?" for c in safe_updates)
            cx.execute(f"UPDATE people SET {set_clause} WHERE id=?",
                       list(safe_updates.values()) + [keeper_id])

    # Delete the dupe row
    cx.execute("DELETE FROM people WHERE id=?", (dupe_id,))

    return {
        "keeper_id": keeper_id,
        "deleted_dupe_id": dupe_id,
        "fields_filled": fields_filled,
        "tags_added": tags_added_count,
        "counts_summed": counts_summed,
    }


def _household_slug(name, head_first_name="", existing=None):
    """URL-safe stable identifier for a household. Immutable after creation
    (renames update name, never slug). Returns lowercase, hyphen-separated."""
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "household"
    if existing is None:
        return base
    if base not in existing:
        return base
    # Collision: try appending head's first name
    if head_first_name:
        candidate = f"{base}-{re.sub(r'[^a-z0-9]+', '-', head_first_name.lower()).strip('-')}"
        if candidate and candidate not in existing:
            return candidate
    # Numeric suffix fallback
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def _candidate_dedup_key(person_ids):
    """Stable dedup key for household_candidates rows. Sorting ensures the
    same cluster produces the same key across detection runs regardless
    of input ordering."""
    return ",".join(str(i) for i in sorted(int(p) for p in person_ids))


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
    # Multi-tag filter (free-text): substring match, all tags AND.
    for tag in [t.strip() for t in params.get("tags", "").split(",") if t.strip()]:
        clauses.append("tags LIKE ?")
        args.append(f"%{tag}%")
    # Contact-type filter: exact JSON-element match (the quotes prevent
    # type:practitioner from substring-matching type:practitioner-cold).
    typ = params.get("type", "").strip()
    if typ:
        clauses.append("tags LIKE ?")
        args.append(f'%"{typ}"%')
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
    # Feeders pass ?merge_tags=1 for additive, idempotent upserts (union tags,
    # never clobber scalars with blanks). The console editor omits it (overwrite).
    merge_tags = request.args.get("merge_tags", "").lower() in ("1", "true", "yes")
    ts = datetime.now(timezone.utc).isoformat()
    inserted = updated = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for p in items:
            email = (p.get("email") or "").strip().lower()
            if not email:
                continue
            if merge_tags:
                res = _upsert_person_additive(cx, p, ts)
                if res == "inserted":
                    inserted += 1
                elif res == "updated":
                    updated += 1
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


# ── Household endpoints ────────────────────────────────────────────────────────

def _check_console_auth():
    """Admin-only — returns None if authorized, or a (response, status) tuple."""
    if not CONSOLE_SECRET:
        return None
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    if key != CONSOLE_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _check_console_or_scoped_auth():
    """Like _check_console_auth but also accepts workspace-scoped tokens
    (e.g. workspace:shaira). Used by routes that should be reachable by
    Shaira via /workspace/shaira as well as Glen/Rae via /console."""
    ok, _ctx, code = _auth()
    if not ok:
        return jsonify({"error": "Unauthorized" if code == 401 else "Forbidden"}), code
    return None


def _existing_household_slugs(cx):
    return {row[0] for row in cx.execute("SELECT slug FROM households").fetchall()}


def _person_household_slug(cx, person_id):
    """Returns the slug of the household this person is in, or None."""
    row = cx.execute("SELECT tags FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return None
    try:
        tags = json.loads(row[0] or "[]")
    except Exception:
        return None
    for t in tags:
        if t.startswith("household:") and not t.startswith("household-head:"):
            return t.split(":", 1)[1]
    return None


def _mutate_person_tags(cx, person_id, add=None, remove=None):
    """Update a person's tags JSON additively/subtractively. Returns new tags list."""
    add = set(add or [])
    remove = set(remove or [])
    row = cx.execute("SELECT tags FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return []
    try:
        existing = set(json.loads(row[0] or "[]"))
    except Exception:
        existing = set()
    new_tags = sorted((existing | add) - remove)
    cx.execute("UPDATE people SET tags=? WHERE id=?", (json.dumps(new_tags), person_id))
    return new_tags


def _push_household_tags_to_ghl(person_email, slug, is_head, action="add"):
    """Push household and household-head tags to GHL. action='add' or 'remove'.
    Returns (ok_bool, error_msg_or_None)."""
    tags = {f"household:{slug}"}
    if is_head:
        tags.add(f"household-head:{slug}")
    if action == "add":
        _, err = ghl_update_tags(person_email, add=tags)
    else:
        _, err = ghl_update_tags(person_email, remove=tags)
    return (err is None, err)


@app.route("/api/households", methods=["POST"])
def create_household():
    """Create a household, tag members in DB + GHL.

    Body: {name, head_person_id, member_person_ids[], address?, notes?, created_by?}
    Returns 200 with the new household, 409 if any member is already in another household."""
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    head_id = body.get("head_person_id")
    member_ids = body.get("member_person_ids") or []
    if not head_id or head_id not in member_ids:
        return jsonify({"error": "head_person_id must be in member_person_ids"}), 400
    created_by = (body.get("created_by") or "glen").strip()
    address = (body.get("address") or "").strip()
    notes = (body.get("notes") or "").strip()
    ts = datetime.now(timezone.utc).isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Pre-flight: ensure no member is already in a household
        for pid in member_ids:
            existing_slug = _person_household_slug(cx, pid)
            if existing_slug:
                existing_row = cx.execute(
                    "SELECT slug, name FROM households WHERE slug=?", (existing_slug,)
                ).fetchone()
                return jsonify({
                    "error": "member already in household",
                    "person_id": pid,
                    "current_household": {
                        "slug": existing_slug,
                        "name": existing_row[1] if existing_row else existing_slug,
                    },
                }), 409

        # Resolve head's first name for slug-collision fallback
        head_row = cx.execute(
            "SELECT first_name, email FROM people WHERE id=?", (head_id,)
        ).fetchone()
        if not head_row:
            return jsonify({"error": "head person not found"}), 400
        head_first, head_email = head_row

        slug = _household_slug(name, head_first, existing=_existing_household_slugs(cx))

        cx.execute("""
            INSERT INTO households (slug, name, head_person_id, address, notes,
                                    created_at, updated_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (slug, name, head_id, address, notes, ts, ts, created_by))
        household_id = cx.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Tag every member in DB. Head gets both household: and household-head:.
        # Also strip the legacy relationship:family-shared-email tag.
        for pid in member_ids:
            adds = {f"household:{slug}"}
            if pid == head_id:
                adds.add(f"household-head:{slug}")
            _mutate_person_tags(cx, pid, add=adds, remove={"relationship:family-shared-email"})
        cx.commit()

    # Push to GHL outside the lock. Per-member errors collected.
    ghl_errors = []
    with sqlite3.connect(LOG_DB) as cx:
        members = cx.execute("""
            SELECT id, email FROM people WHERE id IN ({})
        """.format(",".join("?" * len(member_ids))), member_ids).fetchall()
    for pid, email in members:
        if not email:
            continue
        is_head = (pid == head_id)
        ok, err = _push_household_tags_to_ghl(email, slug, is_head, action="add")
        if not ok:
            ghl_errors.append({"email": email, "error": str(err)})
        # Also remove the legacy tag from GHL
        try:
            ghl_update_tags(email, remove={"relationship:family-shared-email"})
        except Exception:
            pass
        _time.sleep(0.15)

    return jsonify({
        "ok": True,
        "household": {"id": household_id, "slug": slug, "name": name, "head_person_id": head_id},
        "ghl_errors": ghl_errors,
    })


@app.route("/api/households", methods=["GET"])
def list_households():
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("""
            SELECT h.id, h.slug, h.name, h.head_person_id, h.updated_at,
                   p.first_name AS head_first, p.last_name AS head_last
            FROM households h
            LEFT JOIN people p ON p.id = h.head_person_id
            ORDER BY h.name
        """).fetchall()
        out = []
        for r in rows:
            count = cx.execute(
                "SELECT COUNT(*) FROM people WHERE tags LIKE ?", (f'%"household:{r["slug"]}"%',)
            ).fetchone()[0]
            out.append({
                "id": r["id"],
                "slug": r["slug"],
                "name": r["name"],
                "member_count": count,
                "head": {
                    "id": r["head_person_id"],
                    "name": f'{r["head_first"] or ""} {r["head_last"] or ""}'.strip(),
                },
                "updated_at": r["updated_at"],
            })
    return jsonify({"households": out})


@app.route("/api/households/<slug>", methods=["GET"])
def get_household(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM households WHERE slug=?", (slug,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        members = cx.execute("""
            SELECT id, email, first_name, last_name, phone, tags
            FROM people WHERE tags LIKE ?
            ORDER BY first_name
        """, (f'%"household:{slug}"%',)).fetchall()
        member_list = []
        for m in members:
            try:
                tags = json.loads(m["tags"] or "[]")
            except Exception:
                tags = []
            member_list.append({
                "id": m["id"],
                "email": m["email"],
                "first_name": m["first_name"],
                "last_name": m["last_name"],
                "phone": m["phone"],
                "name": f'{m["first_name"]} {m["last_name"]}'.strip(),
                "is_head": f"household-head:{slug}" in tags,
            })
    return jsonify({
        "id": row["id"], "slug": row["slug"], "name": row["name"],
        "head_person_id": row["head_person_id"], "address": row["address"],
        "notes": row["notes"], "created_at": row["created_at"],
        "updated_at": row["updated_at"], "created_by": row["created_by"],
        "members": member_list,
    })


@app.route("/api/people/<int:person_id>/household", methods=["GET"])
def get_person_household(person_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        slug = _person_household_slug(cx, person_id)
        if not slug:
            return jsonify({"household": None})
    # Reuse the full-household renderer, then wrap so callers get {"household": {...}}
    resp = get_household(slug)
    # get_household returns a Flask Response from jsonify; unwrap and re-wrap
    if isinstance(resp, tuple):  # error case (slug somehow missing)
        return resp
    data = resp.get_json()
    return jsonify({"household": data})


@app.route("/api/household-candidates", methods=["GET"])
def list_household_candidates():
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    status = request.args.get("status", "pending")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT * FROM household_candidates WHERE status=? ORDER BY detected_at DESC",
            (status,)
        ).fetchall()
        out = []
        for r in rows:
            try:
                pids = json.loads(r["person_ids"] or "[]")
            except Exception:
                pids = []
            persons = []
            if pids:
                placeholders = ",".join("?" * len(pids))
                people_rows = cx.execute(
                    f"SELECT id, email, first_name, last_name FROM people WHERE id IN ({placeholders})",
                    pids
                ).fetchall()
                persons = [{"id": p["id"], "email": p["email"],
                            "name": f'{p["first_name"]} {p["last_name"]}'.strip()}
                           for p in people_rows]
            out.append({
                "id": r["id"], "signal": r["signal"], "detected_at": r["detected_at"],
                "person_ids": pids, "persons": persons,
            })
    return jsonify({"candidates": out})


@app.route("/api/household-candidates/<int:cand_id>/confirm", methods=["POST"])
def confirm_household_candidate(cand_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    head_id = body.get("head_person_id")
    if not (name and head_id):
        return jsonify({"error": "name + head_person_id required"}), 400

    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT person_ids, status FROM household_candidates WHERE id=?", (cand_id,)).fetchone()
    if not row:
        return jsonify({"error": "candidate not found"}), 404
    if row[1] != "pending":
        return jsonify({"error": f"candidate is {row[1]}, not pending"}), 409
    try:
        member_ids = json.loads(row[0] or "[]")
    except Exception:
        return jsonify({"error": "candidate has invalid person_ids"}), 500

    # Delegate to create_household via internal call
    with app.test_request_context("/api/households", method="POST",
                                   json={"name": name, "head_person_id": head_id,
                                         "member_person_ids": member_ids},
                                   headers={"X-Console-Key": CONSOLE_SECRET or ""}):
        resp = create_household()
    if isinstance(resp, tuple):
        body, status = resp[0], resp[1]
        if status != 200:
            return body, status
        body = body.get_json()
    else:
        body = resp.get_json()

    # Link candidate to the new household
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            UPDATE household_candidates SET status='confirmed',
                resolved_at=?, resolved_by=?, household_id=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), "glen", body["household"]["id"], cand_id))
        cx.commit()
    return jsonify({"ok": True, "household": body["household"], "ghl_errors": body.get("ghl_errors", [])})


@app.route("/api/household-candidates/<int:cand_id>/dismiss", methods=["POST"])
def dismiss_household_candidate(cand_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        if not cx.execute("SELECT 1 FROM household_candidates WHERE id=?", (cand_id,)).fetchone():
            return jsonify({"error": "candidate not found"}), 404
        cx.execute("""
            UPDATE household_candidates SET status='dismissed', resolved_at=?, resolved_by=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), "glen", cand_id))
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/household-candidates/<int:cand_id>/queue-merge", methods=["POST"])
def queue_merge_from_candidate(cand_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    keeper_id = body.get("keeper_person_id")
    if not keeper_id:
        return jsonify({"error": "keeper_person_id required"}), 400

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT person_ids, status FROM household_candidates WHERE id=?",
                          (cand_id,)).fetchone()
        if not row:
            return jsonify({"error": "candidate not found"}), 404
        if row[1] != "pending":
            return jsonify({"error": f"candidate is {row[1]}, not pending"}), 409
        try:
            person_ids = json.loads(row[0] or "[]")
        except Exception:
            return jsonify({"error": "candidate has invalid person_ids"}), 500
        if len(person_ids) != 2:
            return jsonify({"error": "merge requires exactly 2-person candidate",
                            "actual": len(person_ids)}), 400
        if keeper_id not in person_ids:
            return jsonify({"error": "keeper_person_id must be one of the candidate's persons"}), 400
        dupe_id = next(p for p in person_ids if p != keeper_id)

        now = datetime.now(timezone.utc).isoformat()
        cx.execute("""
            INSERT INTO pending_merges (candidate_id, keeper_person_id, dupe_person_id,
                                         queued_at, queued_by, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (cand_id, keeper_id, dupe_id, now, "glen"))
        merge_id = cx.execute("SELECT last_insert_rowid()").fetchone()[0]
        # Mark candidate as dismissed (it's now a pending_merge, not a household candidate)
        cx.execute("UPDATE household_candidates SET status='dismissed', resolved_at=?, resolved_by=? WHERE id=?",
                   (now, "glen-merge-queue", cand_id))
        cx.commit()
    return jsonify({"ok": True, "merge_id": merge_id, "keeper_id": keeper_id, "dupe_id": dupe_id})


@app.route("/api/pending-merges", methods=["GET"])
def list_pending_merges():
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    status = request.args.get("status", "pending")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("SELECT * FROM pending_merges WHERE status=? ORDER BY queued_at DESC",
                           (status,)).fetchall()
        out = []
        for r in rows:
            keeper = cx.execute("SELECT id, email, first_name, last_name FROM people WHERE id=?",
                                 (r["keeper_person_id"],)).fetchone()
            dupe   = cx.execute("SELECT id, email, first_name, last_name FROM people WHERE id=?",
                                 (r["dupe_person_id"],)).fetchone()
            out.append({
                "id": r["id"], "candidate_id": r["candidate_id"],
                "queued_at": r["queued_at"], "queued_by": r["queued_by"],
                "keeper": dict(keeper) if keeper else {"id": r["keeper_person_id"], "deleted": True},
                "dupe":   dict(dupe)   if dupe   else {"id": r["dupe_person_id"], "deleted": True},
            })
    return jsonify({"merges": out})


@app.route("/api/pending-merges/<int:merge_id>/apply", methods=["POST"])
def apply_pending_merge(merge_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT keeper_person_id, dupe_person_id, status FROM pending_merges WHERE id=?",
                          (merge_id,)).fetchone()
        if not row:
            return jsonify({"error": "merge not found"}), 404
        if row[2] != "pending":
            return jsonify({"error": f"merge is {row[2]}, not pending"}), 409
        try:
            result = _merge_two_people(cx, row[0], row[1])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        cx.execute("UPDATE pending_merges SET status='applied', applied_at=? WHERE id=?",
                   (datetime.now(timezone.utc).isoformat(), merge_id))
        cx.commit()
    return jsonify({"ok": True, "result": result})


@app.route("/api/pending-merges/<int:merge_id>/cancel", methods=["POST"])
def cancel_pending_merge(merge_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        if not cx.execute("SELECT 1 FROM pending_merges WHERE id=?", (merge_id,)).fetchone():
            return jsonify({"error": "merge not found"}), 404
        cx.execute("UPDATE pending_merges SET status='cancelled' WHERE id=?", (merge_id,))
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/households/<slug>", methods=["PATCH"])
def update_household(slug):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    ts = datetime.now(timezone.utc).isoformat()

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM households WHERE slug=?", (slug,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404

        new_name    = body.get("name", row["name"])
        new_address = body.get("address", row["address"])
        new_notes   = body.get("notes", row["notes"])
        new_head    = body.get("head_person_id", row["head_person_id"])

        # Head change requires moving the household-head: tag
        old_head = row["head_person_id"]
        head_changed = new_head != old_head
        old_head_email = new_head_email = None
        if head_changed:
            # Validate new head is in this household
            new_head_slug = _person_household_slug(cx, new_head)
            if new_head_slug != slug:
                return jsonify({"error": "new head must be a current member"}), 400
            _mutate_person_tags(cx, old_head, remove={f"household-head:{slug}"})
            _mutate_person_tags(cx, new_head, add={f"household-head:{slug}"})
            r = cx.execute("SELECT email FROM people WHERE id=?", (old_head,)).fetchone()
            old_head_email = r[0] if r else None
            r = cx.execute("SELECT email FROM people WHERE id=?", (new_head,)).fetchone()
            new_head_email = r[0] if r else None

        cx.execute("""
            UPDATE households SET name=?, address=?, notes=?, head_person_id=?, updated_at=?
            WHERE slug=?
        """, (new_name, new_address, new_notes, new_head, ts, slug))
        cx.commit()

    # GHL sync outside lock
    if head_changed:
        if old_head_email:
            _, err = ghl_update_tags(old_head_email, remove={f"household-head:{slug}"})
            if err: ghl_errors.append({"email": old_head_email, "error": str(err)})
            _time.sleep(0.15)
        if new_head_email:
            _, err = ghl_update_tags(new_head_email, add={f"household-head:{slug}"})
            if err: ghl_errors.append({"email": new_head_email, "error": str(err)})
            _time.sleep(0.15)

    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>/members", methods=["POST"])
def add_household_member(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    person_id = body.get("person_id")
    if not person_id:
        return jsonify({"error": "person_id required"}), 400

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        if not cx.execute("SELECT 1 FROM households WHERE slug=?", (slug,)).fetchone():
            return jsonify({"error": "household not found"}), 404
        existing_slug = _person_household_slug(cx, person_id)
        if existing_slug == slug:
            return jsonify({"ok": True, "already_member": True})
        if existing_slug:
            existing_row = cx.execute(
                "SELECT slug, name FROM households WHERE slug=?", (existing_slug,)
            ).fetchone()
            return jsonify({
                "error": "person already in household",
                "current_household": {"slug": existing_slug,
                                       "name": existing_row["name"] if existing_row else existing_slug},
            }), 409
        person_row = cx.execute("SELECT email FROM people WHERE id=?", (person_id,)).fetchone()
        if not person_row:
            return jsonify({"error": "person not found"}), 404
        email = person_row["email"]
        _mutate_person_tags(cx, person_id, add={f"household:{slug}"},
                            remove={"relationship:family-shared-email"})
        cx.commit()

    if email:
        ok, err = _push_household_tags_to_ghl(email, slug, is_head=False, action="add")
        if not ok: ghl_errors.append({"email": email, "error": str(err)})
        try: ghl_update_tags(email, remove={"relationship:family-shared-email"})
        except Exception: pass
    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>/members/<int:person_id>", methods=["DELETE"])
def remove_household_member(slug, person_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        h_row = cx.execute("SELECT head_person_id FROM households WHERE slug=?", (slug,)).fetchone()
        if not h_row:
            return jsonify({"error": "household not found"}), 404
        if h_row["head_person_id"] == person_id:
            return jsonify({"error": "Cannot remove head — change head first"}), 409
        person_row = cx.execute("SELECT email FROM people WHERE id=?", (person_id,)).fetchone()
        if not person_row:
            return jsonify({"error": "person not found"}), 404
        email = person_row["email"]
        _mutate_person_tags(cx, person_id, remove={f"household:{slug}", f"household-head:{slug}"})
        cx.commit()

    if email:
        ok, err = _push_household_tags_to_ghl(email, slug, is_head=False, action="remove")
        if not ok: ghl_errors.append({"email": email, "error": str(err)})
    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>", methods=["DELETE"])
def disband_household(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        if not cx.execute("SELECT 1 FROM households WHERE slug=?", (slug,)).fetchone():
            return jsonify({"error": "household not found"}), 404
        members = cx.execute("""
            SELECT id, email FROM people WHERE tags LIKE ?
        """, (f'%"household:{slug}"%',)).fetchall()
        for m in members:
            _mutate_person_tags(cx, m["id"],
                                remove={f"household:{slug}", f"household-head:{slug}"})
        # Mark related candidates resolved
        cx.execute("""
            UPDATE household_candidates SET status='dismissed', resolved_at=?
            WHERE household_id=(SELECT id FROM households WHERE slug=?)
        """, (datetime.now(timezone.utc).isoformat(), slug))
        cx.execute("DELETE FROM households WHERE slug=?", (slug,))
        cx.commit()

    for m in members:
        if not m["email"]: continue
        _, err = ghl_update_tags(m["email"],
                                  remove={f"household:{slug}", f"household-head:{slug}"})
        if err: ghl_errors.append({"email": m["email"], "error": str(err)})
        _time.sleep(0.15)
    return jsonify({"ok": True, "ghl_errors": ghl_errors})


def _resync_household_to_ghl(slug):
    """Re-push the household tags for every member to GHL. Returns ghl_errors list."""
    ghl_errors = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        h_row = cx.execute("SELECT head_person_id FROM households WHERE slug=?", (slug,)).fetchone()
        if not h_row:
            return [{"error": "household not found"}]
        head_id = h_row["head_person_id"]
        members = cx.execute("""
            SELECT id, email FROM people WHERE tags LIKE ?
        """, (f'%"household:{slug}"%',)).fetchall()
    for m in members:
        if not m["email"]: continue
        is_head = (m["id"] == head_id)
        ok, err = _push_household_tags_to_ghl(m["email"], slug, is_head, action="add")
        if not ok: ghl_errors.append({"email": m["email"], "error": str(err)})
        _time.sleep(0.15)
    return ghl_errors


@app.route("/api/households/<slug>/resync-ghl", methods=["POST"])
def resync_household_ghl(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    errors = _resync_household_to_ghl(slug)
    return jsonify({"ok": True, "ghl_errors": errors})


def resync_all_households_to_ghl():
    """Iterate every household and push its tags to GHL. Used by daily cron
    for drift recovery. Returns {households_synced, ghl_errors_total}."""
    with sqlite3.connect(LOG_DB) as cx:
        slugs = [r[0] for r in cx.execute("SELECT slug FROM households").fetchall()]
    total_errors = 0
    for slug in slugs:
        errors = _resync_household_to_ghl(slug)
        total_errors += len(errors)
    return {"households_synced": len(slugs), "ghl_errors_total": total_errors}


@app.route("/admin/resync-all-households", methods=["POST"])
def admin_resync_all_households():
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        summary = resync_all_households_to_ghl()
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("resync-all-households failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


def detect_household_candidates():
    """Run all signals against the people table, dedup against existing
    household_candidates rows, insert new pending candidates. Returns:
    {detected, new_pending, skipped_already_household, skipped_dedup}."""
    summary = {"detected": 0, "new_pending": 0,
               "skipped_already_household": 0, "skipped_dedup": 0}
    ts = datetime.now(timezone.utc).isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        people = cx.execute("""
            SELECT id, LOWER(TRIM(email)) AS email_lc, LOWER(TRIM(last_name)) AS last_lc,
                   phone, LOWER(TRIM(city)) AS city_lc, LOWER(TRIM(state)) AS state_lc, tags
            FROM people
        """).fetchall()

        # Mark which people are already in a household
        in_household = set()
        for p in people:
            try:
                tags = json.loads(p["tags"] or "[]")
            except Exception:
                tags = []
            if any(t.startswith("household:") and not t.startswith("household-head:") for t in tags):
                in_household.add(p["id"])

        # Existing dedup keys (any status — pending, confirmed, dismissed)
        existing_keys = set()
        for r in cx.execute("SELECT person_ids FROM household_candidates").fetchall():
            try:
                ids = json.loads(r[0] or "[]")
            except Exception:
                continue
            existing_keys.add(_candidate_dedup_key(ids))

        # ── Signal 1: shared-email ────────────────────────────────────────────
        by_email = {}
        for p in people:
            if not p["email_lc"]: continue
            by_email.setdefault(p["email_lc"], []).append(p["id"])
        # ── Signal 2: shared-phone-lastname ───────────────────────────────────
        by_phone_last = {}
        for p in people:
            if not (p["phone"] and p["last_lc"]): continue
            by_phone_last.setdefault((p["phone"], p["last_lc"]), []).append(p["id"])
        # ── Signal 3: shared-address-lastname ─────────────────────────────────
        by_addr = {}
        for p in people:
            if not (p["city_lc"] and p["state_lc"] and p["last_lc"]): continue
            by_addr.setdefault((p["city_lc"], p["state_lc"], p["last_lc"]), []).append(p["id"])

        def _emit_signal(name, clusters):
            for ids in clusters.values():
                if len(ids) < 2: continue
                summary["detected"] += 1
                if any(i in in_household for i in ids):
                    summary["skipped_already_household"] += 1
                    continue
                key = _candidate_dedup_key(ids)
                if key in existing_keys:
                    summary["skipped_dedup"] += 1
                    continue
                cx.execute("""
                    INSERT INTO household_candidates (detected_at, signal, person_ids, status)
                    VALUES (?, ?, ?, 'pending')
                """, (ts, name, json.dumps(sorted(ids))))
                existing_keys.add(key)   # avoid intra-run dups across signals
                summary["new_pending"] += 1

        _emit_signal("shared-email",            by_email)
        _emit_signal("shared-phone-lastname",   by_phone_last)
        _emit_signal("shared-address-lastname", by_addr)
        cx.commit()
    return summary


@app.route("/admin/detect-household-candidates", methods=["POST"])
def admin_detect_household_candidates():
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        summary = detect_household_candidates()
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("detect_household_candidates failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ── Console AI chat (context-aware) ───────────────────────────────────────────
_OWNER_DESC = {
    "glen":   "Dr. Glen Swartwout, naturopathic optometrist, solopreneur — full access to all systems and data.",
    "rae":    "Rae (Susan Luscombe), business owner and operations partner — full access, handles orders/fulfillment/finance/scheduling.",
    "shaira": "Shaira, technical VA — focused on implementation tasks, GHL/tech integrations.",
}

def _justus_system_prompt(owner: str, extra_context: str = "", extra_directives: str = "") -> str:
    """Build Justus's system prompt for a given owner. Reused by /api/console-ask
    and the per-item workspace thread endpoint."""
    s = (
        f"You are Justus, the AI assistant in the Remedy Match business console. "
        f"You are speaking with: {_OWNER_DESC.get(owner, owner)}\n"
        f"Be concise and action-oriented. Use bullet points for lists. "
        f"Answer questions about clients, business, health protocols, operations, or anything relevant.\n"
    )
    if extra_context:
        s += f"\nCurrent context:\n{extra_context}"
    if extra_directives:
        s += f"\n{extra_directives}"
    return s


def _ask_justus_stream(query: str, system: str, history: list, on_complete=None, history_n: int = 8):
    """SSE generator that streams Justus's reply and (optionally) calls
    on_complete(full_text) once the stream is exhausted."""
    msgs = []
    for h in (history or [])[-history_n:]:
        msgs.append({"role": h.get("role","user"), "content": h.get("content","")})
    msgs.append({"role": "user", "content": query})

    parts: list[str] = []
    try:
        with _cl.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system,
            messages=msgs,
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield f"data: {json.dumps({'text': text})}\n\n"
    except Exception as e:
        app.logger.exception("Justus stream failed")
        yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
        return
    yield f"data: {json.dumps({'done': True})}\n\n"
    if on_complete:
        try:
            on_complete("".join(parts))
        except Exception:
            app.logger.exception("Justus on_complete callback failed")


# ── Tracker tool-use — Justus can edit projects from /console/projects ────────

PROJECT_TOOLS = [
    {
        "name": "add_idea",
        "description": "Add a new idea to the project tracker's Ideas section.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string",
                                     "description": "The idea — 1-500 chars."}},
            "required": ["text"],
        },
    },
    {
        "name": "move_project",
        "description": "Move an existing project to a different section.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Project name (substring match against the bolded name)."},
                "target": {"type": "string",
                            "enum": ["in_process","queued","planning","ideas","completed"]},
            },
            "required": ["name", "target"],
        },
    },
    {
        "name": "set_project_field",
        "description": "Set a field on an existing project (status, effort, value, eta, blockers, where).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name":  {"type": "string"},
                "field": {"type": "string",
                          "description": "Field name, e.g. 'effort', 'value', 'status', 'eta', 'blockers'."},
                "value": {"type": "string"},
            },
            "required": ["name", "field", "value"],
        },
    },
    {
        "name": "drop_project",
        "description": "Remove a project from the tracker entirely.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
]

TODO_TOOLS = [
    {
        "name": "list_todos",
        "description": (
            "List OPEN todos for an owner so you can find the numeric id of one "
            "the user wants to act on. ALWAYS call this first when the user "
            "refers to a todo by description (e.g. 'the Lotika one') rather than "
            "by id. Returns lines like '#42 [high] E4L — Client Messages: title…'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner":    {"type": "string", "enum": ["glen", "rae", "shaira"]},
                "category": {"type": "string",
                             "description": "Optional category substring filter, e.g. 'E4L' or 'Payments'."},
                "limit":    {"type": "integer", "default": 20},
            },
            "required": ["owner"],
        },
    },
    {
        "name": "complete_todo",
        "description": "Mark a todo as done (status='done'). Reversible.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "delegate_todo",
        "description": (
            "Delegate a todo to another team member. Creates a copy on the "
            "delegate's tab and marks the original as 'delegated'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id":   {"type": "integer"},
                "to":   {"type": "string", "enum": ["glen", "rae", "shaira", "justus"]},
                "note": {"type": "string",
                          "description": "Optional context note appended to the delegate's copy."},
            },
            "required": ["id", "to"],
        },
    },
    {
        "name": "dismiss_todo",
        "description": (
            "Dismiss a todo (status='dismissed'). Use for items that aren't "
            "actionable — spam, duplicates, no-longer-relevant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "draft_todo_reply",
        "description": (
            "Generate a draft reply (via Claude) for an actionable email-derived "
            "todo, typically E4L client messages. Returns the draft text — the "
            "user still has to send it via the original channel (Practice Better, "
            "Gmail, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id":       {"type": "integer"},
                "guidance": {"type": "string",
                              "description": "Optional steering text for the draft."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "add_todo",
        "description": "Create a new todo to capture a follow-up without going through Gmail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":    {"type": "string"},
                "owner":    {"type": "string", "enum": ["glen", "rae", "shaira"]},
                "priority": {"type": "string", "enum": ["high", "normal", "low"]},
                "category": {"type": "string"},
                "body":     {"type": "string"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "split_capture",
        "description": (
            "Split a free-form brain-dump into N discrete todos and insert them. "
            "Use when the user gives a long unstructured message — especially "
            "anything starting with [capture] — instead of trying to manually "
            "parse it into items yourself. Returns the count + titles + "
            "priorities. Todos land in category='Idea', source='capture'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text":  {"type": "string",
                          "description": "The free-form capture text. [capture] prefix is optional."},
                "owner": {"type": "string", "enum": ["glen", "rae", "shaira"]},
            },
            "required": ["text"],
        },
    },
]

HOUSEHOLD_TOOLS = [
    {
        "name": "list_household_candidates",
        "description": (
            "List PENDING household candidates — auto-detected clusters of "
            "people that might live together (same last name + address, or "
            "shared email). Each candidate has a numeric id and a list of "
            "person objects {id, name, email}. Call FIRST before "
            "confirm/dismiss/queue_household_merge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "confirm_household_candidate",
        "description": (
            "Confirm a candidate as a real household — creates a `households` "
            "row + tags all members in DB and GHL. Requires the household "
            "name (e.g. 'Savant Family') and which person is the head "
            "(head_person_id, must be one of the candidate's persons)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id":   {"type": "integer"},
                "name":           {"type": "string",
                                    "description": "Household name, e.g. 'Savant Family'."},
                "head_person_id": {"type": "integer",
                                    "description": "Numeric id of the person to designate as household head."},
            },
            "required": ["candidate_id", "name", "head_person_id"],
        },
    },
    {
        "name": "dismiss_household_candidate",
        "description": (
            "Dismiss a household candidate (not actually a household — e.g. "
            "two unrelated people who happen to share a feature). Use when "
            "the candidate is neither a household nor a duplicate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"candidate_id": {"type": "integer"}},
            "required": ["candidate_id"],
        },
    },
    {
        "name": "queue_household_merge",
        "description": (
            "Convert a 2-person household candidate into a pending_merge "
            "(they're actually duplicates of the same person, not a "
            "household). Requires keeper_person_id (which person record to "
            "preserve). Goes to the pending-merges queue for review — does "
            "NOT immediately merge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id":     {"type": "integer"},
                "keeper_person_id": {"type": "integer",
                                      "description": "Which person record to preserve (the other gets merged into it)."},
            },
            "required": ["candidate_id", "keeper_person_id"],
        },
    },
    {
        "name": "list_pending_merges",
        "description": (
            "List queued duplicate-person merges awaiting apply or cancel. "
            "Each merge has keeper + dupe person details. Call FIRST before "
            "apply_pending_merge or cancel_pending_merge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "apply_pending_merge",
        "description": (
            "EXECUTE a queued merge — coalesces dupe's fields into keeper, "
            "unions JSON arrays (tags etc.), sums counts (orders, sessions), "
            "then DELETES the dupe person row. Irreversible. Call only after "
            "the user has explicitly confirmed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"merge_id": {"type": "integer"}},
            "required": ["merge_id"],
        },
    },
    {
        "name": "cancel_pending_merge",
        "description": "Cancel a queued merge (don't apply it). Reversible — does not touch person rows.",
        "input_schema": {
            "type": "object",
            "properties": {"merge_id": {"type": "integer"}},
            "required": ["merge_id"],
        },
    },
]

TRACKER_DIRECTIVES = (
    "TOOLS available — call when the user asks for an action:\n"
    "\nPROJECTS tracker:\n"
    "  add_idea / move_project / set_project_field / drop_project\n"
    "  Sections: in_process / queued / planning / ideas / completed.\n"
    "  Fields: status, where, eta, blockers, effort (1-5 stars; pass '1'–'5' or '★★★'), "
    "value ($/$$/$$$/$$$$), sessions.\n"
    "  Project edits queue and apply within ~10 min via the Mac sync.\n"
    "\nTODOS (the /console tabs):\n"
    "  list_todos (call FIRST to find an id when the user names a todo by description)\n"
    "  complete_todo / delegate_todo / dismiss_todo / draft_todo_reply / add_todo\n"
    "  split_capture (for [capture]-prefixed dumps or any unstructured multi-item input — "
    "don't parse manually, hand the whole text to this tool)\n"
    "  Owners: glen / rae / shaira. Todo actions apply immediately to the DB.\n"
    "\nHOUSEHOLDS + PEOPLE-MERGES (the /console people-merge queue):\n"
    "  list_household_candidates / confirm_household_candidate / dismiss_household_candidate /\n"
    "  queue_household_merge / list_pending_merges / apply_pending_merge / cancel_pending_merge\n"
    "  Two-stage: candidates → (confirm as household | dismiss | queue_household_merge → pending_merges → apply).\n"
    "  apply_pending_merge is IRREVERSIBLE — confirm with the user before calling.\n"
    "\nAfter a tool call, confirm what you did in one short line."
)


def _execute_project_tool(name: str, inp: dict) -> str:
    """Map a Justus tool call to a queued tracker edit."""
    try:
        if name == "add_idea":
            r = _projects.add_pending_edit({"type": "add_idea", "text": inp.get("text", "")})
            return f"Queued idea (id {r['id']})."
        if name == "move_project":
            r = _projects.add_pending_edit({"type": "move",
                                            "name":   inp.get("name", ""),
                                            "target": inp.get("target", "")})
            return f"Queued move (id {r['id']})."
        if name == "set_project_field":
            r = _projects.add_pending_edit({"type":  "set",
                                            "name":  inp.get("name", ""),
                                            "field": inp.get("field", ""),
                                            "value": inp.get("value", "")})
            return f"Queued field update (id {r['id']})."
        if name == "drop_project":
            r = _projects.add_pending_edit({"type": "drop", "name": inp.get("name", "")})
            return f"Queued drop (id {r['id']})."
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


_PROJECT_TOOL_NAMES = {"add_idea", "move_project", "set_project_field", "drop_project"}
_HOUSEHOLD_TOOL_NAMES = {
    "list_household_candidates", "confirm_household_candidate",
    "dismiss_household_candidate", "queue_household_merge",
    "list_pending_merges", "apply_pending_merge", "cancel_pending_merge",
}


def _call_route(route_fn, path: str, method: str = "POST",
                json_body: dict | None = None, **kwargs):
    """Invoke an existing Flask route handler via test_request_context.
    Returns (json_dict, status_code). kwargs are passed to the handler call.

    Pass the CURRENT caller's X-Console-Key through to the nested handler so
    workspace-scoped tokens (e.g. Shaira) keep their identity rather than
    silently escalating to admin. Falls back to CONSOLE_SECRET only when
    invoked outside an HTTP request context."""
    try:
        caller_key = request.headers.get("X-Console-Key", "")
    except RuntimeError:
        caller_key = ""
    caller_key = caller_key or CONSOLE_SECRET or ""
    with app.test_request_context(
        path, method=method,
        json=json_body or {},
        headers={"X-Console-Key": caller_key},
    ):
        resp = route_fn(**kwargs)
    # Flask handlers return either Response or (Response, status) tuple
    if isinstance(resp, tuple):
        body, status = resp[0], resp[1]
    else:
        body, status = resp, 200
    try:
        return body.get_json(), status
    except Exception:
        return {"raw": str(body)}, status


def _execute_household_tool(name: str, inp: dict) -> str:
    """Map a household/merge tool call to the existing route handlers."""
    try:
        if name == "list_household_candidates":
            limit = max(1, min(int(inp.get("limit") or 20), 50))
            data, _ = _call_route(list_household_candidates,
                                   "/api/household-candidates", method="GET")
            cands = (data.get("candidates") or [])[:limit]
            if not cands:
                return "No pending household candidates."
            lines = []
            for c in cands:
                persons = ", ".join(f"{p['name']} (#{p['id']}, {p.get('email','-')})"
                                    for p in c["persons"])
                lines.append(f"#{c['id']} signal='{c['signal']}' detected={c['detected_at'][:10]} "
                             f"persons=[{persons}]")
            return f"{len(cands)} pending household candidate(s):\n" + "\n".join(lines)

        if name == "confirm_household_candidate":
            cid = int(inp["candidate_id"])
            data, status = _call_route(
                confirm_household_candidate,
                f"/api/household-candidates/{cid}/confirm", method="POST",
                json_body={"name": inp["name"], "head_person_id": int(inp["head_person_id"])},
                cand_id=cid,
            )
            if status != 200:
                return f"Confirm failed (HTTP {status}): {data.get('error','?')}"
            hh = data.get("household", {})
            ghl_errs = data.get("ghl_errors", []) or []
            extra = f" (GHL errors: {ghl_errs})" if ghl_errs else ""
            return f"Confirmed candidate #{cid} → household '{hh.get('name')}' (slug={hh.get('slug')}){extra}"

        if name == "dismiss_household_candidate":
            cid = int(inp["candidate_id"])
            data, status = _call_route(
                dismiss_household_candidate,
                f"/api/household-candidates/{cid}/dismiss", method="POST",
                cand_id=cid,
            )
            if status != 200:
                return f"Dismiss failed (HTTP {status}): {data.get('error','?')}"
            return f"Dismissed candidate #{cid}."

        if name == "queue_household_merge":
            cid    = int(inp["candidate_id"])
            keeper = int(inp["keeper_person_id"])
            data, status = _call_route(
                queue_merge_from_candidate,
                f"/api/household-candidates/{cid}/queue-merge", method="POST",
                json_body={"keeper_person_id": keeper},
                cand_id=cid,
            )
            if status != 200:
                return f"Queue-merge failed (HTTP {status}): {data.get('error','?')}"
            return (f"Queued merge #{data.get('merge_id')} from candidate #{cid} "
                    f"(keeper #{data.get('keeper_id')}, dupe #{data.get('dupe_id')}).")

        if name == "list_pending_merges":
            limit = max(1, min(int(inp.get("limit") or 20), 50))
            data, _ = _call_route(list_pending_merges, "/api/pending-merges", method="GET")
            merges = (data.get("merges") or [])[:limit]
            if not merges:
                return "No pending merges."
            lines = []
            for m in merges:
                k = m["keeper"]; d = m["dupe"]
                k_name = f"{k.get('first_name','')} {k.get('last_name','')}".strip() or k.get("email","?")
                d_name = f"{d.get('first_name','')} {d.get('last_name','')}".strip() or d.get("email","?")
                lines.append(f"#{m['id']} keeper=#{k['id']} {k_name} ({k.get('email','-')}) "
                             f"← dupe=#{d['id']} {d_name} ({d.get('email','-')}) "
                             f"queued={m['queued_at'][:10]}")
            return f"{len(merges)} pending merge(s):\n" + "\n".join(lines)

        if name == "apply_pending_merge":
            mid = int(inp["merge_id"])
            data, status = _call_route(
                apply_pending_merge,
                f"/api/pending-merges/{mid}/apply", method="POST",
                merge_id=mid,
            )
            if status != 200:
                return f"Apply failed (HTTP {status}): {data.get('error','?')}"
            r = data.get("result", {})
            return (f"Applied merge #{mid}: kept person #{r.get('keeper_id')}, "
                    f"deleted dupe #{r.get('deleted_dupe_id')}. "
                    f"fields_filled={r.get('fields_filled',[])}, "
                    f"tags_added={r.get('tags_added',0)}, "
                    f"counts_summed={r.get('counts_summed',{})}.")

        if name == "cancel_pending_merge":
            mid = int(inp["merge_id"])
            data, status = _call_route(
                cancel_pending_merge,
                f"/api/pending-merges/{mid}/cancel", method="POST",
                merge_id=mid,
            )
            if status != 200:
                return f"Cancel failed (HTTP {status}): {data.get('error','?')}"
            return f"Cancelled merge #{mid}."

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error in {name}: {type(e).__name__}: {e}"


def _execute_todo_tool(name: str, inp: dict) -> str:
    """Direct in-process SQL — mirrors the same writes the REST handlers use."""
    try:
        if name == "list_todos":
            owner    = (inp.get("owner") or "glen").lower()
            category = (inp.get("category") or "").strip()
            limit    = max(1, min(int(inp.get("limit") or 20), 50))
            with sqlite3.connect(LOG_DB) as cx:
                rows = cx.execute("""
                    SELECT id, title, priority, category, created_at
                    FROM todos
                    WHERE owner=? AND status='open'
                    ORDER BY
                        CASE priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                        created_at DESC
                """, (owner,)).fetchall()
            if category:
                needle = category.lower()
                rows = [r for r in rows if needle in (r[3] or "").lower()]
            rows = rows[:limit]
            if not rows:
                hint = f" matching '{category}'" if category else ""
                return f"No open todos for {owner}{hint}."
            lines = [f"#{r[0]} [{r[2] or 'normal'}] {r[3] or 'General'}: {r[1]}" for r in rows]
            return "Open todos:\n" + "\n".join(lines)

        if name == "complete_todo":
            tid = int(inp["id"])
            ts  = datetime.now(timezone.utc).isoformat()
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                row = cx.execute("SELECT title FROM todos WHERE id=?", (tid,)).fetchone()
                if not row:
                    return f"No todo with id {tid}."
                cx.execute("UPDATE todos SET status='done', done_at=? WHERE id=?", (ts, tid))
                cx.commit()
            return f"Completed #{tid}: {row[0]}"

        if name == "delegate_todo":
            tid  = int(inp["id"])
            to   = (inp.get("to") or "").lower()
            note = (inp.get("note") or "").strip()
            if to not in ("glen", "rae", "shaira", "justus"):
                return f"Invalid delegate target: {to!r}"
            ts = datetime.now(timezone.utc).isoformat()
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                row = cx.execute(
                    "SELECT owner, category, title, body, priority, source, ai_summary, suggested_reply "
                    "FROM todos WHERE id=?", (tid,)
                ).fetchone()
                if not row:
                    return f"No todo with id {tid}."
                cx.execute(
                    "UPDATE todos SET status='delegated', delegated_to=?, delegated_at=? WHERE id=?",
                    (to, ts, tid)
                )
                new_title = f"[From {row[0].title()}] {row[2]}"
                extra_body = f"\n\n📝 Note: {note}" if note else ""
                cx.execute("""
                    INSERT INTO todos (created_at, owner, category, title, body, priority, source,
                                       ai_summary, suggested_reply)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (ts, to, row[1], new_title, (row[3] or "") + extra_body, row[4],
                      row[5], row[6], row[7]))
                cx.commit()
            return f"Delegated #{tid} to {to}: {row[2]}"

        if name == "dismiss_todo":
            tid = int(inp["id"])
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                row = cx.execute("SELECT title FROM todos WHERE id=?", (tid,)).fetchone()
                if not row:
                    return f"No todo with id {tid}."
                cx.execute("UPDATE todos SET status='dismissed' WHERE id=?", (tid,))
                cx.commit()
            return f"Dismissed #{tid}: {row[0]}"

        if name == "draft_todo_reply":
            tid      = int(inp["id"])
            guidance = (inp.get("guidance") or "").strip()
            with sqlite3.connect(LOG_DB) as cx:
                row = cx.execute("SELECT title, body, category FROM todos WHERE id=?",
                                 (tid,)).fetchone()
            if not row:
                return f"No todo with id {tid}."
            title, body, _category = row
            guidance_block = f"\n\nGlen's guidance: {guidance}" if guidance else ""
            prompt = (
                "You are drafting a reply on behalf of Dr. Glen Swartwout, naturopathic "
                "physician and biofield scientist in Hilo, Hawaiʻi. Be warm, concise, and "
                "professional. Sign off naturally as Dr. Glen.\n\n"
                f"Email subject: {title}\n"
                f"Email content:\n{(body or '')[:2000]}"
                f"{guidance_block}\n\n"
                "Draft the reply now:"
            )
            try:
                msg = _cl.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                draft = msg.content[0].text
            except Exception as e:
                return f"Draft generation failed: {e}"
            return f"Draft for #{tid}:\n\n{draft}"

        if name == "add_todo":
            title = (inp.get("title") or "").strip()
            if not title:
                return "Title required."
            owner    = (inp.get("owner") or "glen").lower()
            category = (inp.get("category") or "General")
            priority = (inp.get("priority") or "normal")
            body     = (inp.get("body") or "")
            ts = datetime.now(timezone.utc).isoformat()
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cur = cx.execute("""
                    INSERT INTO todos (created_at, owner, category, title, body, priority, source)
                    VALUES (?,?,?,?,?,?,?)
                """, (ts, owner, category, title, body, priority, "justus"))
                cx.commit()
                new_id = cur.lastrowid
            return f"Added #{new_id} ({owner}, {priority}): {title}"

        if name == "split_capture":
            text  = (inp.get("text") or "").strip()
            owner = (inp.get("owner") or "glen").lower()
            if not text:
                return "Capture text required."
            try:
                result = _do_capture_split(text, owner)
            except ValueError as e:
                return f"Split failed: {e}"
            except Exception as e:
                return f"Split failed: {type(e).__name__}: {e}"
            items = result.get("items", [])
            if not items:
                return "Split produced no items (capture may have been too short or duplicated)."
            lines = [f"  • [{it['priority']}] {it['title']}" for it in items]
            return f"Split into {result['count']} todos ({owner}):\n" + "\n".join(lines)

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error in {name}: {e}"


def _execute_console_tool(name: str, inp: dict) -> str:
    """Dispatch to projects / households-merges / todos based on tool name."""
    if name in _PROJECT_TOOL_NAMES:
        return _execute_project_tool(name, inp)
    if name in _HOUSEHOLD_TOOL_NAMES:
        return _execute_household_tool(name, inp)
    return _execute_todo_tool(name, inp)


import dashboard.justus_adapter as _ja


def _register_justus_actions():
    """Register each Justus WRITE tool as a BOS action whose executor wraps the
    existing _execute_console_tool, so dispatch_action audits + governs it."""
    from dashboard.actions import action as _act, get_action as _get
    from dashboard import rbac as _bos_rbac
    for _name, (_key, _module, _tier) in _ja.JUSTUS_WRITE_ACTIONS.items():
        if _get(_key):
            continue

        def _make(nm):
            def _exec(params, ctx):
                return {"message": _execute_console_tool(nm, params or {})}
            return _exec

        _act(key=_key, module=_module, title=_key,
             description=f"Justus action: {_name}", risk_tier=_tier,
             permission=(_bos_rbac.OWNER, _bos_rbac.OPS, _bos_rbac.VA))(_make(_name))


_register_justus_actions()


def _justus_tool_dispatch(actor):
    """Return tool_dispatch(name, input)->str for _ask_justus_stream_tools.
    READ tools run direct; WRITE tools go through dispatch_action (audit + policy)."""
    def dispatch(name, inp):
        inp = inp or {}
        if _ja.is_read(name):
            return _execute_console_tool(name, inp)
        key = _ja.action_key_for(name)
        if not key:
            return _execute_console_tool(name, inp)
        params = dict(inp)
        if name == "complete_todo" and "id" in params:
            params["todo_id"] = params.pop("id")
        cx = _sqlite3.connect(LOG_DB)
        cx.row_factory = _sqlite3.Row
        try:
            res = _bos_dispatch.dispatch_action(
                cx, key, params, actor, source="justus",
                confirmed=(actor.role in (_bos_rbac.OWNER, _bos_rbac.OPS)))
        finally:
            cx.close()
        return _ja.format_justus_result(name, res)
    return dispatch


# Tools whose result text should be shown to the user inline (Justus's natural-
# language reply alone would hide the actual value — e.g. a drafted email body).
_VISIBLE_TOOL_RESULTS = {"draft_todo_reply", "split_capture", "apply_pending_merge"}


def _ask_justus_stream_tools(query: str, system: str, history: list, tools: list,
                              tool_dispatch, on_complete=None,
                              history_n: int = 6, max_iters: int = 4):
    """SSE generator with a multi-iteration tool-use loop. Streams text from
    each model turn; when a turn ends with tool_use blocks, executes them and
    starts the next turn with the tool_results appended. Tools in
    _VISIBLE_TOOL_RESULTS also emit a `tool_result` SSE event so the widget
    can render their output verbatim (not just Justus's summary)."""
    msgs = []
    for h in (history or [])[-history_n:]:
        msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    msgs.append({"role": "user", "content": query})

    full_text: list[str] = []
    try:
        for _ in range(max_iters):
            with _cl.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=system,
                tools=tools,
                messages=msgs,
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    yield f"data: {json.dumps({'text': text})}\n\n"
                final = stream.get_final_message()

            tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break
            # Build clean content blocks for the assistant turn we're echoing
            # back. `b.model_dump()` on a text block includes SDK-side fields
            # (e.g. parsed_output) that the API rejects on input, so build
            # minimal dicts per block type.
            assistant_content = []
            for b in final.content:
                bt = getattr(b, "type", None)
                if bt == "text":
                    assistant_content.append({"type": "text", "text": b.text})
                elif bt == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id":   b.id,
                        "name": b.name,
                        "input": getattr(b, "input", {}) or {},
                    })
                else:
                    assistant_content.append(b.model_dump())
            msgs.append({"role": "assistant", "content": assistant_content})
            tool_results = []
            for tu in tool_uses:
                result = tool_dispatch(tu.name, getattr(tu, "input", {}) or {})
                if tu.name in _VISIBLE_TOOL_RESULTS:
                    yield ("data: " + json.dumps(
                        {"tool_result": {"name": tu.name, "content": result}}
                    ) + "\n\n")
                tool_results.append({"type": "tool_result",
                                      "tool_use_id": tu.id,
                                      "content": result})
            msgs.append({"role": "user", "content": tool_results})
    except Exception as e:
        app.logger.exception("Justus tool-stream failed")
        yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
        return

    yield f"data: {json.dumps({'done': True})}\n\n"
    if on_complete:
        try: on_complete("".join(full_text))
        except Exception: app.logger.exception("Justus on_complete callback failed")


@app.route("/api/console-ask", methods=["POST"])
def console_ask():
    if CONSOLE_SECRET:
        ok, ctx, code = _auth()
        if not ok:
            return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    else:
        ctx = {"scope": "admin"}
    data    = request.get_json(force=True) or {}
    query   = (data.get("query") or "").strip()
    requested_owner = (data.get("owner") or "glen").lower()
    # Scoped tokens are forced to their own owner identity — prevents Shaira from
    # asking Justus "as Glen" and bypassing the per-user context boundary.
    owner   = _scoped_owner(ctx, requested_owner)
    context = (data.get("context") or "")
    history = data.get("history") or []
    page    = (data.get("page") or "")
    if not query:
        return jsonify({"error":"No query"}), 400
    # Always enable tracker + todo tools — Justus follows intent, not URL.
    # Pass the current page so Justus can reason about what's most relevant.
    page_ctx = f"\nCurrent page: {page}" if page else ""
    system = _justus_system_prompt(owner, (context + page_ctx).strip(), TRACKER_DIRECTIVES)
    _actor = _bos_rbac.actor_for_scope((ctx or {}).get("scope", "admin"), owner)
    gen = _ask_justus_stream_tools(query, system, history,
                                    PROJECT_TOOLS + TODO_TOOLS + HOUSEHOLD_TOOLS,
                                    _justus_tool_dispatch(_actor), history_n=6)
    return Response(stream_with_context(gen),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── Capture splitter: [capture] free-text → N discrete todos ──────────────────
_CAPTURE_SPLIT_SYSTEM = (
    "You split a free-form capture into discrete actionable items for a personal "
    "task inbox. Return ONLY a JSON array (no prose, no code fences) of objects "
    "with keys: title (≤80 chars, imperative), body (one sentence of context, "
    "may be empty), priority (one of: low, normal, high). If the capture "
    "describes a single idea, return a one-element array. Preserve the "
    "speaker's intent — don't invent items. Output JSON only."
)


def _do_capture_split(text: str, owner: str) -> dict:
    """Split free-form text into todos and insert them under `owner`.
    Used by /api/capture-split (HTTP) and the split_capture Justus tool.
    Returns {"items": [...], "count": N}; raises ValueError on bad input."""
    stripped = (text or "").strip()
    if stripped.lower().startswith("[capture]"):
        stripped = stripped[len("[capture]"):].strip()
    if not stripped:
        raise ValueError("Capture body empty")

    resp = _cl.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=_CAPTURE_SPLIT_SYSTEM,
        messages=[{"role": "user", "content": stripped}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    items = json.loads(raw)
    if not isinstance(items, list) or not items:
        raise ValueError("Splitter returned no items")

    ts = datetime.now(timezone.utc).isoformat()
    ts_epoch = int(_time.time())
    inserted = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        for i, it in enumerate(items):
            title    = (it.get("title") or "").strip()[:200]
            body     = (it.get("body") or "").strip()
            priority = (it.get("priority") or "normal").lower()
            if priority not in ("low", "normal", "high"):
                priority = "normal"
            if not title:
                continue
            dedup = f"capture:{ts_epoch}:{i}:{title[:40]}"
            try:
                cx.execute("""
                    INSERT INTO todos
                      (created_at, owner, category, title, body, priority, source, dedup_key)
                    VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(dedup_key) DO NOTHING
                """, (ts, owner, "Idea", title, body, priority, "capture", dedup))
                if cx.execute("SELECT changes()").fetchone()[0]:
                    inserted.append({"title": title, "priority": priority})
            except Exception:
                app.logger.exception("capture_split insert failed for item %s", i)
        cx.commit()
    return {"items": inserted, "count": len(inserted)}


@app.route("/api/capture-split", methods=["POST"])
def capture_split():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    data  = request.get_json(force=True) or {}
    text  = (data.get("text") or "").strip()
    owner = (data.get("owner") or "glen").lower()
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        result = _do_capture_split(text, owner)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.exception("capture_split failed")
        return jsonify({"ok": False, "error": f"split failed: {type(e).__name__}: {e}"}), 500
    return jsonify({"ok": True, **result}), 201


# ── Workspace (per-owner focused-item page) ───────────────────────────────────
_SHAIRA_DIRECTIVES = (
    "\nSpecial guidance for Shaira:\n"
    "- If she uses stuck-language ('still working on…', 'deepening my understanding of…', "
    "'reviewing how X should work…') without naming a specific blocker, ask: "
    "'It sounds like you may have hit a snag — what specifically is stopping you?'\n"
    "- If she asks something requiring authority (priority order, publish/no-publish, "
    "contact-eligibility, business decision) and the answer isn't in available context, "
    "end your reply with [ASK:glen] (or [ASK:rae]) followed by a clearly-worded question. "
    "The server routes the question to that person's console.\n"
    "- Stay focused on the current task. Don't open new threads unless she asks.\n"
)


def _ws_auth_ok():
    """Legacy admin-only check. Kept for endpoints that must remain admin-only
    (cron, backup, token issuance). New workspace endpoints use _auth()."""
    if not CONSOLE_SECRET:
        return True
    key = request.headers.get("X-Console-Key","") or request.args.get("key","")
    return key == CONSOLE_SECRET


# ── Per-user access tokens (Phase 2) ──────────────────────────────────────────
# Tokens supersede CONSOLE_SECRET for offshore/per-user access. Admin key
# (CONSOLE_SECRET) continues to have full access for the existing /console flow.

def _auth(required_scope=None):
    """Authenticate + (optionally) check scope.

    Returns (ok: bool, ctx: dict | None, code: int).
      - ctx is None when unauthenticated, otherwise has:
          scope:     'admin' | 'workspace:<owner>'
          user_name: e.g. 'shaira' | None (None for admin)
          user_id:   row id in workspace_users | None for admin
      - code is the HTTP status to return on failure (401 unauth / 403 forbidden).

    If required_scope is set, the caller's scope must equal it OR be 'admin'.
    """
    key = (request.headers.get("X-Console-Key","")
           or request.args.get("key","")).strip()
    if not key:
        return False, None, 401

    # Admin = CONSOLE_SECRET
    if CONSOLE_SECRET and key == CONSOLE_SECRET:
        ctx = {"scope": "admin", "user_name": None, "user_id": None}
        if required_scope is None or required_scope == "admin" or required_scope.startswith("workspace:"):
            return True, ctx, 200
        return False, ctx, 403

    # Per-user access token
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT u.id, u.name, u.scope "
                "FROM access_tokens t JOIN workspace_users u ON u.id = t.user_id "
                "WHERE t.token = ? AND t.revoked_at IS NULL",
                (key,)
            ).fetchone()
    except Exception:
        return False, None, 401
    if not row:
        return False, None, 401
    user_id, user_name, scope = row
    ctx = {"scope": scope, "user_name": user_name, "user_id": user_id}

    # Best-effort last_used_at touch (don't hold the request open if it fails)
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("UPDATE access_tokens SET last_used_at=datetime('now') WHERE token=?", (key,))
            cx.commit()
    except Exception:
        pass

    if required_scope is None or scope == required_scope or scope == "admin":
        return True, ctx, 200
    return False, ctx, 403


def _owner_from_scope(scope: str):
    """'workspace:shaira' -> 'shaira'. Admin -> None."""
    if not scope or not scope.startswith("workspace:"):
        return None
    return scope.split(":", 1)[1]


def _can_access_owner(ctx, owner: str) -> bool:
    """True if ctx (from _auth) is admin or its scope matches `owner`."""
    if not ctx:
        return False
    if ctx.get("scope") == "admin":
        return True
    return _owner_from_scope(ctx.get("scope", "")) == owner


def _scoped_owner(ctx, requested_owner: str) -> str:
    """For endpoints where the caller passes an owner (e.g. /api/todos?owner=X):
    admin gets whatever they asked for; scoped users are silently forced to
    their own owner (matches the 'enumerated by token' principle)."""
    if not ctx or ctx.get("scope") == "admin":
        return (requested_owner or "glen").lower()
    return _owner_from_scope(ctx["scope"]) or (requested_owner or "").lower()


def _todo_owner(cx, todo_id: int):
    row = cx.execute("SELECT owner FROM todos WHERE id=?", (todo_id,)).fetchone()
    return row[0] if row else None


def _close_open_session(cx, owner: str):
    """Close any currently-open time session for this owner. Returns closed session id or None."""
    row = cx.execute(
        "SELECT id, started_at FROM todo_time_sessions "
        "WHERE owner=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
        (owner,)
    ).fetchone()
    if not row:
        return None
    sid, started = row
    now = datetime.now(timezone.utc)
    duration = 0
    try:
        if started:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            duration = max(0, int((now - start_dt).total_seconds()))
    except Exception:
        pass
    cx.execute(
        "UPDATE todo_time_sessions SET ended_at=?, duration_seconds=? WHERE id=?",
        (now.isoformat(), duration, sid)
    )
    return sid


@app.route("/workspace/<owner>")
def workspace_page(owner):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    if not _can_access_owner(ctx, owner):
        return jsonify({"error":"Forbidden"}), 403
    if owner != "shaira":
        return jsonify({"error": "Workspace not available for this owner yet."}), 404
    resp = send_from_directory(STATIC, "shaira-workspace.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/todos/<int:todo_id>/focus", methods=["POST"])
def todo_focus(todo_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        _close_open_session(cx, owner)
        cx.execute(
            "INSERT INTO todo_time_sessions (todo_id, owner) VALUES (?, ?)",
            (todo_id, owner)
        )
        cx.execute(
            "UPDATE todos SET phase='in_process', "
            "first_started_at = CASE WHEN first_started_at IS NULL OR first_started_at='' "
            "                        THEN datetime('now') ELSE first_started_at END "
            "WHERE id=? AND phase!='complete'",
            (todo_id,)
        )
        cx.execute(
            "INSERT INTO owner_state (owner, focused_todo_id, updated_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(owner) DO UPDATE SET "
            "  focused_todo_id=excluded.focused_todo_id, updated_at=excluded.updated_at",
            (owner, todo_id)
        )
        cx.commit()
    return jsonify({"ok": True, "todo_id": todo_id, "owner": owner})


@app.route("/api/todos/<int:todo_id>/unfocus", methods=["POST"])
def todo_unfocus(todo_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        _close_open_session(cx, owner)
        cx.execute(
            "UPDATE owner_state SET focused_todo_id=NULL, updated_at=datetime('now') "
            "WHERE owner=? AND focused_todo_id=?",
            (owner, todo_id)
        )
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/todos/<int:todo_id>/complete-workspace", methods=["POST"])
def todo_complete_workspace(todo_id):
    """Workspace-style complete: closes the time session, flips phase='complete',
    logs the outcome to the thread. Named -workspace to avoid colliding with the
    existing PATCH-based /api/todos/<id> action='done' used by /console."""
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    outcome = (request.get_json(force=True) or {}).get("outcome","").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        _close_open_session(cx, owner)
        cx.execute(
            "UPDATE todos SET phase='complete', status='done', done_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), todo_id)
        )
        if outcome:
            cx.execute(
                "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
                (todo_id, owner, outcome)
            )
        cx.execute(
            "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
            (todo_id, "system", "Completed.")
        )
        cx.execute(
            "UPDATE owner_state SET focused_todo_id=NULL, updated_at=datetime('now') "
            "WHERE owner=? AND focused_todo_id=?",
            (owner, todo_id)
        )
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/todos/<int:todo_id>/mark-blocked", methods=["POST"])
def todo_mark_blocked(todo_id):
    """Soft-blocked: logs a system message + sets a justus_to_glen row to surface in Glen's console."""
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    reason = (request.get_json(force=True) or {}).get("reason","").strip()
    if not reason:
        return jsonify({"error":"Reason required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        cx.execute(
            "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
            (todo_id, "system", f"Blocked by {owner}: {reason}")
        )
        cx.execute(
            "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
            (todo_id, "justus_to_glen", f"{owner.title()} flagged blocked: {reason}")
        )
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/todos/<int:todo_id>/messages", methods=["GET"])
def todo_messages_get(todo_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    with sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        rows = cx.execute(
            "SELECT id, role, content, created_at FROM todo_messages "
            "WHERE todo_id=? ORDER BY id ASC", (todo_id,)
        ).fetchall()
    msgs = [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
    return jsonify({"messages": msgs})


def _todo_context_for_justus(cx, todo_id: int) -> str:
    row = cx.execute(
        "SELECT title, body, ai_summary FROM todos WHERE id=?", (todo_id,)
    ).fetchone()
    if not row:
        return ""
    title, body, ai_summary = row
    steps = cx.execute(
        "SELECT sequence, done, text FROM todo_steps WHERE todo_id=? ORDER BY sequence ASC",
        (todo_id,)
    ).fetchall()
    parts = [f"Task: {title}"]
    if body:
        parts.append(f"Details:\n{body}")
    if ai_summary:
        parts.append(f"Summary: {ai_summary}")
    if steps:
        steps_str = "\n".join(f"  {'[x]' if s[1] else '[ ]'} {s[2]}" for s in steps)
        parts.append(f"Steps:\n{steps_str}")
    return "\n\n".join(parts)


def _thread_history_for_justus(cx, todo_id: int, limit: int = 10) -> list:
    rows = cx.execute(
        "SELECT role, content FROM todo_messages WHERE todo_id=? ORDER BY id DESC LIMIT ?",
        (todo_id, limit)
    ).fetchall()
    rows = list(reversed(rows))
    history = []
    for role, content in rows:
        if role == "justus":
            history.append({"role": "assistant", "content": content})
        else:
            history.append({"role": "user", "content": f"[{role}] {content}"})
    return history


def _persist_justus_reply(todo_id: int, full_text: str):
    """Store Justus's full reply, parse [ASK:glen|rae] tag → justus_to_X row."""
    import re as _re
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
            (todo_id, "justus", full_text)
        )
        m = _re.search(r"\[ASK:(glen|rae)\]\s*(.+)$", full_text, _re.IGNORECASE | _re.DOTALL)
        if m:
            target = m.group(1).lower()
            question = m.group(2).strip()
            cx.execute(
                "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
                (todo_id, f"justus_to_{target}", question)
            )
        cx.commit()


@app.route("/api/todos/<int:todo_id>/messages", methods=["POST"])
def todo_messages_post(todo_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    data = request.get_json(force=True) or {}
    role = (data.get("role") or "").lower().strip()
    content = (data.get("content") or "").strip()
    # Admin may post as system (used by cron / watcher bots). Humans post as themselves.
    allowed_roles = ("shaira", "glen", "rae", "system") if ctx.get("scope") == "admin" else ("shaira", "glen", "rae")
    if role not in allowed_roles:
        return jsonify({"error":"Invalid role"}), 400
    if not content:
        return jsonify({"error":"Empty content"}), 400

    # Role-spoofing guard: scoped users may only post as themselves.
    if ctx.get("scope") != "admin" and role != ctx.get("user_name"):
        return jsonify({"error":"Forbidden — cannot post as another user"}), 403

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        cx.execute(
            "INSERT INTO todo_messages (todo_id, role, content) VALUES (?, ?, ?)",
            (todo_id, role, content)
        )
        cx.commit()

    if role != "shaira":
        # Glen/Rae replies don't trigger a Justus stream.
        return jsonify({"ok": True})

    # Shaira spoke → stream Justus's reply.
    with sqlite3.connect(LOG_DB) as cx:
        todo_ctx = _todo_context_for_justus(cx, todo_id)
        history = _thread_history_for_justus(cx, todo_id, limit=10)
    # Drop the just-inserted shaira line (it's the new query, included separately by the stream helper)
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    system = _justus_system_prompt("shaira", todo_ctx, _SHAIRA_DIRECTIVES)
    return Response(
        stream_with_context(_ask_justus_stream(
            content, system, history,
            on_complete=lambda t: _persist_justus_reply(todo_id, t),
        )),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
    )


@app.route("/api/todos/<int:todo_id>/steps", methods=["GET"])
def todo_steps_get(todo_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    with sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        rows = cx.execute(
            "SELECT id, sequence, text, done, done_at FROM todo_steps "
            "WHERE todo_id=? ORDER BY sequence ASC", (todo_id,)
        ).fetchall()
    steps = [{"id": r[0], "sequence": r[1], "text": r[2], "done": bool(r[3]), "done_at": r[4]} for r in rows]
    return jsonify({"steps": steps})


@app.route("/api/todos/<int:todo_id>/steps", methods=["POST"])
def todo_steps_post(todo_id):
    """Add a single step manually OR extract via Justus.
    Body {"text": "..."} → manual add.
    Body {"extract": true} → one-shot Justus extraction (replaces un-done steps)."""
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    # Owner check upfront (extract path reads the todo anyway, but we 403 sooner)
    with sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
    if not owner:
        return jsonify({"error":"Not found"}), 404
    if not _can_access_owner(ctx, owner):
        return jsonify({"error":"Forbidden"}), 403
    data = request.get_json(force=True) or {}
    if data.get("extract"):
        return _extract_steps_via_justus(todo_id)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error":"Empty text"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT COALESCE(MAX(sequence), 0) FROM todo_steps WHERE todo_id=?", (todo_id,)).fetchone()
        seq = (row[0] or 0) + 1
        cx.execute(
            "INSERT INTO todo_steps (todo_id, sequence, text, done) VALUES (?, ?, ?, 0)",
            (todo_id, seq, text)
        )
        sid = cx.execute("SELECT last_insert_rowid()").fetchone()[0]
        cx.commit()
    return jsonify({"ok": True, "id": sid, "sequence": seq})


def _extract_steps_via_justus(todo_id: int):
    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT title, body, ai_summary FROM todos WHERE id=?", (todo_id,)
        ).fetchone()
    if not row:
        return jsonify({"error":"Not found"}), 404
    title, body, ai_summary = row
    prompt = (
        f"Task: {title}\n\n"
        f"Details:\n{body or ai_summary or '(no details)'}\n\n"
        "Extract this into a numbered checklist of 3–7 concrete steps the assignee can check off. "
        "Each step is one short action. Reply with ONLY the numbered list, no preamble, no commentary.\n"
        "Example:\n1. Open <URL>\n2. Click <button>\n3. ..."
    )
    try:
        resp = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system="You extract concrete, scannable checklists from task descriptions. No preamble.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as e:
        return jsonify({"error": f"Justus extraction failed: {e}"}), 500
    import re as _re
    extracted = []
    for line in text.splitlines():
        m = _re.match(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+)$", line.strip())
        if m:
            extracted.append(m.group(1).strip())
    if not extracted:
        return jsonify({"error": "Justus returned no steps", "raw": text}), 500
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("DELETE FROM todo_steps WHERE todo_id=? AND done=0", (todo_id,))
        row = cx.execute("SELECT COALESCE(MAX(sequence), 0) FROM todo_steps WHERE todo_id=?", (todo_id,)).fetchone()
        seq = row[0] or 0
        for s in extracted:
            seq += 1
            cx.execute(
                "INSERT INTO todo_steps (todo_id, sequence, text, done) VALUES (?, ?, ?, 0)",
                (todo_id, seq, s)
            )
        cx.commit()
    return jsonify({"ok": True, "steps": extracted})


@app.route("/api/todos/<int:todo_id>/steps/<int:step_id>", methods=["PATCH"])
def todo_steps_patch(todo_id, step_id):
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        owner = _todo_owner(cx, todo_id)
        if not owner:
            return jsonify({"error":"Not found"}), 404
        if not _can_access_owner(ctx, owner):
            return jsonify({"error":"Forbidden"}), 403
        done = bool((request.get_json(force=True) or {}).get("done"))
        cx.execute(
            "UPDATE todo_steps SET done=?, "
            "done_at=CASE WHEN ?=1 THEN datetime('now') ELSE NULL END "
            "WHERE id=? AND todo_id=?",
            (1 if done else 0, 1 if done else 0, step_id, todo_id)
        )
        cx.commit()
    return jsonify({"ok": True})


@app.route("/api/workspace/<owner>/state", methods=["GET"])
def workspace_state(owner):
    """Returns the complete workspace payload for an owner: focused item, in-process,
    plan, recently-completed, plus per-item time/steps aggregates and pending-ask flags."""
    ok, ctx, code = _auth()
    if not ok:
        return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    if not _can_access_owner(ctx, owner):
        return jsonify({"error":"Forbidden"}), 403
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        focused = cx.execute(
            "SELECT focused_todo_id FROM owner_state WHERE owner=?", (owner,)
        ).fetchone()
        focused_id = focused["focused_todo_id"] if focused else None

        all_todos = cx.execute(
            "SELECT id, title, body, ai_summary, priority, phase, first_started_at, "
            "       created_at, done_at, status, action_note "
            "FROM todos WHERE owner=? AND status!='dismissed' "
            "ORDER BY created_at DESC",
            (owner,)
        ).fetchall()

        time_rows = cx.execute(
            "SELECT todo_id, COALESCE(SUM(duration_seconds), 0) AS total, COUNT(*) AS sessions, "
            "       MAX(COALESCE(ended_at, started_at)) AS last_touched "
            "FROM todo_time_sessions WHERE owner=? GROUP BY todo_id",
            (owner,)
        ).fetchall()
        time_by_id = {r["todo_id"]: dict(r) for r in time_rows}

        open_session = cx.execute(
            "SELECT id, todo_id, started_at FROM todo_time_sessions "
            "WHERE owner=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
            (owner,)
        ).fetchone()

        step_rows = cx.execute(
            "SELECT t.id AS todo_id, "
            "       COUNT(s.id) AS total_steps, "
            "       SUM(CASE WHEN s.done=1 THEN 1 ELSE 0 END) AS done_steps "
            "FROM todos t LEFT JOIN todo_steps s ON s.todo_id=t.id "
            "WHERE t.owner=? GROUP BY t.id",
            (owner,)
        ).fetchall()
        steps_by_id = {r["todo_id"]: dict(r) for r in step_rows}

        # Pending ASK = a justus_to_X row newer than the latest X reply on the same todo
        ask_rows = cx.execute("""
            SELECT tm.todo_id, tm.role, tm.content, tm.created_at
            FROM todo_messages tm
            WHERE tm.role IN ('justus_to_glen','justus_to_rae')
              AND tm.created_at > COALESCE((
                SELECT MAX(tm2.created_at) FROM todo_messages tm2
                WHERE tm2.todo_id=tm.todo_id
                  AND tm2.role = CASE tm.role WHEN 'justus_to_glen' THEN 'glen' ELSE 'rae' END
              ), '')
        """).fetchall()
        pending_by_id = {}
        for r in ask_rows:
            pending_by_id.setdefault(r["todo_id"], []).append({
                "target": r["role"].replace("justus_to_",""),
                "content": r["content"],
                "created_at": r["created_at"],
            })

    def _serialize(t):
        d = dict(t)
        tid = d["id"]
        tinfo = time_by_id.get(tid, {})
        sinfo = steps_by_id.get(tid, {})
        d["time_total_seconds"] = int(tinfo.get("total", 0) or 0)
        d["time_sessions"] = int(tinfo.get("sessions", 0) or 0)
        d["last_touched"] = tinfo.get("last_touched") or d["created_at"]
        d["steps_total"] = int(sinfo.get("total_steps", 0) or 0)
        d["steps_done"] = int(sinfo.get("done_steps", 0) or 0)
        d["pending_asks"] = pending_by_id.get(tid, [])
        return d

    serialized = [_serialize(t) for t in all_todos]
    focused_todo = next((t for t in serialized if t["id"] == focused_id), None)
    in_process = [t for t in serialized if t["phase"] == "in_process" and t["id"] != focused_id]
    plan = [t for t in serialized if t["phase"] == "plan"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    complete_recent = [t for t in serialized if t["phase"] == "complete" and (t.get("done_at") or "") >= cutoff]
    # Sort completed by done_at desc
    complete_recent.sort(key=lambda t: t.get("done_at") or "", reverse=True)
    # Sort in-process by last_touched desc
    in_process.sort(key=lambda t: t.get("last_touched") or "", reverse=True)
    # Sort plan by priority then created_at
    plan.sort(key=lambda t: (0 if t.get("priority")=="high" else 1, t.get("created_at") or ""), reverse=False)

    return jsonify({
        "owner": owner,
        "focused": focused_todo,
        "in_process": in_process,
        "plan": plan,
        "complete_recent": complete_recent,
        "open_session": dict(open_session) if open_session else None,
    })


# ── Access tokens — admin-only issuance, list, revoke ─────────────────────────
import secrets as _secrets


def _token_mint_allowed():
    """Refuse to mint tokens against a non-production DB.

    On Render, DATA_DIR is set and LOG_DB lives on the persistent disk
    (same signal the scheduler uses). On a local run DATA_DIR is unset and
    LOG_DB falls back to a repo-local/ephemeral chat_log.db, so a token
    minted there never reaches production and silently fails to authenticate
    (the Shaira 'Unauthorized' incident, 2026-06-04). Set
    ALLOW_LOCAL_TOKEN_MINT=1 to opt in for deliberate local testing.

    Returns (ok: bool, error_message: str|None).
    """
    if os.environ.get("DATA_DIR") or os.environ.get("ALLOW_LOCAL_TOKEN_MINT") == "1":
        return True, None
    return False, (
        "Refusing to mint a token: this is not the production database "
        "(DATA_DIR is unset, so LOG_DB falls back to a local/ephemeral file "
        f"at {LOG_DB}). A token minted here never reaches production and will "
        "fail with 'Unauthorized'. Mint against the live service "
        "(https://glen-knowledge-chat.onrender.com), or set "
        "ALLOW_LOCAL_TOKEN_MINT=1 to override for local testing."
    )


@app.route("/api/access-tokens", methods=["POST"])
def access_token_create():
    """Mint a per-user access token. Admin (CONSOLE_SECRET) only.
    Body: {"name":"shaira","scope":"workspace:shaira","display_name":"Shaira","note":"..."}
    Returns the full token ONCE — never retrievable again."""
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    ok, why = _token_mint_allowed()
    if not ok:
        return jsonify({"error": why, "log_db": str(LOG_DB)}), 409
    data = request.get_json(force=True) or {}
    name  = (data.get("name") or "").lower().strip()
    scope = (data.get("scope") or "").strip()
    if not name or not scope:
        return jsonify({"error":"name + scope required"}), 400
    if scope != "admin" and not scope.startswith("workspace:"):
        return jsonify({"error":"scope must be 'admin' or 'workspace:<owner>'"}), 400
    display_name = (data.get("display_name") or name.title()).strip()
    note = (data.get("note") or "").strip()
    token = _secrets.token_urlsafe(32)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO workspace_users (name, display_name, scope) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET display_name=excluded.display_name, scope=excluded.scope",
            (name, display_name, scope)
        )
        user_id = cx.execute("SELECT id FROM workspace_users WHERE name=?", (name,)).fetchone()[0]
        cx.execute(
            "INSERT INTO access_tokens (token, user_id, note) VALUES (?, ?, ?)",
            (token, user_id, note)
        )
        cx.commit()
    owner = _owner_from_scope(scope) or name
    return jsonify({
        "ok": True,
        "token": token,
        "user": name,
        "display_name": display_name,
        "scope": scope,
        "url": f"/workspace/{owner}?key={token}",
        "note": note,
    })


@app.route("/api/access-tokens", methods=["GET"])
def access_token_list():
    """List all tokens. Admin only. Full token is NEVER returned — only the
    first 8 chars (prefix) for identification + revoke."""
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute(
            "SELECT t.token, t.created_at, t.last_used_at, t.revoked_at, t.note, "
            "       u.name, u.display_name, u.scope "
            "FROM access_tokens t JOIN workspace_users u ON u.id = t.user_id "
            "ORDER BY t.created_at DESC"
        ).fetchall()
    return jsonify({"tokens": [{
        "prefix": r[0][:8],
        "created_at": r[1],
        "last_used_at": r[2],
        "revoked_at": r[3],
        "note": r[4],
        "user": r[5],
        "display_name": r[6],
        "scope": r[7],
    } for r in rows]})


@app.route("/api/access-tokens/<prefix>", methods=["DELETE"])
def access_token_revoke(prefix):
    """Revoke (soft-delete) a token by its 8-char prefix. Admin only."""
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    if not prefix or len(prefix) < 6:
        return jsonify({"error":"prefix too short"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        rows = cx.execute(
            "SELECT token FROM access_tokens WHERE token LIKE ? AND revoked_at IS NULL",
            (prefix + "%",)
        ).fetchall()
        if not rows:
            return jsonify({"error":"no active token matches that prefix"}), 404
        if len(rows) > 1:
            return jsonify({"error":"prefix matches multiple tokens — be more specific","matches":len(rows)}), 409
        full = rows[0][0]
        cx.execute("UPDATE access_tokens SET revoked_at=datetime('now') WHERE token=?", (full,))
        cx.commit()
    return jsonify({"ok": True, "revoked_prefix": prefix})


@app.route("/api/gmail/thread-attachments", methods=["GET"])
def gmail_thread_attachments():
    """Download all real file attachments in a Gmail thread, using the server's
    existing Gmail OAuth token. Admin-only. Used by the Shaira vault watcher,
    which runs on Glen's Mac and writes the bytes into the vault.

    Query: ?thread_id=<id>  (also accepts ?message_id=<id>)
    Returns: {ok, count, attachments: [{message_id, filename, mime_type, size,
              inline, skipped, content_b64}]}
    """
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    thread_id = (request.args.get("thread_id") or request.args.get("message_id") or "").strip()
    if not thread_id:
        return jsonify({"error":"thread_id required"}), 400
    try:
        from dashboard.inbox import get_thread_attachments
        attachments = get_thread_attachments(thread_id)
    except FileNotFoundError as e:
        return jsonify({"error": f"Gmail token not configured: {e}"}), 503
    except Exception as e:
        app.logger.exception("thread-attachments failed")
        return jsonify({"error": f"Gmail fetch failed: {e}"}), 502
    return jsonify({"ok": True, "count": len(attachments), "attachments": attachments})


WORKSPACE_BACKUP_DIR = Path(os.environ.get(
    "WORKSPACE_BACKUP_DIR",
    "/Users/remedymatch/AI-Training/00 System/console-archive"
))


@app.route("/cron/backup-workspace", methods=["POST", "GET"])
def cron_backup_workspace():
    """Dump of workspace tables + recently-touched todos.

    Always returns the full `payload` in the JSON response — the nightly
    backup launchd job runs on Glen's Mac, calls this, and writes the payload
    into the vault locally (the vault is not on the Render filesystem). The
    server-side file write is best-effort and only succeeds if WORKSPACE_BACKUP_DIR
    happens to be writable (i.e. when run on a Mac, not on Render)."""
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        messages = [dict(r) for r in cx.execute(
            "SELECT id, todo_id, role, content, created_at FROM todo_messages ORDER BY id ASC"
        ).fetchall()]
        sessions = [dict(r) for r in cx.execute(
            "SELECT id, todo_id, owner, started_at, ended_at, duration_seconds "
            "FROM todo_time_sessions ORDER BY id ASC"
        ).fetchall()]
        steps = [dict(r) for r in cx.execute(
            "SELECT id, todo_id, sequence, text, done, done_at FROM todo_steps ORDER BY id ASC"
        ).fetchall()]
        owner_state = [dict(r) for r in cx.execute(
            "SELECT owner, focused_todo_id, updated_at FROM owner_state"
        ).fetchall()]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        # Touched = created_at, done_at, or any session/message activity in the window
        todos = [dict(r) for r in cx.execute(
            "SELECT id, created_at, owner, category, title, body, priority, status, "
            "       delegated_to, delegated_at, done_at, source, dedup_key, ai_summary, "
            "       suggested_reply, action_note, received_at, phase, first_started_at "
            "FROM todos "
            "WHERE created_at >= ? OR done_at >= ? OR id IN ("
            "  SELECT DISTINCT todo_id FROM todo_messages WHERE created_at >= ? "
            "  UNION SELECT DISTINCT todo_id FROM todo_time_sessions WHERE started_at >= ? "
            ") "
            "ORDER BY id ASC",
            (cutoff, cutoff, cutoff, cutoff)
        ).fetchall()]
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_30d": cutoff,
        "counts": {
            "messages": len(messages),
            "sessions": len(sessions),
            "steps": len(steps),
            "todos_30d": len(todos),
            "owner_state": len(owner_state),
        },
        "owner_state": owner_state,
        "todos_30d": todos,
        "todo_messages": messages,
        "todo_time_sessions": sessions,
        "todo_steps": steps,
    }
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Best-effort server-side write (only meaningful when run on a Mac, not Render).
    server_wrote = None
    try:
        WORKSPACE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        out_path = WORKSPACE_BACKUP_DIR / f"{date_tag}-workspace.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        server_wrote = str(out_path)
    except Exception:
        server_wrote = None  # expected on Render — caller writes the payload locally
    return jsonify({
        "ok": True,
        "date_tag": date_tag,
        "counts": payload["counts"],
        "server_wrote": server_wrote,
        "payload": payload,
    })


# ── Shaira daily monitoring report (Phase 4) ──────────────────────────────────
@app.route("/cron/shaira-daily-report", methods=["POST", "GET"])
def cron_shaira_daily_report():
    """Generate (or regenerate) today's Shaira daily report. Admin-only.
    Fired daily by a launchd job at ~8 AM HST; idempotent (upserts the day)."""
    if not _ws_auth_ok():
        return jsonify({"error":"Unauthorized"}), 401
    try:
        from dashboard.shaira_daily import generate_and_store
        result = generate_and_store(str(LOG_DB), _cl, "shaira")
    except Exception as e:
        app.logger.exception("shaira-daily-report failed")
        return jsonify({"error": f"Report generation failed: {e}"}), 500
    return jsonify({
        "ok": True,
        "report_date": result["report_date"],
        "markdown": result["markdown"],
        "metrics": result["metrics"],
    })


@app.route("/api/shaira-daily")
def api_shaira_daily():
    """Latest Shaira daily report, shaped for the dashboard 'briefing' card."""
    if CONSOLE_SECRET:
        ok_, _ctx, code = _auth()
        if not ok_:
            return jsonify({"error":"Unauthorized" if code == 401 else "Forbidden"}), code
    from dashboard.shaira_daily import latest_report
    from dashboard import briefing_actions as _ba
    data = latest_report(str(LOG_DB), "shaira")
    if isinstance(data, dict) and data.get("markdown"):
        md = _ba.filter_markdown("shaira-daily", data["markdown"])
        # Promote the model's ### subsections to ## so they get the green border-top
        # rule (the line under the title) and match the other 3 cards' section style.
        md = re.sub(r"(?m)^(\s*)###\s+", r"\1## ", md)
        data["markdown"] = _intel.normalize_title("shaira-daily", _intel.actions_first(md))
    return jsonify({"ok": True, "data": data})


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
        # Certification bonus Biofields: daily sweep at 15:00 UTC (5am HST). Flag-gated
        # (CERT_BONUS_ENABLED) inside _run_biofield_bonuses, so this is a safe no-op until on.
        scheduler.add_job(_run_biofield_bonuses, "cron", hour=15, minute=0,
                          id="biofield_bonuses")
        scheduler.start()
        print("[CRON] Scheduler started — hourly push + daily biofield-bonuses active")
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
    try:
        data = _money.qb_banks()
        # Fold Wise multi-currency balances into the Bank Balances card.
        try:
            data["wise"] = _money.wise_data().get("balances", [])
        except Exception:
            data["wise"] = []
        return ok(data)
    except Exception as e: return fail(e)


# Retained for back-compat / direct callers; the dashboard now folds Wise into
# the Bank Balances card (/api/money/banks) instead of a separate card.
@app.route("/api/money/wise")
@require_console_key
def api_money_wise():
    try: return ok(_money.wise_data())
    except Exception as e: return fail(e)


@app.route("/api/studio/sales")
@require_console_key
def api_studio_sales():
    try:
        from dashboard import studio as _studio
        return ok(_studio.summary())
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


@app.route("/api/heygen/<video_id>/mark-reviewed", methods=["POST"])
@require_console_key
def api_heygen_mark_reviewed(video_id):
    # HeyGen itself has no "reviewed" concept — track it locally.
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO heygen_reviewed (video_id, reviewed_at) VALUES (?,?) "
            "ON CONFLICT(video_id) DO UPDATE SET reviewed_at=excluded.reviewed_at",
            (video_id, ts),
        )
        cx.commit()
    return ok({"video_id": video_id, "reviewed_at": ts})


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


# ── Intelligence briefings ────────────────────────────────────────────────────
from dashboard import intelligence as _intel


@app.route("/api/intelligence/<slug>")
@require_console_key
def api_intelligence_get(slug):
    try:
        from dashboard import briefing_actions as _ba
        data = _intel.read_briefing(slug)
        if data.get("markdown"):
            md = _ba.filter_markdown(slug, data["markdown"])
            data["markdown"] = _intel.normalize_title(slug, _intel.actions_first(md))
        return ok(data)
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/briefing-action", methods=["POST"])
@require_console_key
def api_briefing_action():
    """Record done/dismissed/snoozed state for a single dashboard action item."""
    from dashboard import briefing_actions as _ba
    b = request.get_json(silent=True) or {}
    slug = (b.get("slug") or "").strip()
    text = (b.get("action_text") or "").strip()
    state = (b.get("state") or "").strip()
    if not slug or not text or state not in ("done", "dismissed", "snoozed"):
        return fail("slug, action_text, and a valid state are required", status=400)
    try:
        _ba.set_state(slug, text, state, snooze_days=int(b.get("snooze_days", 0) or 0))
        return ok({"slug": slug, "state": state})
    except Exception as e:
        return fail(e)


@app.route("/api/intelligence/<slug>/upload", methods=["POST"])
@require_console_key
def api_intelligence_upload(slug):
    try: return ok(_intel.write_briefing(slug, request.get_data()))
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e, status=400)


@app.route("/api/intelligence")
@require_console_key
def api_intelligence_list():
    try: return ok(_intel.list_all())
    except Exception as e: return fail(e)


# ── Cron webhooks (called by Render cron services) ───────────────────────────
# Render persistent disks attach to ONE service at a time. The disk `data` is
# mounted on glen-knowledge-chat (web), so cron containers don't have it. To
# avoid sqlite3.OperationalError when the orchestrator opens chat_log.db,
# Render cron just curls this endpoint. The orchestrator runs inside the web
# container where the disk lives.
#
# Auth: X-Cron-Secret header must match CRON_SECRET env (falls back to
# CONSOLE_SECRET so a single shared secret can cover both).

@app.route("/cron/personal-send", methods=["POST"])
def cron_personal_send():
    key = request.headers.get("X-Cron-Secret", "")
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")
    segment = os.environ.get("PERSONAL_EMAIL_SEGMENT", "").strip()
    try:
        if segment:
            # Graduated send to a People-hub segment (opted-in only), ramped.
            from incentive_engine import run_daily_send_for_segment
            cap = int(os.environ.get("SEND_CAP", "25") or "25")
            tags = tuple(t.strip() for t in segment.split(",") if t.strip())
            summary = run_daily_send_for_segment(
                segment_tags=tags, audience="client", cap=cap, dry_run=dry_run)
            return jsonify({"ok": True, "mode": "segment", "segment": tags,
                            "cap": cap, "summary": summary})
        from incentive_engine import run_daily_send_for_beta_cohort
        n = run_daily_send_for_beta_cohort()
        return jsonify({"ok": True, "mode": "beta", "sent": n})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/unsubscribe", methods=["GET"])
def unsubscribe():
    """One-click email unsubscribe (linked in every Personal email footer).
    Verifies the HMAC token, revokes consent in the people hub, pauses sends."""
    email = (request.args.get("email", "") or "").strip().lower()
    token = request.args.get("t", "")
    # Portal-notification unsubscribe: a stable portal token (?token=...) opts the
    # client out of Healing Oasis scan notifications (separate from email-consent).
    portal_token = (request.args.get("token", "") or "").strip()
    if portal_token and not email:
        from dashboard import client_portal as _cp, notify_state as _ns
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            portal = _cp.get_portal_by_token(cx, portal_token)
            if portal:
                _ns.set_opt(cx, portal["email"], "out")
        return (
            "<html><body style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
            "<h2>You're unsubscribed.</h2>"
            "<p>You will no longer receive Healing Oasis scan notifications. "
            "You can re-enable them anytime from your portal.</p>"
            "<p>In wellness,<br>Dr. Glen &amp; Rae</p></body></html>"
        ), 200
    from incentive_engine import verify_unsub_token, revoke_consent
    if not email:
        return "<p>Missing email.</p>", 400
    # Token required only when one was issued; accept tokenless legacy links too.
    if token and not verify_unsub_token(email, token):
        return "<p>This unsubscribe link is invalid or has expired.</p>", 400
    revoke_consent(email)
    return (
        "<html><body style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
        "<h2>You're unsubscribed.</h2>"
        f"<p>{email} will no longer receive these emails. "
        "If this was a mistake, just reply to any past email and we'll turn them back on.</p>"
        "<p>In wellness,<br>Dr. Glen &amp; Rae</p></body></html>"
    ), 200


@app.route("/cron/usps-rate-check", methods=["POST"])
def cron_usps_rate_check():
    """Weekly USPS Flat Rate watcher. Stages pending updates for Glen to confirm."""
    key = request.headers.get("X-Cron-Secret", "")
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        from dashboard.shipping import check_usps_rates
        return jsonify({"ok": True, "summary": check_usps_rates()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/cron/regenerate-briefings", methods=["POST"])
def cron_regenerate_briefings():
    """Daily Intelligence-card regenerator. Gathers live system stats and
    asks Claude to compose all 5 briefings, then writes them to disk so the
    dashboard's Intelligence row serves fresh markdown."""
    key = request.headers.get("X-Cron-Secret", "")
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        from dashboard.briefing_runner import regenerate_all
        return jsonify({"ok": True, "summary": regenerate_all()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regenerate-briefings", methods=["POST"])
@require_console_key
def api_regenerate_briefings():
    """User-triggered briefings regen — same payload as the cron endpoint
    but auth via console-key (the dashboard button uses this)."""
    try:
        from dashboard.briefing_runner import regenerate_all
        return ok({"summary": regenerate_all()})
    except Exception as e:
        return fail(e, status=500)


# ── Subscription email helper ─────────────────────────────────────────────────

def _send_subscription_email(to_email: str, kind: str, data: dict):
    """Send a subscription lifecycle email. Best-effort — NEVER raises.

    kind: 'heads_up' | 'receipt' | 'payment_failed' | 'setup_confirm'
    data: dict with relevant context fields (next_charge_date, amount, etc.)
    Returns (sent_via, error_or_none) same shape as _send_full_report_email.
    """
    try:
        charge_date = data.get("next_charge_date", "")
        amount_str = ""
        raw_cents = data.get("total_cents")
        if raw_cents is not None:
            amount_str = f"${int(raw_cents) / 100:.2f}"

        is_membership = data.get("kind") == "membership"

        if kind == "heads_up":
            if is_membership:
                subject = "Your live group coaching renews in 3 days"
                body = (
                    f"Hi,\n\n"
                    f"Just a quick heads-up about your live group coaching. On {charge_date} "
                    + (f"your card will be charged {amount_str} for the coming month of coaching"
                       if amount_str else "your card will be charged for the coming month of coaching")
                    + f".\n\n"
                    f"If you need to make a change, visit your member portal before that date.\n\n"
                    f"In wellness,\nDr. Glen"
                )
            else:
                subject = f"Reminder: your Remedy Match subscription charges on {charge_date}"
                body = (
                    f"Hi,\n\n"
                    f"Just a quick heads-up — your next Remedy Match subscription order will be "
                    f"charged automatically on {charge_date}.\n\n"
                    f"If you need to skip or pause, visit your subscription portal before that date.\n\n"
                    f"In wellness,\nDr. Glen"
                )
        elif kind == "receipt":
            inv_id = data.get("invoice_id", "")
            if is_membership:
                subject = f"Your live group coaching payment — {amount_str}"
                body = (
                    f"Hi,\n\n"
                    f"Your live group coaching payment has been processed successfully.\n\n"
                    f"Amount charged: {amount_str}\n"
                    + (f"Invoice: {inv_id}\n" if inv_id else "")
                    + f"\nYour next payment date: {charge_date}\n\n"
                    f"Thank you for being part of the group.\n\n"
                    f"In wellness,\nDr. Glen"
                )
            else:
                body = (
                    f"Hi,\n\n"
                    f"Your Remedy Match subscription order has been processed successfully.\n\n"
                    f"Amount charged: {amount_str}\n"
                    + (f"Invoice: {inv_id}\n" if inv_id else "")
                    + f"\nYour next order will ship on {charge_date}.\n\n"
                    f"Thank you for your continued trust.\n\n"
                    f"In wellness,\nDr. Glen"
                )
                subject = f"Your Remedy Match subscription order is confirmed — {amount_str}"
        elif kind == "payment_failed":
            fail_count = data.get("failed_count", 1)
            subject = "Action needed: Remedy Match subscription payment failed"
            body = (
                f"Hi,\n\n"
                f"We were unable to process your Remedy Match subscription payment"
                + (f" ({amount_str})" if amount_str else "")
                + f".\n\n"
                f"This is attempt {fail_count}. After 3 failed attempts your subscription "
                f"will be placed on hold.\n\n"
                f"Please update your payment method or contact us so we can get this sorted out.\n\n"
                f"In wellness,\nDr. Glen"
            )
        elif kind == "setup_confirm":
            subject = "Welcome to Remedy Match Subscribe & Grow"
            body = (
                f"Hi,\n\n"
                f"Your Remedy Match subscription is now active. Your first order has been placed "
                f"and your card is securely vaulted for future cycles.\n\n"
                f"Your next charge date: {charge_date}\n\n"
                f"You can skip, pause, or cancel anytime through your subscription portal.\n\n"
                f"In wellness,\nDr. Glen"
            )
        else:
            subject = f"Remedy Match subscription update"
            body = f"Subscription update ({kind}).\n\nIn wellness,\nDr. Glen"

        return _send_full_report_email(to_email, "", subject, body)
    except Exception as e:
        print(f"[sub-email] kind={kind} to={to_email} failed: {e!r}", flush=True)
        return ("error", str(e))


# ── Daily subscription charge scheduler ──────────────────────────────────────

@app.route("/api/cron/charge-subscriptions", methods=["POST"])
def cron_charge_subscriptions():
    """Daily charge scheduler for active subscriptions.

    Auth: X-Cron-Secret header matching CRON_SECRET (or CONSOLE_SECRET fallback).
    Supports ?dry_run=1 — computes + logs without charging or mutating.

    Two passes:
      1. Heads-up: send 3-day advance notice emails for upcoming charges.
      2. Charge: process due subs (skip skipped, charge active, dun failed).

    Returns {ok, charged, skipped, failed, notified, dry_run}.
    """
    key = request.headers.get("X-Cron-Secret", "")
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401

    if not _subscriptions_enabled():
        return jsonify({"ok": True, "skipped": 0, "charged": 0, "failed": 0,
                        "notified": 0, "dry_run": False,
                        "message": "subscriptions disabled"}), 200

    dry_run = request.args.get("dry_run", "").lower() in ("1", "true", "yes")

    from dashboard import subscriptions as _subs
    from datetime import date as _date

    today = _date.today().isoformat()  # YYYY-MM-DD

    charged = skipped = failed = notified = 0

    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        # Run idempotent migration to ensure failed_count column exists
        _subs.migrate_add_failed_count(cx)
        _subs.migrate_add_membership_columns(cx)

        # ── Pass 1: Heads-up emails (3-day advance notice) ────────────────────
        try:
            upcoming = _subs.list_heads_up_due(cx, as_of=today, lead_days=3)
        except Exception as e:
            print(f"[sub-cron] heads-up query failed: {e!r}", flush=True)
            upcoming = []

        for sub in upcoming:
            try:
                if not dry_run:
                    _send_subscription_email(
                        sub["email"], "heads_up",
                        {"next_charge_date": sub["next_charge_date"],
                         "kind": sub.get("kind", "product"),
                         "total_cents": sub.get("amount_cents")})
                    _subs.set_last_notified_date(cx, sub["id"], sub["next_charge_date"])
                notified += 1
                print(f"[sub-cron] heads-up {'(dry)' if dry_run else ''}"
                      f" sub={sub['id']} date={sub['next_charge_date']}", flush=True)
            except Exception as e:
                print(f"[sub-cron] heads-up sub={sub['id']} error: {e!r}", flush=True)

        # Snapshot the charge list BEFORE consuming skips. list_due (skip_next=0) and
        # list_skip_due (skip_next=1) are disjoint right now; if we queried due AFTER the
        # skip pass, a just-consumed sub (skip cleared, date advanced but possibly still
        # due) would wrongly re-enter the charge loop and get billed for a cycle it skipped.
        try:
            due = _subs.list_due(cx, as_of=today)
        except Exception as e:
            print(f"[sub-cron] list_due query failed: {e!r}", flush=True)
            due = []

        # ── Pass 2a: Skip cycles ───────────────────────────────────────────────
        try:
            skip_due = _subs.list_skip_due(cx, as_of=today)
        except Exception as e:
            print(f"[sub-cron] skip-due query failed: {e!r}", flush=True)
            skip_due = []

        for sub in skip_due:
            try:
                if not dry_run:
                    _subs.consume_skip(cx, sub["id"])
                skipped += 1
                print(f"[sub-cron] skip {'(dry)' if dry_run else ''}"
                      f" sub={sub['id']}", flush=True)
            except Exception as e:
                print(f"[sub-cron] consume_skip sub={sub['id']} error: {e!r}", flush=True)

        # ── Pass 2b: Charge the snapshot of due (non-skip) subs ────────────────
        for sub in due:
            sid = sub["id"]
            try:
                if sub.get("kind") == "membership":
                    amount_cents = int(sub.get("amount_cents") or 0)
                    if amount_cents <= 0:
                        continue
                    if dry_run:
                        print(f"[sub-cron] DRY membership charge sub={sid} "
                              f"email={sub['email']} amount={amount_cents}", flush=True)
                        charged += 1
                        continue
                    res = stripe_pay.charge_off_session(
                        sub["stripe_customer_id"], sub["stripe_payment_method_id"],
                        amount_cents, description="Remedy Match live group coaching",
                        metadata={"sub": str(sid), "kind": "membership"})
                    if res.get("status") == "succeeded":
                        try:
                            cust = qb.find_or_create_customer(sub["email"], "")
                            inv = qb.create_invoice(
                                cust,
                                [{"name": "Live Group Coaching", "amount": amount_cents / 100.0,
                                  "qty": 1, "description": "Live Group Coaching (monthly)"}],
                                allow_online_pay=False, email_to=sub["email"])
                            inv_id = inv.get("Id", "")
                        except Exception as qe:
                            print(f"[sub-cron] membership QBO sub={sid}: {qe!r}", flush=True)
                            inv_id = ""
                        _ingest_order(source="membership", external_ref=res.get("id") or inv_id,
                                      email=sub["email"], items=[], total_cents=amount_cents,
                                      address={}, channel="retail")
                        _subs.advance_after_charge(cx, sid)
                        _subs.reset_failed_count(cx, sid)
                        updated = _subs.get(cx, sid)
                        try:
                            _send_subscription_email(sub["email"], "receipt", {
                                "total_cents": amount_cents, "invoice_id": inv_id,
                                "kind": "membership",
                                "next_charge_date": updated["next_charge_date"] if updated else ""})
                        except Exception as ee:
                            print(f"[sub-cron] membership email sub={sid}: {ee!r}", flush=True)
                        charged += 1
                    else:
                        _subs.bump_failed_count(cx, sid)
                        failed += 1
                    continue

                items = sub.get("items") or []
                ship = sub.get("ship_address") or {}
                order_count = sub.get("order_count", 0)
                tier_pct = _subs.tier_for(order_count)

                # Price the order
                try:
                    pc = _price_cart(items, ship=ship, subscriber_tier_pct=tier_pct)
                except CheckoutError as ce:
                    print(f"[sub-cron] price_cart sub={sid}: {ce!r}", flush=True)
                    failed += 1
                    continue

                discount_cents = int(pc.get("discount_cents", 0))
                points_redeemed_cents = int(pc.get("points_redeemed_cents", 0))
                shipping_cents = int(pc.get("shipping_cents", 0))
                # Charge the discounted product subtotal + shipping. GET is absorbed
                # (recorded, not billed to the customer), so use subtotal_cents NOT total_cents
                # (which includes get_cents). This matches the reorder invoice total.
                total_cents = int(pc["priced"].get("subtotal_cents", 0)) + shipping_cents

                if dry_run:
                    print(f"[sub-cron] DRY charge sub={sid} email={sub['email']}"
                          f" amount={total_cents}", flush=True)
                    charged += 1
                    continue

                # Charge off-session
                res = stripe_pay.charge_off_session(
                    sub["stripe_customer_id"],
                    sub["stripe_payment_method_id"],
                    total_cents,
                    description="Remedy Match subscription",
                    metadata={"sub": str(sid)},
                )

                if res.get("status") == "succeeded":
                    # Build QBO invoice
                    try:
                        cust = qb.find_or_create_customer(sub["email"], "")
                        inv = qb.create_invoice(
                            cust,
                            pc["qbo_lines"] + _shipping_line(shipping_cents),
                            allow_online_pay=False,
                            email_to=sub["email"],
                            discount_cents=discount_cents + points_redeemed_cents,
                        )
                        inv_id = inv.get("Id", "")
                    except Exception as qe:
                        print(f"[sub-cron] QBO invoice sub={sid}: {qe!r}", flush=True)
                        inv_id = ""

                    # Record the order
                    _ingest_order(
                        source="subscription",
                        external_ref=res.get("id") or inv_id,
                        email=sub["email"],
                        items=pc.get("items_rec") or [],
                        total_cents=total_cents,
                        address=ship,
                        channel="retail",
                        get_cents=pc["priced"].get("get_cents", 0),
                        discount_cents=discount_cents,
                        points_redeemed_cents=points_redeemed_cents,
                        shipping_cents=shipping_cents,
                    )

                    _subs.advance_after_charge(cx, sid)
                    _subs.reset_failed_count(cx, sid)

                    # Advance to get next_charge_date for receipt email
                    updated = _subs.get(cx, sid)
                    _send_subscription_email(sub["email"], "receipt", {
                        "total_cents": total_cents,
                        "invoice_id": inv_id,
                        "kind": sub.get("kind", "product"),
                        "next_charge_date": updated["next_charge_date"] if updated else "",
                    })
                    charged += 1
                    print(f"[sub-cron] charged sub={sid} pi={res.get('id')}"
                          f" amount={total_cents}", flush=True)

                else:
                    # Failed or requires_action — dun
                    _subs.bump_failed_count(cx, sid)
                    updated = _subs.get(cx, sid)
                    new_fail_count = (updated or {}).get("failed_count", 1)

                    if new_fail_count >= 3:
                        _subs.set_status(cx, sid, "past_due")
                        print(f"[sub-cron] past_due sub={sid} after {new_fail_count} failures",
                              flush=True)

                    _send_subscription_email(sub["email"], "payment_failed", {
                        "total_cents": total_cents,
                        "failed_count": new_fail_count,
                        "status": res.get("status"),
                        "decline_code": res.get("decline_code"),
                    })
                    failed += 1
                    print(f"[sub-cron] payment failed sub={sid}"
                          f" status={res.get('status')} fail_count={new_fail_count}",
                          flush=True)

            except Exception as e:
                print(f"[sub-cron] sub={sid} unexpected error: {e!r}", flush=True)
                failed += 1

    return jsonify({
        "ok": True,
        "charged": charged,
        "skipped": skipped,
        "failed": failed,
        "notified": notified,
        "dry_run": dry_run,
    })


# ── One-time Gmail token upload (helper for first-time setup on Render) ───────
# Local token at ~/.config/google/token.json gets POSTed here once and
# persisted to /data/google-token.json on the web service's disk.
@app.route("/admin/upload-gmail-token", methods=["POST"])
@require_console_key
def admin_upload_gmail_token():
    try:
        body = request.get_json(silent=True) or {}
        token_json = body.get("token")
        if not token_json:
            return fail("token field required (paste contents of ~/.config/google/token.json)", status=400)
        # Validate it parses + has expected shape
        if isinstance(token_json, str):
            import json as _json
            token_json = _json.loads(token_json)
        for required in ("token", "refresh_token", "client_id", "client_secret"):
            if required not in token_json:
                return fail(f"token JSON missing field: {required}", status=400)
        target = os.environ.get("GMAIL_TOKEN_PATH", "/data/google-token.json")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(target, "w") as f:
            _json.dump(token_json, f)
        os.chmod(target, 0o600)
        return ok({
            "saved_to": target,
            "scopes": token_json.get("scopes", []),
            "client_id_suffix": token_json.get("client_id", "")[-12:],
        })
    except Exception as e: return fail(e, status=500)


# ── Inbox (Gmail in console) ──────────────────────────────────────────────────
# /console/inbox  — full Gmail thread list + read + reply, behind console auth
# /api/inbox/*    — JSON API behind require_console_key
from dashboard import inbox as _inbox


@app.route("/console/inbox")
def console_inbox_page():
    resp = send_from_directory(STATIC, "console-inbox.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/inbox/threads", methods=["GET"])
@require_console_key
def api_inbox_list_threads():
    try:
        q = request.args.get("q", "in:inbox")
        max_results = int(request.args.get("max", 50))
        category = (request.args.get("category") or "").strip().lower()
        threads = _inbox.list_threads(query=q, max_results=max_results)
        if category:
            # Comma-separated multi-select supported: ?category=important,inbox
            wanted = {c.strip() for c in category.split(",") if c.strip()}
            threads = [t for t in threads if t.get("category") in wanted]
        return ok(threads)
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>", methods=["GET"])
@require_console_key
def api_inbox_get_thread(thread_id):
    try: return ok(_inbox.get_thread(thread_id))
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/reply", methods=["POST"])
@require_console_key
def api_inbox_reply(thread_id):
    try:
        body = (request.get_json(silent=True) or {})
        reply_body = (body.get("body") or "").strip()
        if not reply_body:
            return fail("body is required", status=400)
        override_to = (body.get("to") or "").strip() or None
        return ok(_inbox.send_reply(thread_id, reply_body, override_to=override_to))
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/archive", methods=["POST"])
@require_console_key
def api_inbox_archive(thread_id):
    try:
        _inbox.archive_thread(thread_id)
        return ok({"archived": thread_id})
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/star", methods=["POST"])
@require_console_key
def api_inbox_star(thread_id):
    try:
        body = request.get_json(silent=True) or {}
        if body.get("starred"):
            _inbox.star_thread(thread_id)
        else:
            _inbox.unstar_thread(thread_id)
        return ok({"thread_id": thread_id, "starred": bool(body.get("starred"))})
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/read", methods=["POST"])
@require_console_key
def api_inbox_read(thread_id):
    try:
        body = request.get_json(silent=True) or {}
        if body.get("read"):
            _inbox.mark_read(thread_id)
        else:
            _inbox.mark_unread(thread_id)
        return ok({"thread_id": thread_id, "read": bool(body.get("read"))})
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/ai", methods=["POST"])
@require_console_key
def api_inbox_ai(thread_id):
    """Generate summary + numbered actions + initial suggested reply in one call."""
    try:
        from dashboard import inbox_ai as _ai
        thread = _inbox.get_thread(thread_id)
        msgs = thread.get("messages") or []
        if not msgs:
            return fail("thread is empty", status=400)
        last = msgs[-1]
        body_clean = last.get("body_clean") or last.get("body_plain") or ""
        sender = last.get("from", "")
        # Pull a few of Glen's recent sent emails as voice context
        try:
            voice_samples = _inbox.list_recent_sent(max_results=3)
        except Exception:
            voice_samples = []
        summary = _ai.summarize(body_clean)
        draft = _ai.draft_reply(body_clean, sender=sender, voice_samples=voice_samples)
        return ok({
            "essence": summary.get("essence", ""),
            "summary": summary.get("summary", []),
            "actions": summary.get("actions", []),
            "draft": draft,
            "body_clean": body_clean,
            "sender": sender,
        })
    except Exception as e: return fail(e)


@app.route("/api/inbox/hide-sender", methods=["POST"])
@require_console_key
def api_inbox_hide_sender():
    try:
        body = request.get_json(silent=True) or {}
        sender = (body.get("sender") or "").strip()
        if not sender:
            return fail("sender field required", status=400)
        return ok(_inbox.hide_sender(sender))
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/inbox/unhide-sender", methods=["POST"])
@require_console_key
def api_inbox_unhide_sender():
    try:
        body = request.get_json(silent=True) or {}
        sender = (body.get("sender") or "").strip()
        if not sender:
            return fail("sender field required", status=400)
        return ok(_inbox.unhide_sender(sender))
    except Exception as e: return fail(e)


@app.route("/api/inbox/hidden-senders", methods=["GET"])
@require_console_key
def api_inbox_list_hidden():
    try: return ok(_inbox.list_hidden_senders())
    except Exception as e: return fail(e)


@app.route("/api/inbox/threads/<thread_id>/regenerate-reply", methods=["POST"])
@require_console_key
def api_inbox_regenerate(thread_id):
    """Re-draft suggested reply using Glen's prompt instructions."""
    try:
        from dashboard import inbox_ai as _ai
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
        prior = (body.get("prior_draft") or "").strip()
        if not prompt:
            return fail("prompt is required", status=400)
        thread = _inbox.get_thread(thread_id)
        msgs = thread.get("messages") or []
        if not msgs:
            return fail("thread is empty", status=400)
        last = msgs[-1]
        body_clean = last.get("body_clean") or last.get("body_plain") or ""
        sender = last.get("from", "")
        try:
            voice_samples = _inbox.list_recent_sent(max_results=3)
        except Exception:
            voice_samples = []
        new_draft = _ai.regenerate_reply(
            body_clean, prior_draft=prior, prompt=prompt,
            sender=sender, voice_samples=voice_samples,
        )
        return ok({"draft": new_draft})
    except Exception as e: return fail(e)


# ── Shipping (Order-Flow Plumbing) ────────────────────────────────────────────
# /admin/shipping  — Glen + Rae manage bottle catalog, box-fit matrix, USPS rates
# /orders/new      — Rae enters a phone/email order; tool auto-picks box + cost
# /api/shipping/*  — JSON API behind require_console_key
from dashboard import shipping as _shipping


@app.route("/admin/shipping")
def admin_shipping_page():
    resp = send_from_directory(STATIC, "admin-shipping.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/orders/new")
def order_new_page():
    resp = send_from_directory(STATIC, "order-new.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/shipping/bottles", methods=["GET"])
@require_console_key
def api_shipping_list_bottles():
    try: return ok(_shipping.list_bottle_types())
    except Exception as e: return fail(e)


@app.route("/api/shipping/bottles", methods=["POST"])
@require_console_key
def api_shipping_add_bottle():
    try:
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return fail("name is required", status=400)
        notes = (body.get("notes") or "").strip() or None
        new_id = _shipping.add_bottle_type(name, notes=notes)
        return ok({"id": new_id})
    except sqlite3.IntegrityError:
        return fail("bottle type already exists", status=409)
    except Exception as e: return fail(e)


@app.route("/api/shipping/bottles/<int:bid>", methods=["DELETE"])
@require_console_key
def api_shipping_delete_bottle(bid):
    try:
        _shipping.delete_bottle_type(bid)
        return ok({"deleted": bid})
    except Exception as e: return fail(e)


@app.route("/api/shipping/bottles/<int:bid>", methods=["PATCH"])
@require_console_key
def api_shipping_update_bottle(bid):
    try:
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return fail("name is required", status=400)
        notes = (body.get("notes") or "").strip() or None
        _shipping.update_bottle_type(bid, name, notes=notes)
        return ok({"id": bid})
    except sqlite3.IntegrityError:
        return fail("bottle name already exists", status=409)
    except Exception as e: return fail(e)


@app.route("/api/shipping/matrix", methods=["GET"])
@require_console_key
def api_shipping_matrix():
    try: return ok(_shipping.get_capacity_matrix())
    except Exception as e: return fail(e)


@app.route("/api/shipping/capacity", methods=["POST"])
@require_console_key
def api_shipping_set_capacity():
    try:
        body = request.get_json(silent=True) or {}
        bid = int(body.get("bottle_type_id"))
        size = body.get("box_size")
        qty = int(body.get("qty"))
        _shipping.set_box_capacity(bid, size, qty)
        return ok({"bottle_type_id": bid, "box_size": size, "qty": qty})
    except (TypeError, ValueError) as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/shipping/rates", methods=["GET"])
@require_console_key
def api_shipping_rates():
    try:
        return ok({
            "current": _shipping.get_current_rates(),
            "pending": _shipping.list_pending_rate_updates(),
        })
    except Exception as e: return fail(e)


@app.route("/api/shipping/rates/propose", methods=["POST"])
@require_console_key
def api_shipping_propose_rate():
    try:
        body = request.get_json(silent=True) or {}
        rate_id = _shipping.propose_rate_update(
            box_size=body["box_size"],
            usps_retail_cents=int(body["usps_retail_cents"]),
            source_url=body.get("source_url", ""),
            effective_date=body["effective_date"],
        )
        return ok({"id": rate_id})
    except (KeyError, ValueError, TypeError) as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/shipping/rates/<int:rid>/confirm", methods=["POST"])
@require_console_key
def api_shipping_confirm_rate(rid):
    try:
        body = request.get_json(silent=True) or {}
        confirmed_by = (body.get("confirmed_by") or "glen").strip()
        _shipping.confirm_rate_update(rid, confirmed_by=confirmed_by)
        return ok({"confirmed": rid, "by": confirmed_by})
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e)


@app.route("/api/shipping/quote", methods=["POST"])
@require_console_key
def api_shipping_quote():
    """Order-entry helper: bottles_by_type → {box_size, shipping_cents}."""
    try:
        body = request.get_json(silent=True) or {}
        bottles = body.get("bottles") or {}
        # sanitize: positive ints only
        clean = {str(k): int(v) for k, v in bottles.items() if int(v) > 0}
        return ok(_shipping.quote(clean))
    except _shipping.UnknownBottleType as e:
        return fail(f"unknown bottle type: {e}", status=400)
    except (TypeError, ValueError) as e: return fail(e, status=400)
    except Exception as e: return fail(e)


# ─────────────────────────────────────────────────────────────────────────────
# /console/settings — collapsible settings panel (Shipping, Active-Mac, etc.)
# ─────────────────────────────────────────────────────────────────────────────
from dashboard import settings as _settings


@app.route("/console/settings")
def console_settings_page():
    resp = send_from_directory(STATIC, "console-settings.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/settings/active-mac", methods=["GET"])
@require_console_key
def api_settings_get_active_mac():
    try: return ok(_settings.active_mac_payload())
    except Exception as e: return fail(e)


@app.route("/api/settings/active-mac", methods=["POST"])
@require_console_key
def api_settings_set_active_mac():
    try:
        body = request.get_json(silent=True) or {}
        hostname = (body.get("hostname") or "").strip()
        _settings.set_active_mac(hostname)
        return ok(_settings.active_mac_payload())
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e)


# ─────────────────────────────────────────────────────────────────────────────
# /api/admin/chat-log-export — incremental query_log dump for local mirror.
# Render's free-tier disk is ephemeral; chat_log.db is wiped on every restart.
# A local mirror at ~/AI-Training/chat_log_mirror.db (populated hourly via
# 02 Skills/pull-chat-log.py) preserves durable history. Read-only, paginated.
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/admin/chat-log-export", methods=["GET"])
@require_console_key
def api_admin_chat_log_export():
    try:
        since = int(request.args.get("since", "0"))
    except ValueError:
        return fail("since must be an integer rowid", status=400)
    try:
        limit = min(int(request.args.get("limit", "5000")), 10000)
    except ValueError:
        return fail("limit must be an integer", status=400)
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            rows = cx.execute(
                "SELECT id, ts, query, level, answer, rating, rated_at, "
                "session_id, email, ghl_contact_id, mode, full_answer, "
                "name, user_agent, referer, image_count, email_sent_at "
                "FROM query_log WHERE id > ? ORDER BY id ASC LIMIT ?",
                (since, limit),
            ).fetchall()
        out = [dict(r) for r in rows]
        return ok({
            "rows": out,
            "count": len(out),
            "last_id": out[-1]["id"] if out else since,
            "has_more": len(out) == limit,
        })
    except Exception as e:
        return fail(e)


# ─────────────────────────────────────────────────────────────────────────────
# /console/projects — kanban view of 00 System/PROJECTS.md from the vault
# ─────────────────────────────────────────────────────────────────────────────
from dashboard import projects as _projects


@app.route("/console/projects")
def console_projects_page():
    resp = send_from_directory(STATIC, "console-projects.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/projects", methods=["GET"])
@require_console_key
def api_projects_kanban():
    try: return ok(_projects.kanban_payload())
    except Exception as e: return fail(e)


@app.route("/api/projects/upload", methods=["POST"])
@require_console_key
def api_projects_upload():
    try: return ok(_projects.write_projects(request.get_data()))
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e, status=500)


@app.route("/api/projects/add-idea", methods=["POST"])
@require_console_key
def api_projects_add_idea():
    try:
        data = request.get_json(force=True, silent=True) or {}
        return ok(_projects.add_pending_idea(data.get("text", "")))
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e, status=500)


@app.route("/api/projects/pending-ideas", methods=["GET"])
@require_console_key
def api_projects_pending_ideas():
    try: return ok({"ideas": _projects.pending_ideas()})
    except Exception as e: return fail(e)


@app.route("/api/projects/pending-ideas/clear", methods=["POST"])
@require_console_key
def api_projects_pending_clear():
    try:
        data = request.get_json(force=True, silent=True) or {}
        return ok(_projects.clear_pending_ideas(data.get("ids", [])))
    except Exception as e: return fail(e, status=500)


@app.route("/api/projects/edit", methods=["POST"])
@require_console_key
def api_projects_edit():
    """Queue any typed tracker edit (add_idea / move / set / drop)."""
    try:
        return ok(_projects.add_pending_edit(request.get_json(force=True) or {}))
    except ValueError as e: return fail(e, status=400)
    except Exception as e: return fail(e, status=500)


@app.route("/api/projects/pending-edits", methods=["GET"])
@require_console_key
def api_projects_pending_edits():
    try: return ok({"edits": _projects.pending_edits()})
    except Exception as e: return fail(e)


# ─────────────────────────────────────────────────────────────────────────────
# Practitioner Finder
# ─────────────────────────────────────────────────────────────────────────────
import scrapers.practitioner_finder.db as pf_db
import scrapers.practitioner_finder.geocode as pf_geocode
from scrapers.practitioner_finder.geocode import MapboxError as PfMapboxError


# A whole-country radius cap: larger than the great-circle span of any single
# country, so the earthdistance circle effectively becomes "everything in the
# selected country" while the SELECT's ORDER BY distance_miles still sorts by
# proximity to the typed place.
PF_COUNTRYWIDE_RADIUS_MILES = 12500.0


def _pf_parse_radius(raw: str) -> float:
    """Map the radius_miles param to miles. Accepts a number, the 'country-wide'
    sentinel, or the legacy '9999' Nationwide value."""
    raw = (raw or "25").strip().lower()
    if raw in ("country-wide", "countrywide", "nationwide", "9999"):
        return PF_COUNTRYWIDE_RADIUS_MILES
    return float(raw)


@app.route("/api/practitioner-finder/search", methods=["GET", "OPTIONS"])
def practitioner_finder_search():
    if request.method == "OPTIONS":
        return "", 200

    # `location` is the free-text place; `zip` is kept as a back-compat alias.
    location = request.args.get("location", "").strip() or request.args.get("zip", "").strip()
    if not location:
        return jsonify({"error": "location (or zip) query param is required"}), 400

    # country: ISO-2 (default US). "ANY"/empty => international, no country filter.
    country = request.args.get("country", "US").strip().upper()
    international = country in ("", "ANY")

    try:
        radius_miles = _pf_parse_radius(request.args.get("radius_miles", "25"))
    except ValueError:
        return jsonify({"error": "radius_miles must be a number or 'country-wide'"}), 400

    specialties = request.args.getlist("specialties[]") or None
    tiers = request.args.getlist("tier[]") or None
    fellowship_only = request.args.get("fellowship_only", "").lower() in ("1", "true", "yes")

    # Geocode the typed place to a search centre, biased to the chosen country.
    try:
        lat, lng = pf_geocode.geocode_place(location, None if international else country)
    except PfMapboxError as e:
        return jsonify({"error": f"geocoding failed: {e}"}), 502
    if lat is None or lng is None:
        return jsonify({"error": f"could not locate '{location}'"}), 404

    results = pf_db.run_search(
        lat=lat, lng=lng, radius_miles=radius_miles,
        specialties=specialties, tiers=tiers, limit=200,
        fellowship_only=fellowship_only,
        countries=None if international else [country],
    )
    # Certification-track practitioners: never expose raw email/phone in the
    # public payload — contact is routed through the inquiry form (which fetches
    # the address server-side). Website + cert level stay visible.
    for r in results:
        if r.get("tier") in ("panel_in_cert", "panel_certified") and not r.get("show_contact"):
            r["email"] = None
            r["phone"] = None
        r.pop("show_contact", None)
    return jsonify({"count": len(results), "practitioners": results,
                    "search_center": {"lat": lat, "lng": lng}})


# Friendly names for the countries that actually appear in the data. Anything
# not listed falls back to its raw code so the dropdown never shows a blank.
PF_COUNTRY_NAMES = {
    "US": "United States", "CA": "Canada", "GB": "United Kingdom",
    "AU": "Australia", "NZ": "New Zealand", "IE": "Ireland",
    "KR": "South Korea", "JP": "Japan", "CN": "China", "HK": "Hong Kong",
    "SG": "Singapore", "MY": "Malaysia", "ID": "Indonesia", "IN": "India",
    "AE": "United Arab Emirates", "SA": "Saudi Arabia", "IL": "Israel",
    "TR": "Turkey", "EG": "Egypt", "ZA": "South Africa", "MX": "Mexico",
    "BR": "Brazil", "CO": "Colombia", "CL": "Chile", "EC": "Ecuador",
    "VE": "Venezuela", "PY": "Paraguay", "GT": "Guatemala", "CR": "Costa Rica",
    "DO": "Dominican Republic", "PR": "Puerto Rico", "DE": "Germany",
    "FR": "France", "ES": "Spain", "IT": "Italy", "PT": "Portugal",
    "NL": "Netherlands", "BE": "Belgium", "CH": "Switzerland", "AT": "Austria",
    "CZ": "Czechia", "PL": "Poland", "HU": "Hungary", "SK": "Slovakia",
    "SI": "Slovenia", "HR": "Croatia", "GR": "Greece", "RO": "Romania",
    "LV": "Latvia", "GE": "Georgia", "CY": "Cyprus", "LU": "Luxembourg",
    "LB": "Lebanon", "JO": "Jordan", "IQ": "Iraq", "IR": "Iran",
    "TZ": "Tanzania", "ZW": "Zimbabwe", "MU": "Mauritius", "JM": "Jamaica",
    "BB": "Barbados", "GY": "Guyana", "PH": "Philippines",
}


@app.route("/api/practitioner-finder/countries", methods=["GET"])
def practitioner_finder_countries():
    """List countries present in the (geocodable) practitioner data, with counts,
    for the finder's country selector."""
    from db_supabase import supabase_cursor
    try:
        with supabase_cursor() as cur:
            cur.execute(
                "SELECT country, count(*) AS n FROM v_practitioners_public "
                "WHERE lat IS NOT NULL AND country IS NOT NULL "
                "GROUP BY country ORDER BY n DESC"
            )
            rows = cur.fetchall()
    except Exception as e:
        print(f"[pf-countries] error: {e!r}", flush=True)
        return jsonify({"countries": []}), 200
    countries = [
        {"code": r["country"],
         "name": PF_COUNTRY_NAMES.get(r["country"], r["country"]),
         "count": r["n"]}
        for r in rows
    ]
    return jsonify({"countries": countries})


@app.route("/practitioner-finder", methods=["GET"])
def practitioner_finder_page():
    """Serve the finder page with Mapbox public token injected."""
    token = os.environ.get("MAPBOX_PUBLIC_TOKEN", "")
    html_path = Path(__file__).parent / "static" / "practitioner-finder.html"
    html = html_path.read_text()
    # Inject token via window global — searched in <script> by the page
    # Token is a Mapbox public pk.* — safe to expose to browser
    injection = f"<script>window.__MAPBOX_TOKEN__ = {token!r};</script>"
    html = html.replace("</head>", injection + "\n</head>")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


def _fetch_practitioners_by_ids(ids: list) -> list:
    """Fetch practitioner records from Supabase by a list of id strings.

    Returns a list of dicts with at minimum: id, email, name, accepts_inquiries.
    Mirrors the connection style of practitioner_finder_search (pf_db / supabase_cursor).
    Unknown ids are silently omitted — callers check len(result) vs len(ids).
    """
    if not ids:
        return []
    from db_supabase import supabase_cursor
    # Use %s placeholders; psycopg2 fills them positionally
    placeholders = ", ".join(["%s"] * len(ids))
    sql = f"""
        SELECT id, name, email, accepts_inquiries
        FROM practitioners
        WHERE id::text = ANY(ARRAY[{placeholders}])
    """
    try:
        with supabase_cursor() as cur:
            cur.execute(sql, list(ids))
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[inquiry] _fetch_practitioners_by_ids error: {e!r}", flush=True)
        return []


def _set_practitioner_accepts_inquiries(practitioner_id, value, verified=False):
    """UPDATE practitioners.accepts_inquiries in Supabase.

    If verified=True also sets claim_verified_at=now().
    Returns True on success, False on failure (never raises).
    """
    from db_supabase import supabase_cursor
    try:
        with supabase_cursor() as cur:
            if verified:
                cur.execute(
                    "UPDATE practitioners SET accepts_inquiries=%s, claim_verified_at=now() "
                    "WHERE id::text=%s",
                    (value, str(practitioner_id))
                )
            else:
                cur.execute(
                    "UPDATE practitioners SET accepts_inquiries=%s WHERE id::text=%s",
                    (value, str(practitioner_id))
                )
        return True
    except Exception as e:
        print(f"[inquiry] _set_practitioner_accepts_inquiries error: {e!r}", flush=True)
        return False


@app.route("/api/practitioner-finder/inquiry", methods=["POST", "OPTIONS"])
def practitioner_finder_inquiry():
    """POST /api/practitioner-finder/inquiry

    Accept a multi-cast inquiry from a site visitor.  Fan-out one email per
    selected practitioner, record the inquiry in SQLite, fire a
    journey_events row, and return {inquiry_id, sent_count, skipped}.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(force=True) or {}

    # ── Validation ────────────────────────────────────────────────────────────
    client_name      = (data.get("client_name") or "").strip()
    client_email     = (data.get("client_email") or "").strip().lower()
    client_phone     = (data.get("client_phone") or "").strip()
    main_challenge   = (data.get("main_challenge") or "").strip()
    main_goal        = (data.get("main_goal") or "").strip()
    practitioner_ids = data.get("practitioner_ids") or []

    missing = []
    if not client_name:
        missing.append("client_name")
    if not client_email:
        missing.append("client_email")
    elif "@" not in client_email or "." not in client_email:
        return jsonify({"error": "client_email must be a valid email address"}), 400
    if not main_challenge:
        missing.append("main_challenge")
    if not main_goal:
        missing.append("main_goal")
    if not practitioner_ids:
        missing.append("practitioner_ids")
    if missing:
        return jsonify({"error": f"missing required fields: {', '.join(missing)}"}), 400
    if not isinstance(practitioner_ids, list):
        return jsonify({"error": "practitioner_ids must be a list"}), 400

    # ── Rate limit 1: max 20 practitioners ───────────────────────────────────
    if len(practitioner_ids) > 20:
        return jsonify({"error": "max 20 practitioners per inquiry"}), 400

    # ── Session handling (mirrors /begin/unlock pattern exactly) ─────────────
    session_id     = (request.cookies.get("amg_session") or "").strip()
    minted_session = not session_id
    if minted_session:
        session_id = secrets.token_urlsafe(16)

    ref_slug = (request.cookies.get("rm_ref") or "").strip()
    ip       = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()

    sorted_ids = sorted(str(i) for i in practitioner_ids)
    pcount     = len(sorted_ids)

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Ensure ip column exists (defensive ALTER — mirrors schema-evolution pattern)
        try:
            cx.execute("ALTER TABLE inquiries ADD COLUMN ip TEXT")
            cx.commit()
        except sqlite3.OperationalError:
            pass  # already exists

        # ── Rate limit 2: one inquiry per session per 24h (different email or set) ──
        prior = cx.execute(
            "SELECT id, client_email, practitioner_count FROM inquiries "
            "WHERE session_id=? AND created_at > datetime('now','-24 hour')",
            (session_id,)
        ).fetchone()
        if prior:
            prior_id, prior_email, prior_count = prior
            # Check if this is a true de-dupe (same email + same set)
            is_dedup = False
            if prior_email == client_email and prior_count == pcount:
                prior_pids = sorted(
                    row[0] for row in cx.execute(
                        "SELECT practitioner_id FROM inquiry_practitioners WHERE inquiry_id=?",
                        (prior_id,)
                    ).fetchall()
                )
                if prior_pids == sorted_ids:
                    is_dedup = True
            if is_dedup:
                # Idempotent replay — return existing inquiry_id, skip all sends
                resp = jsonify({
                    "inquiry_id": prior_id,
                    "sent_count": 0,
                    "skipped": [],
                    "deduped": True,
                })
                return resp, 200
            else:
                return jsonify({"error": "one inquiry per day"}), 429

        # ── Rate limit 3: 3 per IP per 24h ───────────────────────────────────
        if ip:
            ip_count = cx.execute(
                "SELECT COUNT(*) FROM inquiries WHERE ip=? AND created_at > datetime('now','-24 hour')",
                (ip,)
            ).fetchone()[0]
            if ip_count >= 3:
                return jsonify({"error": "too many inquiries from this network today"}), 429

    # ── Fetch practitioners + build to_send / skipped lists ──────────────────
    records     = _fetch_practitioners_by_ids(sorted_ids)
    records_map = {str(r["id"]): r for r in records}

    # Check opt-outs in batch
    with sqlite3.connect(LOG_DB) as cx:
        opted_out_emails = {
            row[0] for row in cx.execute(
                "SELECT email FROM practitioner_inquiry_opt_outs"
            ).fetchall()
        }

    to_send = []
    skipped = []

    for pid in sorted_ids:
        rec = records_map.get(pid)
        if rec is None:
            skipped.append({"practitioner_id": pid, "reason": "not_found"})
            continue
        ai = rec.get("accepts_inquiries")
        if ai is False:
            skipped.append({"practitioner_id": pid, "reason": "opted_out_at_listing"})
            continue
        email_addr = (rec.get("email") or "").strip()
        if not email_addr:
            skipped.append({"practitioner_id": pid, "reason": "no_email"})
            continue
        if email_addr in opted_out_emails:
            skipped.append({"practitioner_id": pid, "reason": "globally_opted_out"})
            continue
        to_send.append(rec)

    # ── Inserts (single transaction) ─────────────────────────────────────────
    inquiry_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    ts_now     = datetime.utcnow()
    base_url   = request.host_url.rstrip("/")

    # Per-recipient tokens (minted before the DB write)
    send_tokens = []   # [(rec, plain_reply, plain_optout, plain_claim)]
    for rec in to_send:
        plain_reply   = secrets.token_urlsafe(32)
        plain_optout  = secrets.token_urlsafe(32)
        plain_claim   = secrets.token_urlsafe(32)
        send_tokens.append((rec, plain_reply, plain_optout, plain_claim))

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # inquiries row
        cx.execute(
            "INSERT INTO inquiries "
            "(id, created_at, session_id, client_email, client_name, client_phone, "
            "ref_slug, main_challenge, main_goal, practitioner_count, ip) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (inquiry_id, created_at, session_id, client_email, client_name, client_phone,
             ref_slug or None, main_challenge, main_goal, pcount, ip or None)
        )
        # inquiry_practitioners rows for skipped recipients too, so the dedupe SELECT
        # sees the full requested set (else a replay with all-skipped ids returns 429)
        for s in skipped:
            cx.execute(
                "INSERT INTO inquiry_practitioners "
                "(id, inquiry_id, practitioner_id, practitioner_email, status, email_sent_at) "
                "VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), inquiry_id, s["practitioner_id"],
                 (records_map.get(s["practitioner_id"], {}).get("email") or "").strip(),
                 "skipped_" + s["reason"], None)
            )
        # inquiry_practitioners rows for to_send (the ones we actually email)
        for rec, plain_reply, plain_optout, plain_claim in send_tokens:
            pid        = str(rec["id"])
            email_addr = rec["email"].strip()
            ip_row_id  = str(uuid.uuid4())
            cx.execute(
                "INSERT INTO inquiry_practitioners "
                "(id, inquiry_id, practitioner_id, practitioner_email, status, email_sent_at) "
                "VALUES (?,?,?,?,?,?)",
                (ip_row_id, inquiry_id, pid, email_addr, "sent", created_at)
            )
            # inquiry_reply_tokens
            expires_reply = (ts_now + timedelta(days=30)).isoformat() + "Z"
            cx.execute(
                "INSERT OR REPLACE INTO inquiry_reply_tokens "
                "(token_hash, inquiry_id, practitioner_id, created_at, expires_at) "
                "VALUES (?,?,?,?,?)",
                (_hash_token(plain_reply), inquiry_id, pid, created_at, expires_reply)
            )
            # auth_tokens: practitioner_optout (365d)
            expires_optout = (ts_now + timedelta(days=365)).isoformat() + "Z"
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?)",
                (_hash_token(plain_optout), email_addr, "practitioner_optout",
                 json.dumps({"practitioner_id": pid}),
                 created_at, expires_optout)
            )
            # auth_tokens: practitioner_claim (7d)
            expires_claim = (ts_now + timedelta(days=7)).isoformat() + "Z"
            cx.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?)",
                (_hash_token(plain_claim), email_addr, "practitioner_claim",
                 json.dumps({"practitioner_id": pid}),
                 created_at, expires_claim)
            )
        cx.commit()

    # ── Fan-out emails ────────────────────────────────────────────────────────
    client_first = client_name.split(None, 1)[0] if client_name else "Someone"
    sent_count   = 0

    for rec, plain_reply, plain_optout, plain_claim in send_tokens:
        pid              = str(rec["id"])
        pract_email      = rec["email"].strip()
        pract_name       = (rec.get("name") or pract_email)
        reply_url        = f"{base_url}/inquiries/{inquiry_id}/{pid}/reply?token={plain_reply}"
        optout_url       = f"{base_url}/practitioner-optout/{plain_optout}"
        claim_url        = f"{base_url}/practitioner-claim/{plain_claim}"

        subject = "A potential client is asking about your work"
        body = (
            f"Hi {pract_name},\n\n"
            f"{client_first} found your listing on RemedyMatch.com and asked us to share "
            f"their question with you.\n\n"
            f"What they are working through:\n{main_challenge}\n\n"
            f"What success looks like for them:\n{main_goal}\n\n"
            f"How to respond:\n"
            f"Reply by email (just hit Reply) - it goes straight to {client_email}\n\n"
            f"Or reply via secure link:\n{reply_url}\n\n"
            f"---\n"
            f"You received this because your listing appears on RemedyMatch.com. "
            f"To stop receiving future inquiries:\n{optout_url}\n\n"
            f"To claim your listing and display a 'Verified Responsive' badge:\n{claim_url}\n\n"
            f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
            f"This message was sent on behalf of {client_first}; "
            f"you can reply directly to {client_email}.\n"
        )

        ok = _send_inquiry_email(
            to_email=pract_email,
            subject=subject,
            body=body,
            reply_to=client_email,
        )
        if ok:
            sent_count += 1
        else:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute(
                    "UPDATE inquiry_practitioners SET status='failed' "
                    "WHERE inquiry_id=? AND practitioner_id=?",
                    (inquiry_id, pid)
                )
                cx.commit()

    # ── Journey signal (fire-and-forget) ─────────────────────────────────────
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO journey_events "
                "(ts, session_id, email, trigger, detail, rung_before, rung_after) "
                "VALUES (?, ?, ?, 'practitioner_inquiry', ?, '', '')",
                (created_at, session_id, client_email,
                 json.dumps({"count": len(to_send), "ref_slug": ref_slug or ""}))
            )
            cx.commit()
    except Exception as e:
        print(f"[inquiry] journey_events insert failed: {e!r}", flush=True)

    # ── Client receipt (transactional, fire-and-forget) ──────────────────────
    try:
        if to_send:
            _send_client_receipt(client_email, client_name, [r for r, *_ in send_tokens], base_url)
    except Exception as e:
        print(f"[inquiry] client receipt send failed: {e!r}", flush=True)

    # ── Response ──────────────────────────────────────────────────────────────
    resp = jsonify({
        "inquiry_id": inquiry_id,
        "sent_count": sent_count,
        "skipped": skipped,
    })
    if minted_session:
        resp.set_cookie(
            "amg_session", session_id, max_age=60 * 60 * 24 * 365,
            httponly=True, samesite="Lax", secure=request.is_secure,
        )
    return resp, 200


# ── Slice 3: token-gated practitioner + reply routes ─────────────────────────

def _render_static_template(filename, **ctx):
    """Load a file from static/ and render it as a Jinja2 template string.

    Flask autoescape is on for .html files rendered via render_template_string,
    so all {{ var }} substitutions are HTML-escaped automatically.
    """
    path = Path(__file__).parent / "static" / filename
    text = path.read_text(encoding="utf-8")
    return render_template_string(text, **ctx)


def _validate_auth_token(cx, token_plain, purpose):
    """Validate a plaintext auth token against the DB.

    Returns (row_dict, error_html_or_None).
    row_dict includes: token_hash, email, purpose, extra, consumed_at, expires_at.
    On failure returns (None, html_string) — caller must return the html with a 4xx code.
    """
    th = _hash_token(token_plain)
    row = cx.execute(
        "SELECT token_hash, email, purpose, extra, consumed_at, expires_at "
        "FROM auth_tokens WHERE token_hash=? AND purpose=?",
        (th, purpose)
    ).fetchone()
    if not row:
        return None, (_render_static_template(
            "practitioner-claim.html" if purpose == "practitioner_claim" else "practitioner-optout.html",
            status="error"), 400)
    consumed_at, expires_at = row[4], row[5]
    if consumed_at:
        return None, (_render_static_template(
            "practitioner-claim.html" if purpose == "practitioner_claim" else "practitioner-optout.html",
            status="error"), 410)
    try:
        exp_dt = datetime.fromisoformat(expires_at.rstrip("Z"))
        # make naive comparison consistent
        now_cmp = datetime.utcnow()
        if exp_dt < now_cmp:
            return None, (_render_static_template(
                "practitioner-claim.html" if purpose == "practitioner_claim" else "practitioner-optout.html",
                status="error"), 410)
    except Exception:
        return None, (_render_static_template(
            "practitioner-claim.html" if purpose == "practitioner_claim" else "practitioner-optout.html",
            status="error"), 400)
    return {"token_hash": row[0], "email": row[1], "purpose": row[2],
            "extra": row[3], "consumed_at": row[4], "expires_at": row[5]}, None


@app.route("/practitioner-claim/<token>", methods=["GET"])
def practitioner_claim_get(token):
    """Render the claim form for a practitioner."""
    token = token.strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row, err = _validate_auth_token(cx, token, "practitioner_claim")
    if err:
        html, code = err
        return html, code, {"Content-Type": "text/html; charset=utf-8"}

    extra = json.loads(row["extra"] or "{}")
    pid = extra.get("practitioner_id", "")
    recs = _fetch_practitioners_by_ids([pid]) if pid else []
    name = recs[0].get("name", "") if recs else ""
    html = _render_static_template("practitioner-claim.html",
                                   status="form", token=token, name=name)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/practitioner-claim/<token>", methods=["POST"])
def practitioner_claim_post(token):
    """Consume the claim token and flip accepts_inquiries=True in Supabase."""
    token = token.strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row, err = _validate_auth_token(cx, token, "practitioner_claim")
        if err:
            html, code = err
            return html, code, {"Content-Type": "text/html; charset=utf-8"}
        # Mark consumed atomically inside the same lock/connection
        th = row["token_hash"]
        now_iso = _now_utc().isoformat()
        cx.execute(
            "UPDATE auth_tokens SET consumed_at=? WHERE token_hash=? AND consumed_at IS NULL",
            (now_iso, th)
        )
        cx.commit()

    extra = json.loads(row["extra"] or "{}")
    pid = extra.get("practitioner_id", "")
    _set_practitioner_accepts_inquiries(pid, True, verified=True)
    html = _render_static_template("practitioner-claim.html",
                                   status="confirmed", token=token, name="")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/practitioner-optout/<token>", methods=["GET"])
def practitioner_optout(token):
    """Record opt-out and flip accepts_inquiries=False in Supabase."""
    token = token.strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row, err = _validate_auth_token(cx, token, "practitioner_optout")
        if err:
            html, code = err
            return html, code, {"Content-Type": "text/html; charset=utf-8"}
        th = row["token_hash"]
        email = row["email"]
        extra = json.loads(row["extra"] or "{}")
        pid = extra.get("practitioner_id", "")
        now_iso = _now_utc().isoformat()
        cx.execute(
            "INSERT OR REPLACE INTO practitioner_inquiry_opt_outs (email, ts, practitioner_id) "
            "VALUES (?, ?, ?)",
            (email, now_iso, pid or None)
        )
        cx.execute(
            "UPDATE auth_tokens SET consumed_at=? WHERE token_hash=? AND consumed_at IS NULL",
            (now_iso, th)
        )
        cx.commit()

    _set_practitioner_accepts_inquiries(pid, False, verified=False)
    html = _render_static_template("practitioner-optout.html", status="done")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/inquiries/<inquiry_id>/<practitioner_id>/reply", methods=["GET"])
def inquiry_reply_get(inquiry_id, practitioner_id):
    """Render the reply form and record an impression."""
    token_plain = (request.args.get("token") or "").strip()
    if not token_plain:
        html = _render_static_template("inquiry-reply.html", status="error")
        return html, 400, {"Content-Type": "text/html; charset=utf-8"}

    th = _hash_token(token_plain)
    now_iso = _now_utc().isoformat()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    ua = request.headers.get("User-Agent", "")

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Validate reply token
        tok_row = cx.execute(
            "SELECT token_hash, expires_at FROM inquiry_reply_tokens "
            "WHERE token_hash=? AND inquiry_id=? AND practitioner_id=?",
            (th, inquiry_id, practitioner_id)
        ).fetchone()
        if not tok_row:
            html = _render_static_template("inquiry-reply.html", status="error")
            return html, 404, {"Content-Type": "text/html; charset=utf-8"}
        try:
            exp_dt = datetime.fromisoformat(tok_row[1].rstrip("Z"))
            if exp_dt < datetime.utcnow():
                html = _render_static_template("inquiry-reply.html", status="error")
                return html, 410, {"Content-Type": "text/html; charset=utf-8"}
        except Exception:
            html = _render_static_template("inquiry-reply.html", status="error")
            return html, 400, {"Content-Type": "text/html; charset=utf-8"}

        # Record impression
        cx.execute(
            "INSERT INTO inquiry_reply_impressions (ts, inquiry_id, practitioner_id, ip, user_agent) "
            "VALUES (?, ?, ?, ?, ?)",
            (now_iso, inquiry_id, practitioner_id, ip or None, ua or None)
        )

        # Fetch inquiry context
        inq = cx.execute(
            "SELECT main_challenge, main_goal, client_email, client_name "
            "FROM inquiries WHERE id=?",
            (inquiry_id,)
        ).fetchone()
        cx.commit()

    if not inq:
        html = _render_static_template("inquiry-reply.html", status="error")
        return html, 404, {"Content-Type": "text/html; charset=utf-8"}

    main_challenge, main_goal, _client_email, _client_name = inq
    html = _render_static_template(
        "inquiry-reply.html",
        status="form",
        main_challenge=main_challenge,
        main_goal=main_goal,
        inquiry_id=inquiry_id,
        practitioner_id=practitioner_id,
        token=token_plain,
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/inquiries/<inquiry_id>/<practitioner_id>/reply", methods=["POST"])
def inquiry_reply_post(inquiry_id, practitioner_id):
    """Insert reply, forward to client via email, mark status=replied."""
    token_plain = (request.form.get("token") or "").strip()
    body = (request.form.get("body") or "").strip()

    if not token_plain:
        html = _render_static_template("inquiry-reply.html", status="error")
        return html, 400, {"Content-Type": "text/html; charset=utf-8"}
    if not body:
        html = _render_static_template("inquiry-reply.html", status="error",
                                       main_challenge="", main_goal="",
                                       inquiry_id=inquiry_id, practitioner_id=practitioner_id,
                                       token=token_plain)
        return html, 400, {"Content-Type": "text/html; charset=utf-8"}

    th = _hash_token(token_plain)
    now_iso = _now_utc().isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Validate reply token
        tok_row = cx.execute(
            "SELECT token_hash, expires_at FROM inquiry_reply_tokens "
            "WHERE token_hash=? AND inquiry_id=? AND practitioner_id=?",
            (th, inquiry_id, practitioner_id)
        ).fetchone()
        if not tok_row:
            html = _render_static_template("inquiry-reply.html", status="error")
            return html, 404, {"Content-Type": "text/html; charset=utf-8"}
        try:
            exp_dt = datetime.fromisoformat(tok_row[1].rstrip("Z"))
            if exp_dt < datetime.utcnow():
                html = _render_static_template("inquiry-reply.html", status="error")
                return html, 410, {"Content-Type": "text/html; charset=utf-8"}
        except Exception:
            html = _render_static_template("inquiry-reply.html", status="error")
            return html, 400, {"Content-Type": "text/html; charset=utf-8"}

        # Fetch inquiry context
        inq = cx.execute(
            "SELECT main_challenge, main_goal, client_email, client_name "
            "FROM inquiries WHERE id=?",
            (inquiry_id,)
        ).fetchone()
        if not inq:
            html = _render_static_template("inquiry-reply.html", status="error")
            return html, 404, {"Content-Type": "text/html; charset=utf-8"}
        main_challenge, main_goal, client_email, client_name = inq

        # Fetch practitioner email from inquiry_practitioners
        pract_row = cx.execute(
            "SELECT practitioner_email FROM inquiry_practitioners "
            "WHERE inquiry_id=? AND practitioner_id=?",
            (inquiry_id, practitioner_id)
        ).fetchone()
        practitioner_email = pract_row[0] if pract_row else None

        # Insert reply
        reply_id = str(uuid.uuid4())
        cx.execute(
            "INSERT INTO inquiry_replies (id, inquiry_id, practitioner_id, body, reply_method, received_at) "
            "VALUES (?, ?, ?, ?, 'form', ?)",
            (reply_id, inquiry_id, practitioner_id, body, now_iso)
        )
        # Mark inquiry_practitioners status=replied
        cx.execute(
            "UPDATE inquiry_practitioners SET status='replied' "
            "WHERE inquiry_id=? AND practitioner_id=?",
            (inquiry_id, practitioner_id)
        )
        cx.commit()

    # Forward reply to client via email
    client_first = client_name.split(None, 1)[0] if client_name else "there"
    forward_subject = "New reply from a practitioner about your inquiry"
    forward_body = (
        f"Hi {client_first},\n\n"
        f"A practitioner you reached out to through RemedyMatch.com has replied to your inquiry.\n\n"
        f"Their reply:\n{body}\n\n"
        f"---\n"
        f"Your original inquiry:\n"
        f"What you are working through: {main_challenge}\n"
        f"What success looks like: {main_goal}\n\n"
        f"You can reply directly to this practitioner by responding to this email.\n\n"
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
    )
    _send_inquiry_email(
        to_email=client_email,
        subject=forward_subject,
        body=forward_body,
        reply_to=practitioner_email,
    )

    html = _render_static_template(
        "inquiry-reply.html",
        status="confirmed",
        client_first=client_first,
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ── Phase 2b: share-with-practitioner ────────────────────────────────────────

def _validate_share_token(token):
    """Return (email, row) for a valid practitioner_share token, or (None, None)."""
    if not token:
        return None, None
    th = _hash_token(token)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT email, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose='practitioner_share'",
            (th,)
        ).fetchone()
    if not row:
        return None, None
    if row["consumed_at"]:
        return None, None
    try:
        exp_dt = datetime.fromisoformat(row["expires_at"].rstrip("Z"))
        if exp_dt < datetime.utcnow():
            return None, None
    except Exception:
        return None, None
    return row["email"], row


@app.route("/share-with-practitioner/<token>", methods=["GET"])
def share_with_practitioner_get(token):
    email, _row = _validate_share_token(token)
    if not email:
        html = _render_static_template("practitioner-share.html", status="error")
        return html, 410, {"Content-Type": "text/html; charset=utf-8"}

    payload = _scoreapp_payload_for(email)
    recent  = _recent_inquiry_practitioner_ids(email)

    # Read globally-opted-out emails so we hide them from the recipient list
    with sqlite3.connect(LOG_DB) as cx:
        opted_out = {r[0] for r in cx.execute(
            "SELECT email FROM practitioner_inquiry_opt_outs").fetchall()}

    # Filter: not opted-out at the email level AND not already shared
    eligible = [
        (iid, pid, pemail) for (iid, pid, pemail, shared_at) in recent
        if pemail and pemail not in opted_out and not shared_at
    ]
    if not payload or not eligible:
        html = _render_static_template(
            "practitioner-share.html", status="empty", token=token)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    # Load practitioner display rows for each unique id
    unique_pids = sorted({pid for (_, pid, _) in eligible})
    records = _fetch_practitioners_by_ids(unique_pids)
    rec_map = {str(r["id"]): r for r in records}
    practitioners = []
    for pid in unique_pids:
        r = rec_map.get(pid, {})
        name = (r.get("name") or "(name unavailable)").strip() or "(name unavailable)"
        city = (r.get("city") or "").strip()
        state = (r.get("state") or "").strip()
        loc = ", ".join(p for p in [city, state] if p)
        practitioners.append({"name": name, "location": loc})

    data = (payload.get("data", payload) or {})
    score = (data.get("total_score") or {}).get("percent") or data.get("score") or ""
    quiz_questions = data.get("quiz_questions") or []
    questions = []
    for q in quiz_questions:
        qt = (q.get("question") or "").strip()
        answers = q.get("answers") or []
        atext = ", ".join((a.get("answer") or "").strip() for a in answers if a.get("answer"))
        if qt and atext:
            questions.append({"question": qt, "answer": atext})

    first_row = None
    try:
        first_row = next(iter(c for c in [data.get("first_name")] if c), "") or ""
    except Exception:
        first_row = ""
    client_first = (first_row.strip() or email.split("@", 1)[0])

    html = _render_static_template(
        "practitioner-share.html",
        status="ready",
        token=token,
        client_first=client_first,
        score=score,
        questions=questions,
        practitioners=practitioners,
        practitioner_count=len(practitioners),
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/share-with-practitioner/<token>", methods=["POST"])
def share_with_practitioner_post(token):
    email, _row = _validate_share_token(token)
    if not email:
        html = _render_static_template("practitioner-share.html", status="error")
        return html, 410, {"Content-Type": "text/html; charset=utf-8"}

    payload = _scoreapp_payload_for(email)
    if not payload:
        html = _render_static_template(
            "practitioner-share.html", status="empty", token=token)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    recent = _recent_inquiry_practitioner_ids(email)
    with sqlite3.connect(LOG_DB) as cx:
        opted_out = {r[0] for r in cx.execute(
            "SELECT email FROM practitioner_inquiry_opt_outs").fetchall()}
    eligible = [
        (iid, pid, pemail) for (iid, pid, pemail, shared_at) in recent
        if pemail and pemail not in opted_out and not shared_at
    ]
    if not eligible:
        html = _render_static_template(
            "practitioner-share.html", status="empty", token=token)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    # Pull the original inquiry context per (inquiry_id) to populate the share
    # email body.
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        inq_ids = sorted({iid for (iid, _, _) in eligible})
        placeholders = ",".join(["?"] * len(inq_ids))
        inq_rows = cx.execute(
            f"SELECT id, client_name, client_email, main_challenge, main_goal "
            f"FROM inquiries WHERE id IN ({placeholders})",
            inq_ids
        ).fetchall()
    inq_map = {r["id"]: r for r in inq_rows}

    sent_ok = 0
    now_iso = datetime.utcnow().isoformat() + "Z"
    for (iid, pid, pemail) in eligible:
        inq = inq_map.get(iid)
        if not inq:
            continue
        client_name = inq["client_name"] or ""
        client_email = inq["client_email"] or email
        client_first = (client_name.split(None, 1)[0] if client_name
                        else client_email.split("@", 1)[0])
        body = _compose_share_email(
            client_first=client_first,
            client_email=client_email,
            main_challenge=inq["main_challenge"] or "",
            main_goal=inq["main_goal"] or "",
            scoreapp_payload=payload,
        )
        try:
            ok = _send_inquiry_email(
                to_email=pemail,
                subject=f"Follow-up from {client_first}: assessment context for your reply",
                body=body,
                reply_to=client_email,
            )
        except Exception as e:
            print(f"[share] send failed for {pemail}: {e!r}", flush=True)
            ok = False
        if ok:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute(
                    "UPDATE inquiry_practitioners SET shared_at=? "
                    "WHERE inquiry_id=? AND practitioner_id=?",
                    (now_iso, iid, pid)
                )
            sent_ok += 1

    # Mark the share token consumed
    th = _hash_token(token)
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "UPDATE auth_tokens SET consumed_at=? "
            "WHERE token_hash=? AND consumed_at IS NULL",
            (now_iso, th)
        )

    html = _render_static_template(
        "practitioner-share.html",
        status="confirmed",
        sent_count=sent_ok,
        token=token,
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe uploaded audio via Whisper (reuses journal_blueprint._whisper_transcribe)."""
    import tempfile, os as _os
    import journal_blueprint
    if "audio" not in request.files:
        return jsonify({"error": "no audio"}), 400
    audio_file = request.files["audio"]
    if (request.content_length or 0) > 26 * 1024 * 1024:
        return jsonify({"error": "audio too large"}), 413
    suffix = _os.path.splitext(audio_file.filename or "clip.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        audio_file.save(tf.name)
        audio_path = tf.name
    try:
        result = journal_blueprint._whisper_transcribe(audio_path)
        return jsonify({"text": (result.get("text") or "").strip()})
    except Exception as e:
        print(f"[transcribe] {e!r}", flush=True)
        return jsonify({"error": "transcription failed"}), 500
    finally:
        try: _os.unlink(audio_path)
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Cache pre-warm — runs per gunicorn worker boot.
# QB banks has a 5-min in-memory cache; on a cold dyno the first request must
# succeed or the dashboard card 500s. Warm it in the background so the user
# never sees a cold-cache failure.
# ─────────────────────────────────────────────────────────────────────────────
def _prewarm_caches():
    def _warm():
        import time as _t
        _t.sleep(3)  # let worker finish booting before hitting upstream APIs
        try:
            _money.qb_banks()
            print("[prewarm] money.qb_banks ✓")
        except Exception as e:
            print(f"[prewarm] money.qb_banks failed: {e}")
    threading.Thread(target=_warm, daemon=True, name="prewarm-money").start()

_prewarm_caches()


# ── Membership admin routes (Slice 2) ─────────────────────────────────────────

@app.route("/admin/membership/grant", methods=["POST"])
@require_console_key
def admin_membership_grant():
    import json as _json, uuid
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    source = (data.get("source") or "").strip()
    truly_vip_ref = (data.get("truly_vip_ref") or "").strip() or None
    notes = data.get("notes")
    days_raw = data.get("days")

    allowed_sources = {
        "video", "cash", "studio_credit",
        "bonus_biofield", "bonus_cert", "bonus_one_to_one",
        "bonus_healing_oasis", "bonus_hawaii", "bonus_consultant",
    }
    if not email or "@" not in email:
        return jsonify({"error": "email required"}), 400
    if source not in allowed_sources:
        return jsonify({"error": f"unknown source; allowed={sorted(allowed_sources)}"}), 400
    try:
        days = int(days_raw) if days_raw is not None else 30
    except (TypeError, ValueError):
        return jsonify({"error": "days must be an integer"}), 400
    if days <= 0 or days > 3650:
        return jsonify({"error": "days out of range"}), 400

    membership_id = str(uuid.uuid4())
    granted_by = request.headers.get("X-Console-Granted-By", "glen")
    granted_at = datetime.utcnow().isoformat() + "Z"
    expires_at = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO memberships "
            "(id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (membership_id, email, granted_at, expires_at, granted_by, source,
             truly_vip_ref, notes)
        )

    plain = _mint_membership_magic_link(email)
    base = request.host_url.rstrip("/")
    magic_link_url = f"{base}/coaching/auth/{plain}"

    subject = "Your Remedy Match coaching access is open"
    body = (
        f"Hi,\n\n"
        f"Your Remedy Match coaching access has been opened for the next {days} days.\n\n"
        f"Click here to sign in:\n{magic_link_url}\n\n"
        f"You'll land in your member dashboard with the AI agent loaded for your context. "
        f"You can chat 24/7, request a direct video reply from Glen on tricky questions, "
        f"and join the monthly group Zoom call.\n\n"
        f"When you'd like to renew for another 30 days, record a fresh 3-5 minute video at "
        f"https://truly.vip/Results.\n\n"
        f"---\n"
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
    )
    try:
        _send_inquiry_email(
            to_email=email,
            subject=subject,
            body=body,
            reply_to=RM_INBOUND_INQUIRY_EMAIL,
        )
    except Exception as e:
        print(f"[membership-grant] email send failed: {e!r}", flush=True)

    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO journey_events "
                "(ts, session_id, email, trigger, detail, rung_before, rung_after) "
                "VALUES (?, ?, ?, 'membership_granted', ?, '', '')",
                (granted_at, "", email,
                 _json.dumps({"source": source, "days": days,
                              "membership_id": membership_id,
                              "truly_vip_ref": truly_vip_ref or ""}))
            )
            cx.commit()
    except Exception as e:
        print(f"[membership-grant] journey_events insert failed: {e!r}", flush=True)

    return jsonify({
        "membership_id": membership_id,
        "magic_link_url": magic_link_url,
        "expires_at": expires_at,
    }), 200


@app.route("/admin/escalations", methods=["GET"])
@require_console_key
def admin_escalations_list():
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT id, created_at, email, query_text, ai_response, flag_reason "
            "FROM escalation_queue WHERE status='pending' ORDER BY created_at"
        ).fetchall()
    return jsonify([dict(r) for r in rows]), 200


# ── Slice 5: member-triggered escalation + admin queue detail + admin reply ───

@app.route("/coaching/escalate", methods=["POST"])
def coaching_escalate():
    import json as _json, uuid
    email = request.cookies.get("rm_member_email", "").strip().lower()
    membership = _active_membership_for_email(email) if email else None
    if not membership:
        return jsonify({"error": "membership required"}), 403
    data = request.get_json(silent=True) or {}
    query_text = (data.get("query_text") or "").strip()
    ai_response = (data.get("ai_response") or "").strip() or None
    if not query_text:
        return jsonify({"error": "query_text required"}), 400
    eid = str(uuid.uuid4())
    now_iso = datetime.utcnow().isoformat() + "Z"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "INSERT INTO escalation_queue "
            "(id, created_at, email, query_text, ai_response, ai_confidence, "
            " flag_reason, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (eid, now_iso, email, query_text, ai_response, None,
             "member_request", "pending")
        )
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                "VALUES (?,?,?,'membership_escalation_filed',?,'','')",
                (now_iso, request.cookies.get("amg_session", ""), email,
                 _json.dumps({"escalation_id": eid}))
            )
            cx.commit()
    except Exception as e:
        print(f"[coaching-escalate] journey_events insert failed: {e!r}", flush=True)
    return jsonify({
        "message": "Glen has been notified; a reply will arrive in your inbox within ~7 days.",
        "escalation_id": eid,
    }), 200


@app.route("/admin/escalations/<eid>", methods=["GET"])
@require_console_key
def admin_escalation_detail(eid):
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT id, created_at, email, query_text, ai_response, flag_reason, status "
            "FROM escalation_queue WHERE id=?",
            (eid,)
        ).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    body = dict(row)
    body["member_context"] = _member_context_for_email(body["email"])
    return jsonify(body), 200


@app.route("/admin/escalations/<eid>/reply", methods=["POST"])
@require_console_key
def admin_escalation_reply(eid):
    import json as _json
    data = request.get_json(silent=True) or {}
    video_url = (data.get("video_url") or "").strip()
    text = (data.get("text") or "").strip()
    if not video_url or not text:
        return jsonify({"error": "video_url and text required"}), 400
    now_iso = datetime.utcnow().isoformat() + "Z"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute(
            "SELECT email, query_text, status FROM escalation_queue WHERE id=?",
            (eid,)
        ).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        email, query_text, status = row
        if status != "pending":
            return jsonify({"error": "row not pending"}), 400
        cx.execute(
            "UPDATE escalation_queue SET status='replied', glen_reply_url=?, "
            "glen_reply_text=?, replied_at=? WHERE id=?",
            (video_url, text, now_iso, eid)
        )
    base = request.host_url.rstrip("/")
    subject = "Glen replied to your question"
    body = (
        f"Hi,\n\n"
        f"You asked: {query_text}\n\n"
        f"Glen recorded a reply for you here:\n{video_url}\n\n"
        f"Glen says: {text}\n\n"
        f"You can return to your coaching dashboard any time:\n{base}/coaching\n\n"
        f"---\n"
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
    )
    try:
        _send_inquiry_email(
            to_email=email, subject=subject, body=body,
            reply_to=RM_INBOUND_INQUIRY_EMAIL,
        )
    except Exception as e:
        print(f"[escalation-reply] email send failed: {e!r}", flush=True)
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                "VALUES (?,?,?,'membership_escalation_replied',?,'','')",
                (now_iso, "", email, _json.dumps({"escalation_id": eid}))
            )
            cx.commit()
    except Exception as e:
        print(f"[escalation-reply] journey_events insert failed: {e!r}", flush=True)
    return jsonify({"ok": True}), 200


# ── Slice 3: member auth + coaching dashboard ─────────────────────────────────

@app.route("/coaching/auth/<token>", methods=["GET"])
def coaching_auth_token(token):
    email = _validate_membership_magic_link(token)
    if not email:
        html = _render_static_template("coaching.html", status="error")
        return html, 410, {"Content-Type": "text/html; charset=utf-8"}
    # Consume the token
    th = _hash_token(token)
    now_iso = datetime.utcnow().isoformat() + "Z"
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            "UPDATE auth_tokens SET consumed_at=? WHERE token_hash=? AND consumed_at IS NULL",
            (now_iso, th)
        )
    # Mint amg_session if absent, then set rm_member_email cookie
    session_id = request.cookies.get("amg_session", "")
    minted = not session_id
    if minted:
        session_id = secrets.token_urlsafe(16)
    resp = redirect("/coaching", code=302)
    if minted:
        resp.set_cookie("amg_session", session_id,
                        max_age=60 * 60 * 24 * 365, httponly=True,
                        samesite="Lax", secure=request.is_secure)
    resp.set_cookie("rm_member_email", email,
                    max_age=60 * 60 * 24 * 365, httponly=True,
                    samesite="Lax", secure=request.is_secure)
    return resp


# ── Slice 6: studio.com credit intent + daily renewal-reminder cron ──────────

@app.route("/coaching/studio-credit", methods=["GET"])
def coaching_studio_credit_get():
    html = _render_static_template("coaching.html", status="studio_credit")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/coaching/studio-credit", methods=["POST"])
def coaching_studio_credit_post():
    import uuid
    data = request.get_json(silent=True) or request.form or {}
    email = (data.get("email") or "").strip().lower()
    studio_ref = (data.get("studio_ref") or "").strip() or None
    if email and "@" in email:
        sid = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat() + "Z"
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO studio_credit_intents (id, created_at, email, studio_ref) "
                "VALUES (?,?,?,?)",
                (sid, now_iso, email, studio_ref)
            )
        subject = "studio.com credit intent submitted"
        body = (
            f"A visitor reported a studio.com purchase and asked for the 30-day credit.\n\n"
            f"Email: {email}\n"
            f"studio_ref: {studio_ref or '(not provided)'}\n"
            f"Submitted: {now_iso}\n\n"
            f"To verify and grant 30 days, POST /admin/membership/grant with "
            f"source=studio_credit, email={email}, notes=studio_ref.\n"
        )
        try:
            _send_inquiry_email(
                to_email=RM_INBOUND_INQUIRY_EMAIL,
                subject=subject, body=body,
                reply_to=None,
            )
        except Exception as e:
            print(f"[studio-credit] glen notification failed: {e!r}", flush=True)
    html = _render_static_template("coaching.html", status="studio_credit_submitted")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/cron/membership-renewals", methods=["POST"])
@require_console_key
def cron_membership_renewals():
    import json as _json
    now_iso = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT id, email, expires_at, last_reminder_at FROM memberships "
            "WHERE datetime(expires_at) > datetime('now') "
            "AND datetime(expires_at) < datetime('now', '+3 days') "
            "LIMIT 500"
        ).fetchall()
    reminded = 0
    for r in rows:
        last = r["last_reminder_at"]
        if last:
            try:
                last_dt = datetime.fromisoformat(last.rstrip("Z"))
                if (datetime.utcnow() - last_dt) < timedelta(hours=24):
                    continue
            except Exception:
                pass
        try:
            exp_dt = datetime.fromisoformat(r["expires_at"].rstrip("Z"))
            days_left = max(0, (exp_dt - datetime.utcnow()).days)
        except Exception:
            days_left = 0
        s_days = "s" if days_left != 1 else ""
        subject = f"Your Remedy Match coaching access ends in {days_left} day{s_days}"
        body = (
            f"Hi,\n\n"
            f"Your Remedy Match coaching access ends in {days_left} day{s_days}"
            f" on {r['expires_at']}.\n\n"
            f"To renew for another 30 days, record a fresh 3-5 minute video at:\n"
            f"https://truly.vip/Results\n\n"
            f"Glen reviews each submission and re-opens access on acceptance.\n\n"
            f"---\n"
            f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
        )
        try:
            ok_sent = _send_inquiry_email(
                to_email=r["email"], subject=subject, body=body,
                reply_to=RM_INBOUND_INQUIRY_EMAIL,
            )
        except Exception as e:
            print(f"[renewal-cron] send failed for {r['email']}: {e!r}", flush=True)
            ok_sent = False
        if ok_sent:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute(
                    "UPDATE memberships SET last_reminder_at=? WHERE id=?",
                    (now_iso, r["id"])
                )
            try:
                with sqlite3.connect(LOG_DB) as cx:
                    cx.execute(
                        "INSERT INTO journey_events "
                        "(ts, session_id, email, trigger, detail, rung_before, rung_after) "
                        "VALUES (?,?,?,'membership_renewal_reminder',?,'','')",
                        (now_iso, "", r["email"], _json.dumps({"days_left": days_left}))
                    )
                    cx.commit()
            except Exception as e:
                print(f"[renewal-cron] journey_events insert failed: {e!r}", flush=True)
            reminded += 1
    return jsonify({"reminded": reminded}), 200


@app.route("/coaching/login-request", methods=["POST"])
def coaching_login_request():
    data = request.get_json(silent=True) or request.form or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        # Still return the same shape (defense-in-depth)
        return jsonify({"message": "If an active membership exists for that email, a sign-in link is on its way."}), 200
    # Mint and send REGARDLESS of active status (defense-in-depth: don't leak membership status)
    plain = _mint_membership_magic_link(email)
    base = request.host_url.rstrip("/")
    magic_link_url = f"{base}/coaching/auth/{plain}"
    subject = "Your Remedy Match sign-in link"
    body = (
        f"Hi,\n\n"
        f"Use this link to sign in to your Remedy Match coaching dashboard:\n{magic_link_url}\n\n"
        f"The link is good for 15 minutes. If you don't have an active membership, "
        f"recording a fresh 3-5 minute video at https://truly.vip/Results will earn you "
        f"30 days of access.\n\n"
        f"---\n"
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
    )
    try:
        _send_inquiry_email(
            to_email=email, subject=subject, body=body,
            reply_to=RM_INBOUND_INQUIRY_EMAIL,
        )
    except Exception as e:
        print(f"[coaching-login] email send failed: {e!r}", flush=True)
    return jsonify({"message": "If an active membership exists for that email, a sign-in link is on its way."}), 200


@app.route("/coaching", methods=["GET"])
def coaching_dashboard():
    email = request.cookies.get("rm_member_email", "").strip().lower()
    membership = _active_membership_for_email(email) if email else None
    if not membership:
        html = _render_static_template("coaching.html", status="lapsed")
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    # Resolve first name from inbound_leads (best effort) or email local-part
    client_first = email.split("@", 1)[0]
    try:
        with sqlite3.connect(LOG_DB) as cx:
            row = cx.execute(
                "SELECT first_name FROM inbound_leads "
                "WHERE email=? AND first_name IS NOT NULL AND first_name != '' "
                "ORDER BY id DESC LIMIT 1",
                (email,)
            ).fetchone()
            if row and row[0]:
                client_first = row[0]
    except Exception:
        pass
    # Pull last 5 Glen replies
    glen_replies = []
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.row_factory = sqlite3.Row
            for r in cx.execute(
                "SELECT id, query_text, glen_reply_url, glen_reply_text, replied_at "
                "FROM escalation_queue WHERE email=? AND status='replied' "
                "ORDER BY replied_at DESC LIMIT 5",
                (email,)
            ).fetchall():
                glen_replies.append(dict(r))
    except Exception:
        pass
    html = _render_static_template(
        "coaching.html",
        status="active",
        client_first=client_first,
        days_remaining=membership.get("days_remaining", 0),
        glen_replies=glen_replies,
    )
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ───────────────────────── Knowledge Atlas ─────────────────────────
import atlas_store
import atlas_ask as _atlas_ask

# Seed the mutable concept/pending files onto the persistent disk on first boot so admin
# approvals survive redeploys (no-op locally / once seeded).
try:
    if atlas_store.reseed_from_repo():
        print("[atlas] seeded persistent concept files from repo", flush=True)
except Exception as _e:
    print(f"[atlas] reseed skipped: {_e}", flush=True)


@app.route("/atlas")
def atlas_page():
    resp = send_from_directory(STATIC, "atlas.html")
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.route("/atlas.js")
def atlas_js():
    resp = send_from_directory(STATIC, "atlas.js")
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.route("/atlas.css")
def atlas_css():
    resp = send_from_directory(STATIC, "atlas.css")
    resp.headers["Content-Type"] = "text/css"
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.route("/atlas/data")
def atlas_data():
    return jsonify(atlas_store.build_graph())


@app.route("/atlas/ask", methods=["POST"])
def atlas_ask_route():
    question = (request.get_json(silent=True) or {}).get("question", "")
    concepts = atlas_store.build_graph()["concepts"]

    def _answer(q, ids):
        try:
            vec = embed(q)
            matches = query_all_namespaces(vec)[:6]
            ctx = "\n".join(m.metadata.get("text", "")[:400] for m in matches)
            labels = [c["label"] for c in concepts if c["id"] in ids]
            lead = ("This relates to " + ", ".join(labels) + ". ") if labels else ""
            return lead + (ctx[:600] if ctx else "See the linked concepts on the map.")
        except Exception:
            return "See the highlighted concepts on the map."

    return jsonify(_atlas_ask.atlas_ask(question, concepts, answer_fn=_answer))


@app.route("/admin/atlas")
def admin_atlas_page():
    return send_from_directory(STATIC, "admin-atlas.html")


@app.route("/admin/atlas/pending", methods=["GET"])
@require_console_key
def admin_atlas_pending():
    return ok({"concepts": atlas_store.load_pending().get("concepts", [])})


@app.route("/admin/atlas/approve", methods=["POST"])
@require_console_key
def admin_atlas_approve():
    cid = (request.get_json(silent=True) or {}).get("id")
    if not cid:
        return fail("id required", 400)
    try:
        atlas_store.approve_concept(cid)
    except KeyError:
        return fail("unknown concept id", 404)
    return ok({"approved": cid})


@app.route("/admin/atlas/reject", methods=["POST"])
@require_console_key
def admin_atlas_reject():
    cid = (request.get_json(silent=True) or {}).get("id")
    if not cid:
        return fail("id required", 400)
    atlas_store.reject_concept(cid)
    return ok({"rejected": cid})


@app.route("/admin/atlas/reseed", methods=["POST"])
@require_console_key
def admin_atlas_reseed():
    # Republish the git-committed build onto the persistent disk (overwrites live curation).
    # Use after an intentional rebuild. force defaults true here (explicit admin action).
    force = (request.get_json(silent=True) or {}).get("force", True)
    seeded = atlas_store.reseed_from_repo(force=force)
    return ok({"reseeded": seeded, "force": force})


# ── Tier-2 wholesale: application → approval ──────────────────────────────────
# Apply (resale-license holders) → Glen/Rae approve in /admin/wholesale → the same
# wholesale_unlocked_at gate is set and ordering opens. Sits alongside the existing
# licensed-on-register and coach-on-module auto-unlock paths.

def _send_wholesale_decision_email(to_email, name, approved, notes=""):
    """Approval / rejection email to a wholesale applicant."""
    if approved:
        subject = "Your Remedy Match wholesale application is approved"
        body = (
            f"Hi {name or 'there'},\n\n"
            f"Good news — your wholesale application is approved. You now have wholesale "
            f"ordering in your practitioner portal: {PUBLIC_BASE_URL}/practitioner/login-request\n\n"
            f"In wellness,\nDr. Glen\n")
    else:
        subject = "Your Remedy Match wholesale application"
        body = (
            f"Hi {name or 'there'},\n\n"
            f"Thank you for applying for wholesale access. We're not able to approve it "
            f"at this time.\n"
            + (f"\nNote: {notes}\n" if notes else "")
            + f"\nIf you think this is in error, just reply to this email.\n\n— Remedy Match\n")
    return _send_full_report_email(to_email, name or "there", subject, body)


@app.route("/wholesale/apply")
def wholesale_apply_page():
    return send_from_directory(STATIC, "wholesale-apply.html")


@app.route("/api/wholesale/apply", methods=["POST", "OPTIONS"])
def api_wholesale_apply():
    if request.method == "OPTIONS":
        return "", 200
    clean, err = _pp.validate_wholesale_application(request.get_json(silent=True) or {})
    if err:
        return jsonify({"ok": False, "error": err}), 400
    try:
        pid, already = _pp.submit_wholesale_application(clean)
    except Exception as e:
        print(f"[wholesale-apply] submit failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "Could not submit your application. Please try again."}), 500

    # Applying also makes them a Tier-1 Member (they agreed to the ToS here).
    sid = request.cookies.get("amg_session") or uuid.uuid4().hex
    parts = clean["name"].split(None, 1)
    try:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            begin_funnel.init_journey_tables(cx)
            begin_funnel.record_unlock(
                cx, session_id=sid, trigger="tos", email=clean["email"],
                first_name=parts[0], last_name=(parts[1] if len(parts) > 1 else ""),
                tos=True, tos_version=BEGIN_TOS_VERSION)
    except Exception as e:
        print(f"[wholesale-apply] tos record failed: {e!r}", flush=True)

    # Magic link so they can sign in and track status.
    try:
        magic = _pp.create_magic_link_token(pid, clean["email"])
        _send_practitioner_magic_link(
            clean["email"], clean["name"],
            f"{PUBLIC_BASE_URL}/practitioner/login-verify?token={magic}")
    except Exception as e:
        print(f"[wholesale-apply] magic link failed: {e!r}", flush=True)

    # Notify Rae of a new application to review (unless they already had access).
    if not already:
        try:
            _send_full_report_email(
                os.environ.get("RAE_EMAIL", "suerae1111@gmail.com"), "Rae",
                f"New wholesale application: {clean['name']}",
                f"{clean['name']} ({clean['email']}) applied for wholesale access.\n"
                f"Resale license: {clean['resale_license_number']}"
                + (f" ({clean['license_state']})" if clean['license_state'] else "") + "\n"
                + (f"Practice: {clean['practice_name']}\n" if clean['practice_name'] else "")
                + f"\nReview pending applications at {PUBLIC_BASE_URL}/admin/wholesale\n")
        except Exception:
            pass

    msg = ("You already have wholesale access — check your email for a sign-in link."
           if already else
           "Application received. We'll email you when it's approved. "
           "Check your email for a sign-in link to track your status.")
    resp = jsonify({"ok": True, "already_unlocked": already, "message": msg})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", sid, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/admin/wholesale")
def admin_wholesale_page():
    return send_from_directory(STATIC, "admin-wholesale.html")


@app.route("/admin/wholesale/pending", methods=["GET"])
@require_console_key
def admin_wholesale_pending():
    return ok({"applications": _pp.list_pending_applications()})


@app.route("/admin/wholesale/approve", methods=["POST"])
@require_console_key
def admin_wholesale_approve():
    data = request.get_json(silent=True) or {}
    pid = data.get("id")
    if not pid:
        return fail("id required", 400)
    who = _pp.decide_application(pid, approve=True, notes=(data.get("notes") or ""))
    if not who:
        return fail("unknown application id", 404)
    try:
        _send_wholesale_decision_email(who["email"], who["name"], approved=True)
    except Exception:
        pass
    return ok({"approved": pid})


@app.route("/admin/wholesale/reject", methods=["POST"])
@require_console_key
def admin_wholesale_reject():
    data = request.get_json(silent=True) or {}
    pid = data.get("id")
    if not pid:
        return fail("id required", 400)
    notes = (data.get("notes") or "").strip()
    who = _pp.decide_application(pid, approve=False, notes=notes)
    if not who:
        return fail("unknown application id", 404)
    try:
        _send_wholesale_decision_email(who["email"], who["name"], approved=False, notes=notes)
    except Exception:
        pass
    return ok({"rejected": pid})


@app.route("/admin/qbo/taxprefs", methods=["GET"])
@require_console_key
def admin_qbo_taxprefs():
    """Read QBO Preferences.TaxPrefs to confirm Automated Sales Tax (AST) is on.
    Runs where the live QBO token is valid (Render). PartnerTaxEnabled=true → AST."""
    try:
        from dashboard import money
        pref = money.qb_get(money.qb_refresh(), "/preferences")
        tp = ((pref.get("Preferences") or pref).get("TaxPrefs") or {})
        from dashboard import tax as _tax
        return ok({
            "partner_tax_enabled": tp.get("PartnerTaxEnabled"),
            "using_sales_tax": tp.get("UsingSalesTax"),
            "app_tax_enabled": _tax.tax_enabled(),
            "app_retail_rate": _tax.retail_rate(),
            "app_wholesale_rate": _tax.wholesale_rate(),
            "tax_prefs": tp,
        })
    except Exception as e:
        return fail(f"qbo preferences read failed: {e}", 502)


@app.route("/admin/tax")
def admin_tax_page():
    return send_from_directory(STATIC, "admin-tax.html")


@app.route("/admin/tax/get-report", methods=["GET"])
@require_console_key
def admin_tax_get_report():
    """Absorbed Hawai'i GET owed over a period, in G-45 buckets (HI retail / HI
    wholesale / out-of-state / unknown-state). ?format=csv for a download."""
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _bos_orders.init_orders_table(cx)
        rep = _bos_orders.get_tax_report(cx, date_from=date_from, date_to=date_to)
    finally:
        cx.close()
    if (request.args.get("format") or "").lower() == "csv":
        import io as _io, csv as _csv
        buf = _io.StringIO(); w = _csv.writer(buf)
        w.writerow(["bucket", "orders", "gross_usd", "get_usd"])
        for k in ("hi_retail", "hi_wholesale", "out_of_state", "unknown_state"):
            b = rep[k]
            w.writerow([k, b["orders"], f"{b['gross_cents']/100:.2f}", f"{b['get_cents']/100:.2f}"])
        w.writerow(["total_get_owed", "", "", f"{rep['total_get_cents']/100:.2f}"])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=get-report.csv"})
    return ok(rep)


# ── Clips review admin ────────────────────────────────────────────────────────
@app.route("/admin/clips")
def admin_clips_page():
    return send_from_directory(STATIC, "admin-clips.html")


@app.route("/admin/clips/pending", methods=["GET"])
@require_console_key
def admin_clips_pending():
    try:
        res = _idx.query(
            vector=[0.0] * 1536,
            top_k=100,
            namespace="clips",
            include_metadata=True,
            filter={"status": "pending"},
        )
        items = [{"id": m.id, **(m.metadata or {})} for m in res.matches]
    except Exception as e:
        return fail(str(e), 500)
    return ok({"clips": items})


@app.route("/admin/clips/approve", methods=["POST"])
@require_console_key
def admin_clips_approve():
    cid = (request.get_json(silent=True) or {}).get("id")
    if not cid:
        return fail("id required", 400)
    _idx.update(id=cid, set_metadata={"status": "approved"}, namespace="clips")
    return ok({"approved": cid})


@app.route("/admin/clips/reject", methods=["POST"])
@require_console_key
def admin_clips_reject():
    d = request.get_json(silent=True) or {}
    cid = d.get("id")
    if not cid:
        return fail("id required", 400)
    _idx.delete(ids=[cid], namespace="clips")
    if d.get("r2_key"):
        try:
            _r2().delete_object(Bucket=os.environ.get("R2_BUCKET", "rm-clips"), Key=d["r2_key"])
        except Exception:
            pass
    return ok({"rejected": cid})


# ── Business OS spine ─────────────────────────────────────────────────────────
# Event log + action registry population.
import sqlite3 as _sqlite3
import dashboard  # noqa: F401 (exposes dashboard.CONSOLE_SECRET for _bos_actor)
from dashboard import events as _bos_events
from dashboard import dispatch as _bos_dispatch
from dashboard import rbac as _bos_rbac
import dashboard.actions_tasks  # noqa: F401  (registers tasks.* actions)
import dashboard.actions_rewards  # noqa: F401  (registers rewards.process_payout MONEY_SEND action)
import dashboard.signals as _bos_signals  # noqa: F401 (registers module signals)
import dashboard.orders as _bos_orders  # noqa: F401 (registers order actions + signal)
import dashboard.finance as _bos_finance  # noqa: F401 (registers money signal + finance actions)
import dashboard.crm as _bos_crm  # noqa: F401 (registers the CRM home signal)
import dashboard.module_signals as _bos_module_signals  # noqa: F401 (registers 5 cell signals)
import dashboard.products as _bos_products  # noqa: F401 (registers products signal + action; replaces module_signals' products signal)
import dashboard.easypost as _bos_easypost  # noqa: F401
import dashboard.ghl_queue as _bos_ghl_queue  # noqa: F401 (registers crm enqueue actions)
from dashboard import stripe_alerts as _stripe_alerts  # noqa: F401 (Stripe-failure alerting)


def _alert_stripe(context, err):
    """Best-effort: record a Stripe session-create failure (throttled owner email
    + console signal). Runs inside checkout except blocks — must never raise."""
    try:
        cx = _sqlite3.connect(LOG_DB)
        try:
            _stripe_alerts.record_failure(cx, context, str(err))
        finally:
            cx.close()
    except Exception as _e:
        print(f"[stripe-alert] skipped: {_e!r}", flush=True)


def _init_bos_ghl_queue():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_ghl_queue.init_ghl_queue_table(cx)
    finally:
        cx.close()


_init_bos_ghl_queue()


def _init_bos_events():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_events.init_event_tables(cx)
    finally:
        cx.close()


_init_bos_events()


def _init_bos_orders():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_orders.init_orders_table(cx)
        _bos_orders.init_fulfillments_table(cx)
        _stripe_alerts.init_stripe_alerts_table(cx)
    finally:
        cx.close()


_init_bos_orders()


def _normalize_ship_address(addr, fallback_name=""):
    """Normalize a posted shipping address to the shape easypost.build_shipment
    reads (street/city/state/zip/country). State is upper-cased for the GET
    tax check. Returns {} when nothing usable was provided."""
    addr = addr or {}
    street = (addr.get("street") or addr.get("street1") or addr.get("address") or "").strip()
    state = (addr.get("state") or "").strip().upper()
    zipc = (addr.get("zip") or addr.get("postal") or addr.get("zipcode") or "").strip()
    city = (addr.get("city") or "").strip()
    if not (street or city or state or zipc):
        return {}
    return {
        "name": (addr.get("name") or fallback_name or "").strip(),
        "street": street, "city": city, "state": state, "zip": zipc,
        "country": (addr.get("country") or "US").strip().upper(),
    }


def _ingest_order(*, source, external_ref, email="", name="", phone="",
                  items=None, total_cents=0, address=None, channel="retail",
                  get_cents=0, discount_cents=0, points_redeemed_cents=0, shipping_cents=0):
    """Best-effort: record an order into the BOS orders table. Never raises into
    a checkout path. get_cents = absorbed Hawai'i GET owed (recorded, not charged)."""
    try:
        cx = _sqlite3.connect(LOG_DB)
        try:
            _bos_orders.upsert_order(
                cx, source=source, external_ref=external_ref, email=email, name=name,
                phone=phone, items=items or [], total_cents=int(total_cents or 0),
                address=address or {}, channel=channel, get_cents=int(get_cents or 0),
                discount_cents=int(discount_cents or 0),
                points_redeemed_cents=int(points_redeemed_cents or 0),
                shipping_cents=int(shipping_cents or 0))
        finally:
            cx.close()
    except Exception as e:
        print(f"[orders] ingest {source}/{external_ref}: {e!r}", flush=True)


def _bos_actor():
    """Resolve the calling actor. Owner master key (CONSOLE_SECRET) for now;
    scoped token->role mapping is added in the RBAC-UX task of Phase 1.
    Unlike the legacy @require_console_key decorator, BOS routes return 401 when CONSOLE_SECRET is unset (resolve_actor returns None) rather than passing through; this is intentional."""
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    return _bos_rbac.resolve_actor(key, console_secret=dashboard.CONSOLE_SECRET)


@app.route("/api/action/<path:key>", methods=["POST"])
def bos_action(key):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    confirmed = bool(body.pop("confirmed", False))
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        res = _bos_dispatch.dispatch_action(
            cx, key, dict(body), actor, source="panel", confirmed=confirmed)
    finally:
        cx.close()
    return jsonify(res)


# ── In-house order entry (Phase 1: proposed invoice) — OWNER only ───────────────
@app.route("/api/customers/search", methods=["GET"])
def api_customers_search():
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import customers as _cust
    from dashboard import points as _points
    q = (request.args.get("q") or "").strip()
    cx = _sqlite3.connect(LOG_DB)
    try:
        people = _cust.find_people(cx, q, limit=10)
        _points.init_points_table(cx)
        # Fill a missing street address from the contact's last shipped-to order,
        # and surface the contact's redeemable points balance (email-keyed, lowercased).
        for p in people:
            if not (p.get("address1") or "").strip():
                last = _cust.last_address_for(cx, p.get("email"))
                for k in ("address1", "address2", "city", "state", "zip", "country"):
                    if last.get(k) and not (p.get(k) or "").strip():
                        p[k] = last[k]
            email = (p.get("email") or "").strip().lower()
            p["points_balance_cents"] = _points.balance(cx, email) if email else 0
    finally:
        cx.close()
    return jsonify({"ok": True, "people": people})


@app.route("/api/orders/manual", methods=["POST"])
def api_orders_manual():
    """Create a proposed invoice (in-house order entry). Computes rule-based
    pricing (discount/shipping/GET) with per-line + order-level override, builds
    structured line items, links/saves the customer record, inserts a 'proposed'
    order. Payment is recorded later via the orders.record_payment action."""
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import customers as _cust
    body = request.get_json(silent=True) or {}
    customer = body.get("customer") or {}
    lines_in = body.get("lines") or []
    if not lines_in:
        return jsonify({"ok": False, "error": "no line items"}), 400
    addr_in = customer.get("address") or {}
    ship = {
        "name": customer.get("name") or "",
        "street": addr_in.get("address1") or addr_in.get("street") or "",
        "address2": addr_in.get("address2") or "",
        "city": addr_in.get("city") or "", "state": addr_in.get("state") or "",
        "zip": addr_in.get("zip") or "", "country": (addr_in.get("country") or "US").upper(),
    }
    # Build cart + structured line items. FF capsules ($69.97) are priced at the
    # order-wide volume rate per line (see _inhouse_ff_unit_cents); an explicit
    # unit override wins; everything else is its list price.
    from dashboard import pricing as _pricing
    settings = _pricing.load_settings(_pricing_settings())
    total_ff_qty = _inhouse_total_ff_qty(lines_in)
    cart, items_rec, subtotal_list = [], [], 0
    for ln in lines_in:
        slug = (ln.get("slug") or "").strip()
        p = _get_product(slug)
        if not p:
            continue
        qty = max(1, min(int(ln.get("qty") or 1), 99))
        unit_cents = _inhouse_line_unit_cents(p, ln.get("unit_cents"), total_ff_qty, settings)
        line_cents = unit_cents * qty
        subtotal_list += line_cents
        cart.append({"slug": slug, "qty": qty})
        items_rec.append({"slug": slug, "name": p["name"], "qty": qty,
                          "unit_cents": unit_cents, "line_cents": line_cents})
    if not cart:
        return jsonify({"ok": False, "error": "no valid products"}), 400
    # Volume is already baked into the per-line FF prices above, so the order does
    # NOT apply the engine's months-volume discount — only a manual order discount.
    # _price_cart is used solely for shipping + absorbed GET.
    try:
        pc = _price_cart(cart, ship=ship, channel="retail")
        shipping_cents = int(pc.get("shipping_cents") or 0)
        get_cents = int((pc.get("priced") or {}).get("get_cents") or 0)
    except CheckoutError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        print(f"[orders.manual] pricing fell back: {e!r}", flush=True)
        shipping_cents, get_cents = 0, 0
    discount_cents = (max(0, int(body.get("discount_cents")))
                      if body.get("discount_cents") not in (None, "") else 0)
    total_cents = max(0, subtotal_list - discount_cents) + shipping_cents
    # Points apply at the bottom of the invoice, against the total. Server is the
    # ONLY authority on the amount: capped to the total AND to the customer's actual
    # redeemable balance (email-keyed, lowercased) — never trust the client number.
    points_redeemed_cents = 0
    if body.get("points_redeem_cents") not in (None, ""):
        from dashboard import points as _points
        _pemail = (customer.get("email") or "").strip().lower()
        _pcx = _sqlite3.connect(LOG_DB)
        try:
            _points.init_points_table(_pcx)
            _pbal = _points.balance(_pcx, _pemail) if _pemail else 0
        finally:
            _pcx.close()
        points_redeemed_cents = max(0, min(int(body.get("points_redeem_cents")), total_cents, _pbal))
        total_cents -= points_redeemed_cents
    person_id = customer.get("person_id")
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        if not person_id and (customer.get("email") or "").strip():
            person_id = _cust.find_or_create_by_email(
                cx, email=customer.get("email"), name=customer.get("name"),
                phone=customer.get("phone"))
        if person_id:
            _cust.upsert_person_address(cx, person_id, {**addr_in, "phone": customer.get("phone")})
        ext = "INH-" + uuid.uuid4().hex[:10].upper()
        oid = _bos_orders.upsert_order(
            cx, source="in-house", external_ref=ext, status="proposed",
            email=(customer.get("email") or ""), name=(customer.get("name") or ""),
            phone=(customer.get("phone") or ""), person_id=person_id,
            items=items_rec, total_cents=total_cents,
            address={"name": ship["name"], "street": ship["street"],
                     "address2": ship["address2"], "city": ship["city"],
                     "state": ship["state"], "zip": ship["zip"], "country": ship["country"]},
            channel="retail", get_cents=get_cents,
            discount_cents=discount_cents, shipping_cents=shipping_cents,
            points_redeemed_cents=points_redeemed_cents)
    finally:
        cx.close()
    return jsonify({"ok": True, "order_id": oid, "external_ref": ext,
                    "method": (body.get("method") or ""),
                    "totals": {"subtotal_cents": subtotal_list, "discount_cents": discount_cents,
                               "shipping_cents": shipping_cents, "get_cents": get_cents,
                               "points_redeemed_cents": points_redeemed_cents,
                               "total_cents": total_cents},
                    "lines": items_rec})


@app.route("/api/orders/price-preview", methods=["POST"])
def api_orders_price_preview():
    """OWNER: live per-line pricing for the order-entry form. Same authority as
    /api/orders/manual — FF capsules get the order-wide volume rate, others list
    price, owner overrides honored. Server is the single source of truth."""
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import pricing as _pricing
    settings = _pricing.load_settings(_pricing_settings())
    lines_in = (request.get_json(silent=True) or {}).get("lines") or []
    total_ff_qty = _inhouse_total_ff_qty(lines_in)
    vol_pct = round(float(_pricing.volume_pct(total_ff_qty, settings) or 0), 2)
    out_lines, subtotal = [], 0
    for ln in lines_in:
        slug = (ln.get("slug") or "").strip()
        p = _get_product(slug)
        if not p:
            continue
        qty = max(1, min(int(ln.get("qty") or 1), 99))
        ov = ln.get("unit_cents")
        is_ff = _qty_eligible(p)
        list_cents = int(p.get("price_cents") or 0)
        unit = _inhouse_line_unit_cents(p, ov, total_ff_qty, settings)
        overridden = ov not in (None, "")
        line_cents = unit * qty
        subtotal += line_cents
        out_lines.append({
            "slug": slug, "qty": qty, "is_ff": is_ff, "list_cents": list_cents,
            "effective_unit_cents": unit, "line_cents": line_cents,
            "vol_pct": (vol_pct if (is_ff and not overridden) else 0),
            "savings_cents": max(0, list_cents - unit)})
    return jsonify({"ok": True, "total_ff_qty": total_ff_qty,
                    "subtotal_cents": subtotal, "lines": out_lines})


# ── Customer invoice pay-link (Phase 3) — PUBLIC, token-scoped ─────────────────
# Auth is the per-order token ONLY (never the console key). All pricing is
# recomputed server-side; client-sent prices are ignored; the Stripe amount comes
# from the order's server-side total. One token ⇄ one order.
def _invoice_order_for_token(token):
    oid = _pp.order_id_from_invoice_token(token)
    if not oid:
        return None
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        return _bos_orders.get_order(cx, int(oid))
    finally:
        cx.close()


def _invoice_points_balance(order):
    """Customer's redeemable points balance (email-keyed, lowercased to match the ledger)."""
    from dashboard import points as _points
    email = (order.get("email") or "").strip().lower()
    if not email:
        return 0
    cx = _sqlite3.connect(LOG_DB)
    try:
        _points.init_points_table(cx)
        return _points.balance(cx, email)
    finally:
        cx.close()


def _invoice_summary(order):
    """Customer-safe view — no person_id, notes, phone, or full address."""
    lines = order.get("items") or []
    subtotal = sum(int(l.get("line_cents") or 0) for l in lines)
    return {
        "points_balance_cents": _invoice_points_balance(order),
        "external_ref": order.get("external_ref"),
        "name": order.get("name") or "",
        "lines": [{"slug": l.get("slug"), "name": l.get("name"), "qty": int(l.get("qty") or 0),
                   "unit_cents": int(l.get("unit_cents") or 0),
                   "line_cents": int(l.get("line_cents") or 0)} for l in lines],
        "subtotal_cents": subtotal,
        "discount_cents": int(order.get("discount_cents") or 0),
        "shipping_cents": int(order.get("shipping_cents") or 0),
        "get_cents": int(order.get("get_cents") or 0),
        "points_redeemed_cents": int(order.get("points_redeemed_cents") or 0),
        "total_cents": int(order.get("total_cents") or 0),
        "status": order.get("status"),
        "pay_status": order.get("pay_status") or "unpaid",
        "editable": order.get("status") in ("proposed", "confirmed"),
        "pay_enabled": bool(INVOICE_PAYLINK_ENABLED and _STRIPE_ACTIVE),
        "paylink_enabled": bool(INVOICE_PAYLINK_ENABLED),
    }


@app.route("/invoice/<token>")
def invoice_page(token):
    if not _pp.order_id_from_invoice_token(token):
        return ("<!doctype html><meta charset=utf-8><title>Invoice</title>"
                "<div style='font-family:sans-serif;max-width:480px;margin:80px auto;text-align:center'>"
                "<h2>This invoice link is invalid or has expired.</h2>"
                "<p>Please contact us for a fresh link.</p></div>"), 403
    resp = send_from_directory(STATIC, "invoice.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/invoice/<token>", methods=["GET"])
def api_invoice_get(token):
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    return jsonify({"ok": True, "order": _invoice_summary(order)})


@app.route("/api/invoice/<token>/update", methods=["POST"])
def api_invoice_update(token):
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    if order.get("status") not in ("proposed", "confirmed"):
        return jsonify({"ok": False, "error": "this invoice can no longer be changed"}), 409
    body = request.get_json(silent=True) or {}
    cart, items_rec, subtotal_list = [], [], 0
    for ln in (body.get("lines") or []):
        slug = (ln.get("slug") or "").strip()
        p = _get_product(slug)
        if not p:
            continue
        qty = max(0, min(int(ln.get("qty") or 0), 99))
        if qty <= 0:
            continue
        unit_cents = int(p.get("price_cents") or 0)  # SERVER price only — ignore client
        line_cents = unit_cents * qty
        subtotal_list += line_cents
        cart.append({"slug": slug, "qty": qty})
        items_rec.append({"slug": slug, "name": p["name"], "qty": qty,
                          "unit_cents": unit_cents, "line_cents": line_cents})
    if not cart:
        return jsonify({"ok": False, "error": "no valid items"}), 400
    ship = order.get("address") or {}
    try:
        pc = _price_cart(cart, ship=ship, channel="retail")
        discount_cents = int(pc.get("discount_cents") or 0)
        shipping_cents = int(pc.get("shipping_cents") or 0)
        get_cents = int((pc.get("priced") or {}).get("get_cents") or 0)
    except Exception as e:
        print(f"[invoice.update] pricing fell back: {e!r}", flush=True)
        discount_cents, shipping_cents, get_cents = 0, 0, 0
    total_cents = max(0, subtotal_list - discount_cents) + shipping_cents
    # Carry points forward, re-capped to the new total AND the live balance.
    points = min(int(order.get("points_redeemed_cents") or 0), total_cents,
                 _invoice_points_balance(order))
    total_cents -= points
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _bos_orders.upsert_order(
            cx, source=order["source"], external_ref=order["external_ref"],
            email=order.get("email") or "", name=order.get("name") or "",
            phone=order.get("phone") or "", person_id=order.get("person_id"),
            items=items_rec, total_cents=total_cents, address=ship, channel="retail",
            get_cents=get_cents, discount_cents=discount_cents,
            points_redeemed_cents=points, shipping_cents=shipping_cents)
        updated = _bos_orders.get_order(cx, order["id"])
    finally:
        cx.close()
    try:
        from dashboard import inbox as _inbox
        _inbox.send_email("drglenswartwout@gmail.com",
                          f"Invoice {order.get('external_ref')} changed by customer",
                          f"Customer updated invoice {order.get('external_ref')} — "
                          f"new total ${total_cents/100:,.2f}.")
    except Exception as _e:
        print(f"[invoice] owner notify skipped: {_e!r}", flush=True)
    return jsonify({"ok": True, "order": _invoice_summary(updated)})


@app.route("/api/invoice/<token>/apply-points", methods=["POST"])
def api_invoice_apply_points(token):
    """Customer applies points against the invoice total. Server is the ONLY
    authority on the amount: capped to the pre-points total AND the live balance.
    Persisted as points_redeemed_cents so it's redeemed at payment."""
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    if order.get("status") not in ("proposed", "confirmed"):
        return jsonify({"ok": False, "error": "this invoice can no longer be changed"}), 409
    requested = max(0, int((request.get_json(silent=True) or {}).get("cents") or 0))
    # Work from the pre-points (gross) total so re-applying is idempotent.
    gross_total = int(order.get("total_cents") or 0) + int(order.get("points_redeemed_cents") or 0)
    new_points = max(0, min(requested, gross_total, _invoice_points_balance(order)))
    new_total = gross_total - new_points
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _bos_orders.upsert_order(
            cx, source=order["source"], external_ref=order["external_ref"],
            email=order.get("email") or "", name=order.get("name") or "",
            phone=order.get("phone") or "", person_id=order.get("person_id"),
            items=order.get("items"), total_cents=new_total, address=order.get("address") or {},
            channel="retail", get_cents=int(order.get("get_cents") or 0),
            discount_cents=int(order.get("discount_cents") or 0),
            points_redeemed_cents=new_points, shipping_cents=int(order.get("shipping_cents") or 0))
        updated = _bos_orders.get_order(cx, order["id"])
    finally:
        cx.close()
    return jsonify({"ok": True, "order": _invoice_summary(updated)})


@app.route("/api/invoice/<token>/pay", methods=["POST"])
def api_invoice_pay(token):
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    if order.get("pay_status") == "paid" or order.get("status") not in ("proposed", "confirmed"):
        return jsonify({"ok": False, "error": "this invoice is already paid"}), 409
    if not INVOICE_PAYLINK_ENABLED:
        return jsonify({"ok": False, "error": "online payment is not enabled yet"}), 403
    method = ((request.get_json(silent=True) or {}).get("method") or "card").strip().lower()
    amount = int(order.get("total_cents") or 0)  # SERVER amount only
    if method == "card":
        _card_unavailable = ("Card payment is temporarily unavailable — "
                             "please use Zelle or Wise, or contact us.")
        if not _STRIPE_ACTIVE:
            return jsonify({"ok": False, "error": _card_unavailable}), 503
        from dashboard import stripe_pay as _sp
        ext = order.get("external_ref")
        # A Stripe failure (bad key, API error, timeout) must NOT 500 a paying
        # customer — degrade to a clean message pointing at Zelle/Wise.
        try:
            sess = _sp.create_checkout_session(
                amount, customer_email=order.get("email") or "",
                description=f"Remedy Match invoice {ext}",
                metadata={"kind": "in-house", "order_id": str(order["id"]),
                          "invoice_id": ext, "customer_id": ""},
                success_url=f"{PUBLIC_BASE_URL}/begin/checkout-return?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{PUBLIC_BASE_URL}/invoice/{token}")
            url = sess.get("url")
        except Exception as e:
            app.logger.warning(f"[invoice.pay] stripe session failed for {ext}: {e!r}")
            _alert_stripe("invoice-card", e)
            url = None
        if not url:
            return jsonify({"ok": False, "error": _card_unavailable}), 502
        return jsonify({"ok": True, "method": "card", "url": url})
    if method in ("zelle", "wise"):
        amt = "$%.2f" % (amount / 100)
        if method == "zelle":
            to = os.environ.get("ZELLE_PAY_TO", "drglenswartwout@gmail.com")
            instr = f"Send {amt} via Zelle to {to}, then tap “I’ve sent payment”."
        else:
            link = os.environ.get("WISE_PAY_LINK", "")
            instr = (f"Pay {amt} via Wise" + (f": {link}" if link else " — we’ll email you Wise details")
                     + ", then tap “I’ve sent payment”.")
        return jsonify({"ok": True, "method": method, "instructions": instr})
    return jsonify({"ok": False, "error": "unknown payment method"}), 400


@app.route("/api/invoice/<token>/claim-paid", methods=["POST"])
def api_invoice_claim_paid(token):
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    if order.get("pay_status") == "paid" or order.get("status") not in ("proposed", "confirmed"):
        return jsonify({"ok": False, "error": "this invoice is already paid"}), 409
    method = ((request.get_json(silent=True) or {}).get("method") or "").strip().lower()
    if method not in ("zelle", "wise"):
        return jsonify({"ok": False, "error": "unknown payment method"}), 400
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _bos_orders.set_order_payment_claimed(cx, order["id"], method=method)
    finally:
        cx.close()
    try:
        from dashboard import inbox as _inbox
        _inbox.send_email(
            "drglenswartwout@gmail.com",
            f"Payment claimed: invoice {order.get('external_ref')}",
            f"{order.get('name') or 'A customer'} reports sending "
            f"${int(order.get('total_cents') or 0)/100:,.2f} via {method} for invoice "
            f"{order.get('external_ref')}. Confirm receipt, then Record payment.")
    except Exception as _e:
        print(f"[invoice] claim notify skipped: {_e!r}", flush=True)
    return jsonify({"ok": True, "message": "Thanks — we’ll confirm your payment shortly."})


@app.route("/api/invoice/<token>/chat", methods=["POST"])
def api_invoice_chat(token):
    if not _pp.order_id_from_invoice_token(token):
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    body = request.get_json(silent=True) or {}
    catalog = _build_ff_catalog()
    result = _chat.scoped_reply(body.get("message") or "", body.get("history") or [], catalog)
    suggestions = []
    for slug in (result.get("suggested_slugs") or []):
        p = _get_product(slug)
        if not p:
            continue
        suggestions.append({"slug": slug, "name": p.get("name") or slug,
                            "price_cents": p.get("price_cents") or 0})
    return jsonify({"ok": True, "reply": result.get("reply", ""), "suggestions": suggestions})


@app.route("/api/events", methods=["GET"])
def bos_events():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        try:
            _limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            _limit = 50
        _limit = max(1, min(_limit, 200))
        rows = _bos_events.list_events(
            cx, limit=_limit,
            status=request.args.get("status"),
            module=request.args.get("module"))
    finally:
        cx.close()
    return jsonify({"ok": True, "data": rows})


@app.route("/api/events/<int:event_id>/approve", methods=["POST"])
def bos_event_approve(event_id):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        res = _bos_dispatch.approve_event(cx, event_id, actor)
    finally:
        cx.close()
    return jsonify(res)


@app.route("/api/events/<int:event_id>/cancel", methods=["POST"])
def bos_event_cancel(event_id):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        res = _bos_dispatch.cancel_event(cx, event_id)
    finally:
        cx.close()
    return jsonify(res)


@app.route("/api/home/signals", methods=["GET"])
def bos_home_signals():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        cells = _bos_signals.aggregate_signals(cx, actor)
    finally:
        cx.close()
    return jsonify({"ok": True, "data": cells})


@app.route("/console/home")
def bos_home_page():
    # Retired: the Business OS home board's 9 signal cells + pending approvals now
    # live on the /console front door (Mission Control banner). Redirect old
    # links/bookmarks, preserving the ?key= so the console stays unlocked.
    qs = request.query_string.decode()
    return redirect("/console" + ("?" + qs if qs else ""), code=302)


@app.route("/console/orders")
def bos_orders_page():
    resp = send_from_directory(STATIC, "console-orders.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/console/cert")
def bos_cert_page():
    resp = send_from_directory(STATIC, "console-cert.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/products")
def bos_products_list():
    if _bos_actor() is None:
        return jsonify({"error": "unauthorized"}), 401
    # ?all=1 → every sellable SKU (remedymatch.com catalog), not just the
    # ingredient-enriched subset. Used by the in-house order-entry picker.
    all_skus = request.args.get("all") in ("1", "true", "yes")
    return jsonify({"products": _bos_products.catalog(with_ingredients_only=not all_skus)})


@app.route("/api/products/stale")
def bos_products_stale():
    if _bos_actor() is None:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"stale": _bos_products.stale_pages()})


@app.route("/console/products")
def bos_products_page():
    resp = send_from_directory(STATIC, "console-products.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/orders", methods=["GET", "POST"])
def bos_orders_create():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if request.method == "GET":
        cx = _sqlite3.connect(LOG_DB)
        cx.row_factory = _sqlite3.Row
        try:
            try:
                rows = _bos_orders.list_orders(
                    cx, status=request.args.get("status"),
                    limit=min(int(request.args.get("limit", 200) or 200), 500))
            except (TypeError, ValueError):
                rows = _bos_orders.list_orders(cx)
            # Annotate each order with per-line fulfillment + backorder units, using a
            # single grouped query over order_fulfillments (no per-order round-trips).
            try:
                filled = {}
                for e in cx.execute(
                        "SELECT order_id, line_index, COALESCE(SUM(qty),0) FROM order_fulfillments "
                        "GROUP BY order_id, line_index").fetchall():
                    filled[(int(e[0]), int(e[1]))] = int(e[2])
                for o in rows:
                    oid = int(o.get("id"))
                    back = 0
                    flines = []
                    for i, it in enumerate(o.get("items") or []):
                        ordered = int(it.get("qty") or 0)
                        f = filled.get((oid, i), 0)
                        bo = max(0, ordered - f)
                        back += bo
                        flines.append({"index": i, "ordered": ordered, "fulfilled": f, "backordered": bo})
                    o["fulfillment"] = flines
                    o["backorder_units"] = back
            except Exception as _e:
                print(f"[orders] backorder annotate skipped: {_e!r}", flush=True)
        finally:
            cx.close()
        return jsonify({"ok": True, "data": rows})
    # --- existing POST body unchanged below ---
    b = request.get_json(silent=True) or {}
    ref = str(b.get("external_ref") or f"manual-{_bos_orders._now()}")
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        oid = _bos_orders.upsert_order(
            cx, source="manual", external_ref=ref, email=b.get("email", ""),
            name=b.get("name", ""), phone=b.get("phone", ""), items=b.get("items") or [],
            total_cents=int(b.get("total_cents") or 0), address=b.get("address") or {},
            channel=b.get("channel", "retail"))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        cx.close()
    return jsonify({"ok": True, "order_id": oid})


@app.route("/api/backorders", methods=["GET"])
def bos_backorders():
    """Reorder worklist: per-product units backordered across open orders."""
    actor = _bos_actor()
    if actor is None or actor.role not in (_bos_rbac.OWNER, _bos_rbac.OPS):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        data = _bos_orders.backorder_rollup(cx)
    finally:
        cx.close()
    return jsonify({"ok": True, "data": data})


@app.route("/console/finance")
def bos_finance_page():
    resp = send_from_directory(STATIC, "console-finance.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/console/crm")
def bos_crm_page():
    resp = send_from_directory(STATIC, "console-crm.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/finance/ar", methods=["GET"])
def bos_finance_ar():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        rows = _bos_finance.open_invoices()
        summary = _bos_finance.finance_summary()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    return jsonify({"ok": True, "data": rows, "summary": summary})


@app.route("/api/pricing/preview", methods=["POST"])
def api_pricing_preview():
    # Public by design: read-only price preview for the storefront cart JS. Returns only
    # prices (already public) — no auth needed, no data mutated, no PII echoed.
    from dashboard import pricing as _pricing, tax as _tax
    data = request.get_json(silent=True) or {}
    settings = _pricing.load_settings(_pricing_settings())
    items = []
    for it in (data.get("items") or []):
        p = _get_product(it.get("slug"))
        if not p:
            continue                      # unavailable/inactive → skip, never silently mis-price
        try:                              # tolerate junk qty from a public body, default to 1
            qty = max(1, int(float(it.get("qty") or 1)))
        except (TypeError, ValueError):
            qty = 1
        # base = TRUE single-unit list (volume is now a discount candidate, not a base price)
        items.append({
            "slug": p["slug"], "name": p.get("name", p["slug"]), "qty": qty, "product": p,
            "unit_cents": int(p.get("price_cents") or 0),
            "months": qty * int(p.get("months_per_unit", 1)),   # 30-cap bottle = 1 month
            "volume_eligible": bool(p.get("volume_eligible", True)),  # Pure Powders set False
        })
    result = _pricing.compute(
        items, settings=settings,
        subscriber_tier_pct=data.get("subscriber_tier_pct"),
        coupon_pct=data.get("coupon_pct"),
        points_to_redeem_cents=data.get("points_to_redeem_cents") or 0,
        channel=data.get("channel", "retail"),
        ship_to_state=data.get("ship_to_state"),
        tax_fn=_tax.compute_get_cents)
    return jsonify(result)


@app.route("/api/console/pricing-settings", methods=["GET", "POST"])
def api_console_pricing_settings():
    """Console-gated read/write of the global pricing + rewards tunables, persisted to
    pricing-settings.json in DATA_DIR (live-reloaded by _pricing_settings)."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import pricing_settings as _ps
    if request.method == "GET":
        raw = _pricing_settings()
        return jsonify({"saved": raw, "effective": _ps.effective(raw),
                        "defaults": _ps.defaults_view()})
    # POST
    payload = request.get_json(silent=True) or {}
    clean, errors = _ps.validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    import tempfile
    _PRICING_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(_PRICING_SETTINGS_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean, f, indent=2)
        os.replace(tmp, _PRICING_SETTINGS_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    _PRICING_SETTINGS_CACHE["mtime"] = None      # force re-read on next access
    raw = _pricing_settings()
    return jsonify({"saved": raw, "effective": _ps.effective(raw)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
