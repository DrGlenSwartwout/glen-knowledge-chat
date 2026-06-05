"""Journey-state engine for the /begin progressive-disclosure funnel.

Pure functions over a sqlite3 connection. Routes in app.py manage the
connection + _db_lock; tests pass their own connection. See
docs/superpowers/specs/2026-05-28-progressive-disclosure-funnel-design.md
"""

import json
import sqlite3
import urllib.parse

from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_journey_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_state (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            email           TEXT,
            first_name      TEXT,
            ref_slug        TEXT,
            current_rung    TEXT    DEFAULT 'arrival',
            unlocked_gates  TEXT    DEFAULT '[]',
            awareness_stage TEXT    DEFAULT 'unknown',
            path            TEXT    DEFAULT 'none',
            tos_agreed_at   TEXT,
            tos_version     TEXT,
            last_signal     TEXT,
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_session ON journey_state(session_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_email   ON journey_state(email)")
    # Slice 2 additive migration — awareness classification timestamp.
    try:
        cx.execute("ALTER TABLE journey_state ADD COLUMN awareness_classified_at TEXT")
    except Exception:
        pass  # already exists
    # Piece 3 additive migration — explicit last name capture.
    try:
        cx.execute("ALTER TABLE journey_state ADD COLUMN last_name TEXT")
    except Exception:
        pass  # already exists
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            session_id  TEXT,
            email       TEXT,
            trigger     TEXT NOT NULL,
            detail      TEXT DEFAULT '',
            rung_before TEXT,
            rung_after  TEXT
        )
    """)
    cx.execute("""
        CREATE TABLE IF NOT EXISTS affiliate_social_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, slug TEXT NOT NULL, email TEXT,
            url TEXT NOT NULL, platform TEXT DEFAULT '',
            points INTEGER, views INTEGER, likes INTEGER, shares INTEGER,
            reviewed_at TEXT
        )
    """)
    cx.commit()


RUNGS = ["arrival", "listening", "inquire", "personalize", "free_tier",
         "explore_voice", "assess", "choose_path", "ascend", "advocate"]
RUNG_INDEX = {r: i for i, r in enumerate(RUNGS)}

# All accepted unlock triggers. The page (slice 1) only fires the first six;
# the rest are accepted so the engine spine is forward-compatible with later
# slices (rooms built in slices 4-6).
VALID_TRIGGERS = {
    "load", "video", "scroll", "question", "name", "email", "tos",
    "voice", "scan", "quiz", "paid_fork", "purchase", "share_video",
    "deep_link",
}

# Gate keys stored in unlocked_gates (email/tos drive their own columns, but
# are still recorded as gates for completeness).
GATE_TRIGGERS = VALID_TRIGGERS - {"load", "deep_link"}

# ---------------------------------------------------------------------------
# Awareness-stage inference (Slice 2)
# ---------------------------------------------------------------------------

AWARENESS_RANK = {"unknown": 0, "problem": 1, "solution": 2, "product": 3, "most": 4}

# Deliberately MINIMAL seed lists (refined from data over time, not hand-tuned
# here). Case-insensitive substring match on recent chat text.
_PRODUCT_KEYWORDS = ["e4l", "evox", "neuro magnesium", "retina renew", "zyto",
                     "voice scan", "bioenergetic"]
_SOLUTION_KEYWORDS = ["detox", "cleanse", "frequency", "remedy", "supplement",
                      "protocol", "natural healing", "biofield", "energetic"]
_PROBLEM_KEYWORDS = ["tired", "fatigue", "pain", "can't sleep", "cant sleep",
                     "insomnia", "anxious", "anxiety", "bloated", "headache",
                     "stress", "vision"]
_PRODUCT_GATES = {"paid_fork", "purchase", "scan", "quiz", "voice"}


def _max_awareness(a, b):
    return a if AWARENESS_RANK.get(a, 0) >= AWARENESS_RANK.get(b, 0) else b


def infer_awareness_heuristic(want, gates, query_texts):
    """Cold-start awareness signal from explicit intent, gates opened, and recent
    chat text. Returns one of AWARENESS_RANK's keys."""
    if want:
        return "most"
    gates = set(gates or ())
    if gates & _PRODUCT_GATES:
        return "product"
    text = " ".join(query_texts or []).lower()
    if any(k in text for k in _PRODUCT_KEYWORDS):
        return "product"
    if any(k in text for k in _SOLUTION_KEYWORDS):
        return "solution"
    if any(k in text for k in _PROBLEM_KEYWORDS):
        return "problem"
    return "unknown"


WANT_TARGETS = {
    "e4l":     "https://truly.vip/E4L",
    "quiz":    "https://healing.scoreapp.com",
    "join":    "https://truly.vip/Join",
    "results": "https://truly.vip/Results",
    "voice":   "/begin/voice",
    "path":    "/begin/path",
    "ascend":  "/begin/ascend",
}


def resolve_want(want, ref=""):
    """Return the threaded external URL for a live ?want= target, else None."""
    key = (want or "").strip().lower()
    base = WANT_TARGETS.get(key)
    if not base:
        return None
    if base.startswith("/"):      # internal target — same-origin, no utm
        return base
    slug = (ref or "remedy-match").strip() or "remedy-match"
    sep = "&" if "?" in base else "?"
    return (f"{base}{sep}utm_source={urllib.parse.quote(slug)}"
            f"&utm_medium=affiliate&utm_campaign=begin-deeplink-{key}")


def compute_rung(gates, email, tos_agreed):
    """Derive the highest rung reached. Monotonic in ladder order. The
    free_tier rung specifically requires BOTH an email and ToS agreement."""
    gates = set(gates or ())
    rung = "arrival"
    if "video" in gates or "scroll" in gates:
        rung = "listening"
    if "question" in gates:
        rung = "inquire"
    if "name" in gates:
        rung = "personalize"
    if email and tos_agreed:
        rung = "free_tier"
    if "voice" in gates:
        rung = "explore_voice"
    if "scan" in gates or "quiz" in gates:
        rung = "assess"
    if "paid_fork" in gates:
        rung = "choose_path"
    if "purchase" in gates:
        rung = "ascend"
    if "share_video" in gates:
        rung = "advocate"
    return rung


_RUNG_LAYERS = {
    "arrival":      ["layer0"],
    "listening":    ["layer0", "layer1"],
    "inquire":      ["layer0", "layer1", "layer2"],
    "personalize":  ["layer0", "layer1", "layer2", "layer3"],
    "free_tier":    ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"],
}
# Full unfolding surface (layer0-5) stays visible at every rung at/above free_tier.
_ALL_LAYERS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"]


def reveal_for(rung, awareness="unknown"):
    # gate-skip: product/most-aware visitors see the full unfolding surface
    if AWARENESS_RANK.get(awareness, 0) >= AWARENESS_RANK["product"]:
        return list(_ALL_LAYERS)
    if RUNG_INDEX.get(rung, 0) >= RUNG_INDEX["free_tier"]:
        return list(_ALL_LAYERS)
    return list(_RUNG_LAYERS.get(rung, ["layer0"]))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_for_session(cx, session_id):
    cx.row_factory = sqlite3.Row
    return cx.execute(
        "SELECT * FROM journey_state WHERE session_id=? ORDER BY id DESC LIMIT 1",
        (session_id,)).fetchone()


# ---------------------------------------------------------------------------
# record_unlock — mutating entry point
# ---------------------------------------------------------------------------

def record_unlock(cx, *, session_id, trigger, email="", detail="",
                  first_name="", last_name="", tos=False, ref_slug="", tos_version="",
                  want="", query_texts=None, path=""):
    if trigger not in VALID_TRIGGERS:
        raise ValueError(f"invalid trigger: {trigger!r}")
    cx.row_factory = sqlite3.Row
    now = _now()
    row = _row_for_session(cx, session_id)

    if row is None:
        gates = set()
        existing = dict(email="", first_name="", last_name="", ref_slug="",
                        tos_agreed_at=None, tos_version=None,
                        created_at=now)
    else:
        gates = set(json.loads(row["unlocked_gates"] or "[]"))
        existing = dict(row)

    rung_before = compute_rung(
        gates, existing.get("email") or "",
        bool(existing.get("tos_agreed_at")))

    if trigger in GATE_TRIGGERS:
        gates.add(trigger)
    new_email = (email or existing.get("email") or "").strip().lower()
    new_first = (first_name or existing.get("first_name") or "").strip()
    new_last = (last_name or existing.get("last_name") or "").strip()
    new_ref = (ref_slug or existing.get("ref_slug") or "").strip()
    tos_at = existing.get("tos_agreed_at")
    tos_ver = existing.get("tos_version")
    if trigger == "tos" or tos:
        tos_at = tos_at or now
        tos_ver = tos_version or tos_ver

    rung_after = compute_rung(gates, new_email, bool(tos_at))
    gates_json = json.dumps(sorted(gates))

    _persisted_aw = (existing.get("awareness_stage") if row is not None else "unknown") or "unknown"
    # awareness is inferred from the POST-add gate set (a just-opened
    # product/assessment gate immediately implies product-awareness)
    _new_aw = _max_awareness(_persisted_aw, infer_awareness_heuristic(want, gates, query_texts))

    _persisted_path = (existing.get("path") if row is not None else "none") or "none"
    _new_path = path if (path and path != "none") else _persisted_path

    if row is None:
        cx.execute("""
            INSERT INTO journey_state
              (session_id, email, first_name, last_name, ref_slug, current_rung,
               unlocked_gates, awareness_stage, path, tos_agreed_at,
               tos_version, last_signal, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (session_id, new_email, new_first, new_last, new_ref, rung_after,
              gates_json, _new_aw, _new_path, tos_at, tos_ver, trigger,
              now, now))
    else:
        cx.execute("""
            UPDATE journey_state SET
              email=?, first_name=?, last_name=?, ref_slug=?, current_rung=?,
              unlocked_gates=?, awareness_stage=?, path=?, tos_agreed_at=?,
              tos_version=?, last_signal=?, updated_at=?
            WHERE id=?
        """, (new_email, new_first, new_last, new_ref, rung_after, gates_json,
              _new_aw, _new_path, tos_at, tos_ver, trigger, now, row["id"]))

    cx.execute("""
        INSERT INTO journey_events
          (ts, session_id, email, trigger, detail, rung_before, rung_after)
        VALUES (?,?,?,?,?,?,?)
    """, (now, session_id, new_email, trigger, detail[:500],
          rung_before, rung_after))
    cx.commit()
    state = get_state(cx, session_id=session_id)
    state["rung_before"] = rung_before
    return state


# ---------------------------------------------------------------------------
# get_state — non-destructive read + email aggregation
# ---------------------------------------------------------------------------

def _default_state(session_id, email):
    return {
        "session_id": session_id, "email": email or "", "first_name": "",
        "last_name": "",
        "ref_slug": "", "current_rung": "arrival", "unlocked_gates": [],
        "awareness_stage": "unknown", "path": "none",
        "tos_agreed_at": None, "tos_version": None,
        "reveal": reveal_for("arrival"), "surfaced_cards": [],
    }


def get_state(cx, session_id="", email=""):
    """Return the visitor's aggregated journey state. When an email is known,
    union the gates across ALL rows sharing that email (cross-device
    continuity) plus the current session row. Non-destructive."""
    cx.row_factory = sqlite3.Row
    email = (email or "").strip().lower()
    rows = []
    seen = set()
    if email:
        for r in cx.execute(
                "SELECT * FROM journey_state WHERE LOWER(email)=?", (email,)):
            rows.append(r); seen.add(r["id"])
    if session_id:
        r = _row_for_session(cx, session_id)
        if r is not None and r["id"] not in seen:
            rows.append(r)
    if not rows:
        return _default_state(session_id, email)

    gates = set()
    first_name = ""
    last_name = ""
    ref_slug = ""
    email_final = email
    tos_at = None
    tos_ver = None
    path = "none"
    awareness = "unknown"
    created_at = None
    for r in rows:
        gates |= set(json.loads(r["unlocked_gates"] or "[]"))
        first_name = first_name or (r["first_name"] or "")
        last_name = last_name or (r["last_name"] or "")
        ref_slug = ref_slug or (r["ref_slug"] or "")
        email_final = email_final or (r["email"] or "")
        tos_at = tos_at or r["tos_agreed_at"]
        tos_ver = tos_ver or r["tos_version"]
        if (r["path"] or "none") != "none":
            path = r["path"]
        awareness = _max_awareness(awareness, r["awareness_stage"] or "unknown")
        if created_at is None or (r["created_at"] and r["created_at"] < created_at):
            created_at = r["created_at"]

    rung = compute_rung(gates, email_final, bool(tos_at))
    return {
        "session_id": session_id, "email": email_final, "first_name": first_name,
        "last_name": last_name,
        "ref_slug": ref_slug, "current_rung": rung,
        "unlocked_gates": sorted(gates), "awareness_stage": awareness,
        "path": path, "tos_agreed_at": tos_at, "tos_version": tos_ver,
        "reveal": reveal_for(rung, awareness), "surfaced_cards": [],
    }


def set_awareness(cx, session_id, stage):
    """Persist an awareness stage upward (never regresses) for a session and
    stamp awareness_classified_at. Used by the background Haiku classifier."""
    cx.row_factory = sqlite3.Row
    now = _now()
    rows = cx.execute(
        "SELECT id, awareness_stage FROM journey_state WHERE session_id=?",
        (session_id,)).fetchall()
    for r in rows:
        merged = _max_awareness(r["awareness_stage"] or "unknown", stage)
        cx.execute(
            "UPDATE journey_state SET awareness_stage=?, awareness_classified_at=?, "
            "updated_at=? WHERE id=?",
            (merged, now, now, r["id"]))
    cx.commit()


# ---------------------------------------------------------------------------
# Slice 3 — CARD_CATALOG + card_href + _card
# ---------------------------------------------------------------------------

CARD_CATALOG = {
    "quiz":               {"title": "Discover Your Healing Path",
                           "sub": "Uncover your top wellness opportunities in 3 minutes",
                           "base_url": "https://healing.scoreapp.com", "internal": False},
    "e4l_scan":           {"title": "Your Voice Reveals What Your Body Knows",
                           "sub": "A 10-second scan shows your body's current priorities",
                           "base_url": "https://truly.vip/E4L", "internal": False},
    "intake":             {"title": "Begin Your Journey",
                           "sub": "Personalized guidance starts with understanding your story",
                           "base_url": "https://truly.vip/Join", "internal": False},
    "voice_distinctions": {"title": "Listen Deeper — Your Voice & Frequencies",
                           "sub": "5-Element toning · voice-sample analysis · Bioenergetic Wellness Scan · EVOX",
                           "base_url": "/begin/voice", "internal": True},
    "remedy_match":       {"title": "Find Your Perfect Remedy Match",
                           "sub": "A few questions to match you to your one perfect remedy",
                           "base_url": "/begin/match", "internal": True},
    "product":            {"title": "Formulations Matched to You",
                           "sub": "Explore remedies suited to what your body needs",
                           "base_url": "https://remedymatch.com", "internal": False},
    "ash_course":         {"title": "Learn to Heal Yourself",
                           "sub": "The Accelerated Self-Healing approach, step by step",
                           "base_url": "https://truly.vip/GetWell", "internal": False},
    "ash_masterclass":    {"title": "The Path Deeper: Choose Your Depth",
                           "sub": "From a free orientation to full immersion. See the whole spectrum of how you can go further.",
                           "base_url": "/begin/ascend", "internal": True},
    "pay_forward":        {"title": "Share Your Results, Lift Others",
                           "sub": "Pass your healing forward — and earn as you do",
                           "base_url": "/begin/path", "internal": True},
    "practitioner":       {"title": "Find a Practitioner Near You",
                           "sub": "Connect with a practitioner who fits your path",
                           "base_url": "/practitioner-finder", "internal": True},
}


def card_href(key, ref=""):
    c = CARD_CATALOG[key]
    if c["internal"]:
        return c["base_url"]
    slug = (ref or "remedy-match").strip() or "remedy-match"
    sep = "&" if "?" in c["base_url"] else "?"
    return (f"{c['base_url']}{sep}utm_source={urllib.parse.quote(slug)}"
            f"&utm_medium=affiliate&utm_campaign=begin-card-{key}")


def _card(key, ref=""):
    c = CARD_CATALOG[key]
    return {"key": key, "title": c["title"], "sub": c["sub"], "href": card_href(key, ref)}


# ---------------------------------------------------------------------------
# Explore page — non-linear table of contents (served at /begin/explore)
# ---------------------------------------------------------------------------

# Explore-only entries that are intentionally NOT in CARD_CATALOG. CARD_CATALOG
# drives Slice-3 contextual surfacing; keeping these out of it lets the Explore
# directory list them without altering surfacing behavior. Both are internal.
_EXPLORE_EXTRA = {
    "tone": {
        "title": "5-Element Tone Analyzer",
        "sub": "Hear which of the five elements your voice is calling for",
        "base_url": "/begin/tone", "internal": True},
    "practitioner_apply": {
        "title": "Work With Us",
        "sub": "For practitioners: bring Dr. Glen's formulations and methods into your practice",
        "base_url": "/practitioner", "internal": True},
}

# Ordered Explore layout. Each item is either a CARD_CATALOG key, or an
# _EXPLORE_EXTRA key prefixed with "x:". Sections render top-to-bottom.
_EXPLORE_LAYOUT = [
    {"title": "Start Here",
     "blurb": "Three quick ways to learn what your body is asking for.",
     "items": ["quiz", "e4l_scan", "intake"]},
    {"title": "Listen Deeper",
     "blurb": "Your voice carries the signal.",
     "items": ["voice_distinctions", "x:tone"]},
    {"title": "Match & Remedies",
     "blurb": "Find what fits, then explore the formulations.",
     "items": ["remedy_match", "product"]},
    {"title": "Learn to Heal",
     "blurb": "The Accelerated Self-Healing approach, step by step.",
     "items": ["ash_course"]},
    {"title": "Go Deeper",
     "blurb": "Choose how far you want to take this.",
     "items": ["ash_masterclass"]},
    {"title": "Share & Lift Others",
     "blurb": "Pass your healing forward.",
     "items": ["pay_forward"]},
    {"title": "Find a Practitioner",
     "blurb": "Connect with someone near you.",
     "items": ["practitioner"]},
    {"title": "For Practitioners",
     "blurb": "A different door, for the clinicians among us.",
     "items": ["x:practitioner_apply"], "audience": "practitioner"},
]


def partner_links(trusted_links):
    """Return the external partner/affiliate links for the funnel's 'Recommended
    Tools & Partners' section, sourced from trusted-links.json.

    Only entries flagged `"affiliate": true` are returned (Blushield, Glen's
    Amazon Associates picks); unflagged internal links (E4L, etc.) are excluded
    so they stay name-resolve / auto-open only. `trusted_links` is the parsed
    trusted-links.json dict ({"links": {...}}), injected by the caller so this
    module stays free of app.py / filesystem coupling.

    Each item: {name, url, note, amazon} where `amazon` is True for amzn.to /
    amazon.* links (drives the Amazon Associates disclosure). Order follows the
    JSON insertion order."""
    out = []
    for name, val in (trusted_links.get("links", {}) or {}).items():
        if not isinstance(val, dict) or not val.get("affiliate"):
            continue
        url = val.get("url", "")
        if not url:
            continue
        amazon = "amzn.to" in url or "amazon." in url
        out.append({"name": name, "url": url,
                    "note": val.get("note", ""), "amazon": amazon})
    return out


def explore_sections(ref="", trusted_links=None):
    """Build the ordered, ref-threaded section list for the /begin/explore page.

    Renders from CARD_CATALOG (so it stays in sync with the funnel as rooms are
    added) plus the explore-only extras in _EXPLORE_EXTRA. Returns a list of:
        {title, blurb, audience, cards: [{title, sub, href, external}], disclosure?}
    `external` is True for off-site links (drives target=_blank on the page).
    External hrefs carry the same utm threading as card_href; internal hrefs
    stay bare. All new copy here is em-dash-free (Glen's standing rule).

    Two integration sections are appended after the static layout:
      - "Earn by Sharing": the affiliate-program door (-> /affiliate, ref-threaded
        as ?ref= when a ref is present so attribution carries through).
      - "Recommended Tools & Partners": rendered only when `trusted_links` is
        supplied, from partner_links(); carries an Amazon Associates disclosure
        when any card is an Amazon link."""
    sections = []
    for sec in _EXPLORE_LAYOUT:
        cards = []
        for item in sec["items"]:
            if item.startswith("x:"):
                c = _EXPLORE_EXTRA[item[2:]]
                href = c["base_url"]          # extras are always internal
                external = not c["internal"]
            else:
                c = CARD_CATALOG[item]
                href = card_href(item, ref)
                external = not c["internal"]
            cards.append({"title": c["title"], "sub": c["sub"],
                          "href": href, "external": external})
        sections.append({"title": sec["title"], "blurb": sec["blurb"],
                         "audience": sec.get("audience", "patient"),
                         "cards": cards})

    # Affiliate-program door — its own directory entry (today it only lives
    # inside /begin/path). Ref-threaded so a landing attribution carries through.
    aff_href = "/affiliate"
    if ref:
        aff_href += "?ref=" + urllib.parse.quote(ref)
    sections.append({
        "title": "Earn by Sharing",
        "blurb": "Pass your healing forward, and earn your way deeper as others begin through you.",
        "audience": "patient",
        "cards": [{
            "title": "Become an Affiliate",
            "sub": "Share your link and earn. Points and credit accrue toward your own access as others start their journey through you.",
            "href": aff_href, "external": False}],
    })

    # External partner / affiliate tools (Blushield, Glen's Amazon picks).
    if trusted_links:
        partners = partner_links(trusted_links)
        if partners:
            cards = [{"title": p["name"], "sub": p["note"],
                      "href": p["url"], "external": True} for p in partners]
            disclosure = ("As an Amazon Associate, Healing Oasis earns from "
                          "qualifying purchases.") if any(p["amazon"] for p in partners) else ""
            sections.append({
                "title": "Recommended Tools & Partners",
                "blurb": "Trusted devices and tools Dr. Glen recommends alongside your remedies.",
                "audience": "patient",
                "cards": cards, "disclosure": disclosure})

    return sections


# ---------------------------------------------------------------------------
# Slice 3 Task 2 — surface()
# ---------------------------------------------------------------------------

_REMEDY_MATCH_CUES = ["remedy for", "what helps", "support for", "what should i take",
                      "what should i use", "help with my"]
_GENERIC_PRODUCT_CUES = ["products", "supplements", "what do you sell",
                         "what do you have", "browse", "store", "shop"]
_VOICE_KEYWORDS = ["voice", "frequency", "evox", "toning", "vibration",
                   "5-element", "5 element", "scan"]
_LEARN_KEYWORDS = ["learn", "understand", "course", "how do i", "study", "diy", "myself"]
_SHARE_KEYWORDS = ["share", "refer", "affiliate", "help others", "testimonial"]
# Seed list — substring match; deliberately minimal, refined from data over time.
_PRACTITIONER_KEYWORDS = ["practitioner", "dentist", "doctor", "chiropractor",
                          "find someone", "near me", "local practitioner",
                          "local doctor", "not happy with", "second opinion",
                          "my doctor", "unhappy", "frustrated with my"]
_DEFAULT_TRIO = ["quiz", "e4l_scan", "intake"]


def _match_card_keys(state, query_texts):
    """Return an ordered, deduped list of matched card keys with NO fallback.
    Extracts the signal-matching logic shared by surface() and surface_for_chat()."""
    state = state or {}
    text = " ".join(query_texts or []).lower()
    aw = state.get("awareness_stage", "unknown")
    rung = state.get("current_rung", "arrival")
    gates = set(state.get("unlocked_gates") or [])
    keys = []
    def add(k):
        if k not in keys:
            keys.append(k)
    if any(k in text for k in _PRACTITIONER_KEYWORDS):
        add("practitioner")
    remedy_intent = any(c in text for c in _REMEDY_MATCH_CUES)
    specific_product = (any(k in text for k in _PRODUCT_KEYWORDS) or remedy_intent)
    if remedy_intent:
        add("remedy_match")   # Socratic matcher → /begin/match (before generic browse)
    if specific_product:
        add("product")
    generic_product = (not specific_product
                       and any(c in text for c in _GENERIC_PRODUCT_CUES))
    if any(k in text for k in _VOICE_KEYWORDS) or generic_product:
        add("voice_distinctions")
    if any(k in text for k in _LEARN_KEYWORDS):
        add("ash_course")
    if any(k in text for k in _SHARE_KEYWORDS):
        add("pay_forward")
    if (AWARENESS_RANK.get(aw, 0) >= AWARENESS_RANK["most"]
            or RUNG_INDEX.get(rung, 0) >= RUNG_INDEX["assess"]
            or len(gates) >= 5):
        add("ash_masterclass")
    return keys


def surface(state, query_texts, ref=""):
    """Return an ordered, deduped, capped-at-3 list of card dicts for the visitor's
    signals. Falls back to the default trio when no contextual signal fires."""
    keys = _match_card_keys(state, query_texts)
    if not keys:
        keys = list(_DEFAULT_TRIO)
    return [_card(k, ref) for k in keys[:3]]


def surface_for_chat(state, query_texts, ref=""):
    """Return an ordered, deduped, capped-at-2 list of card dicts for chat context.
    Falls back to a single gentle quiz card when no contextual signal fires."""
    keys = _match_card_keys(state, query_texts)
    if not keys:
        keys = ["quiz"]
    return [_card(k, ref) for k in keys[:2]]


# ---------------------------------------------------------------------------
# Piece 4 -- per-tier ascension catalog (paid tiers 3-8, slug-keyed)
# ---------------------------------------------------------------------------
TIER_CATALOG = {
    "biofield-analysis": {"slug": "biofield-analysis", "n": 3,
        "title": "Causal Biofield Analysis + Program Design + Consultation",
        "price": "$300", "value": "$1,000 value",
        "included": "Your ASH Causal Biofield Analysis, a Functional Formulations™ program designed for you, and a consultation.",
        "cta_label": "Book your consultation"},
    "certification": {"slug": "certification", "n": 4,
        "title": "ASH Certification Training", "price": "~$3,600", "value": "",
        "included": "Train in the Accelerated Self-Healing™ method.",
        "cta_label": "Book your consultation"},
    "one-to-one": {"slug": "one-to-one", "n": 5,
        "title": "ASH 1:1 Live Support + Certification", "price": "~$8,500", "value": "6 months",
        "included": "Six months of live one-to-one support alongside your certification.",
        "cta_label": "Book your consultation"},
    "healing-oasis-tools": {"slug": "healing-oasis-tools", "n": 6,
        "title": "Healing Oasis Tools Installation + Training + Certification",
        "price": "~$14,000", "value": "",
        "included": "Your own Healing Oasis tools installed, full tools training, and ASH certification.",
        "cta_label": "Book your consultation"},
    "hawaii-immersion": {"slug": "hawaii-immersion", "n": 7,
        "title": "Hawaii Island Technology Detox at The Shire", "price": "~$25,000", "value": "1 week",
        "included": "A one-week immersion on Hawaiʻi Island: Healing Oasis package, training, and ASH certification.",
        "cta_label": "Book your consultation"},
    "consultant-package": {"slug": "consultant-package", "n": 8,
        "title": "Complete Consultant Package", "price": "~$50,000", "value": "",
        "included": "Everything, including the software, to run your own Accelerated Self-Healing practice.",
        "cta_label": "Book your consultation"},
}


def tier_for(slug):
    return TIER_CATALOG.get(slug)
