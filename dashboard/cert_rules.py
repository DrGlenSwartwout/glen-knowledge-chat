# dashboard/cert_rules.py
"""Certification work-product rules (pure: no I/O, no Flask).

The single place the certification completion rules live, so "rules subject to
change as the program evolves" stays a one-file edit.

Source of truth: ~/AI-Training/00 Projects/certification/cert-work-product-framework.md
"""

# The 12 module concepts (verbatim labels), id 1..12.
MODULES = [
    {"id": 1,  "label": "Body"},
    {"id": 2,  "label": "Mind"},
    {"id": 3,  "label": "Spirit"},
    {"id": 4,  "label": "Family inheritance / EVOX transgenerational perception reframing"},
    {"id": 5,  "label": "Personal history / retracing"},
    {"id": 6,  "label": "Epigenetics / EVOX / Infoceuticals"},
    {"id": 7,  "label": "Symptoms / embryological tissue layers"},
    {"id": 8,  "label": "Terrain phases"},
    {"id": 9,  "label": "Diagnostic category"},
    {"id": 10, "label": "Therapeutic hierarchy"},
    {"id": 11, "label": "Regulatory response"},
    {"id": 12, "label": "Prognosis / self-fulfilling prophecy / faith / belief"},
]

# Format catalog. `kind` groups formats so the written+video rule is testable.
# kind in {"written","video","audio","visual"}.
FORMATS = [
    {"key": "talking_head_scripted",   "label": "Talking-head video (scripted)",        "kind": "video"},
    {"key": "talking_head_unscripted", "label": "Talking-head video (unscripted)",      "kind": "video"},
    {"key": "slideshow_video",         "label": "Slideshow / screen-share video",       "kind": "video"},
    {"key": "interview_guest",         "label": "Video interview (being interviewed)",  "kind": "video"},
    {"key": "interview_host",          "label": "Video interview (interviewing someone)", "kind": "video"},
    {"key": "demo_video",              "label": "Demonstration / walkthrough video",    "kind": "video"},
    {"key": "webinar",                 "label": "Webinar / workshop recording",         "kind": "video"},
    {"key": "short_form_video",        "label": "Short-form social video",              "kind": "video"},
    {"key": "written_post",            "label": "Written post (social / blog)",         "kind": "written"},
    {"key": "article",                 "label": "Article / feature piece",              "kind": "written"},
    {"key": "white_paper",             "label": "White paper / longer paper",           "kind": "written"},
    {"key": "case_report",             "label": "Case report (single case)",            "kind": "written"},
    {"key": "study_case_control",      "label": "Study — case-control",                 "kind": "written"},
    {"key": "study_observational",     "label": "Study — group observational",          "kind": "written"},
    {"key": "study_controlled",        "label": "Study — controlled group",             "kind": "written"},
    {"key": "literature_review",       "label": "Literature review / synthesis",        "kind": "written"},
    {"key": "protocol_writeup",        "label": "Protocol / program design write-up",   "kind": "written"},
    {"key": "book_chapter",            "label": "Book chapter / ebook contribution",    "kind": "written"},
    {"key": "podcast",                 "label": "Podcast (host or guest)",              "kind": "audio"},
    {"key": "audio_testimonial",       "label": "Audio testimonial / narrated story",   "kind": "audio"},
    {"key": "infographic",             "label": "Infographic / carousel / one-sheet",   "kind": "visual"},
    {"key": "before_after",            "label": "Before/after photo essay (consent)",   "kind": "visual"},
    {"key": "annotated_scan",          "label": "Annotated scan / reading (de-identified)", "kind": "visual"},
    {"key": "conference",              "label": "Conference talk or poster",            "kind": "visual"},
    {"key": "qa_explainer",            "label": "Q&A / FAQ explainer",                  "kind": "written"},
    {"key": "social_thread",           "label": "Social thread / series",               "kind": "written"},
    {"key": "client_testimonial",      "label": "Client testimonial capture (consented)", "kind": "video"},
]

_KIND_BY_KEY = {f["key"]: f["kind"] for f in FORMATS}

MIN_SUBMISSIONS = 12


def kinds_for(format_keys):
    """The set of `kind` values for a list of format keys (unknown keys ignored)."""
    return {_KIND_BY_KEY[k] for k in (format_keys or []) if k in _KIND_BY_KEY}


def evaluate(submissions):
    """Given the student's approved+published submissions, return progress vs the
    completion rules.

    Each submission is a dict with:
      - "credited_modules": list[int]  (the module ids credited on approval)
      - "formats": list[str]           (format keys from FORMATS)

    Returns a dict. `complete` is the AND of the four completion rules:
    >= MIN_SUBMISSIONS approved, no missing modules, has_written, has_video.
    `multi_modality` is informational only (it does NOT gate completion): the
    minimum modality rule is specifically written + video, per the program
    framework's "minimum (B)" requirement.
    """
    subs = list(submissions or [])
    approved_count = len(subs)

    covered = set()
    all_kinds = set()
    for s in subs:
        covered |= {int(m) for m in (s.get("credited_modules") or [])}
        all_kinds |= kinds_for(s.get("formats"))

    all_ids = [m["id"] for m in MODULES]
    modules_missing = [i for i in all_ids if i not in covered]
    has_written = "written" in all_kinds
    has_video = "video" in all_kinds
    multi_modality = len(all_kinds) >= 2

    reasons = []
    if approved_count < MIN_SUBMISSIONS:
        reasons.append(f"Needs at least {MIN_SUBMISSIONS} approved submissions "
                       f"(has {approved_count}).")
    if modules_missing:
        labels = ", ".join(str(i) for i in modules_missing)
        reasons.append(f"{len(modules_missing)} module(s) not yet covered: {labels}.")
    if not has_written:
        reasons.append("No written-format submission yet.")
    if not has_video:
        reasons.append("No video-format submission yet.")

    return {
        "approved_count": approved_count,
        "modules_covered": covered,
        "modules_missing": modules_missing,
        "has_written": has_written,
        "has_video": has_video,
        "multi_modality": multi_modality,
        "complete": not reasons,
        "reasons": reasons,
    }
