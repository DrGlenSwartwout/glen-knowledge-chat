"""Client clinical intake: a declarative form (brought home from Practice Better)
plus pure store logic. No Flask, no network. The form definition is the single
source of truth for the questions; the local tagger consumes the `maps_to` hints.

Response shape: answers is a dict field_id -> value. Scalars for text/number/
scale/single_choice; a list of row-dicts for `table` fields; `terms` is
{"agreed": bool, "signature": str, "date": str}."""
import json

# --- scale option builders (labels are Glen's exact PB wording) ---
def _scale(pairs):
    return [{"value": v, "label": l} for v, l in pairs]

_TERRAIN = _scale([
    (1, "Cancer, Degeneration, Viral or Low Energy"),
    (2, "Rapid Aging, Bacterial, or Parasitic"),
    (3, "Fungal, Deposition, Slow Metabolism, or Low Body Temperature"),
    (4, "Allergy or Toxicity"),
    (5, "Stress or Hormonal Imbalance"),
])
_PENETRATION = _scale([
    (1, "Genetic or epigenetic expression"),
    (2, "Cell metabolism or mitochondrial dysfunction"),
    (3, "Connective tissue, immunity, autonomic or other nerve challenges"),
    (4, "Circulation, lymph drainage issues"),
    (5, "Poor digestion, dysbiosis, or other gut concerns"),
])
_TISSUE = _scale([
    (1, "Urogenital or Muscle"),
    (2, "Connective Tissue, Immune, or Cardiovascular"),
    (3, "Digestive or Respiratory"),
    (4, "Neuroendocrine"),
    (5, "Skin"),
])
_RESPONSE = _scale([
    (1, "No change"),
    (2, "Feel worse before better"),
    (3, "Mixed: some symptoms worse, but others better"),
    (4, "Some gradual improvement"),
    (5, "Rapid improvement"),
])
_COMMITMENT = _scale([(n, str(n)) for n in range(1, 11)])

INTAKE_FORM = {
    "version": "2026-07-07",
    "sections": [
        {"id": "personal", "title": "Personal Information", "fields": [
            {"id": "first_name", "type": "text", "label": "Legal first name", "required": True},
            {"id": "last_name", "type": "text", "label": "Last name", "required": True},
            {"id": "street", "type": "text", "label": "Street"},
            {"id": "unit", "type": "text", "label": "Unit"},
            {"id": "city", "type": "text", "label": "City"},
            {"id": "state", "type": "text", "label": "State"},
            {"id": "postal_code", "type": "text", "label": "Postal code"},
            {"id": "country", "type": "text", "label": "Country"},
            {"id": "email", "type": "email", "label": "Email address", "required": True},
            {"id": "home_phone", "type": "tel", "label": "Home phone"},
            {"id": "mobile_phone", "type": "tel", "label": "Mobile phone"},
            {"id": "dob", "type": "date", "label": "Date of birth", "required": True},
            {"id": "relationship_status", "type": "single_choice", "label": "Relationship status",
             "options": ["Single", "Partnered", "Married", "Divorced", "Widowed", "Prefer not to say"]},
            {"id": "gender", "type": "single_choice", "label": "Gender",
             "options": ["Woman/Girl", "Man/Boy", "Nonbinary", "Prefer not to say"]},
            {"id": "occupation", "type": "text", "label": "Occupation"},
            {"id": "hours_per_week", "type": "number", "label": "Hours per week"},
            {"id": "referred_by", "type": "text", "label": "Referred by"},
            {"id": "favorite_color", "type": "text", "label": "Describe your favorite color"},
        ]},
        {"id": "goals", "title": "Top Health Goals", "fields": [
            {"id": "health_concerns", "type": "table",
             "label": "List your current health concerns in order of importance",
             "help": "Rate how important each concern is to you from 1 to 10.",
             "columns": [
                 {"id": "concern", "type": "text", "label": "Health concern"},
                 {"id": "rating", "type": "number", "label": "Rating (1-10)"},
                 {"id": "years_since_onset", "type": "number", "label": "Years since onset"},
             ]},
        ]},
        {"id": "dimensions", "title": "Key Dimensions of the Clinical Theory of Everything",
         "fields": [
            {"id": "terrain", "type": "scale", "maps_to": "terrain", "required": True,
             "label": "Dominant Terrain",
             "help": "Select the lowest number that applies to current issues.",
             "options": _TERRAIN},
            {"id": "penetration", "type": "scale", "maps_to": "penetration", "required": True,
             "label": "Penetration of the Body Sanctuary",
             "help": "Select the lowest number that applies to current issues.",
             "options": _PENETRATION},
            {"id": "tissue_layer", "type": "scale", "maps_to": "tissue_layer", "required": True,
             "label": "Dominant Embryological Tissue Layer",
             "help": "Select the lowest number that applies to your current issues.",
             "options": _TISSUE},
            {"id": "response", "type": "scale", "maps_to": "response", "required": True,
             "label": "Dominant Healing Response",
             "help": "Your most typical response to well-selected natural therapies.",
             "options": _RESPONSE},
            {"id": "commitment", "type": "scale", "maps_to": "commitment", "required": True,
             "label": "Level of commitment to improving your health",
             "help": "1 is lowest, 10 is highest.", "options": _COMMITMENT},
            {"id": "obstacles", "type": "textarea",
             "label": "Is there anything that will get in the way of following a plan?"},
            {"id": "budget_monthly", "type": "number", "label": "Current budget",
             "help": "Estimated USD per month available to invest in better health."},
        ]},
        {"id": "history", "title": "Personal Health History", "fields": [
            {"id": "sleep", "type": "textarea",
             "label": "Do you have trouble falling asleep, staying asleep, or wake frequently?"},
            {"id": "dental", "type": "textarea", "label": "Dental issues: any amalgams or root canals?"},
            {"id": "vaccinations", "type": "textarea",
             "label": "Vaccinations: any COVID or other recent vaccinations?"},
            {"id": "supplements", "type": "table", "label": "Supplements you take now",
             "help": "Include vitamins, herbs, minerals. Rate how certain you are each is needed, 1 to 10.",
             "columns": [
                 {"id": "brand", "type": "text", "label": "Brand name"},
                 {"id": "name", "type": "text", "label": "Supplement name"},
                 {"id": "reason", "type": "text", "label": "Reason"},
                 {"id": "need", "type": "number", "label": "Need (1-10)"},
             ]},
            {"id": "diagnoses", "type": "table", "label": "Medical diagnoses", "columns": [
                 {"id": "diagnosis", "type": "text", "label": "Diagnosis"},
                 {"id": "current", "type": "single_choice", "label": "Status", "options": ["Current", "Past"]},
                 {"id": "age_onset", "type": "number", "label": "Age at onset"},
            ]},
            {"id": "medications", "type": "table", "label": "Medications you are currently taking",
             "columns": [
                 {"id": "medication", "type": "text", "label": "Medication"},
                 {"id": "reason", "type": "text", "label": "Reason"},
             ]},
            {"id": "surgeries", "type": "table", "label": "Past hospitalizations or surgeries",
             "columns": [
                 {"id": "procedure", "type": "text", "label": "Hospitalization or surgery"},
                 {"id": "reason", "type": "text", "label": "Reason"},
                 {"id": "age", "type": "number", "label": "Age"},
             ]},
            {"id": "allergies", "type": "table",
             "label": "Food or environmental allergies or sensitivities", "columns": [
                 {"id": "sensitivity", "type": "text", "label": "Sensitivity"},
                 {"id": "reaction", "type": "text", "label": "Reaction"},
            ]},
            {"id": "portrait", "type": "textarea",
             "label": "Portrait photo",
             "help": "Link to a photo for our clinical database, or note that one was sent."},
        ]},
        {"id": "consent", "title": "Consent", "fields": [
            {"id": "terms", "type": "consent", "required": True,
             "label": "I agree to the terms of service for Wellness Services at "
                      "remedymatch.com/info/terms-and-conditions."},
        ]},
    ],
}

# --- flat field index for validation ---
def _fields():
    for sec in INTAKE_FORM["sections"]:
        for f in sec["fields"]:
            yield f


def validate_response(answers):
    """Return the ids of required-but-missing or invalid fields (empty = valid).
    Tables are optional in v1 (a client may legitimately have none)."""
    errors = []
    for f in _fields():
        fid, ftype, req = f["id"], f["type"], f.get("required", False)
        val = answers.get(fid)
        if ftype == "scale":
            allowed = {o["value"] for o in f["options"]}
            if val is None:
                if req:
                    errors.append(fid)
            elif val not in allowed:
                errors.append(fid)
        elif ftype == "consent":
            ok = isinstance(val, dict) and val.get("agreed") is True and str(val.get("signature") or "").strip()
            if req and not ok:
                errors.append(fid)
        elif ftype == "table":
            continue  # optional in v1
        else:
            if req and not str(val or "").strip():
                errors.append(fid)
    return errors


def init_intake_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS intake_responses ("
        " email TEXT PRIMARY KEY,"
        " form_version TEXT NOT NULL,"
        " status TEXT NOT NULL,"          # 'draft' | 'submitted'
        " answers_json TEXT NOT NULL,"
        " created_at TEXT NOT NULL,"
        " submitted_at TEXT)")


def _upsert(cx, email, answers, status, now, submitted_at):
    email = (email or "").strip().lower()
    cx.execute(
        "INSERT INTO intake_responses (email, form_version, status, answers_json, created_at, submitted_at)"
        " VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(email) DO UPDATE SET"
        "   form_version=excluded.form_version, status=excluded.status,"
        "   answers_json=excluded.answers_json,"
        "   submitted_at=COALESCE(excluded.submitted_at, intake_responses.submitted_at)",
        (email, INTAKE_FORM["version"], status, json.dumps(answers), now, submitted_at))
    cx.commit()


def save_draft(cx, email, answers, now):
    _upsert(cx, email, answers, "draft", now, None)


def submit(cx, email, answers, now):
    _upsert(cx, email, answers, "submitted", now, now)


def is_submitted(cx, email):
    row = cx.execute("SELECT status FROM intake_responses WHERE email=?",
                     ((email or "").strip().lower(),)).fetchone()
    return bool(row) and row[0] == "submitted"


def _row_to_dict(row):
    d = dict(row)
    d["answers"] = json.loads(d.pop("answers_json"))
    return d


def get_response(cx, email):
    row = cx.execute("SELECT * FROM intake_responses WHERE email=?",
                     ((email or "").strip().lower(),)).fetchone()
    return _row_to_dict(row) if row else None


def list_submitted(cx):
    rows = cx.execute(
        "SELECT * FROM intake_responses WHERE status='submitted' ORDER BY submitted_at").fetchall()
    return [_row_to_dict(r) for r in rows]
