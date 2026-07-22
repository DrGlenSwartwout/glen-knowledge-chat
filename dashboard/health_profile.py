"""Curated read of a client's intake record for the portal's "My Health
Profile" page. Exposes only the self-reported health record — the goals,
clinical dimensions, and personal health history sections of INTAKE_FORM —
never the personal/identity section or the consent signature.

Defensive like the other portal blocks (see _practitioner_finder_block in
portal_view.py): any failure degrades to an empty-but-shaped payload rather
than raising, so a bad intake row never breaks the rest of the portal."""
from dashboard import intake as _intake

# Section ids that make up the editable, self-reported health record. The
# clinical dimensions (terrain/penetration/tissue_layer/response/commitment)
# are included: they are self-reported and expected to change as the client
# heals. `personal` (identity: name/email/address/...) and the `terms`
# consent field are excluded.
_EDITABLE_SECTION_IDS = ("goals", "dimensions", "history")


def _editable_sections():
    return [sec for sec in _intake.INTAKE_FORM["sections"] if sec["id"] in _EDITABLE_SECTION_IDS]


def curated_fields():
    """The curated list of (section, field) dicts from INTAKE_FORM that make
    up the editable health record, grouped by section in form order."""
    return [{"id": sec["id"], "title": sec.get("title", sec["id"]), "fields": sec["fields"]}
            for sec in _editable_sections()]


EDITABLE_FIELD_IDS = {f["id"] for sec in _editable_sections() for f in sec["fields"]}


def build_block(cx, email, enabled):
    """Portal payload block: {"enabled", "status", "sections", "suggestion_count"}.
    Off -> {"enabled": False} only. On -> projects the curated fields (id/label/
    type/value) grouped by section, plus a pending-suggestion count. Degrades
    to an empty-but-shaped record on any error (mirror _practitioner_finder_block)."""
    if not enabled:
        return {"enabled": False}
    try:
        record = _intake.get_response(cx, email)
    except Exception:
        record = None
    answers = (record or {}).get("answers") or {}
    sections = []
    for sec in curated_fields():
        fields = []
        for f in sec["fields"]:
            fid = f["id"]
            fields.append({
                "id": fid,
                "label": f.get("label", fid),
                "type": f.get("type"),
                "value": answers.get(fid),
            })
        sections.append({"title": sec["title"], "fields": fields})
    try:
        from dashboard import health_suggestions as _hs
        suggestion_count = _hs.count_pending(cx, email)
    except Exception:
        suggestion_count = 0
    return {
        "enabled": True,
        "status": "has_record" if record else "empty",
        "sections": sections,
        "suggestion_count": suggestion_count,
    }
