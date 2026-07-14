"""Single source of truth for the operator's 'next action' per record type.

Pure resolvers (normalized record dict -> descriptor) shared by the per-page
buttons and the unified /console Next Action queue. Listers + aggregate live
below. Adding a record type = write a resolver + a lister + register both.
"""

TYPE_PRIORITY = ["biofield_reveal", "handoff", "ff_match_draft"]

_DONE = {"actionable": False}


def resolve_biofield_reveal(rec):
    rid = rec.get("id")
    summary = f"{rec.get('email','')} · scan {rec.get('scan_date','')}"
    age = rec.get("age_ts", "")
    if not rec.get("first_approved"):
        return {
            "type": "biofield_reveal", "id": rid, "actionable": True, "state": "draft",
            "label": "Approve & send",
            "action": {"kind": "dispatch",
                       "keys": ["biofield_reveal.approve", "biofield_reveal.send"],
                       "body": {"id": rid}},
            "confirm": True,
            "secondary": {"label": "Approve only, don't email",
                          "action": {"kind": "dispatch",
                                     "keys": ["biofield_reveal.approve"],
                                     "body": {"id": rid}},
                          "confirm": False},
            "summary": summary, "age_ts": age,
        }
    if not rec.get("notified_at"):
        return {
            "type": "biofield_reveal", "id": rid, "actionable": True,
            "state": "approved_unsent", "label": "Send reveal link",
            "action": {"kind": "dispatch", "keys": ["biofield_reveal.send"],
                       "body": {"id": rid}},
            "confirm": True, "secondary": None,
            "summary": summary, "age_ts": age,
        }
    return dict(_DONE)


def resolve_ff_match_draft(rec):
    if rec.get("status") != "draft":
        return dict(_DONE)
    email = rec.get("email", ""); when = rec.get("scan_date", "")
    return {
        "type": "ff_match_draft", "id": None, "actionable": True, "state": "draft",
        "label": "Publish",
        "action": {"kind": "post", "url": "/api/console/ff-match-drafts/publish",
                   "body": {"email": email, "scan_date": when}},
        "confirm": True,
        "secondary": {"label": "Open to edit",
                      "action": {"kind": "link", "url": "/console/ff-drafts"},
                      "confirm": False},
        "summary": f"{email} · scan {when}", "age_ts": rec.get("age_ts", ""),
    }


def resolve_handoff(rec):
    if rec.get("biofield_status") != "ai_draft":
        return dict(_DONE)
    email = rec.get("email", "")
    return {
        "type": "handoff", "id": None, "actionable": True, "state": "needs_publish",
        "label": "Review & publish",
        "action": {"kind": "post", "url": "/api/console/biofield/publish",
                   "body": {"email": email, "send": False}},
        "confirm": True,
        "secondary": {"label": "Publish & notify client",
                      "action": {"kind": "post", "url": "/api/console/biofield/publish",
                                 "body": {"email": email, "send": True}},
                      "confirm": True},
        "summary": f"{email}", "age_ts": rec.get("age_ts", ""),
    }
