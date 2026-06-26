"""Flatten a recent_comms dict into a single text blob for the stress extractor (B3b)."""


def comms_to_text(context):
    context = context or {}
    parts = []
    s = (context.get("intake_summary") or "").strip()
    if s:
        parts.append(s)
    for inq in context.get("recent_inquiries") or []:
        ch = (inq.get("main_challenge") or "").strip()
        g = (inq.get("main_goal") or "").strip()
        if ch:
            parts.append("challenge: " + ch)
        if g:
            parts.append("goal: " + g)
    for q in context.get("recent_queries") or []:
        qq = (q.get("question") or "").strip()
        if qq:
            parts.append(qq)
    for fb in context.get("recent_feedback") or []:
        s2 = (fb.get("summary") or "").strip()
        if s2:
            parts.append(s2)
        tc = ", ".join([*(fb.get("topics") or []), *(fb.get("conditions") or [])])
        if tc.strip():
            parts.append(tc)
    return "\n".join(parts)
