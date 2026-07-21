"""Gated auto-confirm of AI biofield drafts (Phase 1, B+C).

evaluate_quality = the "B" gate: a draft is publishable only if every layer is
complete and resolvable and no red-flag term appears. should_sample = the "C"
audit: a deterministic slice always goes to human review. maybe_auto_confirm ties
them together and logs every decision. Pure module: no Flask, no network."""
import hashlib
import json


def _dosing_present(layer):
    return any((layer.get(k) or "").strip() for k in ("dosage", "frequency", "timing", "dosing"))


def evaluate_quality(content, *, resolve_slug, red_flag_terms):
    """(ok, reasons). ok only when every check passes; reasons lists each failure."""
    reasons = []
    content = content or {}
    layers = [L for L in (content.get("layers") or []) if (L.get("title") or "").strip()]
    if not layers:
        reasons.append("no titled layer")
    for i, L in enumerate(layers):
        rem = (L.get("remedy") or "").strip()
        if not rem:
            reasons.append(f"layer {i}: empty remedy")
        elif resolve_slug(rem) is None:
            reasons.append(f"layer {i}: remedy not in catalog ({rem!r})")
        if not _dosing_present(L):
            reasons.append(f"layer {i}: missing dosing")
    if red_flag_terms:
        blob = json.dumps(content, ensure_ascii=False).lower()
        for term in red_flag_terms:
            if term and term.lower() in blob:
                reasons.append(f"red_flag term: {term}")
    return (not reasons, reasons)


def should_sample(email, scan_date, pct):
    """Deterministic audit sampling: hash(email|scan_date) mod 100 < pct."""
    try:
        pct = int(pct)
    except (TypeError, ValueError):
        pct = 0
    if pct <= 0:
        return False
    if pct >= 100:
        return True
    key = f"{(email or '').strip().lower()}|{(scan_date or '').strip()}"
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return bucket < pct


def init_autoconfirm_log(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS analysis_autoconfirm_log ("
        "email TEXT, scan_date TEXT, decision TEXT, reasons TEXT, "
        "sampled INTEGER DEFAULT 0, created_at TEXT DEFAULT '', "
        "PRIMARY KEY (email, scan_date))")
    cx.commit()


def _log(cx, email, scan_date, decision, reasons, sampled, now):
    cx.execute(
        "INSERT OR REPLACE INTO analysis_autoconfirm_log "
        "(email, scan_date, decision, reasons, sampled, created_at) VALUES (?,?,?,?,?,?)",
        ((email or "").strip().lower(), (scan_date or "").strip(), decision,
         json.dumps(reasons or []), 1 if sampled else 0, now or ""))
    cx.commit()


def maybe_auto_confirm(cx, email, scan_date, content, *, enabled, sample_pct,
                       resolve_slug, red_flag_terms, confirm_fn, now):
    """Decide confirm-vs-hold for one ai_draft. Returns an outcome string; logs it."""
    if not enabled:
        return "disabled"
    ok, reasons = evaluate_quality(content, resolve_slug=resolve_slug,
                                   red_flag_terms=red_flag_terms)
    if not ok:
        _log(cx, email, scan_date, "held_quality", reasons, False, now)
        return "held_quality"
    if should_sample(email, scan_date, sample_pct):
        _log(cx, email, scan_date, "held_sample", [], True, now)
        return "held_sample"
    confirm_fn(cx, email, scan_date, content)
    _log(cx, email, scan_date, "confirmed", [], False, now)
    return "confirmed"
