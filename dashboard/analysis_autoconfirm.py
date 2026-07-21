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
