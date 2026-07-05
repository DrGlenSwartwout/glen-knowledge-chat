"""Community curated-feed ranking (Layer C, slice C1). Pure logic, no sqlite and
no network: the embedder is injected by the caller. Relevance = cosine(member
interest vector, content vector) + a boost when a liked topic matches; blocked
topics are filtered out; cold start (no member vector) falls back to newest then
most-reacted. The member vector is built from the member's OWN data only."""

import math


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_interest_text(journal_texts, liked_topics, chat_texts):
    parts = list(journal_texts or []) + list(liked_topics or []) + list(chat_texts or [])
    return " ".join(p for p in parts if (p or "").strip()).strip()


def reason_for(item, liked_topics, has_vec, cold_start):
    tags = set(item.get("interest_tags") or [])
    liked = tags & set(liked_topics or [])
    if liked:
        return "Because you liked " + sorted(liked)[0]
    if cold_start or not has_vec:
        return "New in the community"
    return "Related to your recent reflections"


def rank(candidates, member_vec, content_vecs, liked_topics, blocked_topics, *, boost=0.15):
    blocked = set(blocked_topics or [])
    liked = set(liked_topics or [])
    kept = [c for c in candidates if not (set(c.get("interest_tags") or []) & blocked)]
    cold_start = not member_vec
    if cold_start:
        ordered = sorted(kept, key=lambda c: (c.get("published_at") or "",
                                              c.get("reaction_count") or 0), reverse=True)
        return [{**c, "score": 0.0,
                 "reason": reason_for(c, liked, has_vec=False, cold_start=True)}
                for c in ordered]
    scored = []
    for c in kept:
        sim = cosine(member_vec, content_vecs.get(c["id"], []))
        if set(c.get("interest_tags") or []) & liked:
            sim += boost
        scored.append({**c, "score": sim,
                       "reason": reason_for(c, liked, has_vec=True, cold_start=False)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored
