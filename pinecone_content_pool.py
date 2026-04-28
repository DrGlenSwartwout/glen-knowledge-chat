"""
Pinecone content pool for the incentive engine. Fetches candidate
clinical-qa entries by audience and surfaces topic + source-text
for adaptive Personal email content selection.
"""

import os
import json
from functools import lru_cache


# Default candidate pool size — pull this many clinical-qa entries when
# building the audience-filtered topic candidate list. Small enough that
# Pinecone fetches stay fast; large enough that the selector has variety.
POOL_SIZE = 50


def _get_pinecone_index():
    """Lazy-init Pinecone index. Replaceable in tests via monkeypatch."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index("remedy-match-llc")


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@lru_cache(maxsize=1)
def _list_clinical_qa_ids():
    """Cache the list of all clinical-qa entry IDs. Refreshes once per
    process lifetime."""
    idx = _get_pinecone_index()
    all_ids = []
    page_token = None
    while True:
        try:
            resp = idx.list(namespace="clinical-qa", prefix="qa-",
                            pagination_token=page_token)
            ids_chunk = (list(resp.ids) if hasattr(resp, "ids")
                         else list(resp))
        except Exception:
            ids_chunk = []
        all_ids.extend(ids_chunk)
        page_token = (resp.pagination.get("next")
                      if hasattr(resp, "pagination") and resp.pagination
                      else None)
        if not page_token or not ids_chunk:
            break
    return tuple(all_ids)


def fetch_pool_for_audience(audience: str = "both",
                            max_entries: int = POOL_SIZE) -> list:
    """Fetch up to max_entries clinical-qa entries matching the user's
    audience. Returns a list of dicts:
      [{"id", "topic", "topics", "text", "audience"}]

    Audience filter:
      - 'client' → keep entries with audience in ('client', 'both')
      - 'practitioner' → keep entries with audience in ('practitioner', 'both')
      - 'both' (default) → no filter
    """
    idx = _get_pinecone_index()
    all_ids = list(_list_clinical_qa_ids())[:max_entries * 3]
    if not all_ids:
        return []

    # Pinecone fetch has a 512-char limit on the joined ID list. Chunk by
    # fixed count (10) to stay well under that ceiling regardless of ID length.
    FETCH_CHUNK = 10
    all_vectors = {}
    for i in range(0, len(all_ids), FETCH_CHUNK):
        chunk = all_ids[i : i + FETCH_CHUNK]
        try:
            rec = idx.fetch(ids=chunk, namespace="clinical-qa")
            all_vectors.update(rec.vectors)
        except Exception as e:
            # Continue on chunk failure rather than aborting the whole pool
            import os
            if os.environ.get("DEBUG_PINECONE_FETCH"):
                print(f"[pool] chunk {i}-{i+FETCH_CHUNK} failed: {e}", flush=True)

    pool = []
    for vid, v in all_vectors.items():
        meta = v.metadata or {}
        entry_audience = meta.get("audience", "both")
        if audience == "client" and entry_audience == "practitioner":
            continue
        if audience == "practitioner" and entry_audience == "client":
            continue
        # Topic — use first item from topics array as the canonical label;
        # fall back to question or id slug
        topics = meta.get("topics", [])
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except Exception:
                topics = [topics]
        if not isinstance(topics, list):
            topics = []
        primary_topic = topics[0] if topics else (meta.get("question", "")[:40] or vid)
        pool.append({
            "id":       vid,
            "topic":    primary_topic,
            "topics":   topics,
            "text":     meta.get("text", ""),
            "audience": entry_audience,
        })
        if len(pool) >= max_entries:
            break
    return pool


def candidate_topics_for_audience(audience: str = "both") -> list:
    """Return just the list of unique primary topics in the audience pool."""
    pool = fetch_pool_for_audience(audience)
    seen = set()
    out = []
    for entry in pool:
        if entry["topic"] not in seen:
            seen.add(entry["topic"])
            out.append(entry["topic"])
    return out


def fetch_source_text_for_topic(topic: str, audience: str = "both") -> str:
    """Fetch the longest source text for a given topic in the audience
    pool. Used as the grounding passage for generate_personal_email."""
    pool = fetch_pool_for_audience(audience)
    matches = [e for e in pool if e["topic"] == topic
               or topic in e["topics"]]
    if not matches:
        return ""
    # Prefer the longest text (richest grounding)
    matches.sort(key=lambda e: -len(e["text"]))
    return matches[0]["text"]
