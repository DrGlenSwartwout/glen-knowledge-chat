#!/usr/bin/env python3
"""
One-pass backfill — tags every clinical-qa entry in Pinecone with an
'audience' field ('client' | 'practitioner' | 'both'). Default 'both'
unless heuristics or explicit content suggest otherwise.

Heuristics:
  - mentions 'practitioner', 'wholesale', 'dispensary', 'ASH cert',
    'for clinicians', 'in clinical practice'  → audience='practitioner'
  - 2+ DEPTH keywords (mechanism, molecular, pharmacokinetic,
    biochemistry, tight-junction, phytonutrient)  → audience='practitioner'
  - mentions 'I have', 'my symptom', 'I feel', 'self-healing', 'at home'
    → audience='client'
  - default → audience='both'

Re-runnable safely (uses Pinecone metadata-update, not re-embed).

Usage:
  doppler run --project remedy-match --config prd -- \\
    python3 scripts/backfill_audience_tags.py [--dry-run]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRACTITIONER_KEYWORDS = [
    "practitioner",
    "wholesale",
    "dispensary",
    "ASH certif",
    "ASH cert",
    "for clinicians",
    "in clinical practice",
]
DEPTH_KEYWORDS = [
    "mechanism",
    "molecular",
    "pharmacokinetic",
    "biochemistry",
    "tight-junction",
    "phytonutrient",
]
CLIENT_KEYWORDS = [
    "I have",
    "my symptom",
    "I feel",
    "self-healing",
    "at home",
]


def classify_audience(text: str) -> str:
    t = (text or "").lower()
    if any(k.lower() in t for k in PRACTITIONER_KEYWORDS):
        return "practitioner"
    if any(k.lower() in t for k in CLIENT_KEYWORDS):
        return "client"
    depth_score = sum(1 for k in DEPTH_KEYWORDS if k.lower() in t)
    if depth_score >= 2:
        return "practitioner"
    return "both"


def main():
    dry = "--dry-run" in sys.argv
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index("remedy-match-llc")

    audience_counts = {"client": 0, "practitioner": 0, "both": 0}

    # Pinecone v3+ list() is a generator that yields lists of ID strings
    # (one yield per server page).
    all_ids = []
    for page in idx.list(namespace="clinical-qa", prefix="qa-"):
        all_ids.extend(page)

    print(f"Found {len(all_ids)} clinical-qa entries to classify")

    # Batch fetches in chunks of 100 (Pinecone fetch supports up to 1000
    # but smaller chunks keep error messages readable on misconfigured
    # records).
    BATCH = 100
    processed = 0
    for start in range(0, len(all_ids), BATCH):
        batch_ids = all_ids[start : start + BATCH]
        rec = idx.fetch(ids=batch_ids, namespace="clinical-qa")
        for vid in batch_ids:
            v = rec.vectors.get(vid)
            if not v:
                continue
            text = (v.metadata or {}).get("text", "")
            audience = classify_audience(text)
            audience_counts[audience] += 1
            if not dry:
                idx.update(
                    id=vid,
                    namespace="clinical-qa",
                    set_metadata={"audience": audience},
                )
            processed += 1
            if processed % 20 == 0:
                print(f"  [{processed:4d}/{len(all_ids)}] {audience}")

    print(f"\nAudience distribution: {audience_counts}")
    if dry:
        print("[DRY RUN] no metadata updates applied")


if __name__ == "__main__":
    main()
