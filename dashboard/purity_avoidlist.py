"""Loads and validates the versioned excipient avoid-list. Pure: no Flask, no
app import. The avoid-list is code-like config shipped in the repo, so it is
read from a MODULE-RELATIVE path -- never DATA_DIR, which strips repo data files
in prod and in the full test suite."""
import json
import os

_REPO_AVOIDLIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "excipient_avoidlist.json")


def load_avoidlist(path=None):
    with open(path or _REPO_AVOIDLIST, "r", encoding="utf-8") as f:
        al = json.load(f)
    validate(al)
    return al


def validate(al):
    if not isinstance(al, dict) or not al.get("version"):
        raise ValueError("avoid-list needs a non-empty 'version'")
    for bucket in ("red", "yellow"):
        entries = al.get(bucket)
        if not isinstance(entries, list):
            raise ValueError(f"avoid-list '{bucket}' must be a list")
        for e in entries:
            if not e.get("canonical"):
                raise ValueError(f"{bucket} entry missing 'canonical'")
            aliases = e.get("aliases")
            if not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
                raise ValueError(f"{bucket} entry {e.get('canonical')!r} needs non-empty aliases")
            if not (e.get("rationale") or "").strip():
                raise ValueError(f"{bucket} entry {e.get('canonical')!r} needs a rationale")
