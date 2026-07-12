"""Semantic proposals for the formulation-map curation tool: formulations used by e4l
codes SEMANTICALLY SIMILAR to a given code, drawn from the Pinecone `e4l-protocols`
namespace (each e4l_item vector's `formulations` metadata). These are SUGGESTIONS Glen
approves into the curated map (formulation_map.add_mapping) -- never auto-written.
Kept apart from formulation_map.py (pure sqlite) because it needs Pinecone; deps are
injectable so it stays unit-testable without the network.
"""
import hashlib

INDEX = "remedy-match-llc"
NAMESPACE = "e4l-protocols"


def _cid(key):
    """Same id hash the ingest uses (ingest-e4l-db.cid)."""
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _live_deps():
    import os
    import sys
    skills = os.path.expanduser("~/AI-Training/02 Skills")
    if skills not in sys.path:
        sys.path.insert(0, skills)
    from lib import pinecone_client as pc  # noqa: E402
    idx = pc.index(INDEX)
    return {
        "fetch": lambda ids: idx.fetch(ids=ids, namespace=NAMESPACE),
        "query": lambda vec, top_k: pc.query(idx, NAMESPACE, vec, top_k=top_k),
    }


def _vector_values(fetched, vid):
    vecs = fetched.get("vectors") if isinstance(fetched, dict) else getattr(fetched, "vectors", None)
    rec = (vecs or {}).get(vid)
    if rec is None:
        return None
    return rec.get("values") if isinstance(rec, dict) else getattr(rec, "values", None)


def propose_for_code(code, top_k=6, exclude=None, deps=None):
    """[{name, score}] -- formulations that codes similar to `code` are mapped to,
    ranked by the best similarity that surfaced them, excluding names in `exclude`
    (e.g. what the code already maps to). [] on missing vector or any error."""
    code = (code or "").strip()
    if not code:
        return []
    deps = deps or _live_deps()
    excl = {(e or "").strip().lower() for e in (exclude or [])}
    try:
        vid = _cid(f"e4l-item-{code}")
        vals = _vector_values(deps["fetch"]([vid]), vid)
        if not vals:
            return []
        res = deps["query"](vals, max(top_k * 3, 12))
        matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", None)
        best = {}
        for m in matches or []:
            md = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
            score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0) or 0)
            for f in (md.get("formulations") or "").split(","):
                f = f.strip()
                low = f.lower()
                if f and low not in excl:
                    best[f] = max(best.get(f, 0.0), score)
        out = [{"name": n, "score": round(s, 3)} for n, s in best.items()]
        out.sort(key=lambda x: (-x["score"], x["name"].lower()))
        return out[:top_k]
    except Exception as e:
        print(f"[formulation-propose] {code}: {e!r}", flush=True)
        return []
