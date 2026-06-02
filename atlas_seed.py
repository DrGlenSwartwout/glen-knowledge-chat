"""Offline enrichment: hand-seeded concepts -> coords/neighbors/clusters/links -> pending.

Run:  python atlas_seed.py            # reads data/atlas-seed-input.json, writes data/atlas-pending.json
Pure helpers (pca_coords, cosine_neighbors, kmeans_clusters) are import-safe and tested.
The enrich() entrypoint needs OPENAI/PINECONE env and is run manually, not in CI.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED_PATH = DATA_DIR / "atlas-seed-input.json"
PENDING_PATH = DATA_DIR / "atlas-pending.json"
VIDEOS_PATH = DATA_DIR / "atlas-videos.json"
RANDOM_STATE = 42


def pca_coords(vectors):
    """Project to 2-D and min-max scale into the unit square. Deterministic. -> [(x,y), ...]."""
    arr = np.asarray(vectors, dtype=float)
    comps = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(arr)
    out = []
    for axis in (0, 1):
        col = comps[:, axis]
        lo, hi = col.min(), col.max()
        comps[:, axis] = (col - lo) / (hi - lo) if hi > lo else 0.5
    for row in comps:
        out.append((round(float(row[0]), 4), round(float(row[1]), 4)))
    return out


def cosine_neighbors(vectors, ids, k=4):
    """Top-k cosine-nearest ids per id (excluding self). -> {id: [id,...]}."""
    arr = np.asarray(vectors, dtype=float)
    norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)
    result = {}
    for i, cid in enumerate(ids):
        order = np.argsort(sims[i])[::-1][:k]
        result[cid] = [ids[j] for j in order]
    return result


def kmeans_clusters(vectors, k):
    arr = np.asarray(vectors, dtype=float)
    k = max(1, min(k, len(arr)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(arr)
    return [int(x) for x in labels]


def _match_videos(concept, videos, embed_fn):
    """Attach best-matching videos by cosine of concept summary vs video title+description."""
    if not videos:
        return []
    cvec = np.asarray(embed_fn(concept["label"] + " " + concept["summary"]), dtype=float)
    scored = []
    for v in videos:
        vvec = np.asarray(embed_fn(v["title"] + " " + v.get("description", "")), dtype=float)
        sim = float(cvec @ vvec / ((np.linalg.norm(cvec) * np.linalg.norm(vvec)) + 1e-12))
        scored.append((sim, v))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"type": "video", "source": v["platform"], "url": v["url"], "title": v["title"]}
            for sim, v in scored[:2] if sim > 0.78]


def enrich(seed_path=SEED_PATH, out_path=PENDING_PATH):  # pragma: no cover (needs APIs)
    """Manual entrypoint: embed seeds, compute coords/neighbors/clusters, match links."""
    from app import embed  # reuse the app's ada-002 embed
    seeds = json.loads(Path(seed_path).read_text(encoding="utf-8"))["concepts"]
    videos = json.loads(Path(VIDEOS_PATH).read_text(encoding="utf-8")).get("videos", [])
    vecs = [embed(s["label"] + " " + s["summary"]) for s in seeds]
    coords = pca_coords(vecs)
    nbrs = cosine_neighbors(vecs, [s["id"] for s in seeds], k=4)
    labels = kmeans_clusters(vecs, k=max(2, len(seeds) // 6))
    out = []
    for s, (x, y), lab in zip(seeds, coords, labels):
        links = list(s.get("links", [])) + _match_videos(s, videos, embed)
        out.append({
            "id": s["id"], "label": s["label"], "aliases": s.get("aliases", []),
            "summary": s["summary"], "namespaces": s.get("namespaces", []),
            "cluster": s.get("cluster") or f"cluster-{lab}", "parent": s.get("parent"),
            "coords": {"x": x, "y": y}, "neighbors": nbrs[s["id"]],
            "links": links, "status": "pending", "proposed_from": "atlas_seed",
        })
    Path(out_path).write_text(
        json.dumps({"version": seeds and "seeded", "concepts": out}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"wrote {len(out)} pending concepts -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    enrich()
    sys.exit(0)
