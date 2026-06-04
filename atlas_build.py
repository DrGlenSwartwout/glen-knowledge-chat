"""Atlas build orchestrator: seed-input -> enriched live + pending concept files.

Reads data/atlas-seed-input.json (produced by the vault extractors, each concept
tagged disposition auto|queue), embeds label+summary, computes 2-D coords / cosine
neighbors / cluster colors (reusing atlas_seed helpers), matches each concept to the
real video catalog (data/atlas-videos.json), validates, then SPLITS by disposition:
  auto  -> data/atlas-concepts.json  (status "live")
  queue -> data/atlas-pending.json   (status "pending")

Run (manual, needs OPENAI/PINECONE env):  python atlas_build.py
Pure geometry helpers live in atlas_seed.py and stay independently tested.
"""
import json
import sys
from pathlib import Path

import numpy as np

from atlas_seed import pca_coords, cosine_neighbors, kmeans_clusters
from atlas_store import validate_concept, load_videos, REPO_DATA

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED_PATH = DATA_DIR / "atlas-seed-input.json"
# The build always writes the git-committed copies (deploy ships them; the app then seeds
# the persistent disk from these). NOT the runtime persistent paths.
CONCEPTS_PATH = REPO_DATA / "atlas-concepts.json"
PENDING_PATH = REPO_DATA / "atlas-pending.json"
VIDEO_SIM_THRESHOLD = 0.78
NEIGHBORS_K = 4


def _client():
    from app import _oa
    return _oa


def batch_embed(texts, chunk=200):
    """Embed many texts with the app's ada-002 client, in chunks. -> list[vector]."""
    oa, out = _client(), []
    for i in range(0, len(texts), chunk):
        r = oa.embeddings.create(input=texts[i:i + chunk], model="text-embedding-ada-002")
        out.extend(d.embedding for d in r.data)
    return out


def _unit(mat):
    arr = np.asarray(mat, dtype=float)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def match_videos(concept_vecs, videos):
    """Vectorized: precompute video vectors ONCE, cosine vs each concept, top-2 over threshold.
    -> {concept_index: [link, ...]}."""
    if not videos:
        return {}
    vid_vecs = _unit(batch_embed([(v.get("title", "") + " " + v.get("description", "")) for v in videos]))
    cvec = _unit(concept_vecs)
    sims = cvec @ vid_vecs.T                                   # (concepts x videos)
    out = {}
    for i in range(sims.shape[0]):
        order = np.argsort(sims[i])[::-1][:2]
        links = [{"type": "video", "source": videos[j].get("platform", "youtube"),
                  "url": videos[j]["url"], "title": videos[j]["title"]}
                 for j in order if sims[i][j] > VIDEO_SIM_THRESHOLD]
        if links:
            out[i] = links
    return out


def build():
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8")).get("concepts", [])
    if not seed:
        print("no seed concepts — nothing to build")
        return 1
    ids = [c["id"] for c in seed]
    print(f"embedding {len(seed)} concepts...", flush=True)
    vecs = batch_embed([(c["label"] + " " + (c.get("summary") or "")) for c in seed])

    coords = pca_coords(vecs)
    nbrs = cosine_neighbors(vecs, ids, k=NEIGHBORS_K)
    # cluster color: prefer the curated cluster; fall back to kmeans label for any blanks
    km = kmeans_clusters(vecs, k=max(2, len(seed) // 30))
    videos = load_videos()
    print(f"matching against {len(videos)} catalog videos...", flush=True)
    vid_links = match_videos(vecs, videos)

    live, pending, dropped = [], [], 0
    for i, (c, (x, y)) in enumerate(zip(seed, coords)):
        cluster = c.get("cluster") or f"cluster-{km[i]}"
        rec = {
            "id": c["id"], "label": c["label"], "aliases": c.get("aliases", []),
            "summary": c.get("summary", ""), "cluster": cluster, "parent": c.get("parent") or cluster,
            "namespaces": c.get("namespaces", []),
            "coords": {"x": x, "y": y}, "neighbors": nbrs[c["id"]],
            "links": list(c.get("links", [])) + vid_links.get(i, []),
            "status": "live" if c.get("disposition") == "auto" else "pending",
        }
        ok, err = validate_concept(rec)
        if not ok:
            print(f"  drop {c['id']}: {err}", file=sys.stderr)
            dropped += 1
            continue
        (live if rec["status"] == "live" else pending).append(rec)

    CONCEPTS_PATH.write_text(json.dumps({"version": "kb-build", "concepts": live}, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    PENDING_PATH.write_text(json.dumps({"version": "kb-build", "concepts": pending}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    matched = sum(1 for i in range(len(seed)) if i in vid_links)
    print(f"\nlive: {len(live)}  pending: {len(pending)}  dropped: {dropped}  video-matched: {matched}")
    print(f"  -> {CONCEPTS_PATH}\n  -> {PENDING_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
