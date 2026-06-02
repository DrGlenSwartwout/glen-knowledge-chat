import numpy as np
import atlas_seed


def test_pca_coords_in_unit_square_and_deterministic():
    vecs = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], dtype=float)
    a = atlas_seed.pca_coords(vecs)
    b = atlas_seed.pca_coords(vecs)
    assert a == b  # deterministic
    assert len(a) == 4
    for x, y in a:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_cosine_neighbors_returns_closest():
    vecs = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=float)
    ids = ["a", "b", "c"]
    nbrs = atlas_seed.cosine_neighbors(vecs, ids, k=1)
    assert nbrs["a"] == ["b"]   # a closest to b, not c
    assert nbrs["c"] != ["a"] or nbrs["c"] == ["b"]


def test_kmeans_clusters_count_matches_k():
    vecs = np.array([[0, 0], [0.1, 0], [9, 9], [9.1, 9]], dtype=float)
    labels = atlas_seed.kmeans_clusters(vecs, k=2)
    assert len(labels) == 4
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]
