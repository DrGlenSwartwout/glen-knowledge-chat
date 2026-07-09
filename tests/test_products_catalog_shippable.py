"""catalog() exposes the same shippability answer the pricer uses."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_catalog_marks_services_unshippable():
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["biofield-analysis"]["shippable"] is False
    assert rows["evox-session"]["shippable"] is False


def test_catalog_marks_a_real_product_shippable():
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["neuro-magnesium"]["shippable"] is True


def test_every_catalog_row_has_shippable():
    from dashboard.products import catalog
    rows = catalog(with_ingredients_only=False, include_inactive=True)
    assert rows, "catalog must not be empty"
    assert all(isinstance(r.get("shippable"), bool) for r in rows)
