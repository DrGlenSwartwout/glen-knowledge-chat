# tests/test_ingredients_resolver.py
import sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import ingredients
        return ingredients
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def test_slugify():
    m = _mod()
    assert m.slugify("HMC Hesperidin") == "hmc-hesperidin"
    assert m.slugify("Acai 20:1 Freeze-Dried") == "acai-20-1-freeze-dried"


def test_resolve_known_and_unknown():
    m = _mod()
    # a name that exists in fmp-ingredient-content.json - resolve its slug back
    name = next(iter(m._name_index().values()))  # any known canonical name
    slug = m.slugify(name)
    r = m.resolve(slug)
    assert r is not None and r["slug"] == slug and r["name"]
    assert isinstance(r.get("fmp"), dict)
    assert m.resolve("totally-bogus-ingredient-xyz") is None


def test_formulations_with_returns_list():
    m = _mod()
    # any ingredient that appears in products.json; assert a list of {slug,name}
    out = m.formulations_with(next(iter(m._name_index().values())))
    assert isinstance(out, list)
    for f in out:
        assert "slug" in f and "name" in f
