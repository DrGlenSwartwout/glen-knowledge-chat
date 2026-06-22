"""Infoceuticals must resolve to a catalog slug so they are not dropped from
reveals. The matcher pushes the bare code ('EI8'); the catalog name is
'EI8 Microbes-Liver Integrator'. _resolve_remedy_slug must match by code prefix."""
import importlib, sys
from pathlib import Path
import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_bare_infoceutical_code_resolves():
    app = _app()
    assert app._resolve_remedy_slug({"name": "EI8"}) == "ei8-microbes-liver-integrator"
    assert app._resolve_remedy_slug({"name": "MR5"}) == "mr5-harmonize-emotions"
    # case-insensitive bare code
    assert app._resolve_remedy_slug({"name": "mr7"}) == "mr7-calcium-and-terrains"


def test_full_infoceutical_name_still_resolves():
    app = _app()
    assert app._resolve_remedy_slug(
        {"name": "EI8 Microbes-Liver Integrator"}) == "ei8-microbes-liver-integrator"


def test_unknown_remedy_still_drops():
    app = _app()
    assert app._resolve_remedy_slug({"name": "Totally Invented Remedy"}) is None
    assert app._resolve_remedy_slug({"name": "ZZ9"}) is None   # infoceutical-shaped but no SKU
