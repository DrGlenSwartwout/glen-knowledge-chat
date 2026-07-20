import inspect
from dashboard import public_surface as ps


def test_build_demo_view_takes_no_connection():
    """The structural guarantee: no cx parameter means no code path to real data."""
    sig = inspect.signature(ps.build_demo_view)
    assert list(sig.parameters) == []


def test_demo_view_is_marked_as_sample():
    view = ps.build_demo_view()
    assert view["sample"] is True


def test_demo_view_has_expected_shape():
    view = ps.build_demo_view()
    for key in ("greeting", "phase", "practitioner", "layers",
                "findings", "orders", "pricing", "body_map"):
        assert key in view, f"missing {key}"
    assert isinstance(view["layers"], list) and view["layers"]
    assert isinstance(view["findings"], list) and view["findings"]


def test_demo_view_is_a_copy_not_the_fixture():
    """Mutating a returned view must not poison the next caller."""
    first = ps.build_demo_view()
    first["findings"].append({"name": "injected", "note": "x"})
    second = ps.build_demo_view()
    assert all(f["name"] != "injected" for f in second["findings"])


def test_demo_module_does_not_import_sqlite():
    """public_surface builds payloads; it never opens a connection itself."""
    src = inspect.getsource(ps)
    assert "sqlite3.connect" not in src
