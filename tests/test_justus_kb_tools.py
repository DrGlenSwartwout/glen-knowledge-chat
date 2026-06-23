"""Tests for the Justus knowledge-base + web-fetch tools.

Focus is the security-critical surface: a scoped VA (Shaira) must NEVER be able
to reach PHI-bearing client-record namespaces through the KB search tool, and
fetch_url must refuse non-public / internal targets (SSRF guard).
"""
import app
from dashboard.rbac import Actor, VA, OWNER, OPS

# Namespaces that carry client PHI and must be invisible to a scoped VA.
PHI_NAMESPACES = {"consultations", "e4l-protocols", "case-studies", ""}


def test_va_namespaces_exclude_all_phi():
    ns = set(app._kb_namespaces_for(Actor(role=VA)))
    assert ns & PHI_NAMESPACES == set(), f"VA leaked PHI namespaces: {ns & PHI_NAMESPACES}"
    assert "clinical-qa" in ns and "glen-authored-works" in ns


def test_owner_and_ops_get_full_namespace_set():
    owner_ns = set(app._kb_namespaces_for(Actor(role=OWNER)))
    ops_ns = set(app._kb_namespaces_for(Actor(role=OPS)))
    # Owner/Ops see the full RAG set, including client namespaces.
    assert "consultations" in owner_ns
    assert owner_ns == ops_ns


def test_kb_tools_registered_and_routed():
    names = {t["name"] for t in app.KB_TOOLS}
    assert names == {"search_knowledge_base", "fetch_url"}
    assert app._KB_TOOL_NAMES == names


def test_fetch_url_rejects_non_http_scheme():
    assert "http(s)" in app._fetch_url_text("ftp://example.com/x")
    assert "http(s)" in app._fetch_url_text("file:///etc/passwd")


def test_fetch_url_blocks_loopback_and_private():
    # localhost -> 127.0.0.1 (loopback); should be blocked before any request.
    assert "non-public" in app._fetch_url_text("http://localhost:8080/admin")
    assert "non-public" in app._fetch_url_text("http://127.0.0.1/")
    # link-local metadata endpoint (cloud SSRF classic) must be blocked.
    assert "non-public" in app._fetch_url_text("http://169.254.169.254/latest/meta-data/")


def test_search_empty_query_is_guarded():
    assert app._execute_kb_tool("search_knowledge_base", {"query": "  "}, Actor(role=VA)) == "No query provided."
