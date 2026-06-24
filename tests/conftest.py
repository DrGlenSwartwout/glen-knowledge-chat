"""Shared pytest fixtures for the deploy-chat test suite."""

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Path to an empty sqlite db file inside tmp_path. The file does not
    yet exist on disk; the test seeds whatever schema it needs."""
    return str(tmp_path / "chat_log.db")
