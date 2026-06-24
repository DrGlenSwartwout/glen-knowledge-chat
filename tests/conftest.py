"""Shared pytest fixtures for the deploy-chat test suite."""

import pytest
import os
import tempfile
from pathlib import Path


# Create a temp directory for the test database that persists for the session
_test_db_dir = Path(tempfile.mkdtemp(prefix="pytest_deploy_chat_"))
_test_db_path = _test_db_dir / "chat_log.db"


def pytest_configure(config):
    """Set required env vars before any module is imported."""
    os.environ.setdefault("PINECONE_API_KEY", "test-key-for-tests")
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")
    os.environ["DATA_DIR"] = str(_test_db_dir)
    # Ensure the database file exists (sqlite will create schema on first access)
    _test_db_path.touch()


@pytest.fixture
def tmp_db(tmp_path):
    """Path to an empty sqlite db file inside tmp_path. The file does not
    yet exist on disk; the test seeds whatever schema it needs."""
    return str(tmp_path / "chat_log.db")
