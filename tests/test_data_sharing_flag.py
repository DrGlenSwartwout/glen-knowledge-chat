import os
import pytest
import app


def test_flag_off_by_default(monkeypatch):
    """DATA_SHARING_REWARD_ENABLED defaults to False when not set."""
    monkeypatch.delenv("DATA_SHARING_REWARD_ENABLED", raising=False)
    assert app._data_sharing_enabled() is False


def test_flag_on(monkeypatch):
    """DATA_SHARING_REWARD_ENABLED returns True when set to 'true'."""
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "true")
    assert app._data_sharing_enabled() is True
