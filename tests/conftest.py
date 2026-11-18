from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fake_api  # noqa: E402

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


def pytest_collection_modifyitems(config, items):
    live_selected = "live" in (config.getoption("-m") or "")
    skip_live = pytest.mark.skip(reason="live test: run with -m live and WBXSANDBOX_API_KEY")
    for item in items:
        if "live" in item.keywords and not (live_selected and os.environ.get("WBXSANDBOX_API_KEY")):
            item.add_marker(skip_live)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("WBXSANDBOX_API_KEY", "test-key-not-real")
    monkeypatch.delenv("WBXSANDBOX_URL", raising=False)
    return fake_api.install(monkeypatch)


@pytest.fixture
def ws_root():
    """Workspace root on the local machine for the fake API (commands run locally)."""
    root = Path("/tmp") / f"wbx-test-ws-{uuid.uuid4().hex[:10]}"
    yield str(root)
    shutil.rmtree(root, ignore_errors=True)
