import pytest
from src.trackrealties.core.database import db_pool
from unittest.mock import AsyncMock

@pytest.fixture(scope="function", autouse=True)
async def initialize_db_pool(monkeypatch):
    """Stub database pool initialization for tests."""
    monkeypatch.setattr(db_pool, "initialize", AsyncMock())
    monkeypatch.setattr(db_pool, "close", AsyncMock())
    def _dummy_acquire():
        class DummyConn:
            async def __aenter__(self):
                return None
            async def __aexit__(self, exc_type, exc, tb):
                pass
        return DummyConn()

    monkeypatch.setattr(db_pool, "acquire", _dummy_acquire)
    await db_pool.initialize()
    yield
    await db_pool.close()
