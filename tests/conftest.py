import os
os.environ.setdefault("DATABASE_URL", "postgresql://test@test/testdb")
from contextlib import asynccontextmanager

import pytest
from src.trackrealties.core.database import db_pool

from src.trackrealties.agents.context import ContextManager
from src.trackrealties.rag.enhanced_rag_pipeline import TrackRealitiesEnhancedRAG
from src.trackrealties.validation.hallucination import RealEstateHallucinationDetector

@pytest.fixture(scope="function", autouse=True)
async def initialize_db_pool(monkeypatch):
    """Mock database pool initialization for tests."""
    async def _noop():
        return None

    @asynccontextmanager
    async def _fake_acquire():
        class Dummy:
            async def fetch(self, *args, **kwargs):
                return []

            async def fetchrow(self, *args, **kwargs):
                return None

            async def execute(self, *args, **kwargs):
                return None

        yield Dummy()

    monkeypatch.setattr(db_pool, "initialize", _noop)
    monkeypatch.setattr(db_pool, "close", _noop)
    monkeypatch.setattr(db_pool, "acquire", _fake_acquire)

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


@pytest.fixture()
def context_manager():
    """Provides a fresh ContextManager instance."""
    return ContextManager()


@pytest.fixture()
def enhanced_rag_pipeline():
    """Provides the enhanced RAG pipeline."""
    return TrackRealitiesEnhancedRAG()


@pytest.fixture()
def real_estate_hallucination_detector():
    """Provides the real estate hallucination detector."""
    return RealEstateHallucinationDetector()
