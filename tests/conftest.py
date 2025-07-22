import pytest
from src.trackrealties.core.database import db_pool

@pytest.fixture(scope="function", autouse=True)
async def initialize_db_pool():
    """Initialize and close the database pool for each test function."""
    await db_pool.initialize()
    yield
    await db_pool.close()
