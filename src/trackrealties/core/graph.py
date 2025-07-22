"""
Core graph database utilities.
"""
import logging
from typing import Optional

from .config import settings

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = None

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages Neo4j graph database connections and operations."""

    def __init__(self, settings: Optional[dict] = None):
        if settings is None:
            from .config import settings as global_settings
            settings = global_settings
        self.settings = settings
        self._driver: Optional[AsyncDriver] = None

    async def initialize(self) -> None:
        """Initialize the Neo4j driver."""
        if self._driver is not None:
            return
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available. Graph functionality will be limited.")
            return

        if not all([self.settings.NEO4J_URI, self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD]):
            logger.error("Neo4j connection details not configured. Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD)
            )
            logger.info("Neo4j driver initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    async def test_connection(self) -> bool:
        """Test the Neo4j connection."""
        if not self._driver:
            await self.initialize()

        if not self._driver:
            logger.error("Neo4j driver not initialized, cannot test connection.")
            return False

        try:
            async with self._driver.session(database=self.settings.NEO4J_DATABASE) as session:
                logger.info("Testing Neo4j connection...")
                result = await session.run("RETURN 1")
                await result.single()
                logger.info("Neo4j connection test successful.")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}", exc_info=True)
            return False

# Global graph manager instance
graph_manager = GraphManager()

async def test_graph_connection() -> bool:
    """Test the graph database connection."""
    return await graph_manager.test_connection()
