"""
Core graph database utilities for TrackRealties AI Platform.

This module provides connection management and utilities for Neo4j graph database,
enabling knowledge graph operations for real estate data relationships.
"""

import logging
from typing import Optional, AsyncContextManager
from contextlib import asynccontextmanager

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import Neo4jError, ServiceUnavailable
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = None
    Neo4jError = Exception
    ServiceUnavailable = Exception

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Manages Neo4j graph database connections and operations.
    
    This class provides a centralized way to manage Neo4j connections,
    including initialization, testing, and cleanup of database connections.
    """

    def __init__(self, settings=None):
        """
        Initialize the GraphManager.
        
        Args:
            settings: Settings object containing Neo4j configuration.
                     If None, will use the global settings instance.
        """
        if settings is None:
            from .config import get_settings
            settings = get_settings()
        
        self.settings = settings
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False

    async def initialize(self) -> None:

        """
        Initialize the Neo4j driver.
        
        Raises:
            RuntimeError: If Neo4j driver is not available
            ValueError: If Neo4j configuration is missing
        """
        if self._initialized and self._driver is not None:
            logger.debug("Neo4j driver already initialized.")
        """Initialize the Neo4j driver."""
        if self._driver is not None:
            return
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available. Graph functionality will be limited.")
            return
        
        if not NEO4J_AVAILABLE:
            error_msg = (
                "Neo4j driver not available. Please install it with: "
                "pip install neo4j"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validate configuration
        if not all([
            self.settings.NEO4J_URI,
            self.settings.NEO4J_USER,
            self.settings.NEO4J_PASSWORD
        ]):
            error_msg = (
                "Neo4j connection details not configured. "
                "Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info(f"Initializing Neo4j driver with URI: {self.settings.NEO4J_URI}")
            
            self._driver = AsyncGraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60.0,
                # Trust strategy for production deployments
                # encrypted=True,
                # trust=neo4j.TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
            )
            
            # Verify connectivity
            await self._driver.verify_connectivity()
            
            self._initialized = True
            logger.info("Neo4j driver initialized successfully.")
            
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            self._driver = None
            raise RuntimeError(f"Cannot connect to Neo4j at {self.settings.NEO4J_URI}: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None
            raise

    async def close(self) -> None:
        """Close the Neo4j driver and cleanup resources."""
        if self._driver:
            try:
                await self._driver.close()
                logger.info("Neo4j driver closed successfully.")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
            finally:
                self._driver = None
                self._initialized = False
        else:
            logger.debug("No Neo4j driver to close.")

    async def test_connection(self) -> bool:
        """
        Test the Neo4j connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available, cannot test connection.")
            return False
            
        try:
            # Initialize if not already done
            if not self._driver:
                await self.initialize()

            if not self._driver:
                logger.error("Neo4j driver not initialized after initialization attempt.")
                return False

            # Test query
            async with self._driver.session(
                database=self.settings.NEO4J_DATABASE
            ) as session:
                logger.info("Testing Neo4j connection...")
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                
                if record and record["test"] == 1:
                    logger.info("Neo4j connection test successful.")
                    return True
                else:
                    logger.error("Neo4j test query returned unexpected result.")
                    return False
                    
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable during test: {e}")
            return False
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}", exc_info=True)
            return False

    @asynccontextmanager
    async def get_session(self, database: Optional[str] = None):
        """
        Get a Neo4j session context manager.
        
        Args:
            database: Database name to use. If None, uses default from settings.
            
        Yields:
            Neo4j session object
            
        Raises:
            RuntimeError: If driver is not initialized
        """
        if not self._driver:
            await self.initialize()
            
        if not self._driver:
            raise RuntimeError("Neo4j driver not available")
            
        database = database or self.settings.NEO4J_DATABASE
        
        async with self._driver.session(database=database) as session:
            yield session

    async def execute_query(self, query: str, parameters: Optional[dict] = None) -> list:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of query results
            
        Raises:
            RuntimeError: If driver is not initialized
        """
        async with self.get_session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()

    @property
    def is_connected(self) -> bool:
        """Check if Neo4j driver is initialized and connected."""
        return self._driver is not None and self._initialized

    async def health_check(self) -> dict:
        """
        Perform a comprehensive health check.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "available": NEO4J_AVAILABLE,
            "initialized": self._initialized,
            "connected": False,
            "database": self.settings.NEO4J_DATABASE,
            "error": None
        }
        
        if not NEO4J_AVAILABLE:
            health_status["error"] = "Neo4j driver not installed"
            return health_status
            
        try:
            health_status["connected"] = await self.test_connection()
            
            if health_status["connected"]:
                # Get database info
                async with self.get_session() as session:
                    result = await session.run(
                        "CALL dbms.components() YIELD name, versions, edition"
                    )
                    components = await result.single()
                    if components:
                        health_status["neo4j_version"] = components["versions"][0]
                        health_status["neo4j_edition"] = components["edition"]
                        
        except Exception as e:
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")
            
        return health_status


# Global graph manager instance
graph_manager = GraphManager()


async def test_graph_connection() -> bool:
    """
    Test the graph database connection using the global manager.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    return await graph_manager.test_connection()


async def get_graph_health() -> dict:
    """
    Get graph database health status using the global manager.
    
    Returns:
        Dictionary with health status information
    """
    return await graph_manager.health_check()