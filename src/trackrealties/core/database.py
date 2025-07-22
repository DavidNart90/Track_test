
"""
Database utilities for PostgreSQL connection and operations.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from asyncpg.pool import Pool
from .config import settings

logger = logging.getLogger(__name__)

class DatabasePool:
    """Manages PostgreSQL connection pool."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or settings.DATABASE_URL
        if not self.database_url:
            raise ValueError("DATABASE_URL not set in environment or config")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """Create connection pool."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=5,
                    max_size=20,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection

# Global database pool instance
db_pool = DatabasePool()

async def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("Database connection successful.")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
