
"""
API route for health checks.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from ...core.database import test_connection
from ...core.graph import test_graph_connection
from ...models.api import HealthStatus

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    try:
        db_status = await test_connection()
        graph_status = await test_graph_connection()
        
        # Placeholder for other checks
        llm_status = True # Assume ok for now

        overall_status = "healthy"
        if not db_status or not graph_status:
            overall_status = "unhealthy"
        elif not llm_status:
            overall_status = "degraded"

        return HealthStatus(
            status=overall_status,
            database=db_status,
            graph_database=graph_status,
            llm_connection=llm_status,
            version="0.1.0",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")
