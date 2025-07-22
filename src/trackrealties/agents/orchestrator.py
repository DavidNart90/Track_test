
"""
Agent orchestration logic.
"""
import logging
from uuid import UUID
from asyncpg import Connection

from ..data.repository import SessionRepository
from .factory import get_agent_class

logger = logging.getLogger(__name__)

async def run_agent_turn(
    session_id: UUID,
    query: str,
    conn: Connection
):
    """
    Orchestrates a single turn of conversation with the appropriate agent.
    """
    # 1. Get session to determine user role
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(session_id)
    if not session:
        raise ValueError("Session not found or has expired.")

    # 2. Get the appropriate agent class from the factory
    try:
        agent_class = get_agent_class(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    # 3. Instantiate the agent
    agent = agent_class()

    # 4. Run the agent
    response = await agent.run(
        message=query,
        session_id=str(session.id),
        user_id=session.user_id,
        user_role=session.user_role.value
    )
    
    return response

async def stream_agent_turn(
    session_id: UUID,
    query: str,
    conn: Connection
):
    """
    Orchestrates a streaming agent response.
    """
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(session_id)
    if not session:
        raise ValueError("Session not found or has expired.")

    try:
        agent_class = get_agent_class(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    agent = agent_class(session_id=str(session.id), user_id=session.user_id)

    async for chunk in agent.stream(query):
        yield chunk
