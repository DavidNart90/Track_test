
"""
Agent orchestration logic.
"""
import logging
from uuid import UUID
from asyncpg import Connection

from ..data.repository import SessionRepository
from .factory import create_agent, get_agent_class

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

    # 2. Instantiate the appropriate agent using the factory
    try:
        agent = create_agent(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

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
        agent = create_agent(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    # Pass session context to agent if needed
    agent.dependencies.context_manager.get_or_create_context(str(session.id), session.user_id, session.user_role.value)

    async for chunk in agent.stream(query):
        yield chunk
