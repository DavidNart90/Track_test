"""
API routes for agent interactions.
"""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from asyncpg import Connection

from ...agents.context import ContextManager
from ...models.api import ChatRequest, ChatResponse
from ...agents.factory import get_agent_class
from ...agents.base import AgentDependencies
from ..dependencies import get_db_connection
from ...data.repository import SessionRepository, MessageRepository
from ...models.db import MessageRole

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{role}/chat", response_model=ChatResponse)
async def agent_chat(
    role: str,
    request: ChatRequest,
    conn: Connection = Depends(get_db_connection)
):
    """
    Handles a chat message for a specific agent role.
    """
    try:
        session_repo = SessionRepository(conn)
        msg_repo = MessageRepository(conn)

        # 1. Get or create session
        if request.session_id:
            session = await session_repo.get_session(UUID(request.session_id))
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = await session_repo.create_user_session(role=role, user_id=request.user_id)

        # 2. Log user message
        await msg_repo.add_conversation_message(
            session_id=session.id,
            role=MessageRole.USER,
            content=request.message
        )

        # 3. Get agent class and run
        agent_class = get_agent_class(session.user_role)
        
        # Create a new ContextManager for each request to ensure isolation
        context_manager = ContextManager()
        
        agent = agent_class(deps=AgentDependencies(context_manager=context_manager))
        agent_response = await agent.run(
            message=request.message,
            session_id=str(session.id),
            user_id=session.user_id,
            user_role=session.user_role.value
        )

        # 4. Log assistant message
        assistant_message = await msg_repo.add_conversation_message(
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content=agent_response.content,
            metadata={"tools_used": agent_response.tools_used}
        )

        return ChatResponse(
            message=agent_response.content,
            session_id=str(session.id),
            assistant_message_id=str(assistant_message.id),
            tools_used=agent_response.tools_used
        )

    except NotImplementedError:
        raise HTTPException(status_code=400, detail=f"No agent implemented for role: {role}")
    except Exception as e:
        logger.error(f"Agent chat failed for role {role}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat message.")
