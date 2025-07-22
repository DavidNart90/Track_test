"""
API route for chat interactions.
"""
import json
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from asyncpg import Connection

from ...models.api import ChatRequest, ChatResponse, ToolCall, StreamDelta
from ...models.db import MessageRole
from ...data.repository import MessageRepository
from ...agents.orchestrator import run_agent_turn, stream_agent_turn
from ..dependencies import get_db_connection
from ...core.database import db_pool

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    conn: Connection = Depends(get_db_connection)
):
    """Handles a non-streaming chat message."""
    try:
        session_id = UUID(request.session_id)
        msg_repo = MessageRepository(conn)

        # 1. Log user message
        await msg_repo.add_conversation_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )

        # 2. Run agent orchestrator
        agent_response = await run_agent_turn(session_id, request.message, conn)

        # 3. Log assistant message
        assistant_message = await msg_repo.add_conversation_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=agent_response.content,
            metadata={"tools_used": agent_response.tools_used}
        )

        return ChatResponse(
            message=agent_response.content,
            session_id=str(session_id),
            assistant_message_id=str(assistant_message.id),
            tools_used=[ToolCall(**tool) for tool in agent_response.tools_used]
        )

    except ValueError as e:
        logger.warning(f"Value error in chat endpoint: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat message.")

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
):
    """Handles a streaming chat message."""
    
    async def generate_stream():
        session_id = None
        conn = None
        try:
            session_id = UUID(request.session_id)
            conn = await db_pool.pool.acquire()
            msg_repo = MessageRepository(conn)

            # 1. Yield session ID
            yield f"data: {json.dumps(StreamDelta(type='session', content={'session_id': str(session_id)}).dict())}\n\n"

            # 2. Log user message
            await msg_repo.add_conversation_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=request.message
            )

            # 3. Stream agent response
            full_response = ""
            async for chunk in stream_agent_turn(session_id, request.message, conn):
                full_response += chunk
                yield f"data: {json.dumps(StreamDelta(type='text', content=chunk).dict())}\n\n"

            # 4. Log full assistant message
            await msg_repo.add_conversation_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                metadata={"streamed": True}
            )
            
            yield f"data: {json.dumps(StreamDelta(type='end', content=None).dict())}\n\n"

        except ValueError as e:
            logger.warning(f"Value error in chat stream: {e}")
            error_content = {"error": str(e)}
            yield f"data: {json.dumps(StreamDelta(type='error', content=error_content).dict())}\n\n"
        except Exception as e:
            logger.error(f"Chat stream failed: {e}", exc_info=True)
            error_content = {"error": "An internal error occurred."}
            yield f"data: {json.dumps(StreamDelta(type='error', content=error_content).dict())}\n\n"
        finally:
            if conn:
                await db_pool.pool.release(conn)

    return StreamingResponse(generate_stream(), media_type="text/event-stream")