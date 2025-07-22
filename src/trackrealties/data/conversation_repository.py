"""
Data repository for conversation-related database operations.
"""
import logging
import json
from uuid import UUID
from typing import Optional
from src.trackrealties.models.session import ChatSession

import logging
import json
from uuid import UUID
from typing import Optional
from asyncpg import Connection
from src.trackrealties.models.session import ChatSession

logger = logging.getLogger(__name__)

class ConversationRepository:
    """
    Encapsulates database logic for user sessions and conversation messages.
    """
    def __init__(self, db_connection: Connection):
        self.db = db_connection

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = "buyer",
        session_data: Optional[dict] = None,
    ) -> "ChatSession":
        """
        Get an existing session or create a new one.
        """
        if session_id:
            session_record = await self.db.fetchrow(
                "SELECT * FROM user_sessions WHERE id = $1", session_id
            )
            if session_record:
                return ChatSession(**session_record)

        result = await self.db.fetchrow(
            """
            INSERT INTO user_sessions (user_role, user_id)
            VALUES ($1, $2)
            RETURNING *
            """,
            user_role,
            user_id,
        )
        return ChatSession(**result)


    async def log_message(self, session_id: UUID, role: str, content: str) -> None:
        """
        Logs a new message to the conversation_messages table.

        Args:
            session_id: The ID of the session to which the message belongs.
            role: The role of the message author (e.g., 'user', 'assistant').
            content: The text content of the message.
        """
        await self.db.execute(
            """
            INSERT INTO conversation_messages (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_id,
            role,
            content,
        )
        logger.info(f"Logged message for session {session_id} from role '{role}'.")

    async def get_session_role(self, session_id: UUID) -> Optional[str]:
        """
        Retrieves the user role for a given session.

        Args:
            session_id: The ID of the session.

        Returns:
            The user role as a string, or None if the session is not found.
        """
        role = await self.db.fetchval(
            "SELECT user_role FROM user_sessions WHERE id = $1", session_id
        )
        if role:
            logger.info(f"Retrieved role '{role}' for session {session_id}.")
        else:
            logger.warning(f"No session found with ID: {session_id}")
        return role