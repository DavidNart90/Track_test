
"""
Embedder implementations for the RAG module.
"""
import logging
from typing import List
from openai import AsyncOpenAI
from ..core.config import settings

logger = logging.getLogger(__name__)

class DefaultEmbedder:
    """Default embedder using OpenAI."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.model = settings.EMBEDDING_MODEL
        self.initialized = False

    async def initialize(self):
        """Initialize the OpenAI client."""
        if not self.initialized:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.initialized = True
            self.logger.info(f"Initialized OpenAI embedder with model: {self.model}")

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not self.initialized:
            await self.initialize()
        
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not self.initialized:
            await self.initialize()

        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [d.embedding for d in response.data]
