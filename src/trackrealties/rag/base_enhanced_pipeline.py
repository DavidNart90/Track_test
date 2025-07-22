# src/trackrealties/rag/base_enhanced_pipeline.py
"""
Base classes for the enhanced RAG pipeline.
"""

class EnhancedRAGPipeline:
    """
    A base class for enhanced RAG pipelines.
    """
    def __init__(self):
        pass

    async def initialize(self):
        """
        Initializes the pipeline and its components.
        """
        pass

    async def process_query(self, query: str, user_context: dict) -> dict:
        """
        Processes a query and returns a response.
        """
        raise NotImplementedError