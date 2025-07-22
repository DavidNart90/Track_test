# src/trackrealties/rag/intelligent_router.py
"""
An intelligent query router that analyzes queries and determines the best
search strategy.
"""

from typing import Dict, Any

class QueryAnalysis:
    """
    A data class to hold the analysis of a query.
    """
    def __init__(self, primary_strategy: str, secondary_strategy: str = None, confidence: float = 1.0):
        self.primary_strategy = primary_strategy
        self.secondary_strategy = secondary_strategy
        self.confidence = confidence

class IntelligentQueryRouter:
    """
    A more sophisticated query router that can be extended with a trained model.
    """
    async def analyze_query(self, query: str, user_context: Dict[str, Any]) -> QueryAnalysis:
        """
        Analyzes a query and returns a recommended search strategy.
        
        For now, this uses a simple keyword-based approach.
        """
        lower_query = query.lower()

        graph_keywords = ["relationship", "connect", "link", "who", "what is the connection"]
        vector_keywords = ["how many", "what is the average", "compare", "what is the median price"]
        hybrid_keywords = ["invest", "should i", "what about"]

        if any(keyword in lower_query for keyword in graph_keywords):
            return QueryAnalysis(primary_strategy="graph_only")
        
        if any(keyword in lower_query for keyword in vector_keywords):
            return QueryAnalysis(primary_strategy="vector_only")

        if any(keyword in lower_query for keyword in hybrid_keywords):
            return QueryAnalysis(primary_strategy="hybrid")

        return QueryAnalysis(primary_strategy="hybrid")