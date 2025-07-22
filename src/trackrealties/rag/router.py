# src/trackrealties/rag/router.py
"""
This module defines the QueryRouter class, which is responsible for routing
incoming queries to the appropriate search strategy based on their content.
"""

class QueryRouter:
    """
    A router that determines the best search strategy for a given query.
    """

    def route_query(self, query: str) -> str:
        """
        Routes a query to a search strategy based on keywords.

        The routing logic is as follows:
        - If the query contains "relationship," "connect," or "link," it's routed to 'graph_search'.
        - If the query contains "how many," "what is the average," or "compare," it's routed to 'vector_search'.
        - All other queries are routed to 'hybrid_search'.

        Args:
            query (str): The input query string.

        Returns:
            str: The name of the recommended search strategy.
        """
        lower_query = query.lower()

        graph_keywords = ["relationship", "connect", "link"]
        vector_keywords = ["how many", "what is the average", "compare"]

        if any(keyword in lower_query for keyword in graph_keywords):
            return 'graph_search'
        
        if any(keyword in lower_query for keyword in vector_keywords):
            return 'vector_search'

        return 'hybrid_search'