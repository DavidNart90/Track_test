"""
API endpoints for RAG search functionality.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List
import time

from ...models.search import QueryRequest, SearchRequest, SearchResponse, SearchResult
from ...rag.router import QueryRouter
from ...rag.search import HybridSearchEngine
from ...rag.synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_search_engine():
    """Dependency to get a HybridSearchEngine instance."""
    engine = HybridSearchEngine()
    await engine.initialize()
    return engine

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """
    Perform a search using the RAG engine.
    """
    start_time = time.time()
    try:
        logger.info(f"Performing search for query: {request.query}")
        
        results: List[SearchResult] = await search_engine.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
        )
        
        end_time = time.time()
        search_time_ms = int((end_time - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            search_type=request.search_type,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            filters_applied=request.filters,
            sources_searched=["vector_db", "graph_db"], # Placeholder
        )
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search.")


@router.post("/query", response_model=SearchResponse)
async def intelligent_query(
    request: QueryRequest,
    search_engine: HybridSearchEngine = Depends(get_search_engine),
):
    """
    Perform an intelligent search using a query router.
    """
    start_time = time.time()
    try:
        logger.info(f"Performing intelligent query: {request.query}")

        # Instantiate components
        query_router = QueryRouter()
        response_synthesizer = ResponseSynthesizer()

        # Route query to determine search strategy
        search_strategy = query_router.route_query(request.query)
        logger.info(f"Routed query to strategy: {search_strategy}")

        # Perform search using the chosen strategy
        search_results: List[SearchResult] = await search_engine.search(
            query=request.query,
            search_type=search_strategy,
        )

        # Synthesize the response
        # Extracting 'content' from each SearchResult
        result_content = [result.content for result in search_results]
        synthesized_text = response_synthesizer.synthesize_response(
            query=request.query, search_results=result_content
        )

        # Create a single SearchResult for the synthesized response
        synthesized_result = SearchResult(
            result_id="synthesized-response",
            result_type="document",
            title="Synthesized Answer",
            content=synthesized_text,
            relevance_score=1.0,  # Max relevance for synthesized response
            source="RAG Engine",
        )

        end_time = time.time()
        search_time_ms = int((end_time - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            search_type=search_strategy,
            results=[synthesized_result],
            total_results=1,
            search_time_ms=search_time_ms,
        )
    except Exception as e:
        logger.error(f"Error during intelligent query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the query.")