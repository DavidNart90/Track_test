"""
RAG Pipeline Integration with Smart Search Router
Replace existing search.py implementation with optimized routing
"""

import logging
import asyncio

from datetime import datetime
from typing import List, Dict, Any, Optional
from typing import List, Dict, Any, Optional, Tuple
from src.trackrealties.core.graph import graph_manager
from src.trackrealties.core.database import db_pool
from src.trackrealties.models.search import SearchResult
from src.trackrealties.analytics.search import search_analytics, SearchAnalytics
from smart_search_implementation import (
    SmartSearchRouter,
    FixedGraphSearch,
    SearchStrategy,
    RealEstateEntityExtractor
)
from src.trackrealties.rag.embedders import DefaultEmbedder
from src.trackrealties.rag.synthesizer import ResponseSynthesizer
from src.trackrealties.validation.hallucination import RealEstateHallucinationDetector
from src.trackrealties.models.agent import ValidationResult

logger = logging.getLogger(__name__)


class OptimizedVectorSearch:
    """
    Optimized vector search with better error handling and performance
    """
    
    def __init__(self):
        self.embedder = DefaultEmbedder()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the vector search client."""
        await db_pool.initialize()
        await self.embedder.initialize()
        self.initialized = True
        logger.info("Optimized vector search initialized")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Search for relevant content using vector similarity with optimizations
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            query_embedding = await self.embedder.embed_query(query)
            query_embedding_str = str(query_embedding)

            # Build filter query more efficiently
            filter_clauses = []
            filter_values = []
            if filters:
                for key, value in filters.items():
                    filter_clauses.append(f"metadata->>'{key}' = ${len(filter_values) + 3}")
                    filter_values.append(value)
            
            filter_query = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""

            async with db_pool.acquire() as conn:
                # Optimized query with UNION for better performance
                results = await conn.fetch(
                    f"""
                    (
                        SELECT
                            'property_' || id::text as result_id,
                            content,
                            'property' as result_type,
                            1 - (embedding <=> $1) AS similarity,
                            metadata->>'address' as title,
                            'Property Listing' as source
                        FROM property_chunks
                        WHERE 1 - (embedding <=> $1) > $2 {filter_query}
                        ORDER BY similarity DESC
                        LIMIT $3
                    )
                    UNION ALL
                    (
                        SELECT
                            'market_' || id::text as result_id,
                            content,
                            'market_data' as result_type,
                            1 - (embedding <=> $1) AS similarity,
                            metadata->>'region_name' as title,
                            'Market Report' as source
                        FROM market_chunks
                        WHERE 1 - (embedding <=> $1) > $2 {filter_query}
                        ORDER BY similarity DESC
                        LIMIT $3
                    )
                    ORDER BY similarity DESC
                    LIMIT $3
                    """,
                    query_embedding_str,
                    threshold,
                    limit,
                    *filter_values,
                )
            
            return [
                SearchResult(
                    result_id=row["result_id"],
                    content=row["content"],
                    result_type=row["result_type"],
                    similarity_score=row["similarity"],
                    title=row.get("title", "No Title"),
                    source=row.get("source", "No Source")
                )
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


class OptimizedGraphSearch(FixedGraphSearch):
    """
    Enhanced graph search with better entity extraction and error handling
    """
    
    def __init__(self):
        self.driver = None
        self.entity_extractor = RealEstateEntityExtractor()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the graph search client."""
        await graph_manager.initialize()
        self.driver = graph_manager._driver
        self.initialized = True
        logger.info("Optimized graph search initialized")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        max_depth: int = 3
    ) -> List[SearchResult]:
        """
        Enhanced graph search with proper error handling and entity matching
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.driver:
            logger.error("Graph driver not available")
            return []
        
        try:
            entities = await self.entity_extractor.extract_entities(query)
            logger.info(f"Graph search entities: {entities}")
            
            results = []
            
            # Search market data by location
            if entities["locations"]:
                for location in entities["locations"]:
                    if "," in location:
                        city, state = [part.strip() for part in location.split(",")]
                        market_results = await self._search_market_data_safe(city, state, limit)
                        results.extend(market_results)
            
            # Search for properties
            if entities["properties"]:
                for prop_ref in entities["properties"]:
                    property_results = await self._search_property_safe(prop_ref)
                    results.extend(property_results)
            
            # Search for agents
            if entities["agents"]:
                for agent_name in entities["agents"]:
                    agent_results = await self._search_agent_safe(agent_name, limit)
                    results.extend(agent_results)
            
            # If specific metrics mentioned, search for those
            if entities["metrics"] and entities["locations"]:
                metric_results = await self._search_metrics_safe(entities["metrics"], entities["locations"], limit)
                results.extend(metric_results)
            
            # Fallback: general search if no specific entities
            if not any(entities.values()):
                general_results = await self._fallback_search(query, limit)
                results.extend(general_results)
            
            # Convert to SearchResult objects
            search_results = []
            for result in results[:limit]:
                search_results.append(SearchResult(
                    result_id=str(result.get("id", "unknown")),
                    content=result.get("content", "No content available"),
                    result_type=result.get("result_type", "graph_fact"),
                    relevance_score=result.get("score", 0.5),
                    title=result.get("title", "No Title"),
                    source=result.get("source", "Graph Database")
                ))
            
            logger.info(f"Graph search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _search_market_data_safe(self, city: str, state: str, limit: int) -> List[Dict]:
        """Safe market data search with error handling"""
        try:
            async with self.driver.session() as session:
                # Updated query to match actual schema
                query = """
                MATCH (l:Location)
                WHERE toLower(l.city) CONTAINS toLower($city) AND toLower(l.state) = toLower($state)
                OPTIONAL MATCH (md:MarketData)
                WHERE md.region_id CONTAINS $city OR md.content CONTAINS $city
                WITH l, md
                LIMIT $limit
                RETURN 
                    CASE 
                        WHEN md IS NOT NULL THEN md.content
                        ELSE l.content
                    END AS content,
                    CASE 
                        WHEN md IS NOT NULL THEN md.market_data_id
                        ELSE l.location_id
                    END AS id,
                    "market_data" AS result_type,
                    $city + ", " + $state + " Market Data" AS title,
                    "Graph Database" AS source,
                    CASE 
                        WHEN md IS NOT NULL THEN 0.9
                        ELSE 0.7
                    END AS score
                """
                
                result = await session.run(query, city=city, state=state, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Market data search failed for {city}, {state}: {e}")
            return []
    
    async def _search_property_safe(self, property_ref: str) -> List[Dict]:
        """Safe property search"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (p:Property)
                WHERE p.property_id = $prop_ref 
                   OR p.address CONTAINS $prop_ref
                   OR p.content CONTAINS $prop_ref
                OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                RETURN 
                    p.content AS content,
                    p.property_id AS id,
                    "property" AS result_type,
                    coalesce(p.address, "Property " + p.property_id) AS title,
                    "Graph Database" AS source,
                    0.95 AS score
                LIMIT 1
                """
                
                result = await session.run(query, prop_ref=property_ref)
                return await result.data()
        except Exception as e:
            logger.error(f"Property search failed for {property_ref}: {e}")
            return []
    
    async def _search_agent_safe(self, agent_name: str, limit: int) -> List[Dict]:
        """Safe agent search"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (a:Agent)
                WHERE toLower(a.name) CONTAINS toLower($agent_name)
                OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                RETURN 
                    a.content AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - Real Estate Agent" AS title,
                    "Graph Database" AS source,
                    0.9 AS score
                LIMIT $limit
                """
                
                result = await session.run(query, agent_name=agent_name, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Agent search failed for {agent_name}: {e}")
            return []
    
    async def _search_metrics_safe(self, metrics: List[str], locations: List[str], limit: int) -> List[Dict]:
        """Search for specific metrics in locations"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (md:MarketData)
                WHERE ANY(metric IN $metrics WHERE md.content CONTAINS metric)
                  AND ANY(location IN $locations WHERE md.content CONTAINS location OR md.region_id CONTAINS location)
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "metric_data" AS result_type,
                    "Market Metrics" AS title,
                    "Graph Database" AS source,
                    0.8 AS score
                ORDER BY md.date DESC
                LIMIT $limit
                """
                
                result = await session.run(query, metrics=metrics, locations=locations, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Metrics search failed: {e}")
            return []
    
    async def _fallback_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback search when no specific entities found"""
        try:
            async with self.driver.session() as session:
                # Simple content-based search across all node types
                fallback_query = """
                MATCH (n)
                WHERE n.content IS NOT NULL 
                  AND toLower(n.content) CONTAINS toLower($query)
                RETURN 
                    n.content AS content,
                    coalesce(
                        n.property_id, 
                        n.market_data_id, 
                        n.location_id, 
                        n.agent_id,
                        elementId(n)
                    ) AS id,
                    labels(n)[0] AS result_type,
                    "Search Result" AS title,
                    "Graph Database" AS source,
                    0.6 AS score
                LIMIT $limit
                """
                
                result = await session.run(fallback_query, query=query, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []


class OptimizedHybridSearch:
    """
    Optimized hybrid search that intelligently combines vector and graph results
    """
    
    def __init__(self):
        self.vector_search = OptimizedVectorSearch()
        self.graph_search = OptimizedGraphSearch()
        self.initialized = False
    
    async def initialize(self):
        """Initialize both search engines"""
        await self.vector_search.initialize()
        await self.graph_search.initialize()
        self.initialized = True
        logger.info("Optimized hybrid search initialized")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> List[SearchResult]:
        """
        Optimized hybrid search with adaptive weighting
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Run searches in parallel
            vector_results, graph_results = await asyncio.gather(
                self.vector_search.search(query, limit=limit, filters=filters),
                self.graph_search.search(query, limit=limit, filters=filters),
                return_exceptions=True
            )
            
            # Handle potential errors
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(graph_results, Exception):
                logger.error(f"Graph search failed: {graph_results}")
                graph_results = []
            
            # Adaptive weighting based on result quality
            vector_weight, graph_weight = self._adapt_weights(
                vector_results, graph_results, vector_weight, graph_weight
            )
            
            # Combine and rank results
            combined_results = self._intelligent_fusion(
                vector_results, graph_results, vector_weight, graph_weight
            )
            
            logger.info(f"Hybrid search combined {len(vector_results)} vector + {len(graph_results)} graph results")
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            return await self.vector_search.search(query, limit=limit, filters=filters)
    
    def _adapt_weights(self, vector_results: List, graph_results: List, 
                      vector_weight: float, graph_weight: float) -> tuple:
        """Adapt weights based on result quality and quantity"""
        vector_count = len(vector_results)
        graph_count = len(graph_results)
        
        # If one search returns no results, boost the other
        if vector_count == 0 and graph_count > 0:
            return 0.0, 1.0
        elif graph_count == 0 and vector_count > 0:
            return 1.0, 0.0
        
        # Adjust weights based on relative result quality
        if vector_count > graph_count * 2:
            vector_weight *= 1.2
            graph_weight *= 0.8
        elif graph_count > vector_count * 2:
            graph_weight *= 1.2
            vector_weight *= 0.8
        
        # Normalize weights
        total_weight = vector_weight + graph_weight
        return vector_weight / total_weight, graph_weight / total_weight
    
    def _intelligent_fusion(self, vector_results: List[SearchResult], 
                          graph_results: List[SearchResult],
                          vector_weight: float, graph_weight: float) -> List[SearchResult]:
        """Intelligently fuse results from vector and graph search"""
        result_map = {}
        
        # Add vector results with weighted scores
        for result in vector_results:
            score = (result.similarity_score or 0.5) * vector_weight
            result_map[result.result_id] = {
                "result": result,
                "score": score,
                "sources": ["vector"]
            }
        
        # Add graph results with weighted scores
        for result in graph_results:
            score = (result.relevance_score or 0.5) * graph_weight
            if result.result_id in result_map:
                # Boost score for results found in both searches
                result_map[result.result_id]["score"] += score * 1.2
                result_map[result.result_id]["sources"].append("graph")
            else:
                result_map[result.result_id] = {
                    "result": result,
                    "score": score,
                    "sources": ["graph"]
                }
        
        # Sort by combined score
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["result"] for item in sorted_results]


# Updated RAG Pipeline Integration
class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with smart search routing
    """
    
    def __init__(self, analytics: SearchAnalytics | None = None):
        self.smart_router = SmartSearchRouter()
        self.vector_search = OptimizedVectorSearch()
        self.graph_search = OptimizedGraphSearch()
        self.hybrid_search = OptimizedHybridSearch()


        self.analytics = analytics or search_analytics

        self.synthesizer = ResponseSynthesizer()
        self.hallucination_detector = RealEstateHallucinationDetector()

        # Set search engines in router
        self.smart_router.vector_search = self.vector_search
        self.smart_router.graph_search = self.graph_search
        self.smart_router.hybrid_search = self.hybrid_search
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        await self.vector_search.initialize()
        await self.graph_search.initialize()
        await self.hybrid_search.initialize()
        self.initialized = True
        logger.info("Enhanced RAG pipeline initialized")
    
    async def search(self, query: str, user_context: Optional[Dict] = None, 
                    limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """
        Main search method with intelligent routing
        """
        if not self.initialized:
            await self.initialize()

        normalized = query.lower().strip()
        greetings = {"hi", "hello", "hey", "hi agent", "hello agent", "hey agent"}
        if normalized in greetings:
            return []

        try:
            start_time = datetime.utcnow()

            # Determine optimal search strategy
            strategy = await self.smart_router.route_search(query, user_context)

            # Execute search with selected strategy
            results = await self.smart_router.execute_search(
                query, strategy, limit=limit, filters=filters
            )

            response_time = (datetime.utcnow() - start_time).total_seconds()

            # Log search performance
            logger.info(
                f"Search completed: {len(results)} results using {strategy} in {response_time:.2f}s"
            )

            # Send analytics
            if self.analytics:
                await self.analytics.log_search_execution(
                    query, strategy, results, response_time
                )

            return results
            
        except Exception as e:
            logger.error(f"Enhanced RAG search failed: {e}")
            # Ultimate fallback
            return await self.vector_search.search(query, limit=limit, filters=filters)

    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Tuple[str, ValidationResult]:
        """Search, synthesize, and validate a response."""
        results = await self.search(
            query,
            user_context=context.get("user_context"),
            limit=context.get("limit", 10),
            filters=context.get("filters"),
        )

        response = await self.synthesizer.synthesize_response(query, results)
        validation = await self.hallucination_detector.validate(
            response,
            {"search_results": [r.content for r in results], "query": query},
        )
        return response, validation


# Usage Example for Integration
"""
# In your existing search.py file, replace the HybridSearchEngine class with:

class HybridSearchEngine(EnhancedRAGPipeline):
    '''
    Drop-in replacement for existing search engine with smart routing
    '''
    pass

# Or update your existing imports:
from .optimized_search import EnhancedRAGPipeline as HybridSearchEngine
"""