"""
Smart Search Router Implementation for TrackRealties RAG System
Fixes current graph schema issues and implements intelligent search routing
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy options"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


class QueryIntent(Enum):
    """User query intent types"""
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RELATIONSHIP_QUERY = "relationship_query"
    INVESTMENT_ANALYSIS = "investment_analysis"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    PROPERTY_SEARCH = "property_search"


class RealEstateEntityExtractor:
    """
    Enhanced entity extraction specifically for real estate queries
    """
    
    def __init__(self):
        self.location_patterns = [
            r'([A-Z][a-zA-Z\s]+),\s*([A-Z]{2})',  # Austin, TX
            r'([A-Z][a-zA-Z\s]+)\s+([A-Z]{2})\b',   # Austin TX
            r'([A-Z][a-zA-Z\s]+)\s+(metro|area|county)',  # Austin metro
            r'([A-Z][a-zA-Z\s]+)\s+market',  # Austin market
        ]
        
        self.property_patterns = [
            r'(\d+\s+[A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Way|Drive|Dr|Lane|Ln|Boulevard|Blvd))',
            r'property\s+(?:id[:\s]*)?([a-zA-Z0-9\-_]+)',
            r'listing\s+(?:id[:\s]*)?([a-zA-Z0-9\-_]+)',
        ]
        
        self.metric_patterns = [
            r'(median\s+(?:sale\s+)?price)',
            r'(inventory\s+count)',
            r'(days\s+on\s+market)',
            r'(months?\s+(?:of\s+)?supply)',
            r'(price\s+per\s+sq(?:uare)?(?:\s+)?f(?:ee)?t)',
            r'(sales?\s+volume)',
            r'(new\s+listings)',
            r'(roi|return\s+on\s+investment)',
            r'(cash\s+flow)',
            r'(cap\s+rate)',
            r'(appreciation)',
        ]
        
        self.agent_patterns = [
            r'agent\s+([A-Z][a-zA-Z\s]+)',
            r'realtor\s+([A-Z][a-zA-Z\s]+)',
            r'broker\s+([A-Z][a-zA-Z\s]+)',
        ]
    
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract real estate entities from query text"""
        entities = {
            "locations": self._extract_locations(query),
            "properties": self._extract_properties(query),
            "metrics": self._extract_metrics(query),
            "agents": self._extract_agents(query)
        }
        
        # Normalize location entities
        entities["locations"] = self._normalize_locations(entities["locations"])
        
        logger.debug(f"Extracted entities: {entities}")
        return entities
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location names from query"""
        locations = []
        for pattern in self.location_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    city, state = match.groups()
                    locations.append(f"{city.strip()}, {state.upper()}")
                else:
                    locations.append(match.group(1).strip())
        return list(set(locations))
    
    def _extract_properties(self, query: str) -> List[str]:
        """Extract property references from query"""
        properties = []
        for pattern in self.property_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            properties.extend([match.group(1).strip() for match in matches])
        return list(set(properties))
    
    def _extract_metrics(self, query: str) -> List[str]:
        """Extract metric references from query"""
        metrics = []
        for pattern in self.metric_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            metrics.extend([match.group(1).strip().lower() for match in matches])
        return list(set(metrics))
    
    def _extract_agents(self, query: str) -> List[str]:
        """Extract agent names from query"""
        agents = []
        for pattern in self.agent_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            agents.extend([match.group(1).strip() for match in matches])
        return list(set(agents))
    
    def _normalize_locations(self, locations: List[str]) -> List[str]:
        """Normalize location formats"""
        normalized = []
        for location in locations:
            # Handle various formats: "Austin TX" -> "Austin, TX"
            if ',' not in location and len(location.split()) >= 2:
                parts = location.split()
                if len(parts[-1]) == 2 and parts[-1].isupper():
                    state = parts[-1]
                    city = ' '.join(parts[:-1])
                    normalized.append(f"{city}, {state}")
                else:
                    normalized.append(location)
            else:
                normalized.append(location)
        return normalized


class QueryIntentClassifier:
    """
    Classifies user intent to determine optimal search strategy
    """
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL_LOOKUP: [
                r'what\s+is\s+(?:the\s+)?(median|average|current|latest)',
                r'how\s+much\s+(?:is|are|does|do)',
                r'(?:what\s+)?(?:price|cost|value)\s+(?:of|for|in)',
                r'(?:current|latest)\s+(?:price|inventory|count)',
                r'tell\s+me\s+(?:the\s+)?(?:median|average|current)',
            ],
            
            QueryIntent.COMPARATIVE_ANALYSIS: [
                r'compare\s+\w+\s+(?:to|vs|versus|against|with)',
                r'(?:difference|differences)\s+between',
                r'(?:better|best)\s+(?:investment|buy|choice|option)',
                r'(?:pros\s+and\s+cons|advantages\s+and\s+disadvantages)',
                r'which\s+(?:is\s+)?(?:better|best|preferred)',
            ],
            
            QueryIntent.RELATIONSHIP_QUERY: [
                r'who\s+(?:is|are)\s+(?:the\s+)?(?:agent|broker|realtor)',
                r'which\s+(?:agent|office|company|brokerage)',
                r'(?:agent|broker|realtor)\s+(?:for|of)\s+(?:this|that)',
                r'(?:listing|listed)\s+(?:by|with)',
                r'(?:contact|phone|email)\s+(?:for|of)',
            ],
            
            QueryIntent.INVESTMENT_ANALYSIS: [
                r'should\s+I\s+(?:buy|invest|purchase)',
                r'(?:roi|return|cash\s+flow|investment\s+potential)',
                r'(?:profitable|worth\s+it|good\s+(?:deal|investment))',
                r'(?:rental|investment)\s+(?:property|properties)',
                r'(?:analyze|evaluation|analysis)\s+(?:investment|property)',
            ],
            
            QueryIntent.PROPERTY_SEARCH: [
                r'(?:find|show|search)\s+(?:me\s+)?(?:properties|homes|houses)',
                r'(?:looking\s+for|want\s+to\s+find)\s+(?:a\s+)?(?:property|home|house)',
                r'(?:properties|homes|houses)\s+(?:in|near|around)',
                r'(?:3|4|5)\s+bed(?:room)?(?:s)?',
                r'under\s+\$?\d+[kK]?',
            ],
            
            QueryIntent.SEMANTIC_ANALYSIS: [
                r'(?:tell\s+me\s+about|describe|explain)',
                r'(?:overview|summary|analysis)\s+(?:of|for)',
                r'(?:market\s+)?(?:trends|conditions|outlook)',
                r'(?:insights|recommendations|advice)',
                r'(?:what\s+do\s+you\s+think|opinion)',
            ]
        }
    
    async def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a user query"""
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Return the intent with the highest score, or SEMANTIC_ANALYSIS as default
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return QueryIntent.SEMANTIC_ANALYSIS


class QueryClassifier:
    """
    Comprehensive query classification for search strategy determination
    """
    
    def __init__(self):
        self.vector_indicators = [
            "similar", "compare", "like", "analysis", "overview", 
            "recommendations", "insights", "trends", "market conditions",
            "describe", "explain", "tell me about", "summary"
        ]
        
        self.graph_indicators = [
            "relationship", "connected", "related", "history", 
            "agent", "office", "who", "when", "which agent",
            "listing", "contact", "phone", "email"
        ]
        
        self.hybrid_indicators = [
            "best", "recommend", "should I", "investment", 
            "roi", "cash flow", "market analysis", "property analysis",
            "buy", "invest", "purchase", "worth it"
        ]
        
        self.factual_indicators = [
            "what is", "how much", "price", "median", "average",
            "inventory", "count", "number", "specific", "exact",
            "current", "latest"
        ]
    
    async def classify_query_type(self, query: str) -> str:
        """Classify query into general categories"""
        query_lower = query.lower()
        
        scores = {
            "factual": sum(1 for indicator in self.factual_indicators if indicator in query_lower),
            "vector": sum(1 for indicator in self.vector_indicators if indicator in query_lower),
            "graph": sum(1 for indicator in self.graph_indicators if indicator in query_lower),
            "hybrid": sum(1 for indicator in self.hybrid_indicators if indicator in query_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "hybrid"


class SmartSearchRouter:
    """
    Intelligent search routing based on query analysis and entity extraction
    """
    
    def __init__(self, vector_search=None, graph_search=None, hybrid_search=None):
        self.entity_extractor = RealEstateEntityExtractor()
        self.intent_classifier = QueryIntentClassifier()
        self.query_classifier = QueryClassifier()
        
        self.vector_search = vector_search
        self.graph_search = graph_search
        self.hybrid_search = hybrid_search
        
    async def route_search(self, query: str, user_context: Optional[Dict] = None) -> SearchStrategy:
        """
        Determine optimal search strategy for the given query
        """
        # Extract entities and classify intent
        entities = await self.entity_extractor.extract_entities(query)
        intent = await self.intent_classifier.classify_intent(query)
        query_type = await self.query_classifier.classify_query_type(query)
        
        logger.info(f"Query analysis - Intent: {intent}, Type: {query_type}, Entities: {entities}")
        
        # Decision logic based on intent and entities
        strategy = self._determine_strategy(intent, query_type, entities, user_context)
        
        logger.info(f"Selected search strategy: {strategy}")
        return strategy
    
    def _determine_strategy(self, intent: QueryIntent, query_type: str, 
                          entities: Dict[str, List], user_context: Optional[Dict] = None) -> SearchStrategy:
        """
        Core decision logic for search strategy selection
        """
        # Rule 1: Factual lookups with specific locations/metrics -> Graph Only
        if intent == QueryIntent.FACTUAL_LOOKUP and (entities["locations"] or entities["metrics"]):
            return SearchStrategy.GRAPH_ONLY
        
        # Rule 2: Relationship queries (agents, offices, etc.) -> Graph Only
        if intent == QueryIntent.RELATIONSHIP_QUERY:
            return SearchStrategy.GRAPH_ONLY
        
        # Rule 3: Investment analysis or comparative analysis -> Hybrid
        if intent in [QueryIntent.INVESTMENT_ANALYSIS, QueryIntent.COMPARATIVE_ANALYSIS]:
            return SearchStrategy.HYBRID
        
        # Rule 4: Property search with specific criteria -> Hybrid
        if intent == QueryIntent.PROPERTY_SEARCH and entities["locations"]:
            return SearchStrategy.HYBRID
        
        # Rule 5: Pure semantic analysis without specific entities -> Vector Only
        if intent == QueryIntent.SEMANTIC_ANALYSIS and not any(entities.values()):
            return SearchStrategy.VECTOR_ONLY
        
        # Rule 6: Queries with locations but semantic in nature -> Hybrid
        if entities["locations"] and query_type in ["vector", "hybrid"]:
            return SearchStrategy.HYBRID
        
        # Rule 7: Specific property or agent queries -> Graph Only
        if entities["properties"] or entities["agents"]:
            return SearchStrategy.GRAPH_ONLY
        
        # Default: Use hybrid for complex or ambiguous queries
        return SearchStrategy.HYBRID
    
    async def execute_search(self, query: str, strategy: SearchStrategy, 
                           limit: int = 10, filters: Optional[Dict] = None) -> List[Any]:
        """
        Execute the determined search strategy
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            if strategy == SearchStrategy.VECTOR_ONLY:
                results = await self.vector_search.search(query, limit=limit, filters=filters)
            elif strategy == SearchStrategy.GRAPH_ONLY:
                results = await self.graph_search.search(query, limit=limit, filters=filters)
            elif strategy == SearchStrategy.HYBRID:
                results = await self.hybrid_search.search(query, limit=limit, filters=filters)
            else:
                # Fallback to hybrid
                results = await self.hybrid_search.search(query, limit=limit, filters=filters)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Search executed in {execution_time:.2f}s with {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            # Fallback to vector search if others fail
            if strategy != SearchStrategy.VECTOR_ONLY:
                logger.info("Falling back to vector search")
                return await self.vector_search.search(query, limit=limit, filters=filters)
            else:
                return []


# Fixed Graph Queries for Real Estate Domain
FIXED_REAL_ESTATE_QUERIES = {
    "market_data_by_location": """
        MATCH (l:Location)
        WHERE l.city CONTAINS $city AND l.state = $state
        OPTIONAL MATCH (r:Region)-[:HAS_MARKET_DATA]->(md:MarketData)
        WHERE r.name CONTAINS $city OR r.name CONTAINS $state
        RETURN 
            coalesce(md.content, l.content, "No market data available") AS content,
            coalesce(md.market_data_id, l.location_id) AS id,
            "market_data" AS result_type,
            coalesce(md.median_price, 0) AS median_price,
            coalesce(md.inventory_count, 0) AS inventory_count,
            coalesce(md.days_on_market, 0) AS days_on_market,
            1.0 AS score
        ORDER BY md.date DESC
        LIMIT $limit
    """,
    
    "property_by_location": """
        MATCH (l:Location)<-[:LOCATED_IN]-(p:Property)
        WHERE l.city CONTAINS $city AND l.state = $state
        OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
        RETURN 
            p.content AS content,
            p.property_id AS id,
            "property" AS result_type,
            p.address AS title,
            coalesce(a.name, "No agent") AS agent_name,
            p.price AS price,
            0.8 AS score
        ORDER BY p.price DESC
        LIMIT $limit
    """,
    
    "agent_by_name": """
        MATCH (a:Agent)
        WHERE a.name CONTAINS $agent_name
        OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
        RETURN 
            a.content AS content,
            a.agent_id AS id,
            "agent" AS result_type,
            a.name AS title,
            a.phone AS phone,
            a.email AS email,
            COUNT(p) AS property_count,
            0.9 AS score
        ORDER BY property_count DESC
        LIMIT $limit
    """,
    
    "property_by_id": """
        MATCH (p:Property {property_id: $property_id})
        OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
        OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
        RETURN 
            p.content AS content,
            p.property_id AS id,
            "property_detail" AS result_type,
            p.address AS title,
            a.name AS agent_name,
            a.phone AS agent_phone,
            l.city AS city,
            l.state AS state,
            1.0 AS score
        LIMIT 1
    """,
    
    "market_metrics": """
        MATCH (md:MarketData)
        WHERE md.region_id CONTAINS $location_query OR md.content CONTAINS $location_query
        RETURN 
            md.content AS content,
            md.market_data_id AS id,
            "market_metrics" AS result_type,
            md.region_id AS title,
            md.median_price AS median_price,
            md.inventory_count AS inventory_count,
            md.days_on_market AS days_on_market,
            md.months_supply AS months_supply,
            0.95 AS score
        ORDER BY md.date DESC
        LIMIT $limit
    """
}


class FixedGraphSearch:
    """
    Fixed graph search implementation using correct schema
    """
    
    def __init__(self, driver):
        self.driver = driver
        self.entity_extractor = RealEstateEntityExtractor()
    
    async def search(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Any]:
        """
        Execute graph search with fixed queries
        """
        entities = await self.entity_extractor.extract_entities(query)
        
        results = []
        
        # Search for market data if locations are mentioned
        if entities["locations"]:
            for location in entities["locations"]:
                if "," in location:
                    city, state = [part.strip() for part in location.split(",")]
                    market_results = await self._search_market_data(city, state, limit)
                    results.extend(market_results)
        
        # Search for specific properties
        if entities["properties"]:
            for prop_id in entities["properties"]:
                property_results = await self._search_property_by_id(prop_id)
                results.extend(property_results)
        
        # Search for agents
        if entities["agents"]:
            for agent_name in entities["agents"]:
                agent_results = await self._search_agent_by_name(agent_name, limit)
                results.extend(agent_results)
        
        # If no specific entities, search for general market metrics
        if not any(entities.values()):
            general_results = await self._search_general_metrics(query, limit)
            results.extend(general_results)
        
        return results[:limit]
    
    async def _search_market_data(self, city: str, state: str, limit: int) -> List[Any]:
        """Search for market data by location"""
        async with self.driver.session() as session:
            result = await session.run(
                FIXED_REAL_ESTATE_QUERIES["market_data_by_location"],
                city=city, state=state, limit=limit
            )
            return await result.data()
    
    async def _search_property_by_id(self, property_id: str) -> List[Any]:
        """Search for specific property"""
        async with self.driver.session() as session:
            result = await session.run(
                FIXED_REAL_ESTATE_QUERIES["property_by_id"],
                property_id=property_id
            )
            return await result.data()
    
    async def _search_agent_by_name(self, agent_name: str, limit: int) -> List[Any]:
        """Search for agent by name"""
        async with self.driver.session() as session:
            result = await session.run(
                FIXED_REAL_ESTATE_QUERIES["agent_by_name"],
                agent_name=agent_name, limit=limit
            )
            return await result.data()
    
    async def _search_general_metrics(self, query: str, limit: int) -> List[Any]:
        """Search for general market metrics"""
        async with self.driver.session() as session:
            result = await session.run(
                FIXED_REAL_ESTATE_QUERIES["market_metrics"],
                location_query=query, limit=limit
            )
            return await result.data()


