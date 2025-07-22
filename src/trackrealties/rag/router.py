# src/trackrealties/rag/router.py
"""Intelligent routing utilities for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..models.search import SearchResult

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Enumeration of search strategies."""

    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


class QueryIntent(str, Enum):
    """User intent types extracted from a query."""

    FACTUAL_LOOKUP = "factual_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RELATIONSHIP_QUERY = "relationship_query"
    INVESTMENT_ANALYSIS = "investment_analysis"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    PROPERTY_SEARCH = "property_search"


class RealEstateEntityExtractor:
    """Regex based entity extractor for real estate queries."""

    def __init__(self) -> None:
        self.location_patterns = [
            r"([A-Z][a-zA-Z\s]+),\s*([A-Z]{2})",
            r"([A-Z][a-zA-Z\s]+)\s+([A-Z]{2})\b",
            r"([A-Z][a-zA-Z\s]+)\s+(metro|area|county)",
            r"([A-Z][a-zA-Z\s]+)\s+market",
        ]
        self.property_patterns = [
            r"(\d+\s+[A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Way|Drive|Dr|Lane|Ln|Boulevard|Blvd))",
            r"property\s+(?:id[:\s]*)?([a-zA-Z0-9\-_]+)",
            r"listing\s+(?:id[:\s]*)?([a-zA-Z0-9\-_]+)",
        ]
        self.metric_patterns = [
            r"(median\s+(?:sale\s+)?price)",
            r"(inventory\s+count)",
            r"(days\s+on\s+market)",
            r"(months?\s+(?:of\s+)?supply)",
            r"(price\s+per\s+sq(?:uare)?\s?f(?:ee)?t)",
            r"(sales?\s+volume)",
            r"(new\s+listings)",
            r"(roi|return\s+on\s+investment)",
            r"(cash\s+flow)",
            r"(cap\s+rate)",
            r"(appreciation)",
        ]
        self.agent_patterns = [
            r"agent\s+([A-Z][a-zA-Z\s]+)",
            r"realtor\s+([A-Z][a-zA-Z\s]+)",
            r"broker\s+([A-Z][a-zA-Z\s]+)",
        ]

    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {
            "locations": self._extract_locations(query),
            "properties": self._extract_properties(query),
            "metrics": self._extract_metrics(query),
            "agents": self._extract_agents(query),
        }
        entities["locations"] = self._normalize_locations(entities["locations"])
        logger.debug("Extracted entities: %s", entities)
        return entities

    def _extract_locations(self, query: str) -> List[str]:
        locations: List[str] = []
        for pattern in self.location_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                if len(match.groups()) == 2:
                    city, state = match.groups()
                    locations.append(f"{city.strip()}, {state.upper()}")
                else:
                    locations.append(match.group(1).strip())
        return list(set(locations))

    def _extract_properties(self, query: str) -> List[str]:
        properties: List[str] = []
        for pattern in self.property_patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                properties.append(m.group(1).strip())
        return list(set(properties))

    def _extract_metrics(self, query: str) -> List[str]:
        metrics: List[str] = []
        for pattern in self.metric_patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                metrics.append(m.group(1).strip().lower())
        return list(set(metrics))

    def _extract_agents(self, query: str) -> List[str]:
        agents: List[str] = []
        for pattern in self.agent_patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                agents.append(m.group(1).strip())
        return list(set(agents))

    def _normalize_locations(self, locations: List[str]) -> List[str]:
        normalized: List[str] = []
        for loc in locations:
            if "," not in loc and len(loc.split()) >= 2:
                parts = loc.split()
                if len(parts[-1]) == 2 and parts[-1].isupper():
                    city = " ".join(parts[:-1])
                    state = parts[-1]
                    normalized.append(f"{city}, {state}")
                else:
                    normalized.append(loc)
            else:
                normalized.append(loc)
        return normalized


class QueryIntentClassifier:
    """Simple regex based intent classifier."""

    def __init__(self) -> None:
        self.intent_patterns: Dict[QueryIntent, List[str]] = {
            QueryIntent.FACTUAL_LOOKUP: [
                r"what\s+is\s+(?:the\s+)?(median|average|current|latest)",
                r"how\s+much\s+(?:is|are|does|do)",
                r"(?:what\s+)?(?:price|cost|value)\s+(?:of|for|in)",
                r"(?:current|latest)\s+(?:price|inventory|count)",
                r"tell\s+me\s+(?:the\s+)?(?:median|average|current)",
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                r"compare\s+\w+\s+(?:to|vs|versus|against|with)",
                r"(?:difference|differences)\s+between",
                r"(?:better|best)\s+(?:investment|buy|choice|option)",
                r"(?:pros\s+and\s+cons|advantages\s+and\s+disadvantages)",
                r"which\s+(?:is\s+)?(?:better|best|preferred)",
            ],
            QueryIntent.RELATIONSHIP_QUERY: [
                r"who\s+(?:is|are)\s+(?:the\s+)?(?:agent|broker|realtor)",
                r"which\s+(?:agent|office|company|brokerage)",
                r"(?:agent|broker|realtor)\s+(?:for|of)\s+(?:this|that)",
                r"(?:listing|listed)\s+(?:by|with)",
                r"(?:contact|phone|email)\s+(?:for|of)",
            ],
            QueryIntent.INVESTMENT_ANALYSIS: [
                r"should\s+i\s+(?:buy|invest|purchase)",
                r"(?:roi|return|cash\s+flow|investment\s+potential)",
                r"(?:profitable|worth\s+it|good\s+(?:deal|investment))",
                r"(?:rental|investment)\s+(?:property|properties)",
                r"(?:analyze|evaluation|analysis)\s+(?:investment|property)",
            ],
            QueryIntent.PROPERTY_SEARCH: [
                r"(?:find|show|search)\s+(?:me\s+)?(?:properties|homes|houses)",
                r"(?:looking\s+for|want\s+to\s+find)\s+(?:a\s+)?(?:property|home|house)",
                r"(?:properties|homes|houses)\s+(?:in|near|around)",
                r"(?:3|4|5)\s+bed(?:room)?s?",
                r"under\s+\$?\d+[kK]?",
            ],
            QueryIntent.SEMANTIC_ANALYSIS: [
                r"(?:tell\s+me\s+about|describe|explain)",
                r"(?:overview|summary|analysis)\s+(?:of|for)",
                r"(?:market\s+)?(?:trends|conditions|outlook)",
                r"(?:insights|recommendations|advice)",
                r"(?:what\s+do\s+you\s+think|opinion)",
            ],
        }

    async def classify_intent(self, query: str) -> QueryIntent:
        query_lower = query.lower()
        scores: Dict[QueryIntent, int] = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[intent] = score
        if max(scores.values() or [0]) > 0:
            return max(scores, key=scores.get)
        return QueryIntent.SEMANTIC_ANALYSIS


class IntelligentQueryRouter:
    """Analyze queries and execute the best search strategy."""

    def __init__(self, vector_search=None, graph_search=None, hybrid_search=None) -> None:
        self.entity_extractor = RealEstateEntityExtractor()
        self.intent_classifier = QueryIntentClassifier()
        self.vector_search = vector_search
        self.graph_search = graph_search
        self.hybrid_search = hybrid_search

    async def route_search(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SearchStrategy:
        entities = await self.entity_extractor.extract_entities(query)
        intent = await self.intent_classifier.classify_intent(query)
        logger.info(
            "Query analysis - Intent: %s, Entities: %s", intent, entities
        )
        return self._determine_strategy(intent, entities)

    def _determine_strategy(
        self, intent: QueryIntent, entities: Dict[str, List[str]]
    ) -> SearchStrategy:
        if intent == QueryIntent.FACTUAL_LOOKUP and (entities["locations"] or entities["metrics"]):
            return SearchStrategy.GRAPH_ONLY
        if intent == QueryIntent.RELATIONSHIP_QUERY:
            return SearchStrategy.GRAPH_ONLY
        if intent in {QueryIntent.INVESTMENT_ANALYSIS, QueryIntent.COMPARATIVE_ANALYSIS}:
            return SearchStrategy.HYBRID
        if intent == QueryIntent.PROPERTY_SEARCH and entities["locations"]:
            return SearchStrategy.HYBRID
        if intent == QueryIntent.SEMANTIC_ANALYSIS and not any(entities.values()):
            return SearchStrategy.VECTOR_ONLY
        if entities["properties"] or entities["agents"]:
            return SearchStrategy.GRAPH_ONLY
        return SearchStrategy.HYBRID

    async def execute_search(
        self,
        query: str,
        strategy: SearchStrategy,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        start = datetime.utcnow()
        try:
            if strategy == SearchStrategy.VECTOR_ONLY:
                results = await self.vector_search.search(query, limit=limit, filters=filters)
            elif strategy == SearchStrategy.GRAPH_ONLY:
                results = await self.graph_search.search(query, limit=limit, filters=filters)
            else:
                results = await self.hybrid_search.search(query, limit=limit, filters=filters)
            logger.info(
                "Search executed with %s strategy in %.2fs", strategy, (datetime.utcnow() - start).total_seconds()
            )
            return results
        except Exception as exc:
            logger.error("Search execution failed: %s", exc)
            if strategy != SearchStrategy.VECTOR_ONLY:
                logger.info("Falling back to vector search")
                return await self.vector_search.search(query, limit=limit, filters=filters)
            return []


# Backwards compatibility for older imports
QueryRouter = IntelligentQueryRouter
