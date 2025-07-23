"""
Retrieval-Augmented Generation (RAG) module for the TrackRealties AI Platform.

This module provides intelligent search and context-aware responses for real estate data.
"""

from .rag_pipeline_integration import EnhancedRAGPipeline
from .search import VectorSearch, GraphSearch, HybridSearchEngine
from .router import QueryRouter
from .synthesizer import ResponseSynthesizer
from .validation import RealEstateHallucinationDetector
from ..validation import RealEstateHallucinationDetector
from .smart_search import (
    SmartSearchRouter,
    FixedGraphSearch,
    SearchStrategy,
    RealEstateEntityExtractor,
)


__all__ = [
    "VectorSearch",
    'GraphSearch',
    'HybridSearchEngine',
    'QueryRouter',
    "EnhancedRAGPipeline",
    "ResponseSynthesizer",
    'RealEstateHallucinationDetector',
    'SmartSearchRouter',
    'FixedGraphSearch',
    'SearchStrategy',
    'RealEstateEntityExtractor',
]
