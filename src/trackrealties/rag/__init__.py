"""
Retrieval-Augmented Generation (RAG) module for the TrackRealties AI Platform.

This module provides intelligent search and context-aware responses for real estate data.
"""

from .rag_pipeline_integration import EnhancedRAGPipeline
from .router import QueryRouter
from .search import GraphSearch, HybridSearchEngine, VectorSearch
from .synthesizer import ResponseSynthesizer
from .validation import RealEstateHallucinationDetector
from ..validation import RealEstateHallucinationDetector


__all__ = [
    "VectorSearch",
    "GraphSearch",
    "HybridSearchEngine",
    "EnhancedRAGPipeline",
    "QueryRouter",
    "ResponseSynthesizer",
    "RealEstateHallucinationDetector",
]
