"""
JSON Chunker implementation for the TrackRealties AI Platform.
"""

import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

from .enhanced_semantic_chunker import EnhancedSemanticChunker
from ...core.config import get_settings
from .chunk import Chunk
from .market_chunker import MarketDataChunker
from .property_chunker import PropertyListingChunker
from .generic_chunker import GenericChunker
from .utils import generate_chunk_id

logger = logging.getLogger(__name__)
settings = get_settings()


class JSONChunker:
    """
    Chunks JSON data semantically based on its structure.
    
    This class provides methods to break down JSON data into semantically meaningful chunks
    based on the structure of the data, with special handling for property listings and
    market data.
    """
    
    def __init__(self, 
                 max_chunk_size: Optional[int] = None, 
                 chunk_overlap: Optional[int] = None):
        """
        Initialize the JSONChunker.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.enhanced_chunker = EnhancedSemanticChunker(
            max_chunk_size or settings.max_chunk_size,
            chunk_overlap or settings.chunk_overlap
        )
    
    
        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize specialized chunkers
        self.market_chunker = MarketDataChunker(self.max_chunk_size, self.chunk_overlap)
        self.property_chunker = PropertyListingChunker(self.max_chunk_size, self.chunk_overlap)
        self.generic_chunker = GenericChunker(self.max_chunk_size, self.chunk_overlap)



    # Add this new method
    def chunk_json_enhanced(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """Use enhanced semantic chunking"""
        return self.enhanced_chunker.chunk_with_semantic_awareness(data, data_type)
    
    def chunk_market_data(self, market_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk market data into semantically meaningful chunks.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            List of chunks
        """
        return self.market_chunker.chunk_market_data(market_data)
    
    def chunk_property_listing(self, property_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk property listing data into semantically meaningful chunks.
        
        Args:
            property_data: Property listing dictionary
            
        Returns:
            List of chunks
        """
        return self.property_chunker.chunk_property_listing(property_data)
    
    def chunk_json(self, data: Dict[str, Any], data_type: str) -> List[Chunk]:
        """
        Chunk JSON data based on its type.
        
        Args:
            data: JSON data dictionary
            data_type: Type of the data (e.g., 'market', 'property', 'generic')
            
        Returns:
            List of chunks
        """
        self.logger.info(f"Chunking JSON data of type {data_type}")
        try:
            if data_type in ["market_data", "market"]:
                enhanced_chunks = self.enhanced_chunker.chunk_with_semantic_awareness(data, "market_data")
            elif data_type in ["property_listing", "property"]:
                enhanced_chunks = self.enhanced_chunker.chunk_with_semantic_awareness(data, "property_listing")
            else:
                enhanced_chunks = self.enhanced_chunker.chunk_with_semantic_awareness(data, data_type)
            # Convert enhanced chunks to Chunk objects
            chunks = []
            for chunk_data in enhanced_chunks:
                chunk = Chunk(
                    chunk_id=generate_chunk_id(chunk_data['content'], data_type),
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata']
                )
                chunks.append(chunk)
            self.logger.info(f"Enhanced chunking created {len(chunks)} semantic chunks")
            return chunks
        except Exception as e:
            self.logger.warning(f"Enhanced chunking failed, falling back to standard: {e}")
            # Fallback to standard chunking
            if data_type == "market_data":
                return self.market_chunker.chunk_market_data(data)
            elif data_type == "property_listing":
                return self.property_chunker.chunk_property_listing(data)
            else:
                return self.generic_chunker.chunk_generic_json(data, data_type)  
 
    def _get_price_range(self, price: float) -> str:
        """Get price range bucket for a given price."""
        return self.property_chunker._get_price_range(price)
    
    def _get_size_range(self, size: float) -> str:
        """Get size range bucket for a given square footage."""
        return self.property_chunker._get_size_range(size)
    
    def _get_age_range(self, age: int) -> str:
        """Get age range bucket for a given property age in years."""
        return self.property_chunker._get_age_range(age)
    
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract important keywords from a property description."""
        return self.property_chunker._extract_keywords_from_description(description)
    

