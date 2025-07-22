"""
Generic JSON chunking implementation for the TrackRealties AI Platform.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...core.config import get_settings
from .chunk import Chunk
from .utils import generate_chunk_id, enrich_metadata

logger = logging.getLogger(__name__)
settings = get_settings()


class GenericChunker:
    """
    Specialized chunker for generic JSON data.
    """
    
    def __init__(self, max_chunk_size: int, chunk_overlap: int):
        """
        Initialize the GenericChunker.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_generic_json(self, data: Dict[str, Any], data_type: str = "generic") -> List[Chunk]:
        """
        Chunk generic JSON data.
        
        Args:
            data: JSON data dictionary
            data_type: Type of data
            
        Returns:
            List of chunks
        """
        self.logger.info(f"Chunking generic JSON data of type {data_type}")
        
        # Extract source if available
        source = data.get("source", "unknown")
        
        # Create metadata
        metadata = {
            "type": "generic",  # Always use "generic" for unknown types
            "source": source,
            "chunk_type": "main",
            "content_type": "json",
            "chunk_created_at": datetime.utcnow().isoformat(),
            "data_keys": list(data.keys()),
            "data_size": len(json.dumps(data))
        }
        
        # Extract keywords
        keywords = self._extract_keywords_from_json(data)
        if keywords:
            metadata["keywords"] = keywords
        
        # Format content
        content = self._format_generic_json_content(data)
        
        # Create chunk ID
        chunk_id = generate_chunk_id(content, data_type)
        
        # Create chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata
        )
        
        return [chunk]
    
    def _format_generic_json_content(self, data: Dict[str, Any]) -> str:
        """Format generic JSON content for better readability."""
        # Start with a header
        content = "JSON Data:\n\n"
        
        # Add each key-value pair
        for key, value in data.items():
            # Format the key for better readability
            formatted_key = key.replace("_", " ").title()
            
            # Format the value based on its type
            if isinstance(value, dict):
                # For nested dictionaries, format as a nested structure
                content += f"{formatted_key}:\n"
                for sub_key, sub_value in value.items():
                    formatted_sub_key = sub_key.replace("_", " ").title()
                    if isinstance(sub_value, (int, float)) and "price" in sub_key.lower():
                        formatted_sub_value = f"${sub_value:,.2f}"
                    elif isinstance(sub_value, (list, dict)):
                        formatted_sub_value = f"{len(sub_value)} items"
                    else:
                        formatted_sub_value = str(sub_value)
                    content += f"  - {formatted_sub_key}: {formatted_sub_value}\n"
            elif isinstance(value, list):
                # For lists, summarize the content
                content += f"{formatted_key}: {len(value)} items\n"
                # If the list is small, include the items
                if len(value) <= 5:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            content += f"  - Item {i+1}: {len(item)} properties\n"
                        else:
                            content += f"  - Item {i+1}: {item}\n"
            elif isinstance(value, (int, float)) and "price" in key.lower():
                # Format prices with dollar sign and commas
                content += f"{formatted_key}: ${value:,.2f}\n"
            else:
                # For simple values, just convert to string
                content += f"{formatted_key}: {value}\n"
        
        return content
    
    def _extract_keywords_from_json(self, data: Dict[str, Any]) -> List[str]:
        """Extract keywords from JSON data."""
        keywords = set()
        
        # Process the data recursively
        self._process_json_for_keywords(data, keywords)
        
        # Filter out common words
        stop_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "hers"}
        filtered_keywords = [kw for kw in keywords if kw not in stop_words]
        
        return list(filtered_keywords)
    
    def _process_json_for_keywords(self, data: Any, keywords: set):
        """
        Process JSON data recursively to extract keywords.
        
        Args:
            data: JSON data (dict, list, or primitive)
            keywords: Set to add keywords to
        """
        if isinstance(data, dict):
            # Process dictionary
            for key, value in data.items():
                # Add the key itself as a keyword
                keywords.add(key.lower())
                
                # Process the value
                self._process_json_for_keywords(value, keywords)
        elif isinstance(data, list):
            # Process list
            for item in data:
                self._process_json_for_keywords(item, keywords)
        elif isinstance(data, str):
            # Process string
            if len(data) < 30:
                # Add short strings directly
                keywords.add(data.lower())
            else:
                # For longer strings, extract individual words
                words = re.findall(r'\b\w+\b', data.lower())
                for word in words:
                    if len(word) > 2:  # Only add words with more than 2 characters
                        keywords.add(word)