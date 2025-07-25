"""
Enhanced Semantic Chunking Strategy for TrackRealties
Improvements over current implementation
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tiktoken

@dataclass
class ChunkMetadata:
    """Enhanced metadata for chunks with semantic context"""
    chunk_type: str
    semantic_level: str  # 'property_overview', 'location_details', 'financial_details'
    entity_types: List[str]  # ['property', 'location', 'agent', 'price']
    parent_context: str
    chunk_relationships: List[str]  # References to related chunks
    content_density: float  # Information density score
    token_count: int
    semantic_score: float  # How semantically coherent the chunk is


class EnhancedSemanticChunker:
    """
    Advanced chunking with semantic awareness and relationship mapping
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Semantic field importance weights
        self.field_weights = {
            'high_importance': ['price', 'address', 'bedrooms', 'bathrooms', 'square_footage'],
            'medium_importance': ['description', 'property_type', 'listing_agent', 'school_district'],
            'low_importance': ['listing_id', 'last_updated', 'mls_number']
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'price': r'\$[\d,]+',
            'address': r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)',
            'room_count': r'\d+\s*(?:bed|bedroom|bath|bathroom)',
            'area': r'\d+\s*(?:sq|square)\s*(?:ft|feet)',
            'year': r'19\d{2}|20\d{2}'
        }
    
    def chunk_with_semantic_awareness(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """
        Create semantically aware chunks with relationship mapping
        """
        if data_type == "property_listing":
            return self._chunk_property_semantically(data)
        elif data_type == "market_data":
            return self._chunk_market_data_semantically(data)
        else:
            return self._chunk_generic_semantically(data)
    
    def _chunk_property_semantically(self, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for property listings with enhanced context"""
        chunks = []
        
        # 1. Core Property Information Chunk (High Priority)
        core_info = self._extract_core_property_info(property_data)
        if core_info:
            chunks.append({
                'content': core_info,
                'metadata': ChunkMetadata(
                    chunk_type='property_core',
                    semantic_level='primary',
                    entity_types=['property', 'price', 'location'],
                    parent_context=property_data.get('property_id', 'unknown'),
                    chunk_relationships=[],
                    content_density=self._calculate_density(core_info),
                    token_count=len(self.tokenizer.encode(core_info)),
                    semantic_score=self._calculate_semantic_score(core_info, 'property_core')
                ).__dict__
            })
        
        # 2. Location & Neighborhood Chunk
        location_info = self._extract_location_context(property_data)
        if location_info:
            chunks.append({
                'content': location_info,
                'metadata': ChunkMetadata(
                    chunk_type='location_context',
                    semantic_level='secondary',
                    entity_types=['location', 'neighborhood', 'schools'],
                    parent_context=property_data.get('property_id', 'unknown'),
                    chunk_relationships=['property_core'],
                    content_density=self._calculate_density(location_info),
                    token_count=len(self.tokenizer.encode(location_info)),
                    semantic_score=self._calculate_semantic_score(location_info, 'location')
                ).__dict__
            })
        
        # 3. Features & Amenities Chunk
        features_info = self._extract_features_amenities(property_data)
        if features_info:
            chunks.append({
                'content': features_info,
                'metadata': ChunkMetadata(
                    chunk_type='features_amenities',
                    semantic_level='tertiary',
                    entity_types=['features', 'amenities', 'specifications'],
                    parent_context=property_data.get('property_id', 'unknown'),
                    chunk_relationships=['property_core'],
                    content_density=self._calculate_density(features_info),
                    token_count=len(self.tokenizer.encode(features_info)),
                    semantic_score=self._calculate_semantic_score(features_info, 'features')
                ).__dict__
            })
        
        # 4. Financial & Investment Chunk
        financial_info = self._extract_financial_context(property_data)
        if financial_info:
            chunks.append({
                'content': financial_info,
                'metadata': ChunkMetadata(
                    chunk_type='financial_analysis',
                    semantic_level='secondary',
                    entity_types=['price', 'investment', 'taxes', 'costs'],
                    parent_context=property_data.get('property_id', 'unknown'),
                    chunk_relationships=['property_core'],
                    content_density=self._calculate_density(financial_info),
                    token_count=len(self.tokenizer.encode(financial_info)),
                    semantic_score=self._calculate_semantic_score(financial_info, 'financial')
                ).__dict__
            })
        
        return chunks
    
    def _extract_core_property_info(self, data: Dict[str, Any]) -> str:
        """Extract and format core property information"""
        core_fields = ['property_id', 'formatted_address', 'price', 'bedrooms', 
                      'bathrooms', 'square_footage', 'property_type', 'year_built']
        
        content = "PROPERTY OVERVIEW:\n"
        for field in core_fields:
            if field in data and data[field] is not None:
                formatted_field = field.replace('_', ' ').title()
                value = data[field]
                
                # Format specific fields
                if field == 'price' and isinstance(value, (int, float)):
                    value = f"${value:,.2f}"
                elif field == 'square_footage' and isinstance(value, (int, float)):
                    value = f"{value:,} sq ft"
                
                content += f"{formatted_field}: {value}\n"
        
        # Add description if available
        if 'description' in data and data['description']:
            content += f"\nDescription: {data['description'][:300]}{'...' if len(data['description']) > 300 else ''}\n"
        
        return content
    
    def _extract_location_context(self, data: Dict[str, Any]) -> str:
        """Extract location and neighborhood context"""
        location_fields = ['city', 'state', 'zip_code', 'neighborhood', 'school_district']
        geo_fields = ['latitude', 'longitude']
        
        content = "LOCATION & NEIGHBORHOOD:\n"
        
        # Basic location info
        for field in location_fields:
            if field in data and data[field]:
                formatted_field = field.replace('_', ' ').title()
                content += f"{formatted_field}: {data[field]}\n"
        
        # Geographic coordinates
        if all(field in data for field in geo_fields):
            content += f"Coordinates: {data['latitude']}, {data['longitude']}\n"
        
        # Nearby amenities from description
        if 'description' in data:
            amenities = self._extract_location_amenities(data['description'])
            if amenities:
                content += f"Nearby Amenities: {', '.join(amenities)}\n"
        
        return content
    
    def _extract_features_amenities(self, data: Dict[str, Any]) -> str:
        """Extract property features and amenities"""
        feature_fields = ['lot_size', 'garage_spaces', 'parking_spots', 'pool', 
                         'fireplace', 'basement', 'attic', 'deck', 'patio']
        
        content = "FEATURES & AMENITIES:\n"
        
        for field in feature_fields:
            if field in data and data[field] is not None:
                formatted_field = field.replace('_', ' ').title()
                value = data[field]
                
                # Handle boolean fields
                if isinstance(value, bool):
                    content += f"{formatted_field}: {'Yes' if value else 'No'}\n"
                else:
                    content += f"{formatted_field}: {value}\n"
        
        # Extract features from description
        if 'description' in data:
            extracted_features = self._extract_description_features(data['description'])
            if extracted_features:
                content += f"Additional Features: {', '.join(extracted_features)}\n"
        
        return content
    
    def _extract_financial_context(self, data: Dict[str, Any]) -> str:
        """Extract financial and investment information"""
        financial_fields = ['price', 'price_per_sqft', 'hoa_fee', 'property_taxes', 
                           'insurance_cost', 'days_on_market', 'price_history']
        
        content = "FINANCIAL & INVESTMENT INFO:\n"
        
        for field in financial_fields:
            if field in data and data[field] is not None:
                formatted_field = field.replace('_', ' ').title()
                value = data[field]
                
                # Format currency fields
                if 'price' in field or 'fee' in field or 'cost' in field or 'taxes' in field:
                    if isinstance(value, (int, float)):
                        value = f"${value:,.2f}"
                
                content += f"{formatted_field}: {value}\n"
        
        # Calculate investment metrics if possible
        if 'price' in data and 'square_footage' in data:
            try:
                price_per_sqft = data['price'] / data['square_footage']
                content += f"Calculated Price per Sq Ft: ${price_per_sqft:.2f}\n"
            except (ZeroDivisionError, TypeError):
                pass
        
        return content
    
    def _calculate_density(self, content: str) -> float:
        """Calculate information density of content"""
        words = content.split()
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        
        # Calculate metrics
        word_count = len(words)
        unique_ratio = len(unique_words) / word_count if word_count > 0 else 0
        
        # Count important entities
        entity_count = sum(1 for pattern in self.entity_patterns.values() 
                          if re.search(pattern, content, re.IGNORECASE))
        
        # Density score (0-1)
        density = min(1.0, (unique_ratio * 0.4) + (entity_count / word_count * 0.6))
        return round(density, 3)
    
    def _calculate_semantic_score(self, content: str, chunk_type: str) -> float:
        """Calculate semantic coherence score"""
        # Count relevant terms based on chunk type
        relevant_terms = {
            'property_core': ['bedroom', 'bathroom', 'sqft', 'price', 'address'],
            'location': ['city', 'neighborhood', 'school', 'near', 'close'],
            'features': ['kitchen', 'garage', 'pool', 'fireplace', 'updated'],
            'financial': ['price', 'cost', 'tax', 'fee', 'market', 'investment']
        }
        
        terms = relevant_terms.get(chunk_type, [])
        content_lower = content.lower()
        
        # Calculate relevance score
        found_terms = sum(1 for term in terms if term in content_lower)
        relevance_score = found_terms / len(terms) if terms else 0.5
        
        # Factor in content length appropriateness
        length_score = min(1.0, len(content) / 500)  # Optimal around 500 chars
        
        semantic_score = (relevance_score * 0.7) + (length_score * 0.3)
        return round(semantic_score, 3)
    
    def _extract_description_features(self, description: str) -> List[str]:
        """Extract features mentioned in property description"""
        features = []
        feature_keywords = [
            'granite countertops', 'hardwood floors', 'stainless steel', 'walk-in closet',
            'master suite', 'updated kitchen', 'new roof', 'central air', 'fireplace',
            'finished basement', 'deck', 'patio', 'fenced yard', 'garage'
        ]
        
        description_lower = description.lower()
        for feature in feature_keywords:
            if feature in description_lower:
                features.append(feature.title())
        
        return features
    
    def _extract_location_amenities(self, description: str) -> List[str]:
        """Extract location amenities from description"""
        amenities = []
        location_keywords = [
            'school district', 'shopping center', 'park', 'beach', 'downtown',
            'public transportation', 'highway access', 'restaurants', 'hospital'
        ]
        
        description_lower = description.lower()
        for amenity in location_keywords:
            if amenity in description_lower:
                amenities.append(amenity.title())
        
        return amenities


# Usage example
def test_enhanced_chunking():
    """Test the enhanced chunking system"""
    sample_property = {
        "property_id": "12345",
        "formatted_address": "123 Main St, Anytown, ST 12345",
        "price": 450000,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "square_footage": 2100,
        "property_type": "Single Family",
        "year_built": 2010,
        "city": "Anytown",
        "state": "ST",
        "zip_code": "12345",
        "description": "Beautiful updated home with granite countertops, hardwood floors, and finished basement. Close to excellent schools and shopping centers.",
        "garage_spaces": 2,
        "lot_size": 0.25,
        "property_taxes": 8500
    }
    
    chunker = EnhancedSemanticChunker()
    chunks = chunker.chunk_with_semantic_awareness(sample_property, "property_listing")
    
    print(f"Generated {len(chunks)} semantic chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk['metadata']['chunk_type']}):")
        print(f"Content: {chunk['content'][:200]}...")
        print(f"Semantic Score: {chunk['metadata']['semantic_score']}")
        print(f"Content Density: {chunk['metadata']['content_density']}")
        print(f"Token Count: {chunk['metadata']['token_count']}")

if __name__ == "__main__":
    test_enhanced_chunking()