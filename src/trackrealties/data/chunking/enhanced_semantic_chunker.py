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
    
    def _chunk_market_data_semantically(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Create semantic chunks for market data with enhanced context"""
            chunks = []
            # 1. Market Overview Chunk (High Priority)
            overview_content = self._extract_market_overview(market_data)
            if overview_content:
                chunks.append({
                    'content': overview_content,
                    'metadata': {
                        'chunk_type': 'market_overview',
                        'semantic_level': 'primary',
                        'entity_types': ['location', 'price', 'volume'],
                        'source': market_data.get('source', 'unknown'),
                        'parent_context': market_data.get('region_id', market_data.get('location', 'unknown')),
                        'semantic_score': self._calculate_semantic_score(overview_content, 'market_overview'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(overview_content)),
                        'extracted_entities': self._extract_market_entities(market_data),
                    }
                })
            
            # 2. Geographic Location Chunk (NEW!)
            geographic_content = self._extract_geographic_location(market_data)
            if geographic_content:
                chunks.append({
                    'content': geographic_content,
                    'metadata': {
                        'chunk_type': 'geographic_location',
                        'semantic_level': 'secondary',
                        'entity_types': ['location', 'geography', 'coordinates'],
                        'parent_context': market_data.get('region_id', market_data.get('location', 'unknown')),
                        'source': market_data.get('source', 'unknown'),
                        'semantic_score': self._calculate_semantic_score(geographic_content, 'geographic'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(geographic_content)),
                        'extracted_entities': self._extract_geographic_entities(market_data),
                    }
                })
            # 2. Price Trends Chunk
            price_content = self._extract_price_trends(market_data)
            if price_content:
                chunks.append({
                    'content': price_content,
                    'metadata': {
                        'chunk_type': 'price_trends',
                        'semantic_level': 'secondary',
                        'entity_types': ['price', 'trend', 'metrics'],
                        'source': market_data.get('source', 'unknown'),
                        'parent_context': market_data.get('region_id', market_data.get('location', 'unknown')),
                        'semantic_score': self._calculate_semantic_score(price_content, 'price_trends'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(price_content)),
                        'extracted_entities': self._extract_price_entities(market_data),
                    }
                })
            # 3. Inventory Analysis Chunk
            inventory_content = self._extract_inventory_analysis(market_data)
            if inventory_content:
                chunks.append({
                    'content': inventory_content,
                    'metadata': {
                        'chunk_type': 'inventory_analysis',
                        'semantic_level': 'tertiary',
                        'entity_types': ['inventory', 'listing', 'metrics'],
                        'parent_context': market_data.get('region_id', market_data.get('location', 'unknown')),
                        'source': market_data.get('source', 'unknown'),
                        'semantic_score': self._calculate_semantic_score(inventory_content, 'inventory_analysis'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(inventory_content)),
                        'extracted_entities': self._extract_inventory_entities(market_data),
                    }
                })
            return chunks
    
    def _extract_market_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from market data"""
        entities = {}
        
        if 'median_home_price' in data or 'median_price' in data:
            entities['median_price'] = data.get('median_home_price', data.get('median_price'))
        if 'sales_volume' in data:
            entities['sales_volume'] = data['sales_volume']
        if 'location' in data:
            entities['location'] = data['location']
        
        return entities

    def _extract_geographic_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic entities"""
        entities = {}
        
        location = data.get('location', data.get('region_name', ''))
        if location:
            region, county = self._parse_location_string(location)
            if region:
                entities['region'] = region
            if county:
                entities['county'] = county
        
        if 'city' in data:
            entities['city'] = data['city']
        if 'state' in data:
            entities['state'] = data['state']
        if 'latitude' in data and 'longitude' in data:
            entities['coordinates'] = f"{data['latitude']}, {data['longitude']}"
        
        return entities

    def _extract_price_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract price-related entities"""
        entities = {}
        
        if 'price_per_sqft_median' in data:
            entities['price_per_sqft'] = data['price_per_sqft_median']
        if 'price_change_monthly' in data:
            entities['monthly_change'] = data['price_change_monthly']
        if 'price_change_yearly' in data:
            entities['yearly_change'] = data['price_change_yearly']
        
        return entities

    def _extract_inventory_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract inventory-related entities"""
        entities = {}
        
        if 'inventory_count' in data:
            entities['inventory_count'] = data['inventory_count']
        if 'days_on_market_median' in data:
            entities['days_on_market'] = data['days_on_market_median']
        if 'months_supply' in data:
            entities['months_supply'] = data['months_supply']
        if 'new_listings' in data:
            entities['new_listings'] = data['new_listings']
        
        return entities
    
    def _extract_market_overview(self, data: Dict[str, Any]) -> str:
        """Extract market overview information"""
        content = "MARKET OVERVIEW:\n"

        # Location info 
        location = data.get('location', data.get('region_name', 'Unknown Location'))
        content += f"Location: {location}\n"

        # Date info
        date = data.get('date', data.get('data_date',''))
        if date:
            content += f"Date: {date}\n"
        
        # Core metrics
        if 'median_home_price' in data or 'median_price' in data:
            price = data.get('median_home_price', data.get('median_price'))
            content += f"Median Price: ${price:,.2f}\n"
        
        # Sales data
        if 'homes_sold' in data or 'sales_volume' in data:
            sales = data.get('homes_sold', data.get('sales_volume'))
            content += f"Sales Volume: {sales}\n"
        
        return content
    
    def _extract_price_trends(self, data: Dict[str, Any]) -> str:
        """Extract price trend information"""
        content = "PRICE TRENDS:\n"

        # Price per sqft
        if 'price_per_sqft_median' in data or 'price_per_sqft' in data:
            price_sqft = data.get('price_per_sqft_median', data.get('price_per_sqft'))
            content += f"Price Per Sqft: ${price_sqft:.2f}\n"
        # Price changes
        if 'price_change_monthly' in data:
            monthly = data['price_change_monthly'] * 100
            content += f"Monthly Change: {monthly:+.1f}%\n"

        if 'price_change_yearly' in data:
            yearly = data['price_change_yearly'] * 100
            content += f"Yearly Change: {yearly:+.1f}%\n"
        
        return content
    
    def _extract_inventory_analysis(self, data: Dict[str, Any]) -> str:
        """Extract inventory analysis information"""
        content = "INVENTORY ANALYSIS:\n"

        # Inventory count
        if 'inventory_count' in data:
            content += f"Inventory Count: {data['inventory_count']}\n"
        
        # Days on market
        if 'days_on_market_median' in data or 'days_on_market' in data:
            dom = data.get('days_on_market_median', data.get('days_on_market'))
            content += f"Days on Market: {dom}\n"
        
        # Months supply
        if 'months_supply' in data:
            content += f"Months Supply: {data['months_supply']}\n"
        
        # New listings
        if 'new_listings' in data:
            content += f"New Listings: {data['new_listings']}\n"
        return content
    
    def _extract_geographic_location(self, data: Dict[str, Any]) -> str:
        """Extract geographic location information with region and county parsing"""
        content = "GEOGRAPHIC LOCATION:\n"
        
        # Get location string and parse it
        location = data.get('location', data.get('region_name', ''))
        
        if location:
            # Parse region and county from location string
            region, county = self._parse_location_string(location)
            
            if region:
                content += f"Region: {region}\n"
            if county:
                content += f"County: {county}\n"
            
            content += f"Full Location: {location}\n"
        
        # Basic location info
        if 'city' in data:
            content += f"City: {data['city']}\n"
        if 'state' in data:
            content += f"State: {data['state']}\n"

        # Coordinates
        if 'latitude' in data and 'longitude' in data:
            content += f"Coordinates: {data['latitude']}, {data['longitude']}\n"
        
        return content

    def _parse_location_string(self, location: str) -> Tuple[str, str]:
        """Parse region and county from location string like 'Ruston, LA metro area'"""
        region = ""
        county = ""
        
        if not location:
            return region, county
        
        # Common patterns for parsing location strings
        location_lower = location.lower()
        
        # Extract region (city/area name)
        if ',' in location:
            parts = location.split(',')
            region = parts[0].strip()  # "Ruston"
            
            # Look for county indicators in remaining parts
            remaining = ','.join(parts[1:]).strip()
            
            # Check for county patterns
            county_patterns = [
                r'(\w+)\s+county',
                r'(\w+)\s+co\.',
                r'(\w+)\s+parish',  # Louisiana uses parishes
            ]
            
            for pattern in county_patterns:
                import re
                match = re.search(pattern, remaining, re.IGNORECASE)
                if match:
                    county = match.group(1).strip()
                    break
            
            # If no county found but we have state info, try to infer
            if not county and len(parts) >= 2:
                state_part = parts[1].strip()
                # If it's just state code, the region might include county info
                if len(state_part) == 2:  # State code like "LA"
                    # For Louisiana, many areas are parishes
                    if state_part.upper() == 'LA':
                        county = f"{region} Parish"
        else:
            region = location
        
        return region, county


    def _chunk_generic_semantically(self, data: Dict[str, Any], data_type: str = "generic") -> List[Dict[str, Any]]:
        """Create semantic chunks for generic data"""
        chunks = []
    
        # 1. Main Data Chunk
        main_content = self._extract_generic_main_content(data)
        if main_content:
            chunks.append({
                'content': main_content,
                'metadata': {
                    'chunk_type': 'generic_main',
                    'semantic_level': 'primary',
                    'entity_types': self._extract_entity_types_from_data(data),
                    'parent_context': data.get('id', data.get('identifier', 'unknown')),
                    'semantic_score': self._calculate_semantic_score(main_content, 'generic'),
                    'chunk_created_at': datetime.now().isoformat(),
                    'token_count': len(self.tokenizer.encode(main_content)),
                    'data_type': data_type,
                    'data_keys': list(data.keys()) if isinstance(data, dict) else []
                }
            })
        
        # 2. If data is large, create additional chunks for complex nested objects
        if isinstance(data, dict) and len(str(data)) > self.max_chunk_size:
            nested_chunks = self._chunk_large_generic_data(data, data_type)
            chunks.extend(nested_chunks)
        
        return chunks

    def _extract_generic_main_content(self, data: Dict[str, Any]) -> str:
        """Extract main content from generic data"""
        content = "DATA SUMMARY:\n"
        
        # Handle the data based on its structure
        if isinstance(data, dict):
            # Prioritize important fields first
            important_fields = ['id', 'name', 'title', 'description', 'type', 'status', 'date', 'created_at', 'updated_at']
            
            # Add important fields first
            for field in important_fields:
                if field in data and data[field] is not None:
                    formatted_field = field.replace('_', ' ').title()
                    value = data[field]
                    
                    # Format specific field types
                    if 'date' in field.lower() or 'time' in field.lower():
                        content += f"{formatted_field}: {value}\n"
                    elif isinstance(value, (int, float)) and any(keyword in field.lower() for keyword in ['price', 'cost', 'amount', 'fee']):
                        content += f"{formatted_field}: ${value:,.2f}\n"
                    elif isinstance(value, str) and len(value) > 100:
                        # Truncate long text fields
                        content += f"{formatted_field}: {value[:100]}...\n"
                    else:
                        content += f"{formatted_field}: {value}\n"
            
            # Add other fields (excluding the ones already processed)
            remaining_fields = [k for k in data.keys() if k not in important_fields]
            for field in remaining_fields[:10]:  # Limit to first 10 remaining fields
                if data[field] is not None:
                    formatted_field = field.replace('_', ' ').title()
                    value = data[field]
                    
                    # Skip complex nested objects for main content
                    if isinstance(value, (dict, list)):
                        if isinstance(value, list):
                            content += f"{formatted_field}: {len(value)} items\n"
                        else:
                            content += f"{formatted_field}: {len(value)} properties\n"
                    else:
                        # Handle simple values
                        if isinstance(value, str) and len(value) > 50:
                            content += f"{formatted_field}: {value[:50]}...\n"
                        else:
                            content += f"{formatted_field}: {value}\n"
        
        return content

    def _chunk_large_generic_data(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """Create additional chunks for large generic data with nested objects"""
        chunks = []
        
        # Look for complex nested objects that deserve their own chunks
        for key, value in data.items():
            if isinstance(value, dict) and len(str(value)) > 200:
                # Create a chunk for this nested object
                nested_content = self._format_nested_object(key, value)
                
                chunks.append({
                    'content': nested_content,
                    'metadata': {
                        'chunk_type': 'generic_nested',
                        'semantic_level': 'secondary',
                        'entity_types': [key.lower(), 'nested_data'],
                        'parent_context': data.get('id', data.get('identifier', 'unknown')),
                        'nested_field': key,
                        'semantic_score': self._calculate_semantic_score(nested_content, 'nested'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(nested_content)),
                        'data_type': data_type
                    }
                })
            
            elif isinstance(value, list) and len(value) > 5:
                # Create a chunk for large lists
                list_content = self._format_list_data(key, value)
                
                chunks.append({
                    'content': list_content,
                    'metadata': {
                        'chunk_type': 'generic_list',
                        'semantic_level': 'secondary',
                        'entity_types': [key.lower(), 'list_data'],
                        'parent_context': data.get('id', data.get('identifier', 'unknown')),
                        'list_field': key,
                        'list_size': len(value),
                        'semantic_score': self._calculate_semantic_score(list_content, 'list'),
                        'chunk_created_at': datetime.now().isoformat(),
                        'token_count': len(self.tokenizer.encode(list_content)),
                        'data_type': data_type
                    }
                })
        
        return chunks

    def _format_nested_object(self, key: str, obj: Dict[str, Any]) -> str:
        """Format a nested object for chunking"""
        content = f"{key.replace('_', ' ').title().upper()}:\n\n"
        
        for sub_key, sub_value in obj.items():
            formatted_key = sub_key.replace('_', ' ').title()
            
            if isinstance(sub_value, (dict, list)):
                if isinstance(sub_value, list):
                    content += f"{formatted_key}: {len(sub_value)} items\n"
                else:
                    content += f"{formatted_key}: {len(sub_value)} properties\n"
            elif isinstance(sub_value, str) and len(sub_value) > 100:
                content += f"{formatted_key}: {sub_value[:100]}...\n"
            else:
                content += f"{formatted_key}: {sub_value}\n"
        
        return content

    def _format_list_data(self, key: str, lst: List[Any]) -> str:
        """Format list data for chunking"""
        content = f"{key.replace('_', ' ').title().upper()} ({len(lst)} items):\n\n"
        
        # Show first few items
        for i, item in enumerate(lst[:5]):
            if isinstance(item, dict):
                # For dict items, show key info
                content += f"Item {i+1}:\n"
                for item_key, item_value in list(item.items())[:3]:  # First 3 keys
                    formatted_key = item_key.replace('_', ' ').title()
                    content += f"  {formatted_key}: {item_value}\n"
            else:
                content += f"Item {i+1}: {item}\n"
        
        if len(lst) > 5:
            content += f"... and {len(lst) - 5} more items\n"
        
        return content

    def _extract_entity_types_from_data(self, data: Dict[str, Any]) -> List[str]:
        """Extract entity types from generic data based on field names"""
        entity_types = ['data']
        
        # Look for common entity indicators in field names
        field_indicators = {
            'user': ['user', 'person', 'customer', 'client'],
            'location': ['address', 'city', 'state', 'country', 'location', 'geo'],
            'financial': ['price', 'cost', 'amount', 'fee', 'payment', 'revenue'],
            'temporal': ['date', 'time', 'created', 'updated', 'timestamp'],
            'contact': ['email', 'phone', 'contact', 'website'],
            'product': ['product', 'item', 'service', 'offering'],
            'organization': ['company', 'organization', 'business', 'agency']
        }
        
        if isinstance(data, dict):
            field_names = [key.lower() for key in data.keys()]
            
            for entity_type, indicators in field_indicators.items():
                if any(indicator in field_name for field_name in field_names for indicator in indicators):
                    entity_types.append(entity_type)
        
        return entity_types
            
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