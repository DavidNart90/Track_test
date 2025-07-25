"""
Property listing chunking implementation for the TrackRealties AI Platform.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from ...core.config import get_settings
from .chunk import Chunk
from .utils import generate_chunk_id, enrich_metadata

logger = logging.getLogger(__name__)
settings = get_settings()


class PropertyListingChunker:
    """
    Specialized chunker for property listings.
    """
    
    def __init__(self, max_chunk_size: int, chunk_overlap: int):
        """
        Initialize the PropertyListingChunker.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Common property features for keyword extraction
        self.common_features = {
            "updated", "renovated", "remodeled", "new", "modern", "granite", "stainless", 
            "hardwood", "fireplace", "pool", "garage", "basement", "view", "waterfront", 
            "deck", "patio", "balcony", "yard", "garden", "fenced", "gated", "security",
            "central", "air", "heating", "furnished", "appliances", "washer", "dryer",
            "dishwasher", "refrigerator", "microwave", "oven", "stove", "disposal",
            "ceiling", "fan", "walk-in", "closet", "storage", "pantry", "laundry",
            "master", "suite", "bath", "shower", "tub", "jacuzzi", "spa", "sauna",
            "gym", "fitness", "community", "clubhouse", "tennis", "golf", "park",
            "school", "shopping", "restaurant", "transit", "highway", "airport"
        }
    
    def chunk_property_listing(self, property_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk property listing data into semantically meaningful chunks.
        
        Creates multiple chunks based on the property listing structure:
        1. Main Property Info Chunk: Contains core property details
        2. Location Chunk: Contains geo-specific information
        3. Agent and Office Chunk: Contains information about the listing agent and office
        4. History Chunk: Contains the property's history
        
        Args:
            property_data: Property listing dictionary
            
        Returns:
            List of chunks
        """
        self.logger.info(f"Chunking property listing for property {property_data.get('id', 'unknown')}")
        chunks = []
        
        # Extract common metadata
        property_id = property_data.get("property_id", property_data.get("id", "unknown"))
        address = property_data.get("formatted_address", property_data.get("formattedAddress", "Unknown Address"))
        city = property_data.get("city", "Unknown City")
        state = property_data.get("state", "Unknown State")
        property_type = property_data.get("property_type", property_data.get("propertyType", "unknown"))
        status = property_data.get("status", "unknown")
        
        # Base metadata for all chunks
        base_metadata = {
            "type": "property_listing",
            "property_id": property_id,
            "address": address,
            "city": city,
            "state": state,
            "property_type": property_type,
            "status": status,
        }
        
        # 1. Create main property info chunk
        main_chunk_id = f"property_{property_id}_main"
        
        # Create main property content
        main_content = self._format_property_main_content(property_data)
        
        main_metadata = {
            **base_metadata,
            "chunk_type": "main",
            "content_type": "property_overview"
        }
        
        # Add specific property attributes to metadata for better filtering
        for key in ["bedrooms", "bathrooms", "square_footage", "price", "year_built"]:
            if key in property_data:
                main_metadata[key] = property_data[key]
            # Check for camelCase variants
            camel_key = ''.join([key[0], *[c.upper() if i > 0 else c for i, c in enumerate(key.split('_'))]])
            if camel_key in property_data:
                main_metadata[key] = property_data[camel_key]
        
        # Enrich metadata for main property chunk
        main_metadata = self._enrich_property_metadata(main_metadata, property_data)
        
        main_chunk = Chunk(
            chunk_id=main_chunk_id,
            content=main_content,
            metadata=main_metadata
        )
        chunks.append(main_chunk)
        
        # 2. Create location chunk with geo-specific information
        location_chunk_id = f"property_{property_id}_location"
        
        # Extract location data
        location_data = {
            "property_id": property_id,
            "formatted_address": address,
            "address_line1": property_data.get("address_line1", property_data.get("addressLine1")),
            "address_line2": property_data.get("address_line2", property_data.get("addressLine2")),
            "city": city,
            "state": state,
            "zip_code": property_data.get("zip_code", property_data.get("zipCode")),
            "county": property_data.get("county"),
            "latitude": property_data.get("latitude"),
            "longitude": property_data.get("longitude"),
            "neighborhood": property_data.get("neighborhood"),
            "school_district": property_data.get("school_district", property_data.get("schoolDistrict"))
        }
        
        location_content = self._format_property_location_content(location_data)
        
        location_metadata = {
            **base_metadata,
            "chunk_type": "location",
            "content_type": "property_location"
        }
        
        # Add geo coordinates to metadata if available
        if "latitude" in property_data and "longitude" in property_data:
            location_metadata["latitude"] = property_data["latitude"]
            location_metadata["longitude"] = property_data["longitude"]
        
        # Enrich metadata for location chunk
        location_metadata = self._enrich_location_metadata(location_metadata, location_data)
        
        location_chunk = Chunk(
            chunk_id=location_chunk_id,
            content=location_content,
            metadata=location_metadata,
            parent_id=main_chunk_id  # Link to main chunk
        )
        chunks.append(location_chunk)
        
        # 3. Create agent and office chunk if available
        if "listing_agent" in property_data or "listingAgent" in property_data:
            agent_chunk_id = f"property_{property_id}_agent"
            
            # Extract agent data
            listing_agent = property_data.get("listing_agent", property_data.get("listingAgent", {}))
            listing_office = property_data.get("listing_office", property_data.get("listingOffice", {}))
            
            agent_data = {
                "property_id": property_id,
                "address": address,
                "listing_agent": listing_agent,
                "listing_office": listing_office
            }
            
            agent_content = self._format_property_agent_content(agent_data)
            
            agent_metadata = {
                **base_metadata,
                "chunk_type": "agent",
                "content_type": "property_agent"
            }
            
            # Add agent name to metadata if available
            if isinstance(listing_agent, dict) and "name" in listing_agent:
                agent_metadata["agent_name"] = listing_agent["name"]
            
            # Enrich metadata for agent chunk
            agent_metadata = self._enrich_agent_metadata(agent_metadata, agent_data)
            
            agent_chunk = Chunk(
                chunk_id=agent_chunk_id,
                content=agent_content,
                metadata=agent_metadata,
                parent_id=main_chunk_id  # Link to main chunk
            )
            chunks.append(agent_chunk)
        
        # 4. Create history chunk if available
        if "history" in property_data and property_data["history"]:
            history_chunk_id = f"property_{property_id}_history"
            
            history_data = {
                "property_id": property_id,
                "address": address,
                "history": property_data["history"]
            }
            
            history_content = self._format_property_history_content(history_data)
            
            history_metadata = {
                **base_metadata,
                "chunk_type": "history",
                "content_type": "property_history"
            }
            
            # Enrich metadata for history chunk
            history_metadata = self._enrich_property_history_metadata(history_metadata, history_data)
            
            history_chunk = Chunk(
                chunk_id=history_chunk_id,
                content=history_content,
                metadata=history_metadata,
                parent_id=main_chunk_id  # Link to main chunk
            )
            chunks.append(history_chunk)
        
        return chunks
    
    def _format_property_main_content(self, property_data: Dict[str, Any]) -> str:
        """Format main property content for better readability."""
        address = property_data.get("formattedAddress", "Unknown Address")
        property_type = property_data.get("propertyType", "unknown")
        status = property_data.get("status", "unknown")
        price = property_data.get("price", 0)
        
        content = f"""Property Listing: {address}
Status: {status}
Type: {property_type}
Price: ${price:,.2f}

Details:
"""
        
        # Add property details
        bedrooms = property_data.get("bedrooms")
        if bedrooms is not None:
            content += f"- Bedrooms: {bedrooms}\n"
            
        bathrooms = property_data.get("bathrooms")
        if bathrooms is not None:
            content += f"- Bathrooms: {bathrooms}\n"
            
        square_footage = property_data.get("squareFootage")
        if square_footage is not None:
            content += f"- Square Footage: {square_footage:,} sq ft\n"
            
        lot_size = property_data.get("lotSize")
        if lot_size is not None:
            # Format lot size based on its value
            if isinstance(lot_size, (int, float)):
                if lot_size < 1:  # Less than 1 acre
                    content += f"- Lot Size: {lot_size * 43560:,.0f} sq ft\n"
                else:  # 1 acre or more
                    content += f"- Lot Size: {lot_size:,.2f} acres\n"
            else:
                content += f"- Lot Size: {lot_size}\n"
            
        year_built = property_data.get("yearBuilt")
        if year_built is not None:
            content += f"- Year Built: {year_built}\n"
            
        days_on_market = property_data.get("daysOnMarket")
        if days_on_market is not None:
            content += f"- Days on Market: {days_on_market}\n"
            
        # Add features if available
        features = property_data.get("features")
        if features:
            content += "\nFeatures:\n"
            if isinstance(features, list):
                for feature in features:
                    content += f"- {feature}\n"
            elif isinstance(features, dict):
                for category, items in features.items():
                    content += f"\n{category}:\n"
                    if isinstance(items, list):
                        for item in items:
                            content += f"- {item}\n"
                    else:
                        content += f"- {items}\n"
            else:
                content += f"- {features}\n"
                
        # Add description if available
        description = property_data.get("description")
        if description:
            content += f"\nDescription:\n{description}\n"
            
        content += f"\nSource: {property_data.get('source', 'unknown')}"
        source_url = property_data.get("source_url", property_data.get("sourceUrl"))
        if source_url:
            content += f"\nSource URL: {source_url}"
            
        return content
    
    def _format_property_location_content(self, location_data: Dict[str, Any]) -> str:
        """Format property location content for better readability."""
        address = location_data.get("formatted_address", "Unknown Address")
        
        content = f"""Property Location: {address}

Address Details:
"""
        
        if "address_line1" in location_data and location_data["address_line1"]:
            content += f"- Street Address: {location_data['address_line1']}\n"
            
        if "address_line2" in location_data and location_data["address_line2"]:
            content += f"- Unit/Suite: {location_data['address_line2']}\n"
            
        if "city" in location_data and location_data["city"]:
            content += f"- City: {location_data['city']}\n"
            
        if "state" in location_data and location_data["state"]:
            content += f"- State: {location_data['state']}\n"
            
        if "zip_code" in location_data and location_data["zip_code"]:
            content += f"- ZIP Code: {location_data['zip_code']}\n"
            
        if "county" in location_data and location_data["county"]:
            content += f"- County: {location_data['county']}\n"
            
        # Add coordinates if available
        if "latitude" in location_data and "longitude" in location_data:
            if location_data["latitude"] is not None and location_data["longitude"] is not None:
                content += f"\nCoordinates: {location_data['latitude']}, {location_data['longitude']}\n"
                
        # Add neighborhood information if available
        if "neighborhood" in location_data and location_data["neighborhood"]:
            content += f"\nNeighborhood: {location_data['neighborhood']}\n"
            
        # Add school district information if available
        if "school_district" in location_data and location_data["school_district"]:
            content += f"\nSchool District: {location_data['school_district']}\n"
            
        return content
    
    def _format_property_agent_content(self, agent_data: Dict[str, Any]) -> str:
        """Format property agent and office content for better readability."""
        address = agent_data.get("address", "Unknown Address")
        listing_agent = agent_data.get("listing_agent", {})
        listing_office = agent_data.get("listing_office", {})
        
        content = f"""Listing Information for Property: {address}

"""
        
        # Add agent information if available
        if listing_agent:
            content += "Listing Agent:\n"
            
            if isinstance(listing_agent, dict):
                if "name" in listing_agent:
                    content += f"- Name: {listing_agent['name']}\n"
                    
                if "phone" in listing_agent:
                    content += f"- Phone: {listing_agent['phone']}\n"
                    
                if "email" in listing_agent:
                    content += f"- Email: {listing_agent['email']}\n"
                    
                if "license" in listing_agent:
                    content += f"- License #: {listing_agent['license']}\n"
            else:
                content += f"- {listing_agent}\n"
                
        # Add office information if available
        if listing_office:
            content += "\nListing Office:\n"
            
            if isinstance(listing_office, dict):
                if "name" in listing_office:
                    content += f"- Name: {listing_office['name']}\n"
                    
                if "phone" in listing_office:
                    content += f"- Phone: {listing_office['phone']}\n"
                    
                if "address" in listing_office:
                    content += f"- Address: {listing_office['address']}\n"
                    
                if "website" in listing_office:
                    content += f"- Website: {listing_office['website']}\n"
            else:
                content += f"- {listing_office}\n"
                
        return content
    
    def _format_property_history_content(self, history_data: Dict[str, Any]) -> str:
        """Format property history content for better readability."""
        address = history_data.get("address", "Unknown Address")
        history = history_data.get("history", {})
        
        content = f"""Property History for: {address}

"""
        
        if not history:
            content += "No history available."
            return content
            
        # Handle different history formats
        if isinstance(history, dict):
            # Sort by date if possible
            try:
                sorted_dates = sorted(history.keys(), reverse=True)
                
                for date in sorted_dates:
                    event = history[date]
                    content += f"Date: {date}\n"
                    
                    if isinstance(event, dict):
                        # Extract key information from the event
                        event_type = event.get("event", "Unknown Event")
                        price = event.get("price", None)
                        listing_type = event.get("listingType", None)
                        listed_date = event.get("listedDate", None)
                        removed_date = event.get("removedDate", None)
                        days_on_market = event.get("daysOnMarket", None)
                        
                        content += f"- Event: {event_type}\n"
                        
                        if price is not None:
                            content += f"- Price: ${price:,.2f}\n"
                            
                        if listing_type:
                            content += f"- Listing Type: {listing_type}\n"
                            
                        if listed_date:
                            content += f"- Listed Date: {listed_date}\n"
                            
                        if removed_date:
                            content += f"- Removed Date: {removed_date}\n"
                            
                        if days_on_market is not None:
                            content += f"- Days on Market: {days_on_market}\n"
                            
                        # Add any other fields that might be present
                        for key, value in event.items():
                            if key not in ["event", "price", "listingType", "listedDate", "removedDate", "daysOnMarket"]:
                                formatted_key = key.replace("_", " ").title()
                                
                                # Format the value based on its type
                                if isinstance(value, (int, float)) and "price" in key.lower():
                                    formatted_value = f"${value:,.2f}"
                                else:
                                    formatted_value = str(value)
                                    
                                content += f"- {formatted_key}: {formatted_value}\n"
                    else:
                        content += f"- Event: {event}\n"
                        
                    content += "\n"
            except Exception:
                # If sorting fails, just iterate through the dictionary
                for date, event in history.items():
                    content += f"Date: {date}\n"
                    
                    if isinstance(event, dict):
                        for key, value in event.items():
                            formatted_key = key.replace("_", " ").title()
                            content += f"- {formatted_key}: {value}\n"
                    else:
                        content += f"- Event: {event}\n"
                        
                    content += "\n"
        elif isinstance(history, list):
            for event in history:
                if isinstance(event, dict):
                    if "date" in event:
                        content += f"Date: {event['date']}\n"
                    elif "timestamp" in event:
                        content += f"Date: {event['timestamp']}\n"
                    else:
                        content += "Date: Unknown\n"
                        
                    for key, value in event.items():
                        if key not in ["date", "timestamp"]:
                            formatted_key = key.replace("_", " ").title()
                            
                            # Format the value based on its type
                            if isinstance(value, (int, float)) and "price" in key.lower():
                                formatted_value = f"${value:,.2f}"
                            else:
                                formatted_value = str(value)
                                
                            content += f"- {formatted_key}: {formatted_value}\n"
                    
                    content += "\n"
                else:
                    content += f"- {event}\n\n"
        
        return content
    
    def _enrich_property_metadata(self, metadata: Dict[str, Any], property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata for property listing chunk."""
        # Add timestamp
        metadata["chunk_created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Add price range
        price = property_data.get("price")
        if price is not None:
            metadata["price_range"] = self._get_price_range(price)
        
        # Add size range
        square_footage = property_data.get("square_footage", property_data.get("squareFootage"))
        if square_footage is not None:
            metadata["size_range"] = self._get_size_range(square_footage)
        
        # Add property age
        current_year = datetime.now().year
        year_built = property_data.get("year_built", property_data.get("yearBuilt"))
        if year_built is not None:
            try:
                year_built = int(year_built)
                age = current_year - year_built
                metadata["property_age"] = age
                metadata["age_range"] = self._get_age_range(age)
            except (ValueError, TypeError):
                pass
        
        # Extract tags from features
        tags = set()
        features = property_data.get("features")
        if features:
            if isinstance(features, list):
                for feature in features:
                    if isinstance(feature, str):
                        tags.update(self._extract_keywords_from_text(feature))
            elif isinstance(features, dict):
                for category, items in features.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                tags.update(self._extract_keywords_from_text(item))
                    elif isinstance(items, str):
                        tags.update(self._extract_keywords_from_text(items))
        
        # Extract keywords from description
        keywords = []
        description = property_data.get("description")
        if description:
            keywords = self._extract_keywords_from_description(description)
        
        # Add tags and keywords to metadata
        if tags:
            metadata["tags"] = list(tags)
        if keywords:
            metadata["keywords"] = keywords
        
        return metadata
    
    def _enrich_location_metadata(self, metadata: Dict[str, Any], location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata for location chunk."""
        # Add timestamp
        metadata["chunk_created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Add ZIP code
        if "zip_code" in location_data and location_data["zip_code"]:
            metadata["zip_code"] = location_data["zip_code"]
        
        # Add county
        if "county" in location_data and location_data["county"]:
            metadata["county"] = location_data["county"]
        
        # Add neighborhood
        if "neighborhood" in location_data and location_data["neighborhood"]:
            metadata["neighborhood"] = location_data["neighborhood"]
        
        # Add school district
        if "school_district" in location_data and location_data["school_district"]:
            metadata["school_district"] = location_data["school_district"]
        
        # Extract keywords from location data
        keywords = []
        for field in ["neighborhood", "school_district", "county"]:
            if field in location_data and location_data[field]:
                keywords.extend(self._extract_keywords_from_text(str(location_data[field])))
        
        if keywords:
            metadata["keywords"] = list(set(keywords))
        
        return metadata
    
    def _enrich_agent_metadata(self, metadata: Dict[str, Any], agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata for agent chunk."""
        # Add timestamp
        metadata["chunk_created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Add agent information
        listing_agent = agent_data.get("listing_agent", {})
        if isinstance(listing_agent, dict):
            if "phone" in listing_agent:
                metadata["agent_phone"] = listing_agent["phone"]
            if "email" in listing_agent:
                metadata["agent_email"] = listing_agent["email"]
            if "license" in listing_agent:
                metadata["agent_license"] = listing_agent["license"]
        
        # Add office information
        listing_office = agent_data.get("listing_office", {})
        if isinstance(listing_office, dict):
            if "name" in listing_office:
                metadata["office_name"] = listing_office["name"]
            if "phone" in listing_office:
                metadata["office_phone"] = listing_office["phone"]
            if "address" in listing_office:
                metadata["office_address"] = listing_office["address"]
        
        # Extract keywords
        keywords = []
        if isinstance(listing_agent, dict) and "name" in listing_agent:
            keywords.extend(self._extract_keywords_from_text(listing_agent["name"]))
        if isinstance(listing_office, dict) and "name" in listing_office:
            keywords.extend(self._extract_keywords_from_text(listing_office["name"]))
        
        if keywords:
            metadata["keywords"] = list(set(keywords))
        
        return metadata
    
    def _enrich_property_history_metadata(self, metadata: Dict[str, Any], history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata for property history chunk."""
        # Add timestamp
        metadata["chunk_created_at"] = datetime.now(timezone.utc).isoformat()
        
        history = history_data.get("history", {})
        
        # Extract event types
        event_types = set()
        dates = []
        prices = []
        
        if isinstance(history, dict):
            for date, event in history.items():
                dates.append(date)
                
                if isinstance(event, dict):
                    # Extract event type
                    event_type = event.get("event_type") or event.get("type") or event.get("event")
                    if event_type:
                        event_types.add(event_type.lower())
                    
                    # Extract price
                    if "price" in event and isinstance(event["price"], (int, float)):
                        prices.append(event["price"])
        elif isinstance(history, list):
            for event in history:
                if isinstance(event, dict):
                    # Extract date
                    date = event.get("date") or event.get("timestamp")
                    if date:
                        dates.append(date)
                    
                    # Extract event type
                    event_type = event.get("event_type") or event.get("type") or event.get("event")
                    if event_type:
                        event_types.add(event_type.lower())
                    
                    # Extract price
                    if "price" in event and isinstance(event["price"], (int, float)):
                        prices.append(event["price"])
        
        # Add event types to metadata
        if event_types:
            metadata["event_types"] = list(event_types)
        
        # Add date range
        if dates:
            try:
                metadata["date_range_start"] = min(dates)
                metadata["date_range_end"] = max(dates)
                metadata["date_range_count"] = len(dates)
            except Exception:
                pass
        
        # Add price range
        if prices:
            metadata["price_change_min"] = min(prices)
            metadata["price_change_max"] = max(prices)
            
            # Determine price trend
            if len(prices) > 1:
                first_price = prices[0]
                last_price = prices[-1]
                if last_price > first_price:
                    metadata["price_trend"] = "up"
                elif last_price < first_price:
                    metadata["price_trend"] = "down"
                else:
                    metadata["price_trend"] = "stable"
        
        # Extract keywords
        keywords = []
        for event_type in event_types:
            keywords.extend(self._extract_keywords_from_text(event_type))
        
        if keywords:
            metadata["keywords"] = list(set(keywords))
        
        return metadata
    
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract important keywords from a property description."""
        if not description:
            return []
        
        # Convert to lowercase
        text = description.lower()
        
        # Find all words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter for common property features
        keywords = [word for word in words if word in self.common_features]
        
        # Add bigrams and trigrams for common phrases
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if any(feature in bigram for feature in self.common_features):
                bigrams.append(bigram)
        
        # Combine and deduplicate
        all_keywords = list(set(keywords + bigrams))
        
        return all_keywords[:20]  # Limit to 20 keywords
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Find all words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out short words and common stop words
        stop_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "hers"}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _get_price_range(self, price: float) -> str:
        """Get price range bucket for a given price."""
        if price < 100000:
            return "under_100k"
        elif price < 250000:
            return "100k_250k"
        elif price < 500000:
            return "250k_500k"
        elif price < 750000:
            return "500k_750k"
        elif price < 1000000:
            return "750k_1m"
        elif price < 2000000:
            return "1m_2m"
        else:
            return "over_2m"
    
    def _get_size_range(self, size: float) -> str:
        """Get size range bucket for a given square footage."""
        if size < 1000:
            return "under_1000"
        elif size < 1500:
            return "1000_1500"
        elif size < 2000:
            return "1500_2000"
        elif size < 2500:
            return "2000_2500"
        elif size < 3000:
            return "2500_3000"
        elif size < 4000:
            return "3000_4000"
        else:
            return "over_4000"
    
    def _get_age_range(self, age: int) -> str:
        """Get age range bucket for a given property age in years."""
        if age <= 1:
            return "new_construction"
        elif age <= 5:
            return "nearly_new"
        elif age <= 10:
            return "recent"
        elif age <= 20:
            return "established"
        elif age <= 50:
            return "mature"
        else:
            return "historic"