
"""
Entity extraction for the RAG module.
"""
import logging
from typing import List, Dict, Any
import re

class EntityExtractor:
    """
    Extracts entities from text using a simple keyword-based approach.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        # A more robust implementation would use a larger list of locations
        # or a more sophisticated location detection algorithm.
        self.known_locations = [
            "Los Angeles County, CA",
            "Dallas, TX",
            "New York, NY",
            "San Francisco, CA",
            "Chicago, IL",
            "Houston, TX",
            "Phoenix, AZ",
            "Philadelphia, PA",
            "San Antonio, TX",
            "San Diego, CA",
        ]

    async def initialize(self):
        """Initialize the entity extractor."""
        self.initialized = True
        self.logger.info("Entity extractor initialized.")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from text by looking for known locations.
        """
        if not self.initialized:
            await self.initialize()

        self.logger.info(f"Extracting entities from: {text}")

        extracted_entities = []
        for location in self.known_locations:
            if re.search(r'\b' + re.escape(location) + r'\b', text, re.IGNORECASE):
                extracted_entities.append({"name": location, "type": "LOCATION"})

        return extracted_entities
