"""
Enhanced Ingestion Pipeline with JSON Chunking for TrackRealties AI Platform.

This module provides a comprehensive pipeline that handles all aspects of data ingestion,
from chunking JSON data semantically to generating embeddings and building a knowledge graph.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime
import os
import uuid
from dataclasses import dataclass, field

# Add this import at the top
from ..data.embedding.optimized_embedder import OptimizedEmbeddingPipeline, NeonDBEmbeddingManager
from ..core.config import get_settings, Settings
from .chunking.json_chunker import JSONChunker
from .chunking.chunk import Chunk
from .embedding.openai_embedder import OpenAIEmbedder
from .database_integration import DatabaseIntegration
from .graph.graph_builder import GraphBuilder
from .error_logging import log_error

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    total: int
    processed: int
    failed: int
    chunks_created: int
    embeddings_generated: int
    graph_nodes_created: int
    errors: List[str] = field(default_factory=list)


class EnhancedIngestionPipeline:
    """
    Enhanced ingestion pipeline with JSON chunking, embedding generation, and knowledge graph integration.
    
    This class orchestrates the entire ingestion process, from chunking JSON data to generating
    embeddings and building a knowledge graph.
    """
    
    def __init__(self,
                 batch_size: Optional[int] = None,
                 skip_embeddings: bool = False,
                 skip_graph: bool = False,
                 settings: Optional[Settings] = None):
        """
        Initialize the EnhancedIngestionPipeline.
        
        Args:
            batch_size: Number of records to process in a batch
            skip_embeddings: Whether to skip embedding generation
            skip_graph: Whether to skip graph building
            settings: Application settings
        """
                # Use optimized embedding pipeline
        self.optimized_embedder = OptimizedEmbeddingPipeline(
            batch_size=batch_size or 50,
            max_concurrent_batches=3,
            enable_cache_warming=True
        )
        
        # Use enhanced NeonDB manager
        self.neon_manager = NeonDBEmbeddingManager(os.environ.get("DATABASE_URL"))
        self.logger = logging.getLogger(__name__)
        self.settings = settings or get_settings()
        self.batch_size = batch_size or self.settings.ingestion_batch_size
        self.skip_embeddings = skip_embeddings
        self.skip_graph = skip_graph
        
        # Initialize components
        self.chunker = None
        self.embedder = None
        self.db_integration = None
        self.graph_builder = None
        
        # Track initialization status
        self.initialized = False


    
    async def initialize(self, dry_run: bool = False) -> None:
        """Initialize all components of the pipeline."""
        try:
            # Initialize JSON chunker
            self.chunker = JSONChunker(
                max_chunk_size=self.settings.max_chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            self.logger.info("Initialized JSON chunker")
            
            if not dry_run:
                if not self.skip_embeddings:
                    self.embedder = OpenAIEmbedder(
                        model=self.settings.embedding_model,
                        dimensions=self.settings.embedding_dimensions,
                        batch_size=self.settings.embedding_batch_size,
                        use_cache=True,
                        api_key=self.settings.embedding_api_key or self.settings.llm_api_key
                    )
                    await self.embedder.initialize()
                    self.logger.info(f"Initialized embedder with model {self.settings.embedding_model}")
                
                # Initialize database integration
                self.db_integration = DatabaseIntegration()
                await self.db_integration.initialize()
                self.logger.info("Initialized database integration")
                
                # Initialize graph builder - NO PARAMETERS NEEDED!
                if not self.skip_graph:
                    self.graph_builder = GraphBuilder()  # Fixed: no parameters
                    await self.graph_builder.initialize()
                    self.logger.info("Initialized graph builder")
            
            self.initialized = True
            self.logger.info("Enhanced ingestion pipeline initialized successfully")
            
                    # Initialize optimized components
            await self.optimized_embedder.initialize()
            await self.neon_manager.initialize()
            
            # Optimize database for vector search
            await self.neon_manager.optimize_vector_search_performance()
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced ingestion pipeline: {e}")
            raise
    
    async def ingest_market_data(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest market data with proper chunking, embedding, and graph building.
        
        Args:
            source: Source of the data (e.g., 'zillow', 'redfin')
            data: List of market data records
            
        Returns:
            IngestionResult with details of the ingestion
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch)} records)")
            
            for record in batch:
                try:
                    # Chunk the record
                    chunks = self.chunker.chunk_json(record, "market")
                    result.chunks_created += len(chunks)
                    
                    # Generate embeddings
                    if not self.skip_embeddings and self.embedder:
                        chunk_texts = [chunk.content for chunk in chunks]
                        embeddings, token_counts = await self.embedder.generate_embeddings_batch(chunk_texts)
                        # Update chunks with embeddings
                        for chunk, embedding in zip(chunks, embeddings):
                            chunk.embedding = embedding
                        result.embeddings_generated += len(embeddings)
                    
                    # Save to database
                    db_result = await self.db_integration.save_market_data_to_database(
                        record, chunks
                    )
                    
                    if db_result.get("success"):
                        result.processed += 1
                        
                        # Add to knowledge graph
                        if not self.skip_graph and self.graph_builder:
                            graph_result = await self.graph_builder.add_market_data_to_graph(record)
                            if graph_result.get("nodes_created"):
                                result.graph_nodes_created += graph_result["nodes_created"]
                    else:
                        result.failed += 1
                        result.errors.append(f"Failed to save record: {db_result.get('error')}")
                        
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"Error processing record: {str(e)}")
                    self.logger.error(f"Error processing record from {source}: {str(e)}")
        
        self.logger.info(
            f"Market data ingestion complete: {result.processed}/{result.total} processed, "
            f"{result.chunks_created} chunks, {result.embeddings_generated} embeddings, "
            f"{result.graph_nodes_created} graph nodes"
        )
        
        return result
    
    async def ingest_property_listings(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest property listings with proper chunking, embedding, and graph building.
        
        Args:
            source: Source of the data (e.g., 'mls', 'zillow')
            data: List of property listing records
            
        Returns:
            IngestionResult with details of the ingestion
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch)} records)")
            
            for record in batch:
                try:
                    # Chunk the record
                    chunks = self.chunker.chunk_json(record, "property")
                    result.chunks_created += len(chunks)
                    
                    # Generate embeddings
                    if not self.skip_embeddings and self.embedder:
                        chunk_texts = [chunk.content for chunk in chunks]
                        embeddings, token_counts = await self.embedder.generate_embeddings_batch(chunk_texts)
                        # Update chunks with embeddings
                        for chunk, embedding in zip(chunks, embeddings):
                            chunk.embedding = embedding
                        result.embeddings_generated += len(embeddings)
                    
                    # Save to database
                    db_result = await self.db_integration.save_property_to_database(
                        record, chunks
                    )
                    
                    if db_result.get("success"):
                        result.processed += 1
                        
                        # Add to knowledge graph
                        if not self.skip_graph and self.graph_builder:
                            graph_result = await self.graph_builder.add_property_to_graph(record)
                            if graph_result.get("nodes_created"):
                                result.graph_nodes_created += graph_result["nodes_created"]
                    else:
                        result.failed += 1
                        result.errors.append(f"Failed to save record: {db_result.get('error')}")
                        
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"Error processing record: {str(e)}")
                    self.logger.error(f"Error processing record from {source}: {str(e)}")
        
        self.logger.info(
            f"Property listings ingestion complete: {result.processed}/{result.total} processed, "
            f"{result.chunks_created} chunks, {result.embeddings_generated} embeddings, "
            f"{result.graph_nodes_created} graph nodes"
        )
        
        return result
    
    async def validate_market_data(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate market data without saving to database.
        
        Args:
            source: Source of the data
            data: List of market data records
            
        Returns:
            Validation results including errors and warnings
        """
        validation_results = {
            "total": len(data),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": []
        }
        
        required_fields = self.settings.VALIDATION_REQUIRED_FIELDS_MARKET
        
        for i, record in enumerate(data):
            errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate data types and formats
            if "date" in record:
                try:
                    datetime.fromisoformat(record["date"].replace('Z', '+00:00'))
                except:
                    errors.append("Invalid date format")
            
            if "median_price" in record:
                try:
                    float(record["median_price"])
                except:
                    errors.append("median_price must be numeric")
            
            if errors:
                validation_results["invalid"] += 1
                validation_results["errors"].append({
                    "record_index": i,
                    "errors": errors
                })
            else:
                validation_results["valid"] += 1
        
        return validation_results
    
    async def validate_property_listings(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate property listings without saving to database.
        
        Args:
            source: Source of the data
            data: List of property listing records
            
        Returns:
            Validation results including errors and warnings
        """
        validation_results = {
            "total": len(data),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": []
        }
        
        required_fields = self.settings.VALIDATION_REQUIRED_FIELDS_PROPERTY
        
        for i, record in enumerate(data):
            errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate data types and formats
            if "price" in record:
                try:
                    float(record["price"])
                except:
                    errors.append("price must be numeric")
            
            if "bedrooms" in record:
                try:
                    int(record["bedrooms"])
                except:
                    errors.append("bedrooms must be integer")
            
            if errors:
                validation_results["invalid"] += 1
                validation_results["errors"].append({
                    "record_index": i,
                    "errors": errors
                })
            else:
                validation_results["valid"] += 1
                
                # Add warnings for data quality
                if "description" in record and len(record.get("description", "")) < 50:
                    validation_results["warnings"].append({
                        "record_index": i,
                        "warning": "Short description may affect search quality"
                    })
        
        return validation_results
    
    async def close(self) -> None:
        """Close all pipeline components and clean up resources."""
        if self.embedder:
            await self.embedder.close()
        
        if self.graph_builder and hasattr(self.graph_builder, 'close'):
            await self.graph_builder.close()
        
        self.logger.info("Enhanced ingestion pipeline closed")