"""
Enhanced Ingestion Pipeline with JSON Chunking for TrackRealties AI Platform.

This module provides a comprehensive pipeline that handles all aspects of data ingestion,
from chunking JSON data semantically to generating embeddings and building a knowledge graph.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from ..core.config import get_settings, Settings
from ..core.config import get_settings
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
                
                # Initialize graph builder
                if not self.skip_graph:
                    self.graph_builder = GraphBuilder(
                        uri=self.settings.neo4j_uri,
                        user=self.settings.neo4j_user,
                        password=self.settings.neo4j_password,
                        database=self.settings.neo4j_database
                    )
                    await self.graph_builder.initialize()
                    self.logger.info("Initialized graph builder")
            
            self.initialized = True
            self.logger.info("Enhanced ingestion pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced ingestion pipeline: {e}")
            raise
    
    async def ingest_market_data(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest market data with proper chunking, embedding, and graph building.
        
        Args:
            source: Source of the data
            data: List of market data records
            
        Returns:
            IngestionResult with statistics
        """
        if not self.initialized:
            await self.initialize()
        
        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        self.logger.info(f"Starting ingestion of {len(data)} market data records from {source}")
        
        # Process data in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = await self._process_market_data_batch(source, batch)
            
            # Update result
            result.processed += batch_result.processed
            result.failed += batch_result.failed
            result.chunks_created += batch_result.chunks_created
            result.embeddings_generated += batch_result.embeddings_generated
            result.graph_nodes_created += batch_result.graph_nodes_created
            result.errors.extend(batch_result.errors)
            
            self.logger.info(
                f"Processed batch {i//self.batch_size + 1}: "
                f"{batch_result.processed}/{len(batch)} records, "
                f"{batch_result.chunks_created} chunks, "
                f"{batch_result.embeddings_generated} embeddings, "
                f"{batch_result.graph_nodes_created} graph nodes"
            )
        
        self.logger.info(
            f"Completed ingestion of market data: "
            f"{result.processed}/{result.total} records processed, "
            f"{result.failed} failed, "
            f"{result.chunks_created} chunks created, "
            f"{result.embeddings_generated} embeddings generated, "
            f"{result.graph_nodes_created} graph nodes created"
        )
        
        return result
    
    async def ingest_property_listings(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest property listings with proper chunking, embedding, and graph building.
        
        Args:
            source: Source of the data
            data: List of property listing records
            
        Returns:
            IngestionResult with statistics
        """
        if not self.initialized:
            await self.initialize()
        
        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        self.logger.info(f"Starting ingestion of {len(data)} property listings from {source}")
        
        # Process data in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = await self._process_property_listings_batch(source, batch)
            
            # Update result
            result.processed += batch_result.processed
            result.failed += batch_result.failed
            result.chunks_created += batch_result.chunks_created
            result.embeddings_generated += batch_result.embeddings_generated
            result.graph_nodes_created += batch_result.graph_nodes_created
            result.errors.extend(batch_result.errors)
            
            self.logger.info(
                f"Processed batch {i//self.batch_size + 1}: "
                f"{batch_result.processed}/{len(batch)} records, "
                f"{batch_result.chunks_created} chunks, "
                f"{batch_result.embeddings_generated} embeddings, "
                f"{batch_result.graph_nodes_created} graph nodes"
            )
        
        self.logger.info(
            f"Completed ingestion of property listings: "
            f"{result.processed}/{result.total} records processed, "
            f"{result.failed} failed, "
            f"{result.chunks_created} chunks created, "
            f"{result.embeddings_generated} embeddings generated, "
            f"{result.graph_nodes_created} graph nodes created"
        )
        
        return result
    
    async def _process_market_data_batch(self, source: str, batch: List[Dict[str, Any]]) -> IngestionResult:
        """
        Process a batch of market data records.
        
        Args:
            source: Source of the data
            batch: List of market data records
            
        Returns:
            IngestionResult with statistics for this batch
        """
        result = IngestionResult(
            total=len(batch),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        for record in batch:
            try:
                # Add source if not present
                if "source" not in record:
                    record["source"] = source
                
                # Step 1: Chunk the market data
                chunks = self.chunker.chunk_market_data(record)
                result.chunks_created += len(chunks)
                
                # Step 2: Generate embeddings for chunks
                if not self.skip_embeddings:
                    self.logger.info(f"Generating embeddings for {len(chunks)} chunks for record {record.get('region_id', 'N/A')}")
                    await self._generate_embeddings_for_chunks(chunks)
                    result.embeddings_generated += len(chunks)
                
                # Step 3: Save market data and chunks to database
                db_result = await self.db_integration.save_market_data_to_database(record, chunks)
                
                if db_result.get("success", False):
                    # Step 4: Add market data to knowledge graph
                    if not self.skip_graph:
                        self.logger.info(f"Building graph for market data record {record.get('region_id', 'N/A')}")
                        graph_result = await self.graph_builder.add_market_data_to_graph(record)
                        
                        if graph_result.get("success", False):
                            result.graph_nodes_created += graph_result.get("nodes_created", 0)
                            result.processed += 1
                        else:
                            # Log graph error but consider the record processed
                            error_msg = f"Failed to add market data to graph: {graph_result.get('error', 'Unknown error')}"
                            self.logger.warning(error_msg)
                            result.errors.append(error_msg)
                            result.processed += 1
                    else:
                        result.processed += 1
                else:
                    # Database error
                    error_msg = f"Failed to save market data to database: {db_result.get('error', 'Unknown error')}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.failed += 1
                
            except Exception as e:
                error_msg = f"Failed to process market data record: {e}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
                result.failed += 1
                
                # Log detailed error
                log_error(e, {"source": source, "data_type": "market_data"})
        
        return result
    
    async def _process_property_listings_batch(self, source: str, batch: List[Dict[str, Any]]) -> IngestionResult:
        """
        Process a batch of property listing records.
        
        Args:
            source: Source of the data
            batch: List of property listing records
            
        Returns:
            IngestionResult with statistics for this batch
        """
        result = IngestionResult(
            total=len(batch),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0
        )
        
        for record in batch:
            try:
                # Add source if not present
                if "source" not in record:
                    record["source"] = source
                
                # Step 1: Chunk the property listing
                chunks = self.chunker.chunk_property_listing(record)
                result.chunks_created += len(chunks)
                
                # Step 2: Generate embeddings for chunks
                if not self.skip_embeddings:
                    self.logger.info(f"Generating embeddings for {len(chunks)} chunks for property {record.get('id', 'N/A')}")
                    await self._generate_embeddings_for_chunks(chunks)
                    result.embeddings_generated += len(chunks)
                
                # Step 3: Save property listing and chunks to database
                db_result = await self.db_integration.save_property_to_database(record, chunks)
                
                if db_result.get("success", False):
                    # Step 4: Add property listing to knowledge graph
                    if not self.skip_graph:
                        self.logger.info(f"Building graph for property record {record.get('id', 'N/A')}")
                        graph_result = await self.graph_builder.add_property_to_graph(record)
                        
                        if graph_result.get("success", False):
                            result.graph_nodes_created += graph_result.get("nodes_created", 0)
                            result.processed += 1
                        else:
                            # Log graph error but consider the record processed
                            error_msg = f"Failed to add property listing to graph: {graph_result.get('error', 'Unknown error')}"
                            self.logger.warning(error_msg)
                            result.errors.append(error_msg)
                            result.processed += 1
                    else:
                        result.processed += 1
                else:
                    # Database error
                    error_msg = f"Failed to save property listing to database: {db_result.get('error', 'Unknown error')}"
                    self.logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.failed += 1
                
            except Exception as e:
                error_msg = f"Failed to process property listing record: {e}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
                result.failed += 1
                
                # Log detailed error
                log_error(e, {"source": source, "data_type": "property_listing"})
        
        return result
    
    async def _generate_embeddings_for_chunks(self, chunks: List[Chunk]) -> None:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to generate embeddings for
        """
        if not chunks:
            return
        
        # Extract content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings, token_counts = await self.embedder.generate_embeddings_batch(texts)
        
        # Assign embeddings and token counts to chunks
        for chunk, embedding, token_count in zip(chunks, embeddings, token_counts):
            chunk.embedding = embedding
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["token_count"] = token_count
            
    async def validate_market_data(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate market data without saving to database.
        
        Args:
            source: Source of the data
            data: List of market data records
            
        Returns:
            Dictionary with validation results
        """
        if not self.initialized:
            await self.initialize()
        
        result = {
            "total": len(data),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": [],
            "chunking_stats": {
                "total_chunks": 0,
                "chunks_per_record": {}
            }
        }
        
        self.logger.info(f"Validating {len(data)} market data records from {source}")
        
        for i, record in enumerate(data):
            try:
                # Add source if not present
                if "source" not in record:
                    record["source"] = source
                
                # Validate required fields
                required_fields = self.settings.validation_required_fields_market
                missing_fields = [field for field in required_fields if field not in record]
                
                if missing_fields:
                    error_msg = f"Record {i}: Missing required fields: {', '.join(missing_fields)}"
                    self.logger.warning(error_msg)
                    result["errors"].append(error_msg)
                    result["invalid"] += 1
                    continue
                
                # Test chunking
                chunks = self.chunker.chunk_market_data(record)
                result["chunking_stats"]["total_chunks"] += len(chunks)
                result["chunking_stats"]["chunks_per_record"][i] = len(chunks)
                
                # Record is valid
                result["valid"] += 1
                
            except Exception as e:
                error_msg = f"Record {i}: Validation failed: {e}"
                self.logger.error(error_msg)
                result["errors"].append(error_msg)
                result["invalid"] += 1
        
        # Calculate success rate
        result["success_rate"] = result["valid"] / result["total"] if result["total"] > 0 else 0
        
        self.logger.info(
            f"Completed validation of market data: "
            f"{result['valid']}/{result['total']} records valid, "
            f"{result['invalid']} invalid, "
            f"{result['chunking_stats']['total_chunks']} total chunks"
        )
        
        return result
    
    async def validate_property_listings(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate property listings without saving to database.
        
        Args:
            source: Source of the data
            data: List of property listing records
            
        Returns:
            Dictionary with validation results
        """
        if not self.initialized:
            await self.initialize()
        
        result = {
            "total": len(data),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": [],
            "chunking_stats": {
                "total_chunks": 0,
                "chunks_per_record": {}
            }
        }
        
        self.logger.info(f"Validating {len(data)} property listings from {source}")
        
        for i, record in enumerate(data):
            try:
                # Add source if not present
                if "source" not in record:
                    record["source"] = source
                
                # Validate required fields
                required_fields = self.settings.validation_required_fields_property
                missing_fields = [field for field in required_fields if field not in record]
                
                if missing_fields:
                    error_msg = f"Record {i}: Missing required fields: {', '.join(missing_fields)}"
                    self.logger.warning(error_msg)
                    result["errors"].append(error_msg)
                    result["invalid"] += 1
                    continue
                
                # Test chunking
                chunks = self.chunker.chunk_property_listing(record)
                result["chunking_stats"]["total_chunks"] += len(chunks)
                result["chunking_stats"]["chunks_per_record"][i] = len(chunks)
                
                # Record is valid
                result["valid"] += 1
                
            except Exception as e:
                error_msg = f"Record {i}: Validation failed: {e}"
                self.logger.error(error_msg)
                result["errors"].append(error_msg)
                result["invalid"] += 1
        
        # Calculate success rate
        result["success_rate"] = result["valid"] / result["total"] if result["total"] > 0 else 0
        
        self.logger.info(
            f"Completed validation of property listings: "
            f"{result['valid']}/{result['total']} records valid, "
            f"{result['invalid']} invalid, "
            f"{result['chunking_stats']['total_chunks']} total chunks"
        )
        
        return result