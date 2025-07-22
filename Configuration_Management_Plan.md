# Configuration Management Plan for Enhanced Ingestion Pipeline

## Overview

This document outlines the implementation plan for updating the configuration management in the Enhanced Ingestion Pipeline. The goal is to ensure all components use centralized configuration settings and to add validation for required parameters.

## Current Status

Currently, the configuration settings are defined in the `config.py` file and accessed through the `get_settings()` function. However, there are several issues that need to be addressed:

1. Not all components use centralized configuration
2. Some settings are hardcoded in the components
3. There's no validation for required parameters
4. Configuration for new features (embedding, graph, chunking) may not be fully defined

## Implementation Plan

### 1. Update Core Configuration Class

First, we need to update the `Settings` class in `core/config.py` to include all necessary settings for the enhanced ingestion pipeline:

```python
class Settings:
    """Application settings."""
    
    # Database settings
    postgres_uri: str = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/trackrealties")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "trackrealties")
    
    # Neo4j settings
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "./cache/embeddings")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    
    # Chunking settings
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Graph settings
    graph_batch_size: int = int(os.getenv("GRAPH_BATCH_SIZE", "50"))
    
    # Ingestion settings
    ingestion_batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "100"))
    
    # Validation settings
    validation_required_fields_market: List[str] = ["region_id", "region_name", "date", "median_price"]
    validation_required_fields_property: List[str] = ["property_id", "price", "status", "property_type"]
    
    # LLM settings (for future use)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    def validate(self):
        """Validate required settings."""
        errors = []
        
        # Validate database settings
        if not self.postgres_uri and (not self.postgres_host or not self.postgres_db):
            errors.append("Either POSTGRES_URI or POSTGRES_HOST and POSTGRES_DB must be provided")
        
        # Validate Neo4j settings
        if not self.neo4j_uri:
            errors.append("NEO4J_URI must be provided")
        
        # Validate embedding settings
        if not self.embedding_api_key and not self.llm_api_key:
            errors.append("Either EMBEDDING_API_KEY or LLM_API_KEY must be provided for embedding generation")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
```

### 2. Update get_settings Function

Update the `get_settings` function to validate settings when requested:

```python
def get_settings(validate: bool = False) -> Settings:
    """
    Get application settings.
    
    Args:
        validate: Whether to validate required settings
        
    Returns:
        Settings instance
    """
    settings = Settings()
    
    if validate:
        settings.validate()
    
    return settings
```

### 3. Update EnhancedIngestionPipeline to Use Centralized Configuration

Modify the `EnhancedIngestionPipeline` class to use centralized configuration:

```python
class EnhancedIngestionPipeline:
    """
    Enhanced ingestion pipeline with JSON chunking, embedding generation, and knowledge graph integration.
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
        self.settings = settings or get_settings(validate=True)
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
    
    async def initialize(self) -> None:
        """Initialize all components of the pipeline."""
        try:
            # Initialize JSON chunker
            self.chunker = JSONChunker(
                max_chunk_size=self.settings.max_chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            self.logger.info("Initialized JSON chunker")
            
            # Initialize embedder (if not skipped)
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
            else:
                self.logger.info("Skipping embedder initialization as requested")
            
            # Initialize database integration
            self.db_integration = DatabaseIntegration()
            await self.db_integration.initialize()
            self.logger.info("Initialized database integration")
            
            # Initialize graph builder (if not skipped)
            if not self.skip_graph:
                self.graph_builder = GraphBuilder(
                    uri=self.settings.neo4j_uri,
                    user=self.settings.neo4j_user,
                    password=self.settings.neo4j_password,
                    database=self.settings.neo4j_database
                )
                await self.graph_builder.initialize()
                self.logger.info("Initialized graph builder")
            else:
                self.logger.info("Skipping graph builder initialization as requested")
            
            self.initialized = True
            self.logger.info("Enhanced ingestion pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced ingestion pipeline: {e}")
            raise
```

### 4. Update Validation Methods to Use Configuration Settings

Update the validation methods to use the required fields from configuration:

```python
async def validate_market_data(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate market data without saving to database.
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
```

Apply similar changes to the `validate_property_listings` method.

### 5. Update Other Components to Use Centralized Configuration

Ensure all other components (JSONChunker, Embedder, GraphBuilder, DatabaseIntegration) use the centralized configuration settings:

- Update JSONChunker to use settings.max_chunk_size and settings.chunk_overlap
- Update OpenAIEmbedder to use settings.embedding_model, settings.embedding_dimensions, etc.
- Update GraphBuilder to use settings.neo4j_uri, settings.neo4j_user, etc.
- Update DatabaseIntegration to use settings.postgres_uri, etc.

### 6. Add Environment Variables to .env.example

Update the `.env.example` file to include all the new configuration settings:

```
# Database settings
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/trackrealties
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trackrealties

# Neo4j settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Embedding settings
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSIONS=1536
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_CACHE_DIR=./cache/embeddings
EMBEDDING_BATCH_SIZE=100

# Chunking settings
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Graph settings
GRAPH_BATCH_SIZE=50

# Ingestion settings
INGESTION_BATCH_SIZE=100

# LLM settings
LLM_API_KEY=your-openai-api-key
LLM_MODEL=gpt-3.5-turbo
```

## Testing Plan

After implementing these changes, we should test the configuration management:

1. Test with default settings:
   ```
   trackrealties enhanced-ingest sample_market_data.json --data-type market
   ```

2. Test with environment variables:
   ```
   EMBEDDING_MODEL=text-embedding-3-small EMBEDDING_DIMENSIONS=1024 trackrealties enhanced-ingest sample_market_data.json --data-type market
   ```

3. Test validation with missing required settings:
   ```
   # Temporarily rename .env file to simulate missing settings
   mv .env .env.bak
   trackrealties enhanced-ingest sample_market_data.json --data-type market
   # Should show validation errors
   mv .env.bak .env
   ```

## Conclusion

Implementing these changes will ensure that all components of the Enhanced Ingestion Pipeline use centralized configuration settings, making it easier to configure and maintain the system. The validation of required parameters will help prevent runtime errors due to missing configuration.