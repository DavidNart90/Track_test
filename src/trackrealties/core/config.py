import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    """Application settings with enhanced pipeline configuration."""
    
    # Application Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_NAME: str = os.getenv("APP_NAME", "TrackRealties AI Platform")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    
    # PostgreSQL Database Settings
    DATABASE_URL:  Optional[str] = os.getenv("DATABASE_URL")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", 10))
    POSTGRES_URI: Optional[str] = os.getenv("POSTGRES_URI", os.getenv("DATABASE_URL"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trackrealties")
    
    # Neo4j Graph Database Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # OpenAI/LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai:gpt-4-turbo-preview")
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", 4000))
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
    EMBEDDING_API_KEY: Optional[str] = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "./cache/embeddings")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
    
    # Chunking Settings
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", 100))
    
    # Graph Settings
    GRAPH_BATCH_SIZE: int = int(os.getenv("GRAPH_BATCH_SIZE", 50))
    GRAPH_MAX_CONNECTIONS: int = int(os.getenv("GRAPH_MAX_CONNECTIONS", 100))
    
    # Ingestion Settings
    INGESTION_BATCH_SIZE: int = int(os.getenv("INGESTION_BATCH_SIZE", 100))
    INGESTION_MAX_WORKERS: int = int(os.getenv("INGESTION_MAX_WORKERS", 4))
    INGESTION_RETRY_ATTEMPTS: int = int(os.getenv("INGESTION_RETRY_ATTEMPTS", 3))
    INGESTION_RETRY_DELAY: float = float(os.getenv("INGESTION_RETRY_DELAY", 1.0))
    
    # Validation Settings
    VALIDATION_ENABLED: bool = os.getenv("VALIDATION_ENABLED", "true").lower() == "true"
    VALIDATION_REQUIRED_FIELDS_MARKET: List[str] = ["region_id", "region_name", "date", "median_price"]
    VALIDATION_REQUIRED_FIELDS_PROPERTY: List[str] = ["property_id", "price", "status", "property_type"]
    
    # AWS Settings (for future use)
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME", "trackrealties-data")
    
    # Rate Limiting Settings
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 100))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))
    
    # Feature Flags
    FEATURE_ENHANCED_INGESTION: bool = os.getenv("FEATURE_ENHANCED_INGESTION", "true").lower() == "true"
    FEATURE_KNOWLEDGE_GRAPH: bool = os.getenv("FEATURE_KNOWLEDGE_GRAPH", "true").lower() == "true"
    FEATURE_EMBEDDINGS: bool = os.getenv("FEATURE_EMBEDDINGS", "true").lower() == "true"
    
    # Property mappings for backward compatibility and convenience
    @property
    def max_chunk_size(self) -> int:
        return self.MAX_CHUNK_SIZE
    
    @property
    def chunk_overlap(self) -> int:
        return self.CHUNK_OVERLAP
    
    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL
    
    @property
    def embedding_dimensions(self) -> int:
        return self.EMBEDDING_DIMENSIONS
    
    @property
    def embedding_batch_size(self) -> int:
        return self.EMBEDDING_BATCH_SIZE
    
    @property
    def ingestion_batch_size(self) -> int:
        return self.INGESTION_BATCH_SIZE
    
    @property
    def embedding_api_key(self) -> Optional[str]:
        return self.EMBEDDING_API_KEY
    
    @property
    def llm_api_key(self) -> Optional[str]:
        return self.LLM_API_KEY
    
    @property
    def neo4j_uri(self) -> str:
        return self.NEO4J_URI
    
    @property
    def neo4j_user(self) -> str:
        return self.NEO4J_USER
    
    @property
    def neo4j_password(self) -> str:
        return self.NEO4J_PASSWORD
    
    @property
    def neo4j_database(self) -> str:
        return self.NEO4J_DATABASE

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

    def validate_settings(self) -> None:
        """Validate that required settings are properly configured."""
        errors = []
        
        # Check database configuration
        if not self.DATABASE_URL and not self.POSTGRES_URI:
            errors.append("DATABASE_URL or POSTGRES_URI must be set")
        
        # Check API keys for embeddings
        if self.FEATURE_EMBEDDINGS and not self.EMBEDDING_API_KEY and not self.OPENAI_API_KEY:
            errors.append("EMBEDDING_API_KEY or OPENAI_API_KEY must be set for embeddings")
        
        # Check Neo4j configuration if knowledge graph is enabled
        if self.FEATURE_KNOWLEDGE_GRAPH:
            if not self.NEO4J_URI:
                errors.append("NEO4J_URI must be set for knowledge graph feature")
            if not self.NEO4J_USER:
                errors.append("NEO4J_USER must be set for knowledge graph feature")
            if not self.NEO4J_PASSWORD:
                errors.append("NEO4J_PASSWORD must be set for knowledge graph feature")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

settings = Settings()

def get_settings() -> Settings:
    return settings