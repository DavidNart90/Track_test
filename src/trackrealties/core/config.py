import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    """Application settings."""
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai:gpt-4-turbo-preview")
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", 4000))
    VALIDATION_ENABLED: bool = os.getenv("VALIDATION_ENABLED", "true").lower() == "true"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

def get_settings() -> Settings:
    return settings