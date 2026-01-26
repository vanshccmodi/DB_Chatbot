"""
Configuration module for the Schema-Agnostic Database Chatbot.

This module handles all configuration including:
- Database connection settings (MySQL, PostgreSQL, SQLite)
- LLM provider settings (Groq / OpenAI / Local LLaMA)
- Embedding model configuration
- Security settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# Load .env file BEFORE any os.getenv calls
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class DatabaseType(Enum):
    """Supported database types."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"  # FREE!
    OPENAI = "openai"
    LOCAL_LLAMA = "local_llama"


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class DatabaseConfig:
    """
    Database configuration supporting MySQL and PostgreSQL.
    
    All sensitive values are loaded from environment variables.
    """
    # Database type (mysql, postgresql)
    db_type: DatabaseType = field(
        default_factory=lambda: DatabaseType(os.getenv("DB_TYPE", "mysql").lower())
    )
    
    # Common connection settings (for MySQL/PostgreSQL)
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "")))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", "3306"))))
    database: str = field(default_factory=lambda: os.getenv("DB_DATABASE", os.getenv("MYSQL_DATABASE", "")))
    username: str = field(default_factory=lambda: os.getenv("DB_USERNAME", os.getenv("MYSQL_USERNAME", "")))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", "")))
    
    # SSL configuration
    ssl_ca: Optional[str] = field(default_factory=lambda: os.getenv("DB_SSL_CA", os.getenv("MYSQL_SSL_CA", None)))
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string based on database type."""
        if self.db_type == DatabaseType.POSTGRESQL:
            # PostgreSQL connection string
            base_url = f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            if self.ssl_ca:
                return f"{base_url}?sslmode=verify-full&sslrootcert={self.ssl_ca}"
            return base_url
        
        else:  # MySQL (default)
            # MySQL connection string
            base_url = f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            if self.ssl_ca:
                return f"{base_url}?ssl_ca={self.ssl_ca}"
            return base_url
    
    def is_configured(self) -> bool:
        """Check if all required database settings are configured."""
        # MySQL/PostgreSQL need host, database, username, password
        return all([self.host, self.database, self.username, self.password])
    
    @property
    def is_mysql(self) -> bool:
        """Check if using MySQL."""
        return self.db_type == DatabaseType.MYSQL
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.db_type == DatabaseType.POSTGRESQL


@dataclass
class LLMConfig:
    """LLM configuration for query routing and response generation."""
    provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(os.getenv("LLM_PROVIDER", "openai"))
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Local LLaMA settings
    local_model_path: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_PATH", "")
    )
    local_model_name: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_NAME", "llama-2-7b-chat")
    )
    
    # Generation parameters
    temperature: float = 0.1  # Low temperature for more deterministic outputs
    max_tokens: int = 1024
    
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        if self.provider == LLMProvider.OPENAI:
            return bool(self.openai_api_key)
        return bool(self.local_model_path)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration for RAG."""
    provider: EmbeddingProvider = field(
        default_factory=lambda: EmbeddingProvider(
            os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
        )
    )
    
    # OpenAI embedding settings
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Sentence Transformers settings
    st_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    
    # Embedding dimensions (varies by model)
    embedding_dim: int = 384  # Default for all-MiniLM-L6-v2


@dataclass
class SecurityConfig:
    """Security settings for SQL validation and execution."""
    
    # SQL operations whitelist - ONLY SELECT allowed
    allowed_operations: List[str] = field(default_factory=lambda: ["SELECT"])
    
    # Dangerous keywords that should never appear in queries
    forbidden_keywords: List[str] = field(default_factory=lambda: [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "EXEC",
        "INTO OUTFILE", "INTO DUMPFILE", "LOAD_FILE",
        "INFORMATION_SCHEMA.USER_PRIVILEGES"
    ])
    
    # Maximum number of rows to return
    max_result_rows: int = 100
    
    # Default LIMIT clause if not specified
    default_limit: int = 50


@dataclass
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration."""
    
    # FAISS index settings
    faiss_index_path: str = "./faiss_index"
    
    # Number of top results to retrieve
    top_k: int = 5
    
    # Minimum similarity score for relevance
    similarity_threshold: float = 0.3
    
    # Text columns to consider for RAG (common across database types)
    text_column_types: List[str] = field(default_factory=lambda: [
        # MySQL types
        "TEXT", "MEDIUMTEXT", "LONGTEXT", "TINYTEXT", "VARCHAR", "CHAR",
        # PostgreSQL types
        "CHARACTER VARYING", "CHARACTER"
    ])
    
    # Minimum character length to consider a column for RAG
    min_text_length: int = 50
    
    # Chunk size for long text documents
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class ChatConfig:
    """Chat and memory configuration."""
    
    # Short-term memory (in session)
    max_session_messages: int = 20
    
    # Long-term memory table name (will be created if not exists)
    memory_table_name: str = "_chatbot_memory"
    
    # Number of recent messages to include in context
    context_messages: int = 5


class AppConfig:
    """
    Main application configuration aggregator.
    
    Combines all configuration sections and provides
    validation methods.
    """
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.security = SecurityConfig()
        self.rag = RAGConfig()
        self.chat = ChatConfig()
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate all configuration settings.
        
        Returns:
            tuple: (is_valid, list of error messages)
        """
        errors = []
        
        if not self.database.is_configured():
            db_type = self.database.db_type.value.upper()
            errors.append(f"{db_type} configuration incomplete. Check DB_* environment variables.")
        
        if not self.llm.is_configured():
            errors.append(
                f"LLM configuration incomplete for provider: {self.llm.provider.value}. "
                "Check API keys or model paths."
            )
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls()


# Global configuration instance
config = AppConfig.from_env()
