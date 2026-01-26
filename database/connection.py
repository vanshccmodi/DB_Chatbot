"""
Database Connection Module - Multi-Database Support.

This module provides:
- SQLAlchemy engine and session management for MySQL, PostgreSQL, and SQLite
- Connection pooling (for MySQL/PostgreSQL)
- SSL/TLS support
- Connection health checking
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import OperationalError, SQLAlchemyError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig, DatabaseType, config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages database connections with connection pooling.
    
    Supports MySQL, PostgreSQL, and SQLite.
    """
    
    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection manager.
        
        Args:
            db_config: Database configuration. Uses global config if not provided.
        """
        self.config = db_config or config.database
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    def _create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with appropriate settings for each database type.
        
        Returns:
            Configured SQLAlchemy Engine instance
        """
        connect_args = {}
        
        if self.config.db_type == DatabaseType.POSTGRESQL:
            # PostgreSQL-specific settings
            if self.config.ssl_ca:
                connect_args["sslmode"] = "verify-full"
                connect_args["sslrootcert"] = self.config.ssl_ca
            
            engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                connect_args=connect_args,
                echo=False
            )
            
        else:  # MySQL (default)
            # MySQL-specific settings (SSL for Aiven)
            if self.config.ssl_ca:
                connect_args["ssl"] = {
                    "ca": self.config.ssl_ca,
                    "check_hostname": True,
                    "verify_mode": True
                }
            
            engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                connect_args=connect_args,
                echo=False
            )
        
        return engine
    
    @property
    def engine(self) -> Engine:
        """Get or create the SQLAlchemy engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    @property
    def db_type(self) -> DatabaseType:
        """Get the current database type."""
        return self.config.db_type
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Yields:
            SQLAlchemy Session instance
            
        Example:
            with db.get_session() as session:
                result = session.execute(text("SELECT * FROM users"))
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """
        Execute a read-only SQL query and return results.
        
        Args:
            query: SQL query string (must be SELECT)
            params: Optional query parameters for parameterized queries
            
        Returns:
            List of result rows as dictionaries
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            # Convert rows to dictionaries for easier handling
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def execute_write(self, query: str, params: Optional[dict] = None) -> bool:
        """
        Execute a write operation (INSERT, UPDATE, DELETE, CREATE).
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            bool: True if successful
        """
        with self.get_session() as session:
            session.execute(text(query), params or {})
            session.commit()
            return True
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test database connectivity.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()
                if row and row[0] == 1:
                    db_type = self.config.db_type.value.upper()
                    return True, f"{db_type} connection successful"
                return False, "Unexpected result from health check query"
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}")
            return False, f"Unexpected error: {str(e)}"
    
    def close(self):
        """Close all connections and dispose of the engine."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")


# Create a global database connection instance
db_connection = DatabaseConnection()


def get_db() -> DatabaseConnection:
    """Get the global database connection instance."""
    return db_connection
