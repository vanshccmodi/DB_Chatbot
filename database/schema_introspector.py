"""
Dynamic Schema Introspection Module - Multi-Database Support.

This module is the CORE of the schema-agnostic design.
It dynamically discovers:
- All tables in the database
- All columns with their data types
- Primary keys and foreign keys
- Text-like columns for RAG indexing
- Relationships between tables

Supports MySQL, PostgreSQL, and SQLite.
NEVER hardcodes any table or column names.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

from .connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a single database column."""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    max_length: Optional[int] = None
    default_value: Optional[str] = None
    comment: Optional[str] = None
    
    @property
    def is_text_type(self) -> bool:
        """Check if this column contains text data suitable for RAG."""
        text_types = [
            # MySQL
            'text', 'mediumtext', 'longtext', 'tinytext', 'varchar', 'char', 'json',
            # PostgreSQL
            'character varying', 'character', 'text', 'json', 'jsonb',
            # SQLite (column affinity - TEXT)
            'clob', 'nvarchar', 'nchar', 'ntext'
        ]
        data_type_lower = self.data_type.lower().split('(')[0].strip()
        return data_type_lower in text_types
    
    @property
    def is_numeric(self) -> bool:
        """Check if this column contains numeric data."""
        numeric_types = [
            # Common across databases
            'int', 'integer', 'bigint', 'smallint', 'tinyint',
            'decimal', 'numeric', 'float', 'double', 'real',
            # PostgreSQL specific
            'double precision', 'serial', 'bigserial', 'smallserial',
            # SQLite (NUMERIC affinity)
            'bool', 'boolean'
        ]
        data_type_lower = self.data_type.lower().split('(')[0].strip()
        return data_type_lower in numeric_types


@dataclass
class TableInfo:
    """Complete information about a database table."""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # column -> referenced_table.column
    row_count: Optional[int] = None
    comment: Optional[str] = None
    
    @property
    def text_columns(self) -> List[ColumnInfo]:
        """Get columns suitable for text/RAG indexing."""
        return [col for col in self.columns if col.is_text_type]
    
    @property
    def column_names(self) -> List[str]:
        """Get list of all column names."""
        return [col.name for col in self.columns]
    
    def get_column(self, name: str) -> Optional[ColumnInfo]:
        """Get column info by name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None


@dataclass
class SchemaInfo:
    """Complete database schema information."""
    database_name: str
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    
    @property
    def table_names(self) -> List[str]:
        """Get list of all table names."""
        return list(self.tables.keys())
    
    @property
    def all_text_columns(self) -> List[tuple]:
        """Get all text columns across all tables as (table, column) tuples."""
        result = []
        for table_name, table_info in self.tables.items():
            for col in table_info.text_columns:
                result.append((table_name, col.name))
        return result
    
    def to_context_string(self) -> str:
        """
        Generate a natural language description of the schema.
        This is used as context for the LLM.
        """
        lines = [f"Database: {self.database_name}", ""]
        lines.append("Available Tables:")
        lines.append("-" * 40)
        
        for table_name, table_info in self.tables.items():
            lines.append(f"\nTable: {table_name}")
            if table_info.comment:
                lines.append(f"  Description: {table_info.comment}")
            if table_info.row_count is not None:
                lines.append(f"  Approximate rows: {table_info.row_count}")
            
            lines.append("  Columns:")
            for col in table_info.columns:
                pk_marker = " [PRIMARY KEY]" if col.is_primary_key else ""
                nullable = " (nullable)" if col.is_nullable else " (required)"
                lines.append(f"    - {col.name}: {col.data_type}{pk_marker}{nullable}")
                if col.comment:
                    lines.append(f"      Comment: {col.comment}")
            
            if table_info.foreign_keys:
                lines.append("  Foreign Keys:")
                for col, ref in table_info.foreign_keys.items():
                    lines.append(f"    - {col} -> {ref}")
        
        return "\n".join(lines)
    
    def to_sql_ddl(self) -> str:
        """
        Generate SQL-like DDL representation of the schema.
        Useful for SQL generation context.
        """
        ddl_lines = []
        
        for table_name, table_info in self.tables.items():
            ddl_lines.append(f"CREATE TABLE {table_name} (")
            
            col_defs = []
            for col in table_info.columns:
                col_def = f"  {col.name} {col.data_type}"
                if col.is_primary_key:
                    col_def += " PRIMARY KEY"
                if not col.is_nullable:
                    col_def += " NOT NULL"
                col_defs.append(col_def)
            
            ddl_lines.append(",\n".join(col_defs))
            ddl_lines.append(");\n")
        
        return "\n".join(ddl_lines)


class SchemaIntrospector:
    """
    Dynamically introspects database schema.
    
    This is the key component that enables schema-agnostic operation.
    It queries database system catalogs to discover the complete schema.
    Supports MySQL, PostgreSQL, and SQLite.
    """
    
    # System tables to exclude from introspection
    SYSTEM_TABLES = {
        '_chatbot_memory',  # Our own chat history table
        '_chatbot_permanent_memory_v2',
        '_chatbot_user_summaries',
        'schema_migrations',
        'flyway_schema_history',
        # SQLite internal tables
        'sqlite_sequence',
        'sqlite_stat1',
        'sqlite_stat4'
    }
    
    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize the introspector.
        
        Args:
            engine: SQLAlchemy engine. Uses global connection if not provided.
        """
        self.db = get_db()
        self._cached_schema: Optional[SchemaInfo] = None
    
    def introspect(self, force_refresh: bool = False) -> SchemaInfo:
        """
        Perform complete schema introspection.
        
        Args:
            force_refresh: If True, bypass cache and re-introspect
            
        Returns:
            SchemaInfo object with complete schema details
        """
        if self._cached_schema is not None and not force_refresh:
            return self._cached_schema
        
        logger.info("Starting schema introspection...")
        
        # Get database name
        db_name = self._get_database_name()
        
        # Get all user tables
        tables = self._get_tables()
        
        schema = SchemaInfo(database_name=db_name)
        
        for table_name in tables:
            if table_name in self.SYSTEM_TABLES:
                continue
            # Also skip tables that start with underscore (internal tables)
            if table_name.startswith('_chatbot'):
                continue
                
            table_info = self._introspect_table(table_name)
            if table_info:
                schema.tables[table_name] = table_info
        
        self._cached_schema = schema
        logger.info(f"Schema introspection complete. Found {len(schema.tables)} tables.")
        
        return schema
    
    def _get_database_name(self) -> str:
        """Get the current database name."""
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                # For SQLite, return the database file name
                return self.db.config.sqlite_path.split('/')[-1]
            elif db_type.value == "postgresql":
                result = self.db.execute_query("SELECT current_database() as db_name")
                return result[0]['db_name'] if result else "unknown"
            else:  # MySQL
                result = self.db.execute_query("SELECT DATABASE() as db_name")
                return result[0]['db_name'] if result else "unknown"
        except Exception as e:
            logger.error(f"Error getting database name: {e}")
            return "unknown"
    
    def _get_tables(self) -> List[str]:
        """
        Get all user tables from the database.
        Uses database-specific queries for comprehensive discovery.
        """
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                query = """
                    SELECT name as table_name
                    FROM sqlite_master 
                    WHERE type='table' 
                    AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """
                result = self.db.execute_query(query)
                return [row['table_name'] for row in result]
                
            elif db_type.value == "postgresql":
                query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """
                result = self.db.execute_query(query)
                return [row['table_name'] for row in result]
                
            else:  # MySQL
                query = """
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                """
                result = self.db.execute_query(query)
                return [row['TABLE_NAME'] for row in result]
                
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
    
    def _introspect_table(self, table_name: str) -> Optional[TableInfo]:
        """
        Get complete information about a specific table.
        
        Args:
            table_name: Name of the table to introspect
            
        Returns:
            TableInfo object or None if table doesn't exist
        """
        try:
            # Get column information
            columns = self._get_columns(table_name)
            
            # Get primary keys
            primary_keys = self._get_primary_keys(table_name)
            
            # Get foreign keys
            foreign_keys = self._get_foreign_keys(table_name)
            
            # Get approximate row count (fast estimation)
            row_count = self._get_row_count(table_name)
            
            # Get table comment (not available in SQLite)
            comment = self._get_table_comment(table_name)
            
            # Mark primary key columns
            for col in columns:
                col.is_primary_key = col.name in primary_keys
            
            return TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                row_count=row_count,
                comment=comment
            )
            
        except Exception as e:
            logger.error(f"Error introspecting table {table_name}: {e}")
            return None
    
    def _get_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get all columns for a table."""
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                query = f"PRAGMA table_info('{table_name}')"
                result = self.db.execute_query(query)
                
                columns = []
                for row in result:
                    columns.append(ColumnInfo(
                        name=row['name'],
                        data_type=row['type'] or 'TEXT',  # SQLite columns can have no type
                        is_nullable=row['notnull'] == 0,
                        is_primary_key=row['pk'] == 1,
                        max_length=None,
                        default_value=row['dflt_value'],
                        comment=None  # SQLite doesn't support column comments
                    ))
                return columns
                
            elif db_type.value == "postgresql":
                query = """
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        col_description(
                            (SELECT oid FROM pg_class WHERE relname = :table_name),
                            ordinal_position
                        ) as column_comment
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                    ORDER BY ordinal_position
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                
                columns = []
                for row in result:
                    columns.append(ColumnInfo(
                        name=row['column_name'],
                        data_type=row['data_type'],
                        is_nullable=row['is_nullable'] == 'YES',
                        is_primary_key=False,  # Will be set later
                        max_length=row['character_maximum_length'],
                        default_value=row['column_default'],
                        comment=row.get('column_comment')
                    ))
                return columns
                
            else:  # MySQL
                query = """
                    SELECT 
                        COLUMN_NAME,
                        COLUMN_TYPE,
                        IS_NULLABLE,
                        COLUMN_DEFAULT,
                        CHARACTER_MAXIMUM_LENGTH,
                        COLUMN_COMMENT
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = :table_name
                    ORDER BY ORDINAL_POSITION
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                
                columns = []
                for row in result:
                    columns.append(ColumnInfo(
                        name=row['COLUMN_NAME'],
                        data_type=row['COLUMN_TYPE'],
                        is_nullable=row['IS_NULLABLE'] == 'YES',
                        is_primary_key=False,  # Will be set later
                        max_length=row['CHARACTER_MAXIMUM_LENGTH'],
                        default_value=row['COLUMN_DEFAULT'],
                        comment=row['COLUMN_COMMENT'] if row['COLUMN_COMMENT'] else None
                    ))
                return columns
                
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
            return []
    
    def _get_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns for a table."""
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                query = f"PRAGMA table_info('{table_name}')"
                result = self.db.execute_query(query)
                return [row['name'] for row in result if row['pk'] > 0]
                
            elif db_type.value == "postgresql":
                query = """
                    SELECT a.attname as column_name
                    FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = :table_name::regclass
                    AND i.indisprimary
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return [row['column_name'] for row in result]
                
            else:  # MySQL
                query = """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = :table_name
                    AND CONSTRAINT_NAME = 'PRIMARY'
                    ORDER BY ORDINAL_POSITION
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return [row['COLUMN_NAME'] for row in result]
                
        except Exception as e:
            logger.error(f"Error getting primary keys for {table_name}: {e}")
            return []
    
    def _get_foreign_keys(self, table_name: str) -> Dict[str, str]:
        """Get foreign key relationships for a table."""
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                query = f"PRAGMA foreign_key_list('{table_name}')"
                result = self.db.execute_query(query)
                return {
                    row['from']: f"{row['table']}.{row['to']}"
                    for row in result
                }
                
            elif db_type.value == "postgresql":
                query = """
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = :table_name
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return {
                    row['column_name']: f"{row['foreign_table_name']}.{row['foreign_column_name']}"
                    for row in result
                }
                
            else:  # MySQL
                query = """
                    SELECT 
                        COLUMN_NAME,
                        REFERENCED_TABLE_NAME,
                        REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = :table_name
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return {
                    row['COLUMN_NAME']: f"{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}"
                    for row in result
                }
                
        except Exception as e:
            logger.error(f"Error getting foreign keys for {table_name}: {e}")
            return {}
    
    def _get_row_count(self, table_name: str) -> Optional[int]:
        """
        Get approximate row count for a table.
        Uses different strategies per database.
        """
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                # SQLite doesn't have stats table, use max rowid for estimation
                query = f"SELECT MAX(rowid) as row_count FROM \"{table_name}\""
                result = self.db.execute_query(query)
                return result[0]['row_count'] if result and result[0]['row_count'] else 0
                
            elif db_type.value == "postgresql":
                # Use pg_stat_user_tables for fast estimation
                query = """
                    SELECT n_live_tup as row_count
                    FROM pg_stat_user_tables
                    WHERE relname = :table_name
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return result[0]['row_count'] if result else None
                
            else:  # MySQL
                query = """
                    SELECT TABLE_ROWS
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = :table_name
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                return result[0]['TABLE_ROWS'] if result else None
                
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return None
    
    def _get_table_comment(self, table_name: str) -> Optional[str]:
        """Get table comment/description."""
        db_type = self.db.db_type
        
        try:
            if db_type.value == "sqlite":
                # SQLite doesn't support table comments
                return None
                
            elif db_type.value == "postgresql":
                query = """
                    SELECT obj_description(:table_name::regclass, 'pg_class') as table_comment
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                comment = result[0]['table_comment'] if result else None
                return comment if comment else None
                
            else:  # MySQL
                query = """
                    SELECT TABLE_COMMENT
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = :table_name
                """
                result = self.db.execute_query(query, {"table_name": table_name})
                comment = result[0]['TABLE_COMMENT'] if result else None
                return comment if comment else None
                
        except Exception as e:
            logger.error(f"Error getting table comment for {table_name}: {e}")
            return None
    
    def get_text_columns_for_rag(self, min_length: int = 50) -> List[Dict[str, Any]]:
        """
        Get all text columns suitable for RAG indexing.
        
        Args:
            min_length: Minimum max_length for varchar columns to be considered
            
        Returns:
            List of dicts with table name, column name, and metadata
        """
        schema = self.introspect()
        text_columns = []
        
        for table_name, table_info in schema.tables.items():
            for col in table_info.columns:
                if col.is_text_type:
                    # Skip very short varchar columns
                    if col.max_length and col.max_length < min_length:
                        continue
                    
                    text_columns.append({
                        "table": table_name,
                        "column": col.name,
                        "data_type": col.data_type,
                        "primary_keys": table_info.primary_keys,
                        "max_length": col.max_length
                    })
        
        return text_columns
    
    def refresh_cache(self) -> SchemaInfo:
        """Force refresh the cached schema."""
        return self.introspect(force_refresh=True)


# Global introspector instance
_introspector: Optional[SchemaIntrospector] = None


def get_introspector() -> SchemaIntrospector:
    """Get or create the global schema introspector."""
    global _introspector
    if _introspector is None:
        _introspector = SchemaIntrospector()
    return _introspector


def get_schema() -> SchemaInfo:
    """Convenience function to get the current schema."""
    return get_introspector().introspect()
