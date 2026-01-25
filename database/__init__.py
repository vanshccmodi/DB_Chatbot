"""
Database module for the Schema-Agnostic Chatbot.

Provides:
- Database connection management
- Dynamic schema introspection
- Safe query execution
"""

from .connection import DatabaseConnection, get_db, db_connection
from .schema_introspector import (
    SchemaIntrospector,
    SchemaInfo,
    TableInfo,
    ColumnInfo,
    get_introspector,
    get_schema
)

__all__ = [
    "DatabaseConnection",
    "get_db",
    "db_connection",
    "SchemaIntrospector",
    "SchemaInfo",
    "TableInfo",
    "ColumnInfo",
    "get_introspector",
    "get_schema"
]
