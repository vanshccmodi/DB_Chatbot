"""
SQL Validator - Security layer for SQL queries.

Ensures ONLY safe SELECT queries are executed.
Validates against whitelist and blocks dangerous operations.
"""

import logging
import re
from typing import List, Tuple, Optional, Set
import sqlparse
from sqlparse.sql import Statement, Token, Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)


class SQLValidationError(Exception):
    """Raised when SQL validation fails."""
    pass


class SQLValidator:
    """Validates SQL queries for safety before execution."""
    
    FORBIDDEN_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXECUTE', 'EXEC',
        'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE', 'LOAD DATA'
    }
    
    FORBIDDEN_PATTERNS = [
        r'INTO\s+OUTFILE',
        r'INTO\s+DUMPFILE', 
        r'LOAD_FILE\s*\(',
        r'LOAD\s+DATA',
        r';\s*(?:DROP|DELETE|UPDATE|INSERT)',  # Multi-statement attacks
        r'--',  # SQL comments (potential injection)
        r'/\*.*\*/',  # Block comments
    ]
    
    def __init__(self, allowed_tables: Optional[Set[str]] = None, max_limit: int = 100):
        self.allowed_tables = allowed_tables or set()
        self.max_limit = max_limit
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_PATTERNS]
    
    def set_allowed_tables(self, tables: List[str]):
        """Set the whitelist of allowed tables."""
        self.allowed_tables = set(tables)

    def validate(self, sql: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate SQL query for safety.
        
        Returns:
            Tuple of (is_valid, message, sanitized_sql)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query", None
        
        sql = sql.strip()
        
        # Check for forbidden patterns
        for pattern in self._compiled_patterns:
            if pattern.search(sql):
                return False, f"Forbidden pattern detected in query", None
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)
        except Exception as e:
            return False, f"Failed to parse SQL: {e}", None
        
        if not parsed:
            return False, "Failed to parse SQL query", None
        
        # Only allow single statements
        if len(parsed) > 1:
            return False, "Multiple SQL statements not allowed", None
        
        statement = parsed[0]
        
        # Check statement type
        stmt_type = statement.get_type()
        if stmt_type != 'SELECT':
            return False, f"Only SELECT statements allowed, got: {stmt_type}", None
        
        # Check for forbidden keywords in tokens
        sql_upper = sql.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Forbidden keyword detected: {keyword}", None
        
        # Extract and validate tables
        tables = self._extract_tables(statement)
        if self.allowed_tables:
            # Normalize for comparison (remove quotes, lowercase)
            allowed_norm = {t.lower().replace('"', '').replace('`', '') for t in self.allowed_tables}
            tables_norm = {t.lower().replace('"', '').replace('`', '') for t in tables}
            
            invalid_tables = tables_norm - allowed_norm
            if invalid_tables:
                return False, f"Access denied to tables: {invalid_tables}", None
        
        # Ensure LIMIT clause exists
        sanitized = self._ensure_limit(sql)
        
        return True, "Query validated successfully", sanitized
    
    def _extract_tables(self, statement: Statement) -> Set[str]:
        """Extract table names from a SELECT statement using regex."""
        tables = set()
        sql = str(statement)
        
        # Use regex to find tables after FROM and JOIN
        # Pattern: FROM table_name or JOIN table_name, supporting quotes
        # Matches: FROM table, FROM "table", FROM `table`
        from_pattern = re.compile(
            r'\bFROM\s+(?:["`]?)([a-zA-Z0-9_]+)(?:["`]?)',
            re.IGNORECASE
        )
        join_pattern = re.compile(
            r'\bJOIN\s+(?:["`]?)([a-zA-Z0-9_]+)(?:["`]?)',
            re.IGNORECASE
        )
        
        # Find all FROM tables
        for match in from_pattern.finditer(sql):
            tables.add(match.group(1))
        
        # Find all JOIN tables
        for match in join_pattern.finditer(sql):
            tables.add(match.group(1))
        
        return tables

    def _ensure_limit(self, sql: str) -> str:
        """Ensure the query has a LIMIT clause."""
        sql_upper = sql.upper()
        
        if 'LIMIT' in sql_upper:
            # Check if limit is too high
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if limit_match:
                current_limit = int(limit_match.group(1))
                if current_limit > self.max_limit:
                    # Replace with max limit
                    sql = re.sub(
                        r'LIMIT\s+\d+',
                        f'LIMIT {self.max_limit}',
                        sql,
                        flags=re.IGNORECASE
                    )
            return sql
        else:
            # Add LIMIT clause
            sql = sql.rstrip(';').strip()
            return f"{sql} LIMIT {self.max_limit}"


_validator: Optional[SQLValidator] = None


def get_sql_validator() -> SQLValidator:
    global _validator
    if _validator is None:
        _validator = SQLValidator()
    return _validator
