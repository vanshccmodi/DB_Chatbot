"""
Text-to-SQL Generator - Multi-Database Support.

Uses LLM to generate SQL queries from natural language,
with dynamic schema context. Supports MySQL, PostgreSQL, and SQLite.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import re

logger = logging.getLogger(__name__)


def get_sql_dialect(db_type: str) -> str:
    """Get the SQL dialect name for the given database type."""
    dialects = {
        "mysql": "MySQL",
        "postgresql": "PostgreSQL"
    }
    return dialects.get(db_type, "SQL")


def get_dialect_specific_hints(db_type: str) -> str:
    """Get database-specific hints for SQL generation."""
    if db_type == "postgresql":
        return """
PostgreSQL-SPECIFIC NOTES:
- Use ILIKE for case-insensitive pattern matching (instead of LIKE)
- String concatenation uses || operator
- Use LIMIT at the end of queries
- Boolean values are TRUE/FALSE (not 1/0)
- Use double quotes for identifiers with special chars, single quotes for strings
"""
    elif db_type == "sqlite":
        return """
SQLite-SPECIFIC NOTES:
- LIKE is case-insensitive for ASCII characters by default
- Use || for string concatenation
- No ILIKE - use LIKE (case-insensitive) or GLOB (case-sensitive)
- Use LIMIT at the end of queries
- Boolean values are 1/0
- Uses strftime() for date functions instead of DATE_FORMAT
"""
    else:  # MySQL
        return """
MySQL-SPECIFIC NOTES:
- CRITICALLY IMPORTANT: This server runs with ONLY_FULL_GROUP_BY enabled.
- IF YOU USE GROUP BY, EVERY SINGLE COLUMN in the SELECT list MUST be either:
  1. In the GROUP BY clause, OR
  2. Wrapped in an aggregate function (SUM, COUNT, AVT, MAX, MIN).
- EXAMPLE ERROR: "Expression #2 of SELECT list is not in GROUP BY clause..." -> This means you selected a raw column without aggregation.
- FIX: Change `SELECT name, clicks... GROUP BY name` to `SELECT name, SUM(clicks)... GROUP BY name`.
- LIKE is case-insensitive for non-binary strings
- Use CONCAT() for string concatenation
- Use LIMIT at the end of queries
- Boolean values are 1/0
- Use backticks for identifiers with special chars, single quotes for strings
"""


class SQLGenerator:
    """Generates SQL queries from natural language using LLM."""
    
    SYSTEM_PROMPT_TEMPLATE = """You are a SQL expert. Generate {dialect} SELECT queries based on user questions.

RULES:
1. ONLY generate SELECT statements.
2. NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, or TRUNCATE.
3. Always include a LIMIT clause (max 50 rows unless specified). Do NOT use LIMIT 1 for "top" or "best" queries unless explicitly asked for "single" or "one".
4. Use table and column names EXACTLY as shown in the schema.
5. TOP/BEST ITEMS: When asked for 'top', 'highest', or 'best' items (e.g. 'top rated products'), use LIMIT 5 or LIMIT 10 to show potential ties or multiple top candidates. Never use LIMIT 1 for these unless the user explicitly asks for "the number one" or "single best".
6. AMBIGUITY: If the user asks for a category, type, or specific value, and you are unsure which column it belongs to:
   - Check multiple likely columns (e.g., `category`, `sub_category`, `type`, `description`).
   - Use pattern matching for flexibility.
   - Use `OR` to combine multiple column checks.
7. DATA AWARENESS: In footwear databases, specific types like 'Formal', 'Casual', or 'Sports' often appear in `sub_category` OR `category`. Check both if available.
8. SUBJECTIVE/IMPLICIT FILTERS:
   If the user asks for subjective attributes (e.g., "for kids", "summer usage", "rainy season") and no direct column exists:
   - INFER logical mappings using available columns (material, type, category, description).
   - EXAMPLES:
     * "Summer" -> `category` IN ('Sandals', 'Slippers', 'Flip Flops') OR `material` IN ('Canvas', 'Mesh') OR `description` LIKE '%breathable%'
     * "Winter/Rainy" -> `category` IN ('Boots') OR `material` IN ('Leather', 'Synthetic', 'Rubber') OR `description` LIKE '%waterproof%'
     * "Kids" -> `category` IN ('Kids', 'Children', 'Junior') OR `product_name` LIKE '%Junior%' OR `product_name` LIKE '%Infant%' OR (`size` < 6 AND `size` > 0)
   - Use `OR` broadly to capture potential matches.
   - Use pattern matching (`LIKE` / `ILIKE`) on text columns if categories are unclear.

9. Return ONLY the SQL query, no explanations.
10. PAGINATION: If the user asks to "show more", "show other", "see remaining", or similar follow-up:
   - Look at the previous conversation for the original query conditions.
   - Use LIMIT with OFFSET to get the next set of results (e.g., LIMIT 10 OFFSET 10 for the second page).
   - Keep the same WHERE conditions from the previous query.
   - Use LIMIT with OFFSET to get the next set of results (e.g., LIMIT 10 OFFSET 10 for the second page).
   - Keep the same WHERE conditions from the previous query.
11. GROUP BY RULES: If you use GROUP BY, every column in the SELECT list must be either in the GROUP BY clause or wrapped in an aggregate function (SUM, AVG, COUNT, MAX). Do NOT select raw columns like `clicks` or `price` if you are grouping by `product_name`; use SUM(clicks), AVG(price), etc.
12. BUSINESS LOGIC & METRICS:
   - `sales` column is usually QUANTITY (integer). `price` is Unit Price. `mfrcost` is Unit Cost.
   - REVENUE = `sales * price`
   - GROSS PROFIT = `(price - mfrcost) * sales`
   - NET PROFIT (w/ Ad Cost) = `((price - mfrcost) * sales) - adcost`
   - PROFIT MARGIN (%) = `(NET PROFIT / REVENUE) * 100`
   - ROAS = `REVENUE / adcost`
   - If User asks for "Profit" or "Margin" and `adcost` is available, PREFER the NET PROFIT formula that subtracts `adcost`.
   - Always aggregate (SUM) these values when grouping by product/category.

{dialect_hints}

DATABASE SCHEMA:
{schema}

Generate a single {dialect} SELECT query to answer the user's question."""

    def __init__(self, llm_client=None, db_type: str = "mysql"):
        self.llm_client = llm_client
        self.db_type = db_type
    
    def set_llm_client(self, llm_client):
        self.llm_client = llm_client
    
    def set_db_type(self, db_type: str):
        """Set the database type for SQL generation."""
        self.db_type = db_type
    
    def generate(
        self,
        question: str,
        schema_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str]:
        """
        Generate SQL from natural language.
        
        Returns:
            Tuple of (sql_query, explanation)
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured")
        
        dialect = get_sql_dialect(self.db_type)
        dialect_hints = get_dialect_specific_hints(self.db_type)
        
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            dialect=dialect,
            dialect_hints=dialect_hints,
            schema=schema_context
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            for msg in chat_history[-3:]:  # Last 3 exchanges for context
                messages.append(msg)
        
        messages.append({"role": "user", "content": question})
        
        response = self.llm_client.chat(messages)
        
        # Extract SQL from response content
        sql = self._extract_sql(response.content)
        
        # We can optionally pass usage back too, but for strict backward compatibility 
        # let's just use the content in the tuple for now, or update the return type.
        # Since I am updating the chatbot anyway, I will attach usage to the response.
        # However, to avoid breaking other calls immediately, I'll return the response object as the second item
        # instead of just the explanation string, OR I can monkey-patch the explanation string.
        # Better: let's update return type to Tuple[str, LLMResponse].
        
        return sql, response
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Look for SQL in code blocks
        code_block = re.search(r'```(?:sql)?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if code_block:
            return code_block.group(1).strip()
        
        # Look for SELECT statement
        select_match = re.search(
            r'(SELECT\s+.+?(?:;|$))', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if select_match:
            return select_match.group(1).strip().rstrip(';')
        
        return response.strip()


_generator: Optional[SQLGenerator] = None


def get_sql_generator(db_type: str = "mysql") -> SQLGenerator:
    global _generator
    if _generator is None:
        _generator = SQLGenerator(db_type=db_type)
    else:
        _generator.set_db_type(db_type)
    return _generator
