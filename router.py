"""
Query Router - Decides between RAG, SQL, or hybrid approach.

Analyzes user intent and routes to the appropriate handler.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryType(Enum):
    RAG = "rag"           # Semantic search in text
    SQL = "sql"           # Structured query
    HYBRID = "hybrid"     # Both RAG and SQL
    GENERAL = "general"   # General conversation


@dataclass
class RoutingDecision:
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_tables: List[str] = None
    
    def __post_init__(self):
        if self.suggested_tables is None:
            self.suggested_tables = []


class QueryRouter:
    """Routes queries to appropriate handlers based on intent analysis."""
    
    ROUTING_PROMPT = """Analyze this user query and determine the best approach to answer it.

DATABASE SCHEMA:
{schema}

USER QUERY: {query}

Determine if this query needs:
1. RAG - Semantic search through text content (searching for meanings, concepts, descriptions)
2. SQL - Structured database query (counting, filtering, aggregating, specific lookups, OR pagination requests like "show more", "show other", "next results", "remaining items")
3. HYBRID - Both semantic search and structured query
4. GENERAL - General conversation not requiring database access

IMPORTANT: If the user asks to "show more", "show other", "see remaining", "next results", or similar - this is a PAGINATION request and should be routed to SQL, NOT GENERAL.
5. REFERENTIAL/AFFIRMATIVE: If the query is simply "yes", "sure", "ok", "please", or "do it", check if it's likely a confirmation to a previous offer (like "would you like to see 10 more?"). If so, this is likely SQL (pagination or new query). If ambiguous, default to GENERAL.

Respond in this exact format:
TYPE: [RAG|SQL|HYBRID|GENERAL]
CONFIDENCE: [0.0-1.0]
TABLES: [comma-separated list of relevant tables, or NONE]
REASONING: [brief explanation]"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def set_llm_client(self, llm_client):
        self.llm_client = llm_client
    
    def route(self, query: str, schema_context: str, chat_history: Optional[List[Dict]] = None) -> RoutingDecision:
        """Analyze query and determine routing."""
        if not self.llm_client:
            # Fallback to simple heuristics
            return self._heuristic_route(query)
        
        prev_context = ""
        if chat_history and len(chat_history) > 0:
            last_msg = chat_history[-1]
            if last_msg.get("role") == "assistant":
                prev_context = f"\nPREVIOUS ASSISTANT MSG: {last_msg.get('content', '')[:200]}..."
        
        prompt = self.ROUTING_PROMPT.format(schema=schema_context, query=query + prev_context)
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are a query routing assistant."},
                {"role": "user", "content": prompt}
            ])
            return self._parse_routing_response(response)
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using heuristics")
            return self._heuristic_route(query)
    
    def _parse_routing_response(self, response: str) -> RoutingDecision:
        """Parse LLM routing response."""
        lines = response.strip().split('\n')
        
        query_type = QueryType.GENERAL
        confidence = 0.5
        tables = []
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("TYPE:"):
                type_str = line.replace("TYPE:", "").strip().upper()
                query_type = QueryType[type_str] if type_str in QueryType.__members__ else QueryType.GENERAL
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("TABLES:"):
                tables_str = line.replace("TABLES:", "").strip()
                if tables_str.upper() != "NONE":
                    tables = [t.strip() for t in tables_str.split(",")]
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return RoutingDecision(query_type, confidence, reasoning, tables)
    
    def _heuristic_route(self, query: str) -> RoutingDecision:
        """Simple heuristic-based routing when LLM is unavailable."""
        query_lower = query.lower()
        
        # SQL keywords - for structured data retrieval
        sql_keywords = [
            'how many', 'count', 'total', 'average', 'sum', 'max', 'min',
            'list all', 'show all', 'find all', 'get all', 'between',
            'greater than', 'less than', 'equal to', 'top', 'bottom',
            # Data listing patterns
            'what products', 'what customers', 'what orders', 'what items',
            'show me', 'list', 'display', 'give me', 'get me',
            'all products', 'all customers', 'all orders',
            'products do you have', 'customers do you have',
            'from new york', 'from chicago', 'from los angeles',
            # Specific lookups
            'price of', 'cost of', 'stock of', 'quantity',
            'where', 'which', 'who',
            # Pagination / follow-up requests
            'show more', 'show other', 'show rest', 'show remaining',
            'more results', 'next', 'remaining', 'rest of', 'other also',
            'continue', 'keep going', 'see more', 'view more'
        ]
        
        # RAG keywords - for semantic/conceptual questions
        rag_keywords = [
            'what is the policy', 'explain', 'describe', 'tell me about',
            'meaning of', 'definition', 'why', 'how does', 'what does',
            'similar to', 'return policy', 'shipping policy', 'warranty',
            'support', 'help with', 'information about', 'details about'
        ]
        
        sql_score = sum(1 for kw in sql_keywords if kw in query_lower)
        rag_score = sum(1 for kw in rag_keywords if kw in query_lower)
        
        # Boost SQL score for common listing patterns
        if any(word in query_lower for word in ['products', 'customers', 'orders', 'items']):
            if any(word in query_lower for word in ['what', 'show', 'list', 'all', 'have']):
                sql_score += 2
        
        if sql_score > rag_score:
            return RoutingDecision(QueryType.SQL, 0.8, "SQL query for data retrieval")
        elif rag_score > sql_score:
            return RoutingDecision(QueryType.RAG, 0.8, "Semantic search for concepts")
        elif sql_score > 0 and rag_score > 0:
            return RoutingDecision(QueryType.HYBRID, 0.6, "Mixed query type")
        else:
            # Default to SQL for simple questions about data
            if any(word in query_lower for word in ['products', 'customers', 'orders']):
                return RoutingDecision(QueryType.SQL, 0.6, "Default to SQL for data tables")
            return RoutingDecision(QueryType.RAG, 0.5, "Default to semantic search")


_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
