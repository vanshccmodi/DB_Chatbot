"""
Chatbot Core - Main orchestrator for the schema-agnostic database chatbot.

Combines all components:
- Schema introspection
- Query routing
- RAG retrieval
- SQL generation & execution
- Response generation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from database import get_db, get_schema, get_introspector
from rag import get_rag_engine
from sql import get_sql_generator, get_sql_validator
from llm import create_llm_client, LLMClient
from router import get_query_router, QueryType
from memory import ChatMemory, EnhancedChatMemory, create_memory

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from the chatbot."""
    answer: str
    query_type: str
    sources: List[Dict[str, Any]] = None
    sql_query: Optional[str] = None
    sql_results: Optional[List[Dict]] = None
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.token_usage is None:
            self.token_usage = {"input": 0, "output": 0, "total": 0}


class DatabaseChatbot:
    """Main chatbot class orchestrating all components."""
    
    RESPONSE_PROMPT = """You are a helpful database assistant. Answer the user's question based on the provided context.

IMPORTANT: Use the conversation history to understand follow-up questions. If the user refers to "it", "that", "the product", etc., look at the previous messages to understand what they're referring to.

{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY based on the provided context AND conversation history
- Do NOT use outside knowledge, general assumptions, or hallucinate facts
- If the context doesn't contain the answer, explicitly state that the information is not available in the database
- Resolve pronouns using previous messages
- Be concise but complete
- Format data nicely
{language_instruction}

INTERACTION GUIDELINES:
- If the SQL results show a list (e.g., top products) and hit the limit (5, 10, or 50), MENTION this and ASK the user if they want to see more or a specific number. 
  Example: "Here are the top 5 products... Would you like to see the top 10?"
- If the user's question was broad (e.g., "Show me products") and you're showing a limited set, ASK if they want to filter by a specific attribute (e.g., "Would you like to filter by category or price?").
- If the answer is "0 results" for a "top/best" query, suggest looking at the data generally.
- IF SUBJECTIVE INFERENCE WAS USED (e.g., inferred "summer" = sandals), EXPLAIN THIS to the user.
  Example: "I found these products that match 'summer' (based on being Sandals or breathability)..."

YOUR RESPONSE:"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.db = get_db()
        self.introspector = get_introspector()
        self.rag_engine = get_rag_engine()
        # Pass database type to SQL generator for dialect-specific SQL
        db_type = self.db.db_type.value
        self.sql_generator = get_sql_generator(db_type)
        self.sql_validator = get_sql_validator()
        self.router = get_query_router()
        self.llm_client = llm_client
        
        self._schema_initialized = False
        self._rag_initialized = False
    
    def set_llm_client(self, llm_client: LLMClient):
        """Configure the LLM client."""
        self.llm_client = llm_client
        self.sql_generator.set_llm_client(llm_client)
        self.router.set_llm_client(llm_client)
    
    def _get_language_instruction(self, language: str) -> str:
        """Generate language instruction for the response prompt.
        
        Args:
            language: The target language name (e.g., 'Hindi', 'Spanish')
            
        Returns:
            A formatted instruction string for the LLM
        """
        if language == "English":
            return ""  # No special instruction needed for English
        
        # Extract the base language name from display name
        # e.g., "हिन्दी (Hindi)" -> "Hindi"
        base_language = language
        if "(" in language and ")" in language:
            base_language = language.split("(")[1].rstrip(")")
        
        return f"\n- **IMPORTANT: Respond ENTIRELY in {base_language}**. Translate your response to {base_language}. Keep technical terms (like table names, column names, SQL) as-is, but explain everything else in {base_language}."
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize the chatbot by introspecting the database."""
        try:
            # Test connection
            success, msg = self.db.test_connection()
            if not success:
                return False, f"Database connection failed: {msg}"
            
            # Introspect schema
            schema = self.introspector.introspect(force_refresh=True)
            
            # Configure SQL validator with discovered tables
            self.sql_validator.set_allowed_tables(schema.table_names)
            
            self._schema_initialized = True
            
            return True, f"Initialized with {len(schema.tables)} tables"
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False, str(e)
    
    def index_text_data(self, progress_callback=None) -> int:
        """Index all text data for RAG."""
        if not self._schema_initialized:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")
        
        # Use the instance's introspector which might be patched for custom DB
        schema = self.introspector.introspect()
        total_docs = 0
        
        for table_name, table_info in schema.tables.items():
            text_cols = [c.name for c in table_info.text_columns]
            if not text_cols:
                continue
            
            pk = table_info.primary_keys[0] if table_info.primary_keys else None
            cols_to_select = text_cols + ([pk] if pk else [])
            
            # Quote table name based on DB specific rules to handle case sensitivity and special chars
            if self.db.db_type.value == "mysql":
                quoted_table = f"`{table_name}`"
            else:
                quoted_table = f'"{table_name}"'
                
            query = f"SELECT {', '.join(cols_to_select)} FROM {quoted_table} LIMIT 1000"
            
            try:
                # Try the primary query
                query = f"SELECT {', '.join(cols_to_select)} FROM {quoted_table} LIMIT 1000"
                rows = self.db.execute_query(query)
                docs = self.rag_engine.index_table(table_name, rows, text_cols, pk)
                total_docs += docs
                
                if progress_callback:
                    progress_callback(table_name, docs)

            except Exception as e:
                # Fallback mechanism for PostgreSQL if table not found (often due to schema issues)
                if self.db.db_type.value == "postgresql" and "UndefinedTable" in str(e):
                    try:
                        logger.warning(f"Initial query failed for {table_name}, trying 'public' schema prefix...")
                        fallback_query = f"SELECT {', '.join(cols_to_select)} FROM public.\"{table_name}\" LIMIT 1000"
                        rows = self.db.execute_query(fallback_query)
                        docs = self.rag_engine.index_table(table_name, rows, text_cols, pk)
                        total_docs += docs
                        if progress_callback:
                            progress_callback(table_name, docs)
                        continue # Success with fallback
                    except Exception as e2:
                        logger.error(f"Fallback query also failed for {table_name}: {e2}")
                        
                logger.warning(f"Failed to index {table_name}: {e}")
        
        self.rag_engine.save()
        self._rag_initialized = True
        
        return total_docs
    
    def chat(self, query: str, memory: Optional[ChatMemory] = None, ignored_tables: Optional[List[str]] = None, language: str = "English") -> ChatResponse:
        """Process a user query and return a response.
        
        Args:
            query: The user's question
            memory: Optional chat memory for context
            ignored_tables: Tables to exclude from queries
            language: Preferred response language (default: English)
        """
        if not self._schema_initialized:
            return ChatResponse(answer="Chatbot not initialized.", query_type="error",
                              error="Call initialize() first")
        
        if not self.llm_client:
            return ChatResponse(answer="LLM not configured.", query_type="error",
                              error="Configure LLM client first")
        
        try:
            # Use instance introspector
            schema = self.introspector.introspect()
            schema_context = schema.to_context_string(ignored_tables=ignored_tables)
            
            # Calculate allowed tables for RAG and Validator
            allowed_tables = None
            if ignored_tables:
                allowed_tables = [t for t in schema.table_names if t not in ignored_tables]
                # Update validator to only allow these tables
                self.sql_validator.set_allowed_tables(allowed_tables)
            else:
                self.sql_validator.set_allowed_tables(schema.table_names)
            
            # Check for memory commands
            # Check for memory commands
            # Check for memory commands using regex for flexibility
            import re
            save_pattern = re.compile(r"(?:please\s+)?(?:save|remember|memorize)\s+(?:this|that)?\s*(?:to\s+(?:main\s+)?memory)?\s*(?:that)?\s*:?\s*(.*)", re.IGNORECASE)
            match = save_pattern.match(query.strip())
            
            # Check if it looks like a command (starts with command words)
            is_command = bool(match) and (
                query.lower().startswith(("save", "remember", "memorize")) or 
                "saved to" in query.lower() # specific user case "saved to main memory"
            )

            if is_command and memory:
                content_to_save = match.group(1).strip() if match else ""
                
                # If specific content is provided (e.g. "Remember that I like pizza")
                if content_to_save:
                    # Save the explicit content
                    success = memory.save_permanent_context(content_to_save)
                    if success:
                        return ChatResponse(answer=f"💾 I've saved to your permanent memory: '{content_to_save}'", query_type="memory")
                    else:
                        return ChatResponse(answer="❌ Failed to save to permanent memory. Please try again.", query_type="memory")

                # If no content (e.g. "Save this"), save the previous conversation turn
                elif len(memory.messages) >= 2:
                    # [-1] is current command ("save to memory")
                    # [-2] is previous assistant response
                    # [-3] is previous user query (context for the response)
                    
                    msgs_to_save = []
                    # We try to grab the last QA pair: User Prompt + AI Response
                    # memory.messages structure: [User, AI, User, AI, User(current)]
                    
                    if len(memory.messages) >= 3:
                        msg_user = memory.messages[-3]
                        msg_ai = memory.messages[-2]
                        
                        # Verify roles to ensure we are saving a Q&A pair
                        if msg_user.role == "user" and msg_ai.role == "assistant":
                            msgs_to_save = [msg_user, msg_ai]
                            
                    if msgs_to_save:
                        # Format: "User: ... | Assistant: ..."
                        context_str = f"User: {msgs_to_save[0].content} | Assistant: {msgs_to_save[1].content}"
                        success = memory.save_permanent_context(context_str)
                        if success:
                            return ChatResponse(answer="💾 I've saved our last exchange to your permanent memory.", query_type="memory")
                        else:
                            return ChatResponse(answer="❌ Failed to save to permanent memory.", query_type="memory")
                    else:
                        return ChatResponse(answer="⚠️ I couldn't find a clear previous exchange to save. Try saying 'Remember that [fact]'.", query_type="memory")
                else:
                    return ChatResponse(answer="⚠️ Nothing previous to save. Tell me something to remember first!", query_type="memory")

            # Get chat history for context
            history = memory.get_context_messages(5) if memory else []

            # Route the query
            routing = self.router.route(query, schema_context, history)
            
            # Initial usage from routing
            routing_usage = routing.token_usage or {"input": 0, "output": 0, "total": 0}
            
            # Process based on route
            response = None
            if routing.query_type == QueryType.RAG:
                response = self._handle_rag(query, history, allowed_tables, language)
            elif routing.query_type == QueryType.SQL:
                response = self._handle_sql(query, schema_context, history, allowed_tables, language)
            elif routing.query_type == QueryType.HYBRID:
                response = self._handle_hybrid(query, schema_context, history, allowed_tables, language)
            else:
                response = self._handle_general(query, history, language)
                
            # Add routing tokens to total
            if response.token_usage:
                response.token_usage["input"] += routing_usage.get("input", 0)
                response.token_usage["output"] += routing_usage.get("output", 0)
                response.token_usage["total"] += routing_usage.get("total", 0)
            else:
                response.token_usage = routing_usage
                
            return response
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ChatResponse(answer=f"Error: {str(e)}", query_type="error", error=str(e))
    
    def _handle_rag(self, query: str, history: List[Dict], allowed_tables: Optional[List[str]] = None, language: str = "English") -> ChatResponse:
        """Handle RAG-based query."""
        # Check if we have any indexed data
        if self.rag_engine.document_count == 0:
            # Even for this error, we consumed tokens up to the routing decision, but since
            # routing happens before this function, we can't easily track that here.
            # However, we can return empty usage.
            usage = {"input": 0, "output": 0, "total": 0}
            return ChatResponse(
                answer="⚠️ **I can't answer this yet.**\n\nThis looks like a semantic question (searching for meaning/concepts), but you haven't **indexed the text data** yet.\n\nPlease click the **'📚 Index Text Data'** button in the sidebar to enable this functionality.",
                query_type="error",
                error="RAG index is empty",
                token_usage=usage
            )

        context = self.rag_engine.get_context(query, top_k=5, table_filter=allowed_tables)
        
        # Get language instruction
        language_instruction = self._get_language_instruction(language)
        
        prompt = self.RESPONSE_PROMPT.format(
            context=f"RELEVANT DATA:\n{context}", 
            question=query,
            language_instruction=language_instruction
        )
        
        messages = self._construct_messages(
            "You are a helpful database assistant.",
            history, 
            prompt
        )
        
        response = self.llm_client.chat(messages)
        
        usage = {
            "input": response.input_tokens,
            "output": response.output_tokens,
            "total": response.total_tokens
        }
        
        return ChatResponse(answer=response.content, query_type="rag",
                          sources=[{"type": "semantic_search", "context": context[:500]}],
                          token_usage=usage)
    
    def _handle_sql(self, query: str, schema_context: str, history: List[Dict], allowed_tables: Optional[List[str]] = None, language: str = "English") -> ChatResponse:
        """Handle SQL-based query."""
        sql, gen_response = self.sql_generator.generate(query, schema_context, history)
        
        # Initial usage from SQL generation
        total_usage = {
            "input": gen_response.input_tokens,
            "output": gen_response.output_tokens,
            "total": gen_response.total_tokens
        }
        
        # Validate SQL
        is_valid, msg, sanitized_sql = self.sql_validator.validate(sql)
        if not is_valid:
            return ChatResponse(answer=f"Could not generate safe query: {msg}",
                              query_type="sql", error=msg, token_usage=total_usage)
        
        # Execute query
        try:
            results = self.db.execute_query(sanitized_sql)
        except Exception as e:
            return ChatResponse(answer=f"Query execution failed: {e}",
                              query_type="sql", sql_query=sanitized_sql, error=str(e),
                              token_usage=total_usage)
        
        # SMART FALLBACK: If SQL returns nothing, it might be a semantic issue (e.g. wrong column)
        # We try RAG as a fallback if SQL found nothing
        if not results:
            logger.info(f"SQL returned no results for query: '{query}'. Falling back to RAG.")
            rag_response = self._handle_rag(query, history, allowed_tables, language)
            
            # Combine the info: "I couldn't find an exact match in the rows, but here is what I found semantically:"
            rag_response.answer = f"I couldn't find a direct match using a database query, but here is what I found in the product descriptions:\n\n{rag_response.answer}"
            rag_response.query_type = "hybrid_fallback"
            rag_response.sql_query = sanitized_sql
            
            # Add usage from SQL gen to RAG usage
            if rag_response.token_usage:
                rag_response.token_usage["input"] += total_usage["input"]
                rag_response.token_usage["output"] += total_usage["output"]
                rag_response.token_usage["total"] += total_usage["total"]
            else:
                rag_response.token_usage = total_usage
            
            return rag_response

        # Generate response with language instruction
        language_instruction = self._get_language_instruction(language)
        context = f"SQL QUERY:\n{sanitized_sql}\n\nRESULTS:\n{self._format_results(results)}"
        prompt = self.RESPONSE_PROMPT.format(
            context=context, 
            question=query,
            language_instruction=language_instruction
        )
        
        messages = self._construct_messages(
            "You are a helpful database assistant.",
            history, 
            prompt
        )
        
        final_response = self.llm_client.chat(messages)
        
        # Add usage from final response
        total_usage["input"] += final_response.input_tokens
        total_usage["output"] += final_response.output_tokens
        total_usage["total"] += final_response.total_tokens
        
        return ChatResponse(answer=final_response.content, query_type="sql",
                          sql_query=sanitized_sql, sql_results=results[:10],
                          token_usage=total_usage)
    
    def _handle_hybrid(self, query: str, schema_context: str, history: List[Dict], allowed_tables: Optional[List[str]] = None, language: str = "English") -> ChatResponse:
        """Handle hybrid RAG + SQL query."""
        # Get RAG context
        rag_context = self.rag_engine.get_context(query, top_k=3, table_filter=allowed_tables)
        
        # Try SQL as well
        sql_context = ""
        sql_query = None
        
        total_usage = {"input": 0, "output": 0, "total": 0}
        
        try:
            sql, gen_response = self.sql_generator.generate(query, schema_context, history)
            
            # Accumulate usage
            total_usage["input"] += gen_response.input_tokens
            total_usage["output"] += gen_response.output_tokens
            total_usage["total"] += gen_response.total_tokens
            
            is_valid, _, sanitized_sql = self.sql_validator.validate(sql)
            if is_valid:
                results = self.db.execute_query(sanitized_sql)
                sql_context = f"\nSQL RESULTS:\n{self._format_results(results)}"
                sql_query = sanitized_sql
        except Exception as e:
            logger.debug(f"SQL part of hybrid failed: {e}")
        
        # Get language instruction
        language_instruction = self._get_language_instruction(language)
        
        context = f"SEMANTIC SEARCH RESULTS:\n{rag_context}{sql_context}"
        prompt = self.RESPONSE_PROMPT.format(
            context=context, 
            question=query,
            language_instruction=language_instruction
        )
        
        messages = self._construct_messages(
            "You are a helpful database assistant.",
            history, 
            prompt
        )
        
        final_response = self.llm_client.chat(messages)
        
        # Add final usage
        total_usage["input"] += final_response.input_tokens
        total_usage["output"] += final_response.output_tokens
        total_usage["total"] += final_response.total_tokens
        
        return ChatResponse(answer=final_response.content, query_type="hybrid", sql_query=sql_query, token_usage=total_usage)
    
    def _construct_messages(self, system_instruction: str, history: List[Dict], user_content: str) -> List[Dict]:
        """Construct message list, merging system messages from history."""
        # Check if first history item is a system message (from memory)
        additional_context = ""
        filtered_history = []
        
        for msg in history:
            if msg.get("role") == "system":
                additional_context += f"\n\n{msg.get('content')}"
            else:
                filtered_history.append(msg)
                
        full_system_prompt = f"{system_instruction}{additional_context}"
        
        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(filtered_history)
        messages.append({"role": "user", "content": user_content})
        
        return messages

    def _handle_general(self, query: str, history: List[Dict], language: str = "English") -> ChatResponse:
        """Handle conversation."""
        # Get language instruction
        language_instruction = self._get_language_instruction(language)
        
        # Build language suffix for system prompt
        language_suffix = ""
        if language != "English":
            base_language = language
            if "(" in language and ")" in language:
                base_language = language.split("(")[1].rstrip(")")
            language_suffix = f"\n- Respond entirely in {base_language}."
        
        # Use a strict prompt for general conversation as well to prevent hallucinations
        strict_system_prompt = (
            "You are a helpful database assistant.\n"
            "INSTRUCTIONS:\n"
            "- Answer ONLY based on the conversation history and any context provided within it.\n"
            "- Do NOT use outside knowledge, general assumptions, or hallucinate facts.\n"
            "- If the answer is not in the history or context, state that you don't have that information.\n"
            f"- Be concise.{language_suffix}"
        )
        
        messages = self._construct_messages(
            strict_system_prompt,
            history, 
            query
        )
        response = self.llm_client.chat(messages)
        
        usage = {
            "input": response.input_tokens,
            "output": response.output_tokens,
            "total": response.total_tokens
        }
        
        return ChatResponse(answer=response.content, query_type="general", token_usage=usage)
    
    def _format_results(self, results: List[Dict], max_rows: int = 10) -> str:
        """Format SQL results for display."""
        if not results:
            return "No results found."
        
        rows = results[:max_rows]
        lines = []
        
        # Header
        headers = list(rows[0].keys())
        lines.append(" | ".join(headers))
        lines.append("-" * len(lines[0]))
        
        # Rows
        for row in rows:
            values = [str(v)[:50] for v in row.values()]
            lines.append(" | ".join(values))
        
        if len(results) > max_rows:
            lines.append(f"... and {len(results) - max_rows} more rows")
        
        return "\n".join(lines)
    
    def get_schema_summary(self) -> str:
        """Get a summary of the database schema."""
        if not self._schema_initialized:
            return "Schema not loaded."
        return self.introspector.introspect().to_context_string()


def create_chatbot(llm_client: Optional[LLMClient] = None) -> DatabaseChatbot:
    return DatabaseChatbot(llm_client)
