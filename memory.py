"""
Chat Memory - Short-term and long-term memory management.

Supports MySQL, PostgreSQL, and SQLite with dialect-specific DDL.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def get_memory_table_ddl(db_type: str) -> str:
    """Get the DDL for chat memory table based on database type."""
    if db_type == "postgresql":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_memory (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL DEFAULT 'default',
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    elif db_type == "sqlite":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'default',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    else:  # MySQL
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_memory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL DEFAULT 'default',
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id),
            INDEX idx_user (user_id),
            INDEX idx_created (created_at)
        )
        """


def get_permanent_memory_ddl(db_type: str) -> str:
    """Get the DDL for permanent memory table based on database type."""
    if db_type == "postgresql":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_permanent_memory_v2 (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL DEFAULT 'default',
            content TEXT NOT NULL,
            tags VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    elif db_type == "sqlite":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_permanent_memory_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            content TEXT NOT NULL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    else:  # MySQL
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_permanent_memory_v2 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL DEFAULT 'default',
            content TEXT NOT NULL,
            tags VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user (user_id)
        )
        """


def get_summary_table_ddl(db_type: str) -> str:
    """Get the DDL for summary table based on database type."""
    if db_type == "postgresql":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_user_summaries (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            message_count INT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    elif db_type == "sqlite":
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_user_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    else:  # MySQL
        return """
        CREATE TABLE IF NOT EXISTS _chatbot_user_summaries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            summary TEXT NOT NULL,
            message_count INT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY idx_user (user_id)
        )
        """


def get_upsert_summary_query(db_type: str) -> str:
    """Get the upsert query for summary based on database type."""
    if db_type == "postgresql":
        return """
            INSERT INTO _chatbot_user_summaries 
            (user_id, summary, message_count, last_updated)
            VALUES (:user_id, :summary, :message_count, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                summary = EXCLUDED.summary, 
                message_count = EXCLUDED.message_count,
                last_updated = CURRENT_TIMESTAMP
        """
    elif db_type == "sqlite":
        return """
            INSERT INTO _chatbot_user_summaries 
            (user_id, summary, message_count, last_updated)
            VALUES (:user_id, :summary, :message_count, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) 
            DO UPDATE SET 
                summary = excluded.summary, 
                message_count = excluded.message_count,
                last_updated = CURRENT_TIMESTAMP
        """
    else:  # MySQL
        return """
            INSERT INTO _chatbot_user_summaries 
            (user_id, summary, message_count)
            VALUES (:user_id, :summary, :message_count)
            ON DUPLICATE KEY UPDATE 
                summary = :summary, 
                message_count = :message_count,
                last_updated = CURRENT_TIMESTAMP
        """


class ChatMemory:
    """Manages chat history with short-term and long-term storage."""
    
    def __init__(self, session_id: str, user_id: str = "default", max_messages: int = 20, db_connection=None):
        self.session_id = session_id
        self.user_id = user_id
        self.max_messages = max_messages
        self.db = db_connection
        self.messages: List[ChatMessage] = []
        self._db_type = None
        
        if self.db:
            self._db_type = self.db.db_type.value
            self._ensure_tables()
    
    def _ensure_tables(self):
        """Create memory tables if they don't exist."""
        try:
            memory_ddl = get_memory_table_ddl(self._db_type)
            permanent_ddl = get_permanent_memory_ddl(self._db_type)
            
            self.db.execute_write(memory_ddl)
            self.db.execute_write(permanent_ddl)
            
            # Create indexes for SQLite and PostgreSQL (MySQL creates them inline)
            if self._db_type in ("sqlite", "postgresql"):
                self._create_indexes()
            
            # Migration: Ensure user_id column exists (MySQL only for legacy support)
            if self._db_type == "mysql":
                self._migrate_mysql_user_id()
                
        except Exception as e:
            logger.warning(f"Failed to create memory tables: {e}")
    
    def _create_indexes(self):
        """Create indexes for SQLite and PostgreSQL."""
        try:
            if self._db_type == "sqlite":
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_session ON _chatbot_memory(session_id)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_user ON _chatbot_memory(user_id)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_created ON _chatbot_memory(created_at)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_permanent_user ON _chatbot_permanent_memory_v2(user_id)")
            elif self._db_type == "postgresql":
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_session ON _chatbot_memory(session_id)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_user ON _chatbot_memory(user_id)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_memory_created ON _chatbot_memory(created_at)")
                self.db.execute_write("CREATE INDEX IF NOT EXISTS idx_permanent_user ON _chatbot_permanent_memory_v2(user_id)")
        except Exception as e:
            logger.debug(f"Index creation (may already exist): {e}")
    
    def _migrate_mysql_user_id(self):
        """Migrate MySQL table to include user_id column if missing."""
        try:
            check_query = """
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = :db_name 
                AND TABLE_NAME = '_chatbot_memory' 
                AND COLUMN_NAME = 'user_id'
            """
            db_name = self.db.config.database
            result = self.db.execute_query(check_query, {"db_name": db_name})
            
            if not result:
                self.db.execute_write("ALTER TABLE _chatbot_memory ADD COLUMN user_id VARCHAR(255) NOT NULL DEFAULT 'default' AFTER session_id")
                self.db.execute_write("CREATE INDEX idx_user ON _chatbot_memory(user_id)")
                logger.info("Migrated _chatbot_memory to include user_id")
        except Exception as e:
            logger.debug(f"Migration check failed: {e}")
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to memory and optionally persist it."""
        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        
        # Trim if exceeds max (short-term)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
        # Persist to DB (session history)
        if self.db:
            try:
                query = """
                    INSERT INTO _chatbot_memory (session_id, user_id, role, content, metadata)
                    VALUES (:session_id, :user_id, :role, :content, :metadata)
                """
                self.db.execute_write(query, {
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "role": role,
                    "content": content,
                    "metadata": json.dumps(metadata) if metadata else None
                })
            except Exception as e:
                logger.warning(f"Failed to persist message: {e}")
    
    def save_permanent_context(self, content: str, tags: str = "user_saved"):
        """Save specific context explicitly to permanent memory for this user."""
        if not self.db:
            return False, "No database connection"
        
        try:
            query = """
                INSERT INTO _chatbot_permanent_memory_v2 (user_id, content, tags)
                VALUES (:user_id, :content, :tags)
            """
            self.db.execute_write(query, {
                "user_id": self.user_id,
                "content": content, 
                "tags": tags
            })
            return True, "Context saved to permanent memory"
        except Exception as e:
            logger.error(f"Failed to save permanent context: {e}")
            return False, str(e)

    def get_permanent_context(self, limit: int = 5) -> List[str]:
        """Retrieve recent permanent context for this user only."""
        if not self.db:
            return []
            
        try:
            # Use database-agnostic LIMIT syntax
            query = """
                SELECT content FROM _chatbot_permanent_memory_v2 
                WHERE user_id = :user_id 
                ORDER BY created_at DESC LIMIT :limit
            """
            rows = self.db.execute_query(query, {
                "user_id": self.user_id,
                "limit": limit
            })
            return [row['content'] for row in rows]
        except Exception as e:
            logger.warning(f"Failed to load permanent context: {e}")
            return []
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages for LLM context."""
        msgs = self.messages if limit is None else self.messages[-limit:]
        return [m.to_dict() for m in msgs]
    
    def get_context_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """Get recent messages plus permanent context for injection."""
        # Get short-term session messages
        context = self.get_messages(limit=count)
        
        # Inject permanent memory if available
        perm_docs = self.get_permanent_context(limit=3)
        if perm_docs:
            perm_context = f"IMPORTANT CONTEXT FOR USER '{self.user_id}':\n" + "\n".join(perm_docs)
            # Add as a system note at the start
            context.insert(0, {"role": "system", "content": perm_context})
            
        return context
    
    def clear(self):
        """Clear current session memory and remove from DB (temporary history)."""
        self.messages = []
        
        if self.db:
            try:
                # Delete temporary messages for this session
                query = "DELETE FROM _chatbot_memory WHERE session_id = :session_id"
                self.db.execute_write(query, {"session_id": self.session_id})
                logger.info(f"Cleared session memory for {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to clear memory from DB: {e}")

    def clear_user_history(self):
        """Clear ALL temporary history for this user (across all sessions)."""
        self.messages = []
        if self.db:
            try:
                query = "DELETE FROM _chatbot_memory WHERE user_id = :user_id"
                self.db.execute_write(query, {"user_id": self.user_id})
                logger.info(f"Cleared all temporary history for user: {self.user_id}")
            except Exception as e:
                logger.warning(f"Failed to clear user history from DB: {e}")

    def get_user_sessions(self) -> List[Dict[str, Any]]:
        """Get a list of all chat sessions for the current user."""
        if not self.db:
            return []
            
        try:
            # Group by session_id and get the first message time + preview
            # Note: This query needs to be compatible with supported DBs
            query = """
                SELECT session_id, MIN(created_at) as created_at, 
                       (SELECT content FROM _chatbot_memory m2 
                        WHERE m2.session_id = m1.session_id 
                        AND role = 'user' 
                        ORDER BY id ASC LIMIT 1) as title
                FROM _chatbot_memory m1
                WHERE user_id = :user_id
                GROUP BY session_id
                ORDER BY created_at DESC
            """
            rows = self.db.execute_query(query, {"user_id": self.user_id})
            
            sessions = []
            for row in rows:
                title = row.get("title") or "New Chat"
                if len(title) > 30:
                    title = title[:30] + "..."
                
                sessions.append({
                    "id": row.get("session_id"),
                    "created_at": row.get("created_at"),
                    "title": title
                })
            return sessions
        except Exception as e:
            logger.warning(f"Failed to fetch user sessions: {e}")
            return []

    def load_session(self, session_id: str) -> List[Dict]:
        """Load detailed messages for a specific session."""
        if not self.db:
            return []
            
        try:
            query = """
                SELECT role, content, metadata 
                FROM _chatbot_memory 
                WHERE session_id = :session_id AND user_id = :user_id
                ORDER BY id ASC
            """
            rows = self.db.execute_query(query, {
                "session_id": session_id,
                "user_id": self.user_id
            })
            
            messages = []
            for row in rows:
                meta = row.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except:
                        meta = {}
                elif meta is None:
                    meta = {}
                    
                messages.append({
                    "role": row.get("role"),
                    "content": row.get("content"),
                    "metadata": meta
                })
            return messages
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return []


class ConversationSummaryMemory:
    """
    Per-user conversation summary memory using LLM for summarization.
    
    This class maintains a running summary of the conversation, updating it
    periodically (when message count exceeds threshold). This dramatically
    reduces token usage while preserving context for long conversations.
    
    Features:
    - Automatic summarization when threshold is reached
    - Per-user summary storage in database
    - Combines summary + recent messages for optimal context
    - Lazy summarization (only when needed)
    """
    
    SUMMARIZATION_PROMPT = """You are a conversation summarizer. Create a concise summary of the conversation below that captures:
1. Key topics discussed
2. Important facts or preferences mentioned by the user
3. Any decisions or conclusions reached
4. Context needed for follow-up questions

Keep the summary under 300 words but include all important details.

CONVERSATION:
{conversation}

SUMMARY:"""

    INCREMENTAL_SUMMARY_PROMPT = """You are a conversation summarizer. Update the existing summary to incorporate new messages.

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES:
{new_messages}

Create an updated, comprehensive summary that:
1. Incorporates new information from the recent messages
2. Retains important context from the existing summary
3. Removes redundant or outdated information
4. Stays under 300 words

UPDATED SUMMARY:"""

    def __init__(
        self, 
        user_id: str, 
        session_id: str, 
        db_connection=None,
        llm_client=None,
        summary_threshold: int = 10,  # Summarize every N messages
        recent_messages_count: int = 5  # Keep this many recent messages verbatim
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.db = db_connection
        self.llm = llm_client
        self.summary_threshold = summary_threshold
        self.recent_messages_count = recent_messages_count
        self._db_type = None
        
        self._cached_summary: Optional[str] = None
        self._messages_since_summary: int = 0
        
        if self.db:
            self._db_type = self.db.db_type.value
            self._ensure_tables()
            self._load_state()
    
    def _ensure_tables(self):
        """Create summary table if it doesn't exist."""
        try:
            ddl = get_summary_table_ddl(self._db_type)
            self.db.execute_write(ddl)
        except Exception as e:
            logger.warning(f"Failed to create summary table: {e}")
    
    def _load_state(self):
        """Load existing summary state from database (per-user, not per-session)."""
        try:
            query = """
                SELECT summary, message_count FROM _chatbot_user_summaries 
                WHERE user_id = :user_id
            """
            rows = self.db.execute_query(query, {
                "user_id": self.user_id
            })
            if rows:
                self._cached_summary = rows[0].get('summary')
                self._messages_since_summary = 0  # Reset since we loaded
                logger.debug(f"Loaded summary for user {self.user_id}")
        except Exception as e:
            logger.warning(f"Failed to load summary state: {e}")
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for summarization."""
        self.llm = llm_client
    
    def on_message_added(self, message_count: int):
        """
        Called after a message is added to track when to summarize.
        
        Args:
            message_count: Current total number of messages in the conversation
        """
        self._messages_since_summary += 1
        
        # Check if we should summarize
        if self._messages_since_summary >= self.summary_threshold:
            self._trigger_summarization()
    
    def _trigger_summarization(self):
        """Trigger summarization of the conversation."""
        if not self.llm:
            logger.warning("Cannot summarize: No LLM client configured")
            return
        
        if not self.db:
            logger.warning("Cannot summarize: No database connection")
            return
        
        try:
            # Get messages that need to be summarized
            query = """
                SELECT role, content FROM _chatbot_memory 
                WHERE user_id = :user_id AND session_id = :session_id
                ORDER BY created_at ASC
            """
            rows = self.db.execute_query(query, {
                "user_id": self.user_id,
                "session_id": self.session_id
            })
            
            if not rows:
                return
            
            # Format conversation for summarization
            conversation_text = self._format_messages_for_summary(rows)
            
            # Generate summary
            if self._cached_summary:
                # Incremental update
                prompt = self.INCREMENTAL_SUMMARY_PROMPT.format(
                    existing_summary=self._cached_summary,
                    new_messages=conversation_text
                )
            else:
                # Fresh summary
                prompt = self.SUMMARIZATION_PROMPT.format(conversation=conversation_text)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
                {"role": "user", "content": prompt}
            ]
            
            summary = self.llm.chat(messages)
            
            # Save to database
            self._save_summary(summary, len(rows))
            
            self._cached_summary = summary
            self._messages_since_summary = 0
            
            logger.info(f"Generated summary for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
    
    def _format_messages_for_summary(self, messages: List[Dict]) -> str:
        """Format messages as text for summarization."""
        lines = []
        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
    
    def _save_summary(self, summary: str, message_count: int):
        """Save or update summary in database (per-user)."""
        try:
            query = get_upsert_summary_query(self._db_type)
            self.db.execute_write(query, {
                "user_id": self.user_id,
                "summary": summary,
                "message_count": message_count
            })
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def get_summary(self) -> Optional[str]:
        """Get the current conversation summary."""
        return self._cached_summary
    
    def get_context_for_llm(self, recent_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Get optimized context for LLM calls.
        
        Combines the summary (if available) with recent messages for optimal
        token usage while maintaining context.
        
        Args:
            recent_messages: List of recent messages to include verbatim
            
        Returns:
            List of messages with summary prepended as system context
        """
        context_messages = []
        
        # Add summary as system context if available
        if self._cached_summary:
            summary_context = f"""CONVERSATION SUMMARY (previous context):
{self._cached_summary}

Use this summary to understand the conversation history and context for follow-up questions."""
            context_messages.append({
                "role": "system",
                "content": summary_context
            })
        
        # Add recent messages verbatim
        context_messages.extend(recent_messages[-self.recent_messages_count:])
        
        return context_messages
    
    def force_summarize(self):
        """Force immediate summarization regardless of threshold."""
        self._trigger_summarization()
    
    def clear_summary(self):
        """Clear the summary for this user."""
        self._cached_summary = None
        self._messages_since_summary = 0
        
        if self.db:
            try:
                query = "DELETE FROM _chatbot_user_summaries WHERE user_id = :user_id"
                self.db.execute_write(query, {
                    "user_id": self.user_id
                })
                logger.info(f"Cleared summary for user: {self.user_id}")
            except Exception as e:
                logger.warning(f"Failed to clear summary: {e}")
    
    def clear_all_user_summaries(self):
        """Clear all summaries for this user (alias for clear_summary since it's now per-user)."""
        self.clear_summary()


class EnhancedChatMemory(ChatMemory):
    """
    Enhanced ChatMemory with integrated conversation summarization.
    
    Combines the standard ChatMemory functionality with ConversationSummaryMemory
    for automatic summarization and optimized context retrieval.
    """
    
    def __init__(
        self, 
        session_id: str, 
        user_id: str = "default", 
        max_messages: int = 20, 
        db_connection=None,
        llm_client=None,
        enable_summarization: bool = True,
        summary_threshold: int = 10
    ):
        super().__init__(session_id, user_id, max_messages, db_connection)
        
        self.enable_summarization = enable_summarization
        self.summary_memory: Optional[ConversationSummaryMemory] = None
        
        if enable_summarization:
            self.summary_memory = ConversationSummaryMemory(
                user_id=user_id,
                session_id=session_id,
                db_connection=db_connection,
                llm_client=llm_client,
                summary_threshold=summary_threshold
            )
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for summarization."""
        if self.summary_memory:
            self.summary_memory.set_llm_client(llm_client)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message and trigger summarization check."""
        super().add_message(role, content, metadata)
        
        # Notify summary memory of new message
        if self.summary_memory:
            self.summary_memory.on_message_added(len(self.messages))
    
    def get_context_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Get context messages with summary integration.
        
        If summarization is enabled and a summary exists, it will be 
        prepended to provide historical context while keeping recent
        messages verbatim.
        """
        # Get base context from parent
        base_context = super().get_context_messages(count)
        
        # If summarization is enabled, use summary-enhanced context
        if self.summary_memory and self.summary_memory.get_summary():
            # Filter out system messages from base context (we'll add summary separately)
            filtered = [m for m in base_context if m.get("role") != "system"]
            
            # Get summary-enhanced context
            enhanced = self.summary_memory.get_context_for_llm(filtered)
            
            # Re-add permanent memory context if it was present
            for msg in base_context:
                if msg.get("role") == "system" and "IMPORTANT CONTEXT" in msg.get("content", ""):
                    enhanced.insert(0, msg)
            
            return enhanced
        
        return base_context
    
    def get_summary(self) -> Optional[str]:
        """Get the current conversation summary."""
        if self.summary_memory:
            return self.summary_memory.get_summary()
        return None
    
    def force_summarize(self):
        """Force immediate summarization."""
        if self.summary_memory:
            self.summary_memory.force_summarize()
    
    def clear(self):
        """Clear session memory but KEEP the summary (long-term memory)."""
        super().clear()
        # NOTE: Summary is intentionally NOT cleared here
        # Summary acts as long-term memory that persists across chat sessions
    
    def clear_with_summary(self):
        """Clear session memory AND the summary (full reset)."""
        super().clear()
        if self.summary_memory:
            self.summary_memory.clear_summary()
    
    def clear_user_history(self):
        """Clear all user temp history but KEEP summaries."""
        super().clear_user_history()
        # NOTE: Summaries are intentionally NOT cleared
        # They persist as long-term memory for the user
    
    def clear_all_including_summaries(self):
        """Clear ALL user data including summaries (complete wipe)."""
        super().clear_user_history()
        if self.summary_memory:
            self.summary_memory.clear_all_user_summaries()


def create_memory(session_id: str, user_id: str = "default", max_messages: int = 20) -> ChatMemory:
    """Create a standard ChatMemory instance."""
    from database import get_db
    db = get_db()
    return ChatMemory(session_id=session_id, user_id=user_id, max_messages=max_messages, db_connection=db)


def create_enhanced_memory(
    session_id: str, 
    user_id: str = "default", 
    max_messages: int = 20,
    llm_client=None,
    enable_summarization: bool = True,
    summary_threshold: int = 10
) -> EnhancedChatMemory:
    """
    Create an EnhancedChatMemory with summarization support.
    
    Args:
        session_id: Unique session identifier
        user_id: User identifier for per-user memory isolation
        max_messages: Maximum messages to keep in short-term memory
        llm_client: LLM client for summarization (can be set later)
        enable_summarization: Whether to enable automatic summarization
        summary_threshold: Summarize after this many messages
        
    Returns:
        EnhancedChatMemory instance with summarization capabilities
    """
    from database import get_db
    db = get_db()
    return EnhancedChatMemory(
        session_id=session_id,
        user_id=user_id,
        max_messages=max_messages,
        db_connection=db,
        llm_client=llm_client,
        enable_summarization=enable_summarization,
        summary_threshold=summary_threshold
    )
