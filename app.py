"""
Schema-Agnostic Database Chatbot - Streamlit Application

A production-grade chatbot that connects to ANY database
(MySQL, PostgreSQL, SQLite) and provides intelligent querying 
through RAG and Text-to-SQL.

Uses Groq for FREE LLM inference!
"""

import os
from pathlib import Path

# Load .env FIRST before any other imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
import uuid
import time
import io
import csv
import base64
import pandas as pd
from datetime import datetime

# Page config must be first
st.set_page_config(
    page_title="OnceDataBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from config import config, DatabaseConfig, DatabaseType
from database import get_db, get_schema, get_introspector
from database.connection import DatabaseConnection
from llm import create_llm_client
from chatbot import create_chatbot, DatabaseChatbot
from memory import ChatMemory, EnhancedChatMemory
from viz_utils import render_visualization






# Groq models (all FREE!)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Database types
DB_TYPES = {
    "MySQL": "mysql",
    "PostgreSQL": "postgresql",
    "SQLite": "sqlite"
}

# Supported languages for multi-language responses
SUPPORTED_LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "Español (Spanish)": "es",
    "Français (French)": "fr",
    "Deutsch (German)": "de",
    "中文 (Chinese)": "zh",
    "日本語 (Japanese)": "ja",
    "한국어 (Korean)": "ko",
    "Português (Portuguese)": "pt",
    "العربية (Arabic)": "ar",
    "Русский (Russian)": "ru",
    "Italiano (Italian)": "it",
    "Nederlands (Dutch)": "nl",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
    "বাংলা (Bengali)": "bn",
    "ગુજરાতી (Gujarati)": "gu"
}


def create_custom_db_config(db_type: str, **kwargs) -> DatabaseConfig:
    """Create a custom database configuration from user input."""
    return DatabaseConfig(
        db_type=DatabaseType(db_type),
        host=kwargs.get("host", ""),
        port=kwargs.get("port", 3306 if db_type == "mysql" else 5432),
        database=kwargs.get("database", ""),
        username=kwargs.get("username", ""),
        password=kwargs.get("password", ""),
        ssl_ca=kwargs.get("ssl_ca", None)
    )


def create_custom_memory(session_id: str, user_id: str, db_connection, llm_client=None, 
                         enable_summarization=True, summary_threshold=10) -> EnhancedChatMemory:
    """Create enhanced memory with a custom database connection."""
    return EnhancedChatMemory(
        session_id=session_id,
        user_id=user_id,
        max_messages=20,
        db_connection=db_connection,
        llm_client=llm_client,
        enable_summarization=enable_summarization,
        summary_threshold=summary_threshold
    )


def init_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default"
    
    if "enable_summarization" not in st.session_state:
        st.session_state.enable_summarization = True
    
    if "summary_threshold" not in st.session_state:
        st.session_state.summary_threshold = 10
    
    if "memory" not in st.session_state:
        st.session_state.memory = None
        
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    
    if "db_source" not in st.session_state:
        st.session_state.db_source = "environment"  # "environment" or "custom"
    
    if "custom_db_config" not in st.session_state:
        st.session_state.custom_db_config = None
    
    if "custom_db_connection" not in st.session_state:
        st.session_state.custom_db_connection = None
        
    if "ignored_tables" not in st.session_state:
        st.session_state.ignored_tables = set()
    
    if "response_language" not in st.session_state:
        st.session_state.response_language = "English"
    
    if "favorites" not in st.session_state:
        st.session_state.favorites = []  # List of message indices that are favorited


def export_results_to_csv(results: list) -> str:
    """Convert SQL results to CSV format and return as downloadable string."""
    if not results:
        return ""
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


def export_chat_to_text() -> str:
    """Export chat messages to text format."""
    if not st.session_state.messages:
        return "No messages to export."
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"OnceDataBot Chat Export")
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"User: {st.session_state.user_id}")
    lines.append("=" * 50)
    lines.append("")
    
    for i, msg in enumerate(st.session_state.messages):
        role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
        is_favorited = "⭐ " if i in st.session_state.favorites else ""
        lines.append(f"{is_favorited}{role}:")
        lines.append(msg["content"])
        
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            if meta.get("sql_query"):
                lines.append(f"\n📝 SQL Query: {meta['sql_query']}")
            if meta.get("query_type"):
                lines.append(f"📌 Query Type: {meta['query_type']}")
            if meta.get("execution_time"):
                lines.append(f"⏱️ Execution Time: {meta['execution_time']:.2f}s")
        
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def render_copy_button(text: str, key: str):
    """Render a copy to clipboard button using Streamlit."""
    # Using a workaround with st.code which has built-in copy
    st.code(text, language="sql")


def render_database_config():
    """Render database configuration section in sidebar."""
    st.subheader("🗄️ Database Configuration")
    
    # Database source selection
    db_source = st.radio(
        "Database Source",
        options=["Use Environment Variables", "Custom Database"],
        index=0 if st.session_state.db_source == "environment" else 1,
        key="db_source_radio",
        help="Choose to use .env settings or enter custom credentials"
    )
    
    st.session_state.db_source = "environment" if db_source == "Use Environment Variables" else "custom"
    
    if st.session_state.db_source == "environment":
        # Show current environment config
        current_db_type = config.database.db_type.value.upper()
        st.info(f"📌 Using {current_db_type} from environment")
        st.caption(f"Host: {config.database.host}")
        return None
    
    else:
        # Custom database configuration
        st.markdown("##### Enter Database Credentials")
        
        # Database type selector
        db_type_label = st.selectbox(
            "Database Type",
            options=list(DB_TYPES.keys()),
            index=0,
            key="custom_db_type"
        )
        db_type = DB_TYPES[db_type_label]
        
        if db_type == "sqlite":
            # SQLite only needs a file path
            database = st.text_input(
                "SQLite Database File",
                value="ingested_data.db",
                key="db_sqlite_path",
                help="Path to the .db file (will be created if it doesn't exist)"
            )
            return {
                "db_type": db_type,
                "database": database
            }
        
        else:  # MySQL or PostgreSQL
            # MySQL or PostgreSQL
            col1, col2 = st.columns([3, 1])
            with col1:
                host = st.text_input(
                    "Host",
                    value="",
                    key="db_host_input",
                    placeholder="your-database-host.com"
                )
            with col2:
                default_port = 3306 if db_type == "mysql" else 5432
                port = st.number_input(
                    "Port",
                    value=default_port,
                    min_value=1,
                    max_value=65535,
                    key="db_port_input"
                )
            
            database = st.text_input(
                "Database Name",
                value="",
                key="db_name_input",
                placeholder="your_database"
            )
            
            username = st.text_input(
                "Username",
                value="",
                key="db_user_input",
                placeholder="your_username"
            )
            
            password = st.text_input(
                "Password",
                value="",
                type="password",
                key="db_pass_input"
            )
            
            # Optional SSL
            with st.expander("🔒 SSL Settings (Optional)"):
                ssl_ca = st.text_input(
                    "SSL CA Certificate Path",
                    value="",
                    key="ssl_ca_input",
                    help="Path to SSL CA certificate file (for cloud databases like Aiven)"
                )
            
            return {
                "db_type": db_type,
                "host": host,
                "port": int(port),
                "database": database,
                "username": username,
                "password": password,
                "ssl_ca": ssl_ca if ssl_ca else None
            }


def render_sidebar():
    """Render the configuration sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Session Dashboard
        if st.session_state.messages:
            st.markdown("### 📊 Session Stats")
            
            # Calculate stats
            total_msgs = len(st.session_state.messages)
            assistant_msgs = [m for m in st.session_state.messages if m.get("role") == "assistant"]
            sql_queries = sum(1 for m in assistant_msgs if m.get("metadata", {}).get("sql_query"))
            
            total_tokens = 0
            exec_times = []
            for m in assistant_msgs:
                meta = m.get("metadata", {})
                total_tokens += meta.get("token_usage", {}).get("total", 0)
                if meta.get("execution_time"):
                    exec_times.append(meta["execution_time"])
            
            avg_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Queries", sql_queries)
            col_s2.metric("Tokens", f"{total_tokens:,}")
            st.caption(f"⏱️ Avg Time: {avg_time:.2f}s | 💬 Msgs: {total_msgs}")
            st.divider()
        

        
        # User Profile
        st.subheader("👤 User Profile")
        user_id = st.text_input(
            "User ID / Name", 
            value=st.session_state.get("user_id", "default"),
            key="user_id_input",
            help="Your unique ID for private memory storage"
        )
        if user_id != st.session_state.get("user_id"):
            st.session_state.user_id = user_id
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            
            # Recreate memory for new user
            if st.session_state.custom_db_connection:
                st.session_state.memory = create_custom_memory(
                    st.session_state.session_id,
                    user_id,
                    st.session_state.custom_db_connection,
                    st.session_state.get("llm"),
                    st.session_state.enable_summarization,
                    st.session_state.summary_threshold
                )
            elif st.session_state.initialized:
                from memory import create_enhanced_memory
                st.session_state.memory = create_enhanced_memory(
                    st.session_state.session_id,
                    user_id=user_id,
                    enable_summarization=st.session_state.enable_summarization,
                    summary_threshold=st.session_state.summary_threshold
                )
            
            if st.session_state.memory:
                st.session_state.memory.clear_user_history()
            st.rerun()
        
        st.divider()
        
        # Language Selection
        st.subheader("🌐 Response Language")
        selected_language = st.selectbox(
            "Select Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.response_language),
            key="language_selector",
            help="Choose the language for chatbot responses"
        )
        if selected_language != st.session_state.response_language:
            st.session_state.response_language = selected_language
            st.toast(f"🌐 Language changed to {selected_language}")
        
        st.divider()
        
        if st.session_state.messages:
            st.download_button(
                label="📄 Export Chat",
                data=export_chat_to_text(),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download your chat conversation as a text file"
            )
        
        st.divider()
        
        # CSV Ingestion Section
        st.subheader("📥 Ingest CSV Data")
        uploaded_files = st.file_uploader(
            "Upload CSV(s) to create database",
            type=["csv"],
            accept_multiple_files=True,
            help="Your CSVs will be converted to tables in a local SQLite database"
        )
        
        if uploaded_files:
            if st.button("🚀 Upload & Initialize", use_container_width=True):
                with st.spinner("Processing CSVs..."):
                    success_count = 0
                    table_names = []
                    for uploaded_file in uploaded_files:
                        success, name, rows = ingest_csv(uploaded_file)
                        if success:
                            success_count += 1
                            table_names.append(name)
                        else:
                            st.error(f"Failed to ingest {uploaded_file.name}: {name}")
                    
                    if success_count > 0:
                        st.success(f"Successfully ingested {success_count} file(s) as tables: {', '.join(table_names)}")
                        
                        # Now initialize chatbot with this SQLite DB
                        sqlite_params = {
                            "db_type": "sqlite",
                            "database": "ingested_data.db"
                        }
                        # Temporarily set db_source to custom for initialization
                        old_source = st.session_state.db_source
                        st.session_state.db_source = "custom"
                        init_success = initialize_chatbot(sqlite_params, None, None)
                        if not init_success:
                            st.session_state.db_source = old_source
                        else:
                            st.rerun()
        
        st.divider()
        
        # Database Configuration
        custom_db_params = render_database_config()
        
        st.divider()
        
        # LLM Configuration
        st.subheader("🤖 LLM Configuration")
        
        # Show status of API key
        if os.getenv("GROQ_API_KEY"):
            st.success("✓ API Key configured")
        else:
            st.warning("⚠️ GROQ_API_KEY not set in environment")
        
        st.divider()
        
        # Initialize Button
        if st.button("🚀 Connect & Initialize", use_container_width=True, type="primary"):
            with st.spinner("Connecting to database..."):
                success = initialize_chatbot(custom_db_params, None, None)
                if success:
                    st.success("✅ Connected!")
                    st.rerun()
        
        # Index Button (after initialization)
        if st.session_state.initialized:
            if st.button("📚 Index Text Data", use_container_width=True):
                with st.spinner("Indexing text data..."):
                    index_data()
                    st.success("✅ Indexed!")
                    st.rerun()
        
        st.divider()
        
        # Status
        st.subheader("📊 Status")
        if st.session_state.initialized:
            # Show database type
            if st.session_state.custom_db_connection:
                db_type = st.session_state.custom_db_connection.db_type.value.upper()
            else:
                db_type = get_db().db_type.value.upper()
            
            st.success(f"Database: {db_type} ✓")
            
            try:
                schema = get_schema()
                st.info(f"Tables: {len(schema.tables)}")
            except:
                st.warning("Schema not loaded")
            
            if st.session_state.indexed:
                from rag import get_rag_engine
                engine = get_rag_engine()
                st.info(f"Indexed Docs: {engine.document_count}")
        else:
            st.warning("Not connected")
        
        # New Chat
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            if st.session_state.memory:
                st.session_state.memory.clear()
            
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            
            current_user = st.session_state.get("user_id", "default")
            
            if st.session_state.custom_db_connection:
                st.session_state.memory = create_custom_memory(
                    st.session_state.session_id,
                    current_user,
                    st.session_state.custom_db_connection,
                    st.session_state.get("llm"),
                    st.session_state.enable_summarization,
                    st.session_state.summary_threshold
                )
            elif st.session_state.initialized:
                from memory import create_enhanced_memory
                st.session_state.memory = create_enhanced_memory(
                    st.session_state.session_id, 
                    user_id=current_user,
                    enable_summarization=st.session_state.enable_summarization,
                    summary_threshold=st.session_state.summary_threshold
                )
                if st.session_state.get("llm"):
                    st.session_state.memory.set_llm_client(st.session_state.llm)
            
            st.rerun()
        
        # Disconnect button (when using custom DB)
        if st.session_state.initialized and st.session_state.db_source == "custom":
            if st.button("🔌 Disconnect", use_container_width=True):
                if st.session_state.custom_db_connection:
                    st.session_state.custom_db_connection.close()
                st.session_state.custom_db_connection = None
                st.session_state.chatbot = None
                st.session_state.initialized = False
                st.session_state.indexed = False
                st.session_state.memory = None
                st.success("Disconnected!")
                st.rerun()
        
        st.divider()
        
        # Chat History Section
        if st.session_state.memory:
            st.subheader("🕰️ Chat History")
            sessions = st.session_state.memory.get_user_sessions()
            
            if not sessions:
                st.caption("No previous chats found.")
            else:
                for session in sessions:
                    # Highlight current session
                    is_current = session["id"] == st.session_state.session_id
                    icon = "🟢" if is_current else "💬"
                    
                    if st.button(
                        f"{icon} {session['title']}", 
                        key=f"hist_{session['id']}",
                        use_container_width=True,
                        type="secondary" if not is_current else "primary"
                    ):
                        if not is_current:
                            # Load selected session
                            st.session_state.session_id = session["id"]
                            st.session_state.memory.session_id = session["id"]
                            st.session_state.memory.messages = [] # Clear current state local cache
                            
                            # Load from DB
                            msgs = st.session_state.memory.load_session(session["id"])
                            st.session_state.messages = msgs
                            
                            # Re-populate memory object messages list for context
                            # (We need to convert dicts back to ChatMessage objects implicitly or just rely on reload)
                            # Actually, we should probably re-init the memory to be safe or manually populate
                            # Let's manually populate to keep the connection valid
                            from memory import ChatMessage
                            st.session_state.memory.messages = [
                                ChatMessage(
                                    role=m['role'], 
                                    content=m['content'], 
                                    metadata=m.get('metadata')
                                ) for m in msgs
                            ]
                            
                            st.rerun()


def initialize_chatbot(custom_db_params=None, api_key=None, model=None) -> bool:
    """Initialize the chatbot with either environment or custom database."""
    try:
        # Get API key
        groq_api_key = api_key or os.getenv("GROQ_API_KEY", "")
        groq_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not groq_api_key:
            st.error("GROQ_API_KEY not configured. Please enter your API key.")
            return False
        
        # Create LLM client
        llm = create_llm_client("groq", api_key=groq_api_key, model=groq_model)
        
        # Create database connection
        if custom_db_params and st.session_state.db_source == "custom":
            # Validate custom params
            db_type = custom_db_params.get("db_type", "mysql")
            
            if db_type != "sqlite":
                if not all([custom_db_params.get("host"), 
                           custom_db_params.get("database"),
                           custom_db_params.get("username")]):
                    st.error("Please fill in all required database fields.")
                    return False
            else:
                if not custom_db_params.get("database"):
                    st.error("Please specify a SQLite database file path.")
                    return False
            
            # Create custom config
            db_config = create_custom_db_config(**custom_db_params)
            
            # Create custom connection
            custom_connection = DatabaseConnection(db_config)
            
            # Test connection
            success, msg = custom_connection.test_connection()
            if not success:
                st.error(f"Connection failed: {msg}")
                return False
            
            st.session_state.custom_db_connection = custom_connection
            st.session_state.custom_db_config = db_config
            
            # Override the global db connection for the chatbot
            # We need to create a chatbot with this custom connection
            from chatbot import DatabaseChatbot
            from database.schema_introspector import SchemaIntrospector
            from rag import get_rag_engine
            from sql import get_sql_generator, get_sql_validator
            from router import get_query_router
            
            chatbot = DatabaseChatbot.__new__(DatabaseChatbot)
            chatbot.db = custom_connection
            chatbot.introspector = SchemaIntrospector()
            chatbot.introspector.db = custom_connection
            chatbot.rag_engine = get_rag_engine()
            chatbot.sql_generator = get_sql_generator(db_type)
            chatbot.sql_validator = get_sql_validator()
            chatbot.router = get_query_router()
            chatbot.llm_client = llm
            chatbot._schema_initialized = False
            chatbot._rag_initialized = False
            
            # Set LLM client
            chatbot.set_llm_client(llm)
            
            # Initialize (introspect schema)
            schema = chatbot.introspector.introspect(force_refresh=True)
            chatbot.sql_validator.set_allowed_tables(schema.table_names)
            chatbot._schema_initialized = True
            
            st.session_state.chatbot = chatbot
            
        else:
            # Use environment-based connection (existing flow)
            chatbot = create_chatbot(llm)
            chatbot.set_llm_client(llm)
            
            success, msg = chatbot.initialize()
            if not success:
                st.error(f"Initialization failed: {msg}")
                return False
            
            st.session_state.chatbot = chatbot
            st.session_state.custom_db_connection = None
        
        st.session_state.llm = llm
        st.session_state.initialized = True
        st.session_state.indexed = False  # Reset index status on new connection
        
        # Clear RAG index to ensure no data from previous DB connection persists
        if hasattr(chatbot, 'rag_engine') and hasattr(chatbot.rag_engine, 'clear_index'):
            chatbot.rag_engine.clear_index()
        
        # Create memory with appropriate connection
        db_conn = st.session_state.custom_db_connection or get_db()
        st.session_state.memory = create_custom_memory(
            st.session_state.session_id,
            st.session_state.user_id,
            db_conn,
            llm,
            st.session_state.enable_summarization,
            st.session_state.summary_threshold
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def ingest_csv(uploaded_file):
    """Ingest a CSV file into a SQLite database."""
    from sqlalchemy import create_engine
    
    try:
        # 1. Read CSV
        # Reset file pointer to beginning in case it was read before
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # 2. Clean table name from filename
        table_name = Path(uploaded_file.name).stem.replace(" ", "_").replace("-", "_").lower()
        # Ensure it starts with a letter and only contains alphanumeric/underscore
        table_name = "".join([c for c in table_name if c.isalnum() or c == "_"])
        if not table_name[0].isalpha():
            table_name = "t_" + table_name
            
        # 3. Create/Connect to SQLite DB
        db_path = "ingested_data.db"
        engine = create_engine(f"sqlite:///{db_path}")
        
        # 4. Write to DB
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        return True, table_name, len(df)
    except Exception as e:
        return False, str(e), 0


def index_data():
    """Index text data from the database."""
    if st.session_state.chatbot:
        progress = st.progress(0)
        status = st.empty()
        
        # Get schema from the correct introspector
        schema = st.session_state.chatbot.introspector.introspect()
        total_tables = len(schema.tables)
        indexed = 0
        
        def progress_callback(table_name, docs):
            nonlocal indexed
            indexed += 1
            progress.progress(indexed / total_tables)
            status.text(f"Indexed {table_name}: {docs} documents")
        
        total_docs = st.session_state.chatbot.index_text_data(progress_callback)
        st.session_state.indexed = True
        status.text(f"Total: {total_docs} documents indexed")


def render_schema_explorer():
    """Render schema explorer in an expander."""
    if not st.session_state.initialized:
        return
    
    with st.expander("📋 Database Schema", expanded=False):
        try:
            schema = st.session_state.chatbot.introspector.introspect()
            
            tab_list, tab_erd = st.tabs(["📋 Table List", "🕸️ Schema Diagram"])
            
            with tab_list:
                st.markdown("Uncheck tables to exclude them from the chat context.")
                
                for table_name, table_info in schema.tables.items():
                    col1, col2 = st.columns([0.05, 0.95])
                    
                    with col1:
                        is_active = table_name not in st.session_state.ignored_tables
                        active = st.checkbox(
                            "Use", 
                            value=is_active, 
                            key=f"use_{table_name}", 
                            label_visibility="collapsed",
                            help=f"Include {table_name} in chat analysis"
                        )
                        
                        if not active:
                            st.session_state.ignored_tables.add(table_name)
                        else:
                            st.session_state.ignored_tables.discard(table_name)
                    
                    with col2:
                        with st.container():
                            st.markdown(f"**{table_name}** ({table_info.row_count or '?'} rows)")
                            
                            cols = []
                            for col in table_info.columns:
                                pk = "🔑" if col.is_primary_key else ""
                                txt = "📝" if col.is_text_type else ""
                                cols.append(f"`{col.name}` {col.data_type} {pk}{txt}")
                            
                            st.caption(" | ".join(cols))
                            st.divider()
            
            with tab_erd:
                if len(schema.tables) > 50:
                    st.warning("⚠️ Too many tables to visualize effectively (limit: 50).")
                else:
                    try:
                        # Build Graphviz DOT string
                        dot = ['digraph Database {']
                        dot.append('  rankdir=LR;')
                        dot.append('  node [shape=box, style="filled,rounded", fillcolor="#f0f2f6", fontname="Arial", fontsize=10];')
                        dot.append('  edge [fontname="Arial", fontsize=9, color="#666666"];')
                        
                        # Add nodes (tables)
                        for table_name in schema.tables:
                            if table_name not in st.session_state.ignored_tables:
                                dot.append(f'  "{table_name}" [label="{table_name}", fillcolor="#e1effe", color="#1e40af"];')
                            else:
                                dot.append(f'  "{table_name}" [label="{table_name} (ignored)", fillcolor="#f3f4f6", color="#9ca3af", fontcolor="#9ca3af"];')
                        
                        # Add edges (relationships)
                        has_edges = False
                        for table_name, table_info in schema.tables.items():
                            for col_name, ref_str in table_info.foreign_keys.items():
                                # ref_str format: "referenced_table.referenced_column"
                                if "." in ref_str:
                                    ref_table = ref_str.split(".")[0]
                                    # specific_col = ref_str.split(".")[1]
                                    
                                    # Only draw if both tables exist in our schema list
                                    if ref_table in schema.tables:
                                        dot.append(f'  "{table_name}" -> "{ref_table}" [label="{col_name}"];')
                                        has_edges = True
                        
                        dot.append('}')
                        graph_code = "\n".join(dot)
                        st.graphviz_chart(graph_code, width="stretch")
                        
                        if not has_edges:
                            st.info("No foreign key relationships detected in the schema metadata.")
                            
                    except Exception as e:
                        st.error(f"Could not render diagram: {e}")

        except Exception as e:
            st.error(f"Error loading schema: {e}")


def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 OnceDataBot")
    st.caption("Schema-agnostic chatbot • MySQL | PostgreSQL • Powered by Groq (FREE!)")
    
    # Schema explorer
    render_schema_explorer()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                # Create columns for message and favorite button
                msg_col, fav_col = st.columns([0.95, 0.05])
                
                with msg_col:
                    st.markdown(msg["content"])
                
                with fav_col:
                    # Favorite button for assistant messages
                    if msg["role"] == "assistant":
                        is_favorited = i in st.session_state.favorites
                        if st.button(
                            "⭐" if is_favorited else "☆",
                            key=f"fav_{i}",
                            help="Click to favorite/unfavorite this response"
                        ):
                            if is_favorited:
                                st.session_state.favorites.remove(i)
                            else:
                                st.session_state.favorites.append(i)
                            st.rerun()
                
                # Show metadata for assistant messages
                if msg["role"] == "assistant" and "metadata" in msg:
                    meta = msg["metadata"]
                    
                    # Show token usage in a dropdown expander
                    if "token_usage" in meta:
                        usage = meta["token_usage"]
                        total = usage.get('total', 0)
                        
                        with st.expander(f"📊 Token Usage ({total:,} total)", expanded=False):
                            # Create styled token usage boxes using columns
                            st.markdown("""
                            <style>
                            .token-box {
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 12px;
                                padding: 12px 16px;
                                color: white;
                                text-align: center;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                                margin: 4px 0;
                            }
                            .token-box-input {
                                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                                box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
                            }
                            .token-box-output {
                                background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
                                box-shadow: 0 4px 15px rgba(238, 9, 121, 0.3);
                            }
                            .token-box-total {
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                            }
                            .token-label {
                                font-size: 11px;
                                text-transform: uppercase;
                                letter-spacing: 1px;
                                opacity: 0.9;
                                margin-bottom: 4px;
                            }
                            .token-value {
                                font-size: 20px;
                                font-weight: 700;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="token-box token-box-input">
                                    <div class="token-label">📥 Input Tokens</div>
                                    <div class="token-value">{usage.get('input', 0):,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="token-box token-box-output">
                                    <div class="token-label">📤 Output Tokens</div>
                                    <div class="token-value">{usage.get('output', 0):,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="token-box token-box-total">
                                    <div class="token-label">📊 Total Tokens</div>
                                    <div class="token-value">{usage.get('total', 0):,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    if meta.get("query_type"):
                        # Show query type and execution time on same line
                        info_text = f"Query type: {meta['query_type']}"
                        if meta.get("execution_time"):
                            info_text += f" • ⏱️ {meta['execution_time']:.2f}s"
                        st.caption(info_text)
                        
                    # SQL Query expander
                    if meta.get("sql_query"):
                        with st.expander("🛠️ SQL Query & Details"):
                            st.code(meta["sql_query"], language="sql")
                            
                    # Visualizations and CSV export
                    if meta.get("sql_results"):
                        # Only render viz if we have results
                        render_visualization(meta["sql_results"], f"viz_{i}")
                        
                        # CSV Export button
                        csv_data = export_results_to_csv(meta["sql_results"])
                        if csv_data:
                            st.download_button(
                                label="📊 Export to CSV",
                                data=csv_data,
                                file_name=f"query_results_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"csv_export_{i}",
                                help="Download query results as CSV file"
                            )
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        if not st.session_state.initialized:
            st.error("Please connect to a database first!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Calculate memory context for display? No, just render user msg
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                # Add memory interaction
                if st.session_state.memory:
                    st.session_state.memory.add_message("user", prompt)
                
                # Track execution time
                start_time = time.time()
                
                response = st.session_state.chatbot.chat(
                    prompt, 
                    st.session_state.memory,
                    ignored_tables=list(st.session_state.ignored_tables),
                    language=st.session_state.response_language
                )
                
                execution_time = time.time() - start_time
                

                
                # Create metadata dict
                metadata = {
                    "query_type": response.query_type,
                    "sql_query": response.sql_query,
                    "sql_results": response.sql_results,
                    "token_usage": response.token_usage,
                    "execution_time": execution_time
                }
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": metadata
                })
                
                # Set flag to auto-read the latest response
                st.session_state.auto_read_latest = True
                
                # Save to active memory
                if st.session_state.memory:
                    st.session_state.memory.add_message("assistant", response.answer)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.error(traceback.format_exc())


def main():
    """Main application entry point."""
    init_session_state()
    
    # Auto-connect to environment database on first load
    if "auto_connect_attempted" not in st.session_state:
        st.session_state.auto_connect_attempted = True
        if st.session_state.db_source == "environment":
            success = initialize_chatbot()
            if success:
                st.toast("✅ Auto-connected to database!")

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
