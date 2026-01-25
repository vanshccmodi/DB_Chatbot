"""
Schema-Agnostic Database Chatbot - Streamlit Application

A production-grade chatbot that connects to ANY MySQL database
and provides intelligent querying through RAG and Text-to-SQL.

Uses Groq for FREE LLM inference!
"""

import os
from pathlib import Path

# Load .env FIRST before any other imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
import uuid
from datetime import datetime

# Page config must be first
st.set_page_config(
    page_title="Database Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from config import config
from database import get_db, get_schema, get_introspector
from llm import create_llm_client
from chatbot import create_chatbot, DatabaseChatbot
from memory import create_memory, create_enhanced_memory, EnhancedChatMemory


# Groq models (all FREE!)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]


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
        st.session_state.memory = create_enhanced_memory(
            st.session_state.session_id, 
            user_id=st.session_state.user_id,
            enable_summarization=st.session_state.enable_summarization,
            summary_threshold=st.session_state.summary_threshold
        )
        # Clear temporary memory on fresh load/reload
        st.session_state.memory.clear_user_history()
    
    if "indexed" not in st.session_state:
        st.session_state.indexed = False


def render_sidebar():
    """Render the configuration sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # User Profile
        st.subheader("👤 User Profile")
        user_id = st.text_input(
            "User ID / Name", 
            value=st.session_state.get("user_id", "default"),
            key="user_id_input",
            help="Your unique ID for private memory storage"
        )
        if user_id != st.session_state.get("user_id"):
            # USER ID CHANGE - Same behavior as "New Chat":
            # 1. Clear temporary memory (session history) for clean start
            # 2. Permanent memory remains UNTOUCHED (per-user storage)
            st.session_state.user_id = user_id
            st.session_state.session_id = str(uuid.uuid4())  # New session
            st.session_state.messages = []  # Clear UI chat history
            
            # Create memory for new user and clear their temp history (fresh start)
            st.session_state.memory = create_enhanced_memory(
                st.session_state.session_id, 
                user_id=user_id,
                enable_summarization=st.session_state.enable_summarization,
                summary_threshold=st.session_state.summary_threshold
            )
            st.session_state.memory.clear_user_history()  # Clears _chatbot_memory, NOT _chatbot_permanent_memory_v2
            st.rerun()
        
        st.divider()
        
        # Initialize Button
        if st.button("🚀 Connect & Initialize", use_container_width=True, type="primary"):
            with st.spinner("Connecting to database..."):
                success = initialize_chatbot()
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
            st.success("Database: Connected")
            schema = get_schema()
            st.info(f"Tables: {len(schema.tables)}")
            
            if st.session_state.indexed:
                from rag import get_rag_engine
                engine = get_rag_engine()
                st.info(f"Indexed Docs: {engine.document_count}")
        else:
            st.warning("Not connected")
        
        # New Chat (Context Switch)
        # New Chat (Context Switch)
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            # Clear previous session from DB
            if "memory" in st.session_state and st.session_state.memory:
                st.session_state.memory.clear()
            
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())  # Generate new session ID
            
            # Preserve current user ID and memory settings
            current_user = st.session_state.get("user_id", "default")
            st.session_state.memory = create_enhanced_memory(
                st.session_state.session_id, 
                user_id=current_user,
                enable_summarization=st.session_state.enable_summarization,
                summary_threshold=st.session_state.summary_threshold
            )
            # Set LLM client if available
            if "llm" in st.session_state and st.session_state.llm:
                st.session_state.memory.set_llm_client(st.session_state.llm)
            st.rerun()


def initialize_chatbot() -> bool:
    """Initialize the chatbot using environment variables."""
    try:
        # Use Groq as default provider (from environment)
        api_key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not api_key:
            st.error("GROQ_API_KEY not configured. Please set it in your .env file.")
            return False
        
        llm = create_llm_client("groq", api_key=api_key, model=model)
        
        # Create and initialize chatbot
        chatbot = create_chatbot(llm)
        
        # Explicitly set LLM client (also configures router and sql_generator)
        chatbot.set_llm_client(llm)
        
        success, msg = chatbot.initialize()
        
        if success:
            st.session_state.chatbot = chatbot
            st.session_state.llm = llm  # Store LLM separately too
            st.session_state.initialized = True
            
            # Set LLM client on memory for summarization
            if hasattr(st.session_state.memory, 'set_llm_client'):
                st.session_state.memory.set_llm_client(llm)
            
            return True
        else:
            st.error(f"Initialization failed: {msg}")
            return False
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


def index_data():
    """Index text data from the database."""
    if st.session_state.chatbot:
        progress = st.progress(0)
        status = st.empty()
        
        schema = get_schema()
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
        schema = get_schema()
        
        for table_name, table_info in schema.tables.items():
            with st.container():
                st.markdown(f"**{table_name}** ({table_info.row_count or '?'} rows)")
                
                cols = []
                for col in table_info.columns:
                    pk = "🔑" if col.is_primary_key else ""
                    txt = "📝" if col.is_text_type else ""
                    cols.append(f"`{col.name}` {col.data_type} {pk}{txt}")
                
                st.caption(" | ".join(cols))
                st.divider()


def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 Database Copilot")
    st.caption("Schema-agnostic chatbot powered by Groq (FREE!)")
    
    # Schema explorer
    render_schema_explorer()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show metadata for assistant messages
                if msg["role"] == "assistant" and "metadata" in msg:
                    meta = msg["metadata"]
                    if meta.get("query_type"):
                        st.caption(f"Query type: {meta['query_type']}")
                    if meta.get("sql_query"):
                        with st.expander("SQL Query"):
                            st.code(meta["sql_query"], language="sql")
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        if not st.session_state.initialized:
            st.error("Please connect to a database first!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(
                    prompt, 
                    st.session_state.memory
                )
                
                st.markdown(response.answer)
                
                # Show metadata
                if response.query_type != "general":
                    st.caption(f"Query type: {response.query_type}")
                
                if response.sql_query:
                    with st.expander("SQL Query"):
                        st.code(response.sql_query, language="sql")
                
                if response.sql_results:
                    with st.expander("Results"):
                        st.dataframe(response.sql_results)
        
        # Save to memory
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "metadata": {
                "query_type": response.query_type,
                "sql_query": response.sql_query
            }
        })
        st.session_state.memory.add_message("assistant", response.answer)


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
