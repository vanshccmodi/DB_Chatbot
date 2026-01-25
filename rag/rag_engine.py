"""
RAG Engine - Orchestrates the retrieval-augmented generation pipeline.

Handles:
- Automatic indexing of text columns from the database
- Semantic retrieval using FAISS
- Context building for the LLM
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .document_processor import Document, get_document_processor
from .vector_store import VectorStore, get_vector_store
from .embeddings import get_embedding_provider

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine for semantic retrieval from database text."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store()
        self.doc_processor = get_document_processor()
        self.indexed_tables: Dict[str, bool] = {}
    
    def index_table(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        text_columns: List[str],
        primary_key_column: Optional[str] = None
    ) -> int:
        """
        Index text data from a table.
        
        Returns:
            Number of documents indexed
        """
        documents = list(self.doc_processor.process_rows(
            rows, table_name, text_columns, primary_key_column
        ))
        
        if documents:
            self.vector_store.add_documents(documents)
            self.indexed_tables[table_name] = True
            logger.info(f"Indexed {len(documents)} documents from {table_name}")
        
        return len(documents)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        table_filter: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            table_filter: Optional list of tables to search in
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.search(query, top_k=top_k * 2)
        
        if table_filter:
            results = [
                (doc, score) for doc, score in results 
                if doc.table_name in table_filter
            ]
        
        return results[:top_k]
    
    def get_context(
        self, 
        query: str, 
        top_k: int = 5,
        table_filter: Optional[List[str]] = None
    ) -> str:
        """
        Get formatted context for LLM from search results.
        """
        results = self.search(query, top_k, table_filter)
        
        if not results:
            return "No relevant information found in the database."
        
        context_parts = []
        for doc, score in results:
            context_parts.append(doc.to_context_string())
        
        return "\n\n---\n\n".join(context_parts)
    
    def clear_index(self):
        """Clear the entire index."""
        self.vector_store.clear()
        self.indexed_tables = {}
    
    def save(self):
        """Save the index to disk."""
        self.vector_store.save()
    
    @property
    def document_count(self) -> int:
        return len(self.vector_store)


_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
