"""
FAISS Vector Store for RAG.

Manages the FAISS index for semantic search over database text content.
"""

import logging
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .document_processor import Document
from .embeddings import get_embedding_provider, EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(
        self, 
        embedding_provider: Optional[EmbeddingProvider] = None,
        index_path: str = "./faiss_index"
    ):
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.index_path = index_path
        self.dimension = self.embedding_provider.dimension
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Document] = []
        self.id_to_idx: Dict[str, int] = {}
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                # Check file size - if 0 something is wrong
                if os.path.getsize(index_file) > 0:
                    self.index = faiss.read_index(index_file)
                    with open(docs_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    self.id_to_idx = {doc.id: i for i, doc in enumerate(self.documents)}
                    
                    # Verify index dimension matches expected
                    if self.index.d != self.dimension:
                        logger.warning(f"Index dimension mismatch: {self.index.d} != {self.dimension}. Resetting.")
                        raise ValueError("Dimension mismatch")
                        
                    logger.info(f"Loaded index with {len(self.documents)} documents")
                    return
            except (Exception, RuntimeError) as e:
                logger.warning(f"Failed to load index (might be corrupted or memory error): {e}")
                # If loading fails, we should probably backup the broken files or just overwrite
                if os.path.exists(index_file):
                    try:
                        os.rename(index_file, index_file + ".bak")
                        os.rename(docs_file, docs_file + ".bak")
                    except:
                        pass
        
        # Create new index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.id_to_idx = {}
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to the vector store."""
        if not documents:
            return
        
        new_docs = [doc for doc in documents if doc.id not in self.id_to_idx]
        if not new_docs:
            logger.info("No new documents to add")
            return
        
        logger.info(f"Adding {len(new_docs)} documents to index")
        
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            texts = [doc.content for doc in batch]
            
            embeddings = self.embedding_provider.embed_texts(texts)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            start_idx = len(self.documents)
            self.index.add(embeddings)
            
            for j, doc in enumerate(batch):
                self.documents.append(doc)
                self.id_to_idx[doc.id] = start_idx + j
        
        logger.info(f"Index now contains {len(self.documents)} documents")
    
    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_provider.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self):
        """Save the index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        faiss.write_index(self.index, index_file)
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved index with {len(self.documents)} documents")
    
    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.id_to_idx = {}
        
        # Delete files
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        for f in [index_file, docs_file]:
            if os.path.exists(f):
                os.remove(f)
        
        logger.info("Index cleared")
    
    def __len__(self) -> int:
        return len(self.documents)


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
