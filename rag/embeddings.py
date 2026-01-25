"""
Embedding Generation Module.

Supports:
- Sentence Transformers (local, free)
- OpenAI Embeddings (cloud, paid)

Configurable via environment variables.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """
    Sentence Transformers embedding provider.
    
    Uses local models, no API key required.
    Default: all-MiniLM-L6-v2 (384 dimensions)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformer model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Force model load
        return self._dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100)


class OpenAIEmbedding(EmbeddingProvider):
    """
    OpenAI embedding provider.
    
    Uses OpenAI API, requires API key.
    Default: text-embedding-3-small (1536 dimensions)
    """
    
    DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding client.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI embedding model name
        """
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        self._dimension = self.DIMENSION_MAP.get(model_name, 1536)
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai is required. Install with: pip install openai"
                )
        return self._client
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts (batch)."""
        # OpenAI API supports batching up to 2048 inputs
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)


def create_embedding_provider(
    provider_type: str = "sentence_transformers",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> EmbeddingProvider:
    """
    Factory function to create the appropriate embedding provider.
    
    Args:
        provider_type: "sentence_transformers" or "openai"
        model_name: Model name (optional, uses defaults)
        api_key: API key for OpenAI (required if using OpenAI)
        
    Returns:
        Configured EmbeddingProvider instance
    """
    if provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        return OpenAIEmbedding(
            api_key=api_key,
            model_name=model_name or "text-embedding-3-small"
        )
    else:
        return SentenceTransformerEmbedding(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )


# Global embedding provider instance
_embedding_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the global embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        # Default to sentence transformers (free, local)
        _embedding_provider = SentenceTransformerEmbedding()
    return _embedding_provider


def set_embedding_provider(provider: EmbeddingProvider):
    """Set the global embedding provider."""
    global _embedding_provider
    _embedding_provider = provider
