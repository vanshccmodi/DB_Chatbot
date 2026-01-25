"""
Document Processor for RAG.

Converts database rows into semantic documents for embedding.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator
import re

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Semantic document from the database."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    table_name: str = ""
    column_name: str = ""
    primary_key_value: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    
    def __post_init__(self):
        if not self.id:
            hash_input = f"{self.table_name}:{self.column_name}:{self.primary_key_value}:{self.chunk_index}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()
    
    def to_context_string(self) -> str:
        source = f"[Source: {self.table_name}.{self.column_name}"
        if self.primary_key_value:
            source += f" (id: {self.primary_key_value})"
        source += "]"
        return f"{source}\n{self.content}"


class TextChunker:
    """Splits long text into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def chunk_text(self, text: str) -> List[str]:
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        sentences = self.sentence_pattern.split(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]


class DocumentProcessor:
    """Converts database rows into semantic documents."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
    
    def process_row(
        self, row: Dict[str, Any], table_name: str,
        text_columns: List[str], primary_key_column: Optional[str] = None
    ) -> List[Document]:
        documents = []
        pk_value = str(row.get(primary_key_column, "")) if primary_key_column else None
        
        for column_name in text_columns:
            text = row.get(column_name)
            if not text or not isinstance(text, str):
                continue
            
            text = text.strip()
            if not text:
                continue
            
            chunks = self.chunker.chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    id="", content=chunk, table_name=table_name,
                    column_name=column_name, primary_key_value=pk_value,
                    chunk_index=i, total_chunks=len(chunks),
                    metadata={"table": table_name, "column": column_name, "pk": pk_value}
                )
                documents.append(doc)
        
        return documents
    
    def process_rows(
        self, rows: List[Dict[str, Any]], table_name: str,
        text_columns: List[str], primary_key_column: Optional[str] = None
    ) -> Generator[Document, None, None]:
        for row in rows:
            for doc in self.process_row(row, table_name, text_columns, primary_key_column):
                yield doc


def get_document_processor(chunk_size: int = 500, chunk_overlap: int = 50) -> DocumentProcessor:
    return DocumentProcessor(chunk_size, chunk_overlap)
