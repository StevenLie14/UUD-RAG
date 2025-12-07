from .base_chunk import BaseChunk
from typing import Optional

class SimpleChunk(BaseChunk):
    """
    Simple chunk model that only stores the text content without title or summary.
    Used for agentic chunking v2.
    """
    content: str           # The actual chunk text
    index: int             # Index of the chunk in the document
    source: Optional[str] = None  # Source file/document name
    page: Optional[int] = None    # Page number in the document
    
    def get_context(self) -> str:
        """Returns the chunk content with source and page info"""
        context = ""
        if self.source:
            context += f"Sumber: {self.source}\n"
        if self.page is not None:
            context += f"Halaman: {self.page}\n"
        if context:
            context += "\n"
        context += self.content
        return context
    
    def get_payload(self) -> dict:
        """Returns the payload for storage/retrieval"""
        return {
            "chunk_id": self.id,
            "content": self.content,
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata
        }
