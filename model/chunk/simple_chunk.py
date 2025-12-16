from .base_chunk import BaseChunk
from typing import Optional

class SimpleChunk(BaseChunk):
    content: str
    index: int
    source: Optional[str] = None
    page: Optional[int] = None
    
    def get_context(self) -> str:
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
        return {
            "chunk_id": self.id,
            "content": self.content,
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata
        }
