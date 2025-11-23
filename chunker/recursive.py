from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from .base import BaseChunker
import uuid
from model.recursive_chunk import RecursiveChunk
from typing import Dict

class RecursiveChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        docs = self.text_splitter.split_documents(pages)
        for doc in docs:
            id = str(uuid.uuid4())
            metadata = doc.metadata or {}

            chunk_obj = RecursiveChunk(
                id=id,
                content=doc.page_content,
                source=metadata.get("source"),
                page=metadata.get("page"),
                total_pages=metadata.get("total_pages"),
                page_label=metadata.get("page_label"),
            )

            self.chunks[id] = chunk_obj