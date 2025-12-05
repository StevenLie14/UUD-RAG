from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from .base import BaseChunker
import uuid
from model.chunk.recursive_chunk import RecursiveChunk
from typing import Dict, List
from logger import Logger

class RecursiveChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50, cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        if use_cache:
            uncached_pages = self.get_uncached_documents(pages)
            if len(uncached_pages) < len(pages):
                Logger.log(f"Loaded {len(pages) - len(uncached_pages)} documents from cache")
            pages = uncached_pages
        
        if not pages:
            Logger.log("All documents already cached")
            return
        
        Logger.log(f"Processing {len(pages)} uncached documents with recursive chunking...")
        
        # Process each document separately to maintain per-document caching
        for page in pages:
            doc_hash = self._get_document_hash(page)
            chunk_ids = []
            
            # Split this specific page
            split_docs = self.text_splitter.split_documents([page])
            
            for doc in split_docs:
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
                chunk_ids.append(id)
            
            # Cache chunks for this document
            self.document_chunks[doc_hash] = chunk_ids
            if use_cache:
                self._save_chunks_to_cache(doc_hash, chunk_ids)
        
        Logger.log(f"Created {len(self.chunks)} total chunks")