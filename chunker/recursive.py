from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from .base import BaseChunker
import uuid
from model.chunk.recursive_chunk import RecursiveChunk
from typing import Dict, List
from logger import Logger

class RecursiveChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50, cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir, chunker_name="recursive")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        if use_cache:
            self._load_consolidated_cache()
            if len(self.chunks) > 0:
                Logger.log(f"Loaded {len(self.chunks)} chunks from cache")
        
        uncached_pages = self.get_uncached_documents(pages)
        if len(uncached_pages) < len(pages):
            Logger.log(f"Skipping {len(pages) - len(uncached_pages)} already processed documents")
        
        if not uncached_pages:
            Logger.log("All documents already processed")
            return
        
        Logger.log(f"Processing {len(uncached_pages)} new documents with recursive chunking...")
        
        checkpoint_interval = 100
        processed_count = 0
        
        try:
            for idx, page in enumerate(uncached_pages, 1):
                try:
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
                    
                    self.mark_document_processed(page)
                    processed_count += 1
                    
                    if idx % checkpoint_interval == 0:
                        Logger.log(f"Checkpoint: Saving progress ({idx}/{len(uncached_pages)} documents)...")
                        self._save_consolidated_cache()
                
                except Exception as e:
                    Logger.log(f"Error processing document {idx}/{len(uncached_pages)}: {e}")
                    Logger.log(f"Saving progress before continuing...")
                    self._save_consolidated_cache()
                    Logger.log(f"Progress saved. Skipping problematic document.")
                    continue
        
        except KeyboardInterrupt:
            Logger.log(f"\nInterrupted by user. Saving progress...")
            self._save_consolidated_cache()
            Logger.log(f"Progress saved: {processed_count}/{len(uncached_pages)} documents processed")
            raise
        
        Logger.log(f"Total chunks: {len(self.chunks)} (added {processed_count} documents)")
        
        self._save_consolidated_cache()
        Logger.log(f"Created {len(self.chunks)} total chunks")
    
    def _get_chunk_type(self) -> str:
        return 'recursive'
    
    def _reconstruct_chunk(self, chunk_dict: dict) -> RecursiveChunk:
        return RecursiveChunk(**chunk_dict)