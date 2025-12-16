from langchain_core.documents import Document
from model import BaseChunk
from typing import Dict, List, Set
import hashlib
import json
import os
from datetime import datetime

class BaseChunker:
    def __init__(self, cache_dir: str = "./chunk_cache", chunker_name: str = "base"):
        self.chunks : Dict[str, BaseChunk] = {}
        self.cache_dir = cache_dir
        self.chunker_name = chunker_name
        self.processed_doc_hashes: Set[str] = set()  # Track processed documents
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_document_hash(self, page: Document) -> str:
        """Generate unique hash for a document based on its content and metadata"""
        content = page.page_content
        source = page.metadata.get("source", "")
        page_num = page.metadata.get("page", 0)
        hash_input = f"{source}:{page_num}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _get_consolidated_cache_path(self) -> str:
        """Get path for consolidated cache file"""
        return os.path.join(self.cache_dir, f"{self.chunker_name}_cache.json")
    
    def _load_consolidated_cache(self) -> bool:
        """Load all chunks from consolidated cache file"""
        cache_path = self._get_consolidated_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                    chunks_data = cached_data.get('chunks', [])
                    self.processed_doc_hashes = set(cached_data.get('processed_docs', []))
                    chunk_type = cached_data.get('chunk_type', 'base')
                    
                    for chunk_dict in chunks_data:
                        chunk = self._reconstruct_chunk(chunk_dict)
                        if chunk:
                            self.chunks[chunk.id] = chunk
                    
                    return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return False
    
    def _save_consolidated_cache(self):
        """Save all chunks to consolidated cache file"""
        from logger import Logger
        cache_path = self._get_consolidated_cache_path()
        
        try:
            chunks_data = []
            for chunk in self.chunks.values():
                chunks_data.append(chunk.model_dump())
            
            cache_data = {
                'chunk_type': self._get_chunk_type(),
                'chunker_name': self.chunker_name,
                'total_chunks': len(chunks_data),
                'processed_docs': list(self.processed_doc_hashes),
                'last_updated': datetime.now().isoformat(),
                'chunks': chunks_data
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            Logger.log(f"Saved {len(chunks_data)} chunks to {cache_path}")
        except Exception as e:
            Logger.log(f"Failed to save cache: {e}")
    
    def _get_chunk_type(self) -> str:
        """Get the chunk type for this chunker (to be overridden by subclasses)"""
        return 'base'
    
    def _reconstruct_chunk(self, chunk_dict: dict, chunk_type: str) -> BaseChunk:
        """Reconstruct a single chunk object from dict (to be overridden by subclasses)"""
        return None
    
    def get_uncached_documents(self, pages: List[Document]) -> List[Document]:
        """Filter out documents that are already processed"""
        uncached = []
        for page in pages:
            doc_hash = self._get_document_hash(page)
            if doc_hash not in self.processed_doc_hashes:
                uncached.append(page)
        return uncached
    
    def mark_document_processed(self, page: Document):
        """Mark a document as processed"""
        doc_hash = self._get_document_hash(page)
        self.processed_doc_hashes.add(doc_hash)
    
    def is_document_processed_by_source(self, source: str) -> bool:
        """Check if a document source has been processed (for document-level chunking)"""
        # For document-level chunking, we use the source as the hash
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        return source_hash in self.processed_doc_hashes
    
    def mark_document_processed_by_source(self, source: str):
        """Mark a document source as processed (for document-level chunking)"""
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        self.processed_doc_hashes.add(source_hash)
    
    def get_chunks_for_database(self) -> List[BaseChunk]:
        """Get all chunks ready for database storage"""
        return list(self.chunks.values())

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        raise NotImplementedError