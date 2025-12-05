from langchain_core.documents import Document
from model import BaseChunk
from typing import Dict, List
import hashlib
import json
import os
import pickle

class BaseChunker:
    def __init__(self, cache_dir: str = "./chunk_cache"):
        self.chunks : Dict[str, BaseChunk] = {}
        self.cache_dir = cache_dir
        self.document_chunks: Dict[str, List[str]] = {}  # Maps doc hash -> chunk IDs
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_document_hash(self, page: Document) -> str:
        """Generate unique hash for a document based on its content and metadata"""
        content = page.page_content
        source = page.metadata.get("source", "")
        page_num = page.metadata.get("page", 0)
        hash_input = f"{source}:{page_num}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, doc_hash: str) -> str:
        """Get cache file path for a document"""
        return os.path.join(self.cache_dir, f"{doc_hash}.pkl")
    
    def _load_cached_chunks(self, doc_hash: str) -> bool:
        """Load chunks from cache for a document. Returns True if loaded."""
        cache_path = self._get_cache_path(doc_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    chunk_ids = cached_data.get('chunk_ids', [])
                    chunks = cached_data.get('chunks', {})
                    
                    # Add chunks to main collection
                    for chunk_id, chunk in chunks.items():
                        self.chunks[chunk_id] = chunk
                    
                    # Track document-to-chunks mapping
                    self.document_chunks[doc_hash] = chunk_ids
                    return True
            except Exception:
                pass
        return False
    
    def _save_chunks_to_cache(self, doc_hash: str, chunk_ids: List[str]):
        """Save chunks to cache for a document"""
        cache_path = self._get_cache_path(doc_hash)
        try:
            chunks_to_cache = {cid: self.chunks[cid] for cid in chunk_ids if cid in self.chunks}
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'chunk_ids': chunk_ids,
                    'chunks': chunks_to_cache
                }, f)
        except Exception:
            pass
    
    def get_uncached_documents(self, pages: List[Document]) -> List[Document]:
        """Filter out documents that are already cached"""
        uncached = []
        for page in pages:
            doc_hash = self._get_document_hash(page)
            if not self._load_cached_chunks(doc_hash):
                uncached.append(page)
        return uncached
    
    def get_chunks_for_database(self) -> List[BaseChunk]:
        """Get all chunks ready for database storage"""
        return list(self.chunks.values())

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        raise NotImplementedError