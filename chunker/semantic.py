from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from sentence_transformers import SentenceTransformer
from .base import BaseChunker
import uuid
from model.chunk.semantic_chunk import SemanticChunk
from logger import Logger
import re

class SemanticChunker(BaseChunker):
    def __init__(self, 
                 embedding_model_name: str = "LazarusNLP/all-indo-e5-small-v4",
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 95.0,
                 number_of_chunks: int = None,
                 cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir, chunker_name="semantic")
        
        Logger.log(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        
        class SentenceTransformerEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        embeddings_wrapper = SentenceTransformerEmbeddings(self.embedding_model)
        
        self.text_splitter = LangChainSemanticChunker(
            embeddings=embeddings_wrapper,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks
        )
        
        Logger.log(f"Semantic chunker initialized with {breakpoint_threshold_type} threshold: {breakpoint_threshold_amount}")
        
        self.min_chunk_chars = 20
        self.min_alnum_ratio = 0.15

    def _is_meaningful_chunk(self, text: str) -> bool:
        if not text or len(text) < self.min_chunk_chars:
            return False
        alnum = sum(ch.isalnum() for ch in text)
        ratio = alnum / max(len(text), 1)
        if ratio < self.min_alnum_ratio:
            return False
        if re.fullmatch(r"\W+", text):
            return False
        return True

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        try:
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
            
            Logger.log(f"Processing {len(uncached_pages)} new documents with semantic chunking...")
            Logger.log(f"This may take a while - calculating embeddings for all documents...")
            
            docs = self.text_splitter.split_documents(uncached_pages)
            
            Logger.log(f"Semantic chunker created {len(docs)} chunks from {len(uncached_pages)} documents")
            Logger.log(f"Filtering meaningful chunks...")
            
            meaningful_count = 0
            skipped_count = 0
            total_docs = len(docs)
            
            for idx, doc in enumerate(docs, 1):
                if idx % 250 == 0 or idx == total_docs:
                    Logger.log(f"Processing chunks: {idx}/{total_docs} ({idx*100//total_docs}%)")
                
                if not self._is_meaningful_chunk(doc.page_content):
                    skipped_count += 1
                    continue
                
                chunk_id = str(uuid.uuid4())
                metadata = doc.metadata or {}
                
                chunk_obj = SemanticChunk(
                    id=chunk_id,
                    content=doc.page_content,
                    source=metadata.get("source", "Unknown"),
                    page=metadata.get("page", 0),
                    total_pages=metadata.get("total_pages", 0),
                    page_label=metadata.get("page_label", ""),
                )
                
                self.chunks[chunk_id] = chunk_obj
                meaningful_count += 1
            
            if skipped_count > 0:
                Logger.log(f"Skipped {skipped_count} non-meaningful chunks")
            
            for page in uncached_pages:
                self.mark_document_processed(page)
            
            Logger.log(f"Successfully created {len(self.chunks)} total semantic chunks ({meaningful_count} added)")
            
            self._save_consolidated_cache()
            
        except KeyboardInterrupt:
            Logger.log(f"\nInterrupted by user. Saving progress...")
            self._save_consolidated_cache()
            Logger.log(f"Progress saved")
            raise
        except Exception as e:
            Logger.log(f"Error in semantic chunking: {e}")
            self._save_consolidated_cache()
            raise
    
    def get_chunker_info(self):
        return {
            "chunker_type": "semantic",
            "embedding_model": self.embedding_model_name,
            "breakpoint_threshold_type": self.text_splitter.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.text_splitter.breakpoint_threshold_amount,
            "total_chunks": len(self.chunks)
        }
    
    def _get_chunk_type(self) -> str:
        return 'semantic'
    
    def _reconstruct_chunk(self, chunk_dict: dict) -> SemanticChunk:
        return SemanticChunk(**chunk_dict)