from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from sentence_transformers import SentenceTransformer
from .base import BaseChunker
import uuid
from model.chunk.semantic_chunk import SemanticChunk
from typing import Dict, List
from logger import Logger

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

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        try:
            # Load existing cache
            if use_cache:
                self._load_consolidated_cache()
                if len(self.chunks) > 0:
                    Logger.log(f"Loaded {len(self.chunks)} chunks from cache")
            
            # Filter uncached documents
            uncached_pages = self.get_uncached_documents(pages)
            if len(uncached_pages) < len(pages):
                Logger.log(f"Skipping {len(pages) - len(uncached_pages)} already processed documents")
            
            if not uncached_pages:
                Logger.log("All documents already processed")
                return
            
            total_pages = len(uncached_pages)
            Logger.log(f"Processing {total_pages} new documents with semantic chunking...")
            
            checkpoint_interval = 100  # Save every 100 documents
            processed_count = 0
            
            # Process each document separately
            for idx, page in enumerate(uncached_pages, 1):
                try:
                    
                    # Split this specific page
                    split_docs = self.text_splitter.split_documents([page])
                    
                    for doc in split_docs:
                        id = str(uuid.uuid4())
                        metadata = doc.metadata or {}

                        chunk_obj = SemanticChunk(
                            id=id,
                            content=doc.page_content,
                            source=metadata.get("source"),
                            page=metadata.get("page"),
                            total_pages=metadata.get("total_pages"),
                            page_label=metadata.get("page_label"),
                            semantic_score=0.0,  # Default score
                            boundary_type="semantic"  # Semantic boundary type
                        )

                        self.chunks[id] = chunk_obj
                    
                    # Mark document as processed
                    self.mark_document_processed(page)
                    processed_count += 1
                    
                    # Checkpoint save every N documents
                    if idx % checkpoint_interval == 0:
                        Logger.log(f"Checkpoint: Saving progress ({idx}/{total_pages} documents)...")
                        self._save_consolidated_cache()
                    
                    # Show progress every 100 pages
                    if idx % 100 == 0 or idx == total_pages:
                        Logger.log(f"Progress: {idx}/{total_pages} pages processed ({idx*100//total_pages}%)")
                
                except Exception as e:
                    Logger.log(f"Error processing document {idx}/{total_pages}: {e}")
                    Logger.log(f"Saving progress before continuing...")
                    self._save_consolidated_cache()
                    Logger.log(f"Progress saved. Skipping problematic document.")
                    continue
            
            Logger.log(f"Total chunks: {len(self.chunks)} (added {processed_count} documents)")
            
            # Final save
            self._save_consolidated_cache()
            
        except KeyboardInterrupt:
            Logger.log(f"\nInterrupted by user. Saving progress...")
            self._save_consolidated_cache()
            Logger.log(f"Progress saved: {processed_count}/{total_pages} documents processed")
            raise
        except Exception as e:
            Logger.log(f"Error in semantic chunking: {e}")
            Logger.log(f"Saving progress before exit...")
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
    
    def _reconstruct_chunk(self, chunk_dict: dict, chunk_type: str) -> SemanticChunk:
        """Reconstruct SemanticChunk object from dict"""
        return SemanticChunk(**chunk_dict)