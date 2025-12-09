"""
Database Loader Workflow
Handle loading chunks into databases
"""

import json
import os
from typing import Dict, Tuple, Optional

from config import Config
from logger import Logger
from database import Qdrant, FAISS
from model.chunk.recursive_chunk import RecursiveChunk
from model.chunk.semantic_chunk import SemanticChunk
from model.chunk.agentic_chunk import AgenticChunk
from ui import UserInterface


# Constants
CACHE_DIR = "./chunk_cache"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"


class DatabaseLoader:
    """Handle loading chunks into databases"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ui = UserInterface()
    
    async def run(self):
        """Run database loading workflow"""
        self.ui.print_header("LOAD CHUNKS TO DATABASE")
        
        # Select JSON file
        json_file = self._select_json_file()
        if not json_file or not os.path.exists(json_file):
            Logger.log("❌ Invalid file path!")
            return
        
        # Load chunks
        chunks_dict, chunker_type = self._load_chunks(json_file)
        
        # Select database
        db_choice = self.ui.get_choice(
            "Select database:",
            ["FAISS (local)", "Qdrant (cloud)", "Both"]
        )
        
        # Collection name
        default_collection = f"{chunker_type}_chunks"
        collection_name = input(f"\nCollection name (default: {default_collection}): ").strip()
        collection_name = collection_name if collection_name else default_collection
        
        # Clear existing
        clear_db = self.ui.confirm("Clear existing collection?", default=False)
        
        # Store in databases
        self._store_in_databases(chunks_dict, collection_name, db_choice, clear_db)
        
        # Summary
        self.ui.print_subheader("Completed")
        print(f"✓ Chunks stored: {len(chunks_dict)}")
        print(f"✓ Collection: {collection_name}")
        print(f"✓ Database: {['FAISS', 'Qdrant', 'FAISS + Qdrant'][int(db_choice)-1]}")
    
    def _select_json_file(self) -> Optional[str]:
        """Select JSON file"""
        json_files = []
        
        if os.path.exists(CACHE_DIR):
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
            json_files.extend([os.path.join(CACHE_DIR, f) for f in cache_files])
        
        if json_files:
            print("\nAvailable files:")
            for i, file in enumerate(json_files, 1):
                print(f"  {i}. {file}")
            
            choice = input("\nEnter file number or path: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(json_files):
                return json_files[int(choice) - 1]
            return choice
        else:
            return input("\nEnter JSON file path: ").strip()
    
    def _load_chunks(self, json_file: str) -> Tuple[Dict, str]:
        """Load chunks from JSON"""
        Logger.log(f"Loading chunks from {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'chunks' in data:
            chunks_data = data['chunks']
            chunker_type = data.get('chunker_type', 'recursive')
        else:
            chunks_data = data
            chunker_type = 'recursive'
        
        chunks_dict = {}
        
        for chunk_data in chunks_data:
            chunk = self._create_chunk(chunk_data, chunker_type)
            chunks_dict[chunk.id] = chunk
        
        Logger.log(f"✓ Loaded {len(chunks_dict)} chunks")
        return chunks_dict, chunker_type
    
    def _create_chunk(self, chunk_data: Dict, chunker_type: str):
        """Create chunk object from data"""
        chunk_id = chunk_data['id']
        
        if chunker_type == 'semantic':
            return SemanticChunk(
                id=chunk_id,
                content=chunk_data['content'],
                source=chunk_data.get('source'),
                page=chunk_data.get('page'),
                total_pages=chunk_data.get('total_pages'),
                page_label=chunk_data.get('page_label'),
                semantic_score=chunk_data.get('semantic_score', 0.0),
                boundary_type=chunk_data.get('boundary_type', 'semantic')
            )
        elif chunker_type == 'agentic':
            return AgenticChunk(
                id=chunk_id,
                title=chunk_data.get('title', ''),
                summary=chunk_data.get('summary', ''),
                propositions=chunk_data.get('propositions', [chunk_data['content']]),
                index=chunk_data.get('index', 0),
                metadata=chunk_data.get('metadata', {})
            )
        else:
            return RecursiveChunk(
                id=chunk_id,
                content=chunk_data['content'],
                source=chunk_data.get('source'),
                page=chunk_data.get('page'),
                total_pages=chunk_data.get('total_pages'),
                page_label=chunk_data.get('page_label')
            )
    
    def _store_in_databases(self, chunks_dict: Dict, collection_name: str, 
                           db_choice: str, clear_db: bool):
        """Store chunks in selected databases"""
        if db_choice in ['1', '3']:
            self._store_in_faiss(chunks_dict, collection_name, clear_db)
        
        if db_choice in ['2', '3']:
            self._store_in_qdrant(chunks_dict, collection_name, clear_db)
    
    def _store_in_faiss(self, chunks_dict: Dict, collection_name: str, clear_db: bool):
        """Store in FAISS"""
        try:
            Logger.log("📦 Storing in FAISS...")
            faiss_db = FAISS(
                index_path=FAISS_INDEX_PATH,
                dense_model_name=EMBEDDING_MODEL,
                collection_name=collection_name
            )
            
            if clear_db:
                faiss_db.delete_collection()
            
            faiss_db.store_chunks(chunks_dict)
            faiss_db.close()
            Logger.log(f"✓ Stored {len(chunks_dict)} chunks in FAISS")
        except Exception as e:
            Logger.log(f"❌ FAISS error: {e}")
    
    def _store_in_qdrant(self, chunks_dict: Dict, collection_name: str, clear_db: bool):
        """Store in Qdrant"""
        try:
            Logger.log("☁️ Storing in Qdrant...")
            qdrant_db = Qdrant(
                self.config.QDRANT_HOST,
                self.config.QDRANT_API_KEY,
                collection_name
            )
            
            if clear_db:
                qdrant_db.delete_collection()
                qdrant_db._create_collection_if_not_exists()
            
            qdrant_db.store_chunks(chunks_dict)
            qdrant_db.close()
            Logger.log(f"✓ Stored {len(chunks_dict)} chunks in Qdrant")
        except Exception as e:
            Logger.log(f"❌ Qdrant error: {e}")
