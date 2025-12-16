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


CACHE_DIR = "./chunk_cache"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"


class DatabaseLoader:
    def __init__(self, config: Config):
        self.config = config
        self.ui = UserInterface()
    
    async def run(self):
        self.ui.print_header("LOAD CHUNKS TO DATABASE")
        
        json_file = self._select_json_file()
        if not json_file or not os.path.exists(json_file):
            Logger.log("Invalid file path!")
            return
        
        chunks_dict, chunker_type = self._load_chunks(json_file)
        
        total_chunks = len(chunks_dict)
        chunks_to_load = chunks_dict
        
        if total_chunks > 1000:
            if self.ui.confirm(f"Load all {total_chunks} chunks? (No to select batch)", default=True):
                chunks_to_load = chunks_dict
            else:
                chunks_to_load = self._select_batch(chunks_dict)
        
        if not chunks_to_load:
            Logger.log("No chunks selected!")
            return
        
        db_choice = self.ui.get_choice(
            "Select database:",
            ["FAISS", "Qdrant", "Both"]
        )
        
        default_collection = f"{chunker_type}_chunks"
        collection_name = input(f"\nCollection name (default: {default_collection}): ").strip()
        collection_name = collection_name if collection_name else default_collection
        
        clear_db = self.ui.confirm("Clear existing collection?", default=False)
        
        self._store_in_databases(chunks_to_load, collection_name, db_choice, clear_db)
        
        self.ui.print_subheader("Completed")
        print(f"✓ Chunks stored: {len(chunks_to_load)} / {total_chunks}")
        print(f"✓ Collection: {collection_name}")
        print(f"✓ Database: {['FAISS', 'Qdrant', 'FAISS + Qdrant'][int(db_choice)-1]}")
    
    def _select_json_file(self) -> Optional[str]:
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
    
    def _select_batch(self, chunks_dict: Dict) -> Dict:
        total = len(chunks_dict)
        chunk_ids = list(chunks_dict.keys())
        
        print(f"\nTotal chunks: {total}")
        print("\nBatch options:")
        print("1. First N chunks")
        print("2. Last N chunks")
        print("3. Range (start-end)")
        print("4. Every Nth chunk (sample)")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        try:
            if choice == "1":
                n = int(input(f"Load first N chunks (max {total}): ").strip())
                selected_ids = chunk_ids[:min(n, total)]
            elif choice == "2":
                n = int(input(f"Load last N chunks (max {total}): ").strip())
                selected_ids = chunk_ids[-min(n, total):]
            elif choice == "3":
                start = int(input(f"Start index (0-{total-1}): ").strip())
                end = int(input(f"End index ({start+1}-{total}): ").strip())
                selected_ids = chunk_ids[start:end]
            elif choice == "4":
                step = int(input(f"Load every Nth chunk (e.g., 10 = every 10th): ").strip())
                selected_ids = chunk_ids[::step]
            else:
                Logger.log("Invalid choice, loading all chunks")
                return chunks_dict
            
            selected = {cid: chunks_dict[cid] for cid in selected_ids}
            Logger.log(f"Selected {len(selected)} chunks")
            return selected
            
        except (ValueError, IndexError) as e:
            Logger.log(f"Error: {e}. Loading all chunks.")
            return chunks_dict
    
    def _load_chunks(self, json_file: str) -> Tuple[Dict, str]:
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
        
        Logger.log(f"Loaded {len(chunks_dict)} chunks")
        return chunks_dict, chunker_type
    
    def _create_chunk(self, chunk_data: Dict, chunker_type: str):
        chunk_id = chunk_data['id']
        
        if chunker_type == 'semantic':
            return SemanticChunk(
                id=chunk_id,
                content=chunk_data['content'],
                source=chunk_data.get('source'),
                page=chunk_data.get('page'),
                total_pages=chunk_data.get('total_pages'),
                page_label=chunk_data.get('page_label'),
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
        if db_choice in ['1', '3']:
            self._store_in_faiss(chunks_dict, collection_name, clear_db)
        
        if db_choice in ['2', '3']:
            self._store_in_qdrant(chunks_dict, collection_name, clear_db)
    
    def _store_in_faiss(self, chunks_dict: Dict, collection_name: str, clear_db: bool):
        try:
            Logger.log("Storing in FAISS...")
            faiss_db = FAISS(
                index_path=FAISS_INDEX_PATH,
                dense_model_name=EMBEDDING_MODEL,
                collection_name=collection_name
            )
            
            if clear_db:
                faiss_db.delete_collection()
            
            faiss_db.store_chunks(chunks_dict)
            faiss_db.close()
            Logger.log(f"Stored {len(chunks_dict)} chunks in FAISS")
        except Exception as e:
            Logger.log(f"FAISS error: {e}")
    
    def _store_in_qdrant(self, chunks_dict: Dict, collection_name: str, clear_db: bool):
        """Store in Qdrant"""
        try:
            Logger.log("Storing in Qdrant...")
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
            Logger.log(f"Stored {len(chunks_dict)} chunks in Qdrant")
        except Exception as e:
            Logger.log(f"Qdrant error: {e}")
