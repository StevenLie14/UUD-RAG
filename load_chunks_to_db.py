"""
Load Chunks to Database
Loads chunks from JSON file and stores them in FAISS/Qdrant database
"""

import json
import os
from logger import Logger
from database import FAISS, Qdrant
from config import Config
from model.chunk.recursive_chunk import RecursiveChunk
from model.chunk.semantic_chunk import SemanticChunk
from model.chunk.agentic_chunk import AgenticChunk


def load_chunks_from_json(json_file: str, chunker_type: str = None) -> tuple:
    """
    Load chunks from JSON file and convert to chunk objects
    
    Args:
        json_file: Path to JSON file containing chunks
        chunker_type: Type of chunker used ('recursive', 'semantic', 'agentic')
                     If None, will try to read from JSON metadata
    
    Returns:
        Tuple of (chunks_dict, detected_chunker_type)
    """
    Logger.log(f"Loading chunks from {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if new format with metadata
    if isinstance(data, dict) and 'chunks' in data:
        chunks_data = data['chunks']
        detected_type = data.get('chunker_type', chunker_type)
        Logger.log(f"Found {data.get('total_chunks', len(chunks_data))} chunks")
        Logger.log(f"Created at: {data.get('created_at', 'unknown')}")
    else:
        # Old format - just array of chunks
        chunks_data = data
        detected_type = chunker_type
    
    if not chunker_type:
        chunker_type = detected_type
    
    chunks_dict = {}
    
    for chunk_data in chunks_data:
        chunk_id = chunk_data['id']
        
        if chunker_type == 'recursive':
            chunk = RecursiveChunk(
                id=chunk_id,
                content=chunk_data['content'],
                source=chunk_data.get('source'),
                page=chunk_data.get('page'),
                total_pages=chunk_data.get('total_pages'),
                page_label=chunk_data.get('page_label')
            )
        elif chunker_type == 'semantic':
            chunk = SemanticChunk(
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
            chunk = AgenticChunk(
                id=chunk_id,
                title=chunk_data.get('title', ''),
                summary=chunk_data.get('summary', ''),
                propositions=chunk_data.get('propositions', [chunk_data['content']]),
                index=chunk_data.get('index', 0),
                metadata=chunk_data.get('metadata', {})
            )
        else:
            Logger.log(f"Unknown chunker type: {chunker_type}, using RecursiveChunk")
            chunk = RecursiveChunk(
                id=chunk_id,
                content=chunk_data['content'],
                source=chunk_data.get('source'),
                page=chunk_data.get('page')
            )
        
        chunks_dict[chunk_id] = chunk
    
    Logger.log(f"✓ Loaded {len(chunks_dict)} chunks")
    return chunks_dict, chunker_type


def main():
    print("="*80)
    print("LOAD CHUNKS TO DATABASE")
    print("="*80)
    
    config = Config()
    
    # Step 1: Select JSON file
    print("\nStep 1: Select Chunks JSON File")
    print("-" * 80)
    
    # List available JSON files in chunk_cache and current directory
    json_files = []
    
    # Check chunk_cache directory
    if os.path.exists('./chunk_cache'):
        cache_files = [f for f in os.listdir('./chunk_cache') if f.endswith('.json')]
        json_files.extend([os.path.join('./chunk_cache', f) for f in cache_files])
    
    # Check current directory
    current_files = [f for f in os.listdir('.') if f.endswith('_chunks_') and f.endswith('.json')]
    json_files.extend(current_files)
    
    if json_files:
        print("\nAvailable chunk files:")
        for i, file in enumerate(json_files, 1):
            print(f"  {i}. {file}")
        
        file_choice = input("\nEnter file number or full path: ").strip()
        
        if file_choice.isdigit() and 1 <= int(file_choice) <= len(json_files):
            json_file = json_files[int(file_choice) - 1]
        else:
            json_file = file_choice
    else:
        json_file = input("\nEnter JSON file path: ").strip()
    
    if not os.path.exists(json_file):
        print(f"❌ Error: File '{json_file}' not found!")
        return
    
    # Step 2: Detect chunker type
    print("\n" + "="*80)
    print("Step 2: Chunker Type")
    print("-" * 80)
    
    # Try to auto-detect from filename or JSON metadata
    detected_type = None
    
    # Try to read type from JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'chunker_type' in data:
                detected_type = data['chunker_type']
                print(f"\n✓ Detected from JSON metadata: {detected_type}")
    except:
        pass
    
    # Try to detect from filename if not in JSON
    if not detected_type:
        if 'recursive' in json_file.lower():
            detected_type = 'recursive'
        elif 'semantic' in json_file.lower():
            detected_type = 'semantic'
        elif 'agentic' in json_file.lower():
            detected_type = 'agentic'
        else:
            detected_type = 'recursive'
        print(f"\n⚠ Detected from filename: {detected_type}")
    
    print("\nAvailable types:")
    print("  1. recursive")
    print("  2. semantic")
    print("  3. agentic")
    
    type_choice = input(f"\nConfirm type (1-3) or press Enter for '{detected_type}': ").strip()
    
    if type_choice == '1':
        chunker_type = 'recursive'
    elif type_choice == '2':
        chunker_type = 'semantic'
    elif type_choice == '3':
        chunker_type = 'agentic'
    else:
        chunker_type = detected_type
    
    # Step 3: Load chunks
    print("\n" + "="*80)
    print("LOADING CHUNKS")
    print("="*80)
    
    chunks_dict, final_type = load_chunks_from_json(json_file, chunker_type)
    chunker_type = final_type  # Use the final detected type
    
    # Step 4: Select database
    print("\n" + "="*80)
    print("Step 3: Select Database")
    print("-" * 80)
    print("\n1. FAISS (local)")
    print("2. Qdrant (cloud)")
    print("3. Both")
    
    db_choice = input("\nEnter your choice (1-3): ").strip()
    
    # Step 5: Collection name
    print("\n" + "="*80)
    print("Step 4: Collection Name")
    print("-" * 80)
    
    default_collection = f"{chunker_type}_chunks"
    collection_name = input(f"\nEnter collection name (default: {default_collection}): ").strip()
    if not collection_name:
        collection_name = default_collection
    
    # Step 6: Clear existing data
    print("\n" + "="*80)
    print("Step 5: Clear Existing Data")
    print("-" * 80)
    
    clear_choice = input("\nClear existing collection before storing? (y/N): ").strip().lower()
    clear_db = clear_choice == 'y'
    
    # Step 7: Store in databases
    print("\n" + "="*80)
    print("STORING CHUNKS IN DATABASE")
    print("="*80)
    
    if db_choice in ['1', '3']:
        print("\n📦 Storing in FAISS...")
        try:
            faiss_db = FAISS(
                index_path="./faiss_index",
                dense_model_name="LazarusNLP/all-indo-e5-small-v4",
                collection_name=collection_name
            )
            
            if clear_db:
                Logger.log("Clearing existing FAISS collection...")
                faiss_db.delete_collection()
            
            faiss_db.store_chunks(chunks_dict)
            faiss_db.close()
            Logger.log(f"✓ Stored {len(chunks_dict)} chunks in FAISS")
        except Exception as e:
            Logger.log(f"❌ Error storing in FAISS: {e}")
    
    if db_choice in ['2', '3']:
        print("\n☁️  Storing in Qdrant...")
        try:
            qdrant_db = Qdrant(
                config.QDRANT_HOST,
                config.QDRANT_API_KEY,
                collection_name
            )
            
            if clear_db:
                Logger.log("Clearing existing Qdrant collection...")
                qdrant_db.delete_collection()
                # Recreate collection after deletion
                qdrant_db._create_collection_if_not_exists()
            
            qdrant_db.store_chunks(chunks_dict)
            qdrant_db.close()
            Logger.log(f"✓ Stored {len(chunks_dict)} chunks in Qdrant")
        except Exception as e:
            Logger.log(f"❌ Error storing in Qdrant: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"✓ Loaded from: {json_file}")
    print(f"✓ Chunks stored: {len(chunks_dict)}")
    print(f"✓ Chunker type: {chunker_type}")
    print(f"✓ Collection name: {collection_name}")
    
    if db_choice == '1':
        print(f"✓ Database: FAISS")
    elif db_choice == '2':
        print(f"✓ Database: Qdrant")
    else:
        print(f"✓ Database: FAISS + Qdrant")
    
    print("\nNext steps:")
    print("- Use test_all_components.py to evaluate the stored chunks")
    print("- Query the database using your RAG pipeline")
    print("="*80)


if __name__ == "__main__":
    main()
