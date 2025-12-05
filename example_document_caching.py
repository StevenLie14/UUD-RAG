"""
Example: Document-Level Caching and Database Storage

This script demonstrates the new chunking workflow:
1. Load documents
2. Chunk documents (with automatic per-document caching)
3. Send chunks to database

If you run this script multiple times:
- Already processed documents will be loaded from cache (fast)
- Only new documents will be processed (slow)
- All chunks (cached + new) will be sent to the database
"""

import asyncio
from loader import LocalPDFLoader
from chunker import RecursiveChunker, SemanticChunker, AgenticChunker
from database import FAISS, Qdrant
from llm import Gemini
from config import Config
from logger import Logger


async def main():
    config = Config()
    
    # Step 1: Load documents
    Logger.log("="*60)
    Logger.log("STEP 1: Loading documents from test folder")
    Logger.log("="*60)
    
    loader = LocalPDFLoader("./test")
    await loader.load_data()
    Logger.log(f"Loaded {len(loader.pages)} pages")
    
    # Step 2: Choose chunker and process documents
    Logger.log("\n" + "="*60)
    Logger.log("STEP 2: Chunking documents with caching")
    Logger.log("="*60)
    
    print("\nChoose chunker:")
    print("1. Recursive (fast, no LLM needed)")
    print("2. Semantic (medium, uses embeddings)")
    print("3. Agentic (slow, uses LLM)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        chunker = RecursiveChunker(cache_dir="./chunk_cache")
        chunker_name = "recursive"
    elif choice == "2":
        chunker = SemanticChunker(
            embedding_model_name="LazarusNLP/all-indo-e5-small-v4",
            cache_dir="./chunk_cache"
        )
        chunker_name = "semantic"
    elif choice == "3":
        gemini = Gemini("gemini-2.0-flash-exp", config.GOOGLE_API_KEY)
        chunker = AgenticChunker(gemini, cache_dir="./chunk_cache")
        chunker_name = "agentic"
    else:
        print("Invalid choice, using recursive")
        chunker = RecursiveChunker(cache_dir="./chunk_cache")
        chunker_name = "recursive"
    
    # Process documents with caching
    Logger.log(f"\nProcessing documents with {chunker_name} chunker...")
    Logger.log("(Documents already processed will be loaded from cache)")
    
    chunker.load_data_to_chunks(loader.pages, use_cache=True)
    
    Logger.log(f"✓ Total chunks: {len(chunker.chunks)}")
    
    # Step 3: Get chunks ready for database
    Logger.log("\n" + "="*60)
    Logger.log("STEP 3: Preparing chunks for database storage")
    Logger.log("="*60)
    
    chunks_for_db = chunker.get_chunks_for_database()
    Logger.log(f"Prepared {len(chunks_for_db)} chunks for database")
    
    # Convert list to dict format for database storage
    chunks_dict = {chunk.id: chunk for chunk in chunks_for_db}
    Logger.log(f"Converted to dict format with {len(chunks_dict)} entries")
    
    # Step 4: Store in database
    Logger.log("\n" + "="*60)
    Logger.log("STEP 4: Storing chunks in database")
    Logger.log("="*60)
    
    print("\nChoose database:")
    print("1. FAISS (local)")
    print("2. Qdrant (cloud)")
    print("3. Both")
    
    db_choice = input("\nEnter choice (1-3): ").strip()
    
    # Ask about clearing existing data
    print("\nDo you want to clear existing data in the database?")
    print("1. Yes - Clear collection before storing (recommended)")
    print("2. No - Keep existing data (may have duplicates)")
    
    clear_choice = input("\nEnter choice (1-2): ").strip()
    clear_db = clear_choice == "1"
    
    collection_name = f"{chunker_name}_chunks"
    
    if db_choice in ["1", "3"]:
        Logger.log("\nStoring in FAISS...")
        faiss_db = FAISS(
            index_path="./faiss_index",
            dense_model_name="LazarusNLP/all-indo-e5-small-v4",
            collection_name=collection_name
        )
        
        if clear_db:
            Logger.log("Clearing existing FAISS collection...")
            faiss_db.delete_collection()
            Logger.log("✓ FAISS collection cleared")
        
        faiss_db.store_chunks(chunks_dict)
        faiss_db.close()
        Logger.log(f"✓ Stored {len(chunks_dict)} chunks in FAISS")
    
    if db_choice in ["2", "3"]:
        Logger.log("\nStoring in Qdrant...")
        try:
            qdrant_db = Qdrant(
                config.QDRANT_HOST,
                config.QDRANT_API_KEY,
                collection_name
            )
            
            if clear_db:
                Logger.log("Clearing existing Qdrant collection...")
                qdrant_db.delete_collection()
                Logger.log("✓ Qdrant collection cleared")
            
            qdrant_db.store_chunks(chunks_dict)
            qdrant_db.close()
            Logger.log(f"✓ Stored {len(chunks_dict)} chunks in Qdrant")
        except Exception as e:
            Logger.log(f"⚠ Error storing in Qdrant: {e}")
    
    Logger.log("\n" + "="*60)
    Logger.log("COMPLETED!")
    Logger.log("="*60)
    Logger.log("\nNext steps:")
    Logger.log("- Run this script again to see caching in action")
    Logger.log("- Add new PDFs to ./test folder and run again")
    Logger.log("- Only new documents will be processed!")


if __name__ == "__main__":
    asyncio.run(main())
