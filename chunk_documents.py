"""
Document Chunking Script
Processes PDFs and creates chunks using different chunking strategies
Caches chunks per document to avoid reprocessing
"""

import asyncio
import os
from logger import Logger
from loader import LocalPDFLoader
from chunker import RecursiveChunker, SemanticChunker, AgenticChunker, AgenticChunkerV2
from llm import Gemini, GeminiLive, Ollama
from config import Config


async def main():
    print("="*80)
    print("DOCUMENT CHUNKING TOOL")
    print("="*80)
    
    config = Config()
    
    # Step 1: Choose source folder
    print("\nStep 1: Document Source")
    print("-" * 80)
    
    default_folder = "./test"
    folder_path = input(f"Enter PDF folder path (default: {default_folder}): ").strip()
    if not folder_path:
        folder_path = default_folder
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return
    
    # Step 2: Load documents
    print("\n" + "="*80)
    print("LOADING DOCUMENTS")
    print("="*80)
    
    loader = LocalPDFLoader(folder_path)
    await loader.load_data()
    Logger.log(f"Loaded {len(loader.pages)} pages from {folder_path}")
    
    if len(loader.pages) == 0:
        print("❌ No PDF documents found in the folder!")
        return
    
    # Main loop - allow multiple chunking operations
    while True:
        # Step 3: Choose chunking strategy
        print("\n" + "="*80)
        print("Step 2: Choose Chunking Strategy")
        print("-" * 80)
        print("\n1. Recursive Chunker")
        print("   - Fast, rule-based splitting")
        print("   - No LLM required")
        print("   - Good for general documents")
        
        print("\n2. Semantic Chunker")
        print("   - Medium speed, uses embeddings")
        print("   - Splits on semantic boundaries")
        print("   - Good for coherent topics")
        
        print("\n3. Agentic Chunker (V1)")
        print("   - Slow, uses LLM")
        print("   - Intelligent proposition-based chunking")
        print("   - Best quality for complex documents")
        print("   - Includes title and summary generation")
        
        print("\n4. Agentic Chunker V2")
        print("   - Fast, uses LLM")
        print("   - Direct semantic splitting")
        print("   - Includes source and page tracking")
        print("   - No title/summary overhead (~3x faster than V1)")
        
        chunker_choice = input("\nEnter your choice (1-4): ").strip()
        
        cache_dir = "./chunk_cache"
        
        if chunker_choice == "1":
            print("\nRecursive Chunker Configuration:")
            chunk_size = input("  Chunk size (default: 1000): ").strip()
            chunk_overlap = input("  Chunk overlap (default: 50): ").strip()
            
            chunk_size = int(chunk_size) if chunk_size else 1000
            chunk_overlap = int(chunk_overlap) if chunk_overlap else 50
            
            chunker = RecursiveChunker(
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                cache_dir=cache_dir
            )
            chunker_name = "recursive"
            
        elif chunker_choice == "2":
            print("\nSemantic Chunker Configuration:")
            print("  Using default: LazarusNLP/all-indo-e5-small-v4")
            
            chunker = SemanticChunker(
                embedding_model_name="LazarusNLP/all-indo-e5-small-v4",
                cache_dir=cache_dir
            )
            chunker_name = "semantic"
            
        elif chunker_choice == "3":
            print("\nAgentic Chunker V1 Configuration:")
            print("  Choose LLM:")
            print("  1. Gemini (gemini-2.0-flash-lite)")
            print("  2. GeminiLive (gemini-2.0-flash-exp)")
            print("  3. Ollama (gemma3:12b)")
            
            llm_choice = input("\n  Enter your choice (1-3): ").strip()
            
            if llm_choice == "2":
                llm = GeminiLive("gemini-2.0-flash-exp", config.GOOGLE_API_KEY)
            elif llm_choice == "3":
                llm = Ollama("gemma3:12b", base_url="http://10.22.208.138:11434")
            else:
                llm = Gemini("gemini-2.0-flash-lite", config.GOOGLE_API_KEY)
            
            chunker = AgenticChunker(llm, cache_dir=cache_dir)
            chunker_name = "agentic"
            
        elif chunker_choice == "4":
            print("\nAgentic Chunker V2 Configuration:")
            print("  Choose LLM:")
            print("  1. Gemini (gemini-2.0-flash-lite)")
            print("  2. GeminiLive (gemini-2.0-flash-exp)")
            print("  3. Ollama (gemma3:12b)")
            
            llm_choice = input("\n  Enter your choice (1-3): ").strip()
            
            if llm_choice == "2":
                llm = GeminiLive("gemini-2.0-flash-exp", config.GOOGLE_API_KEY)
            elif llm_choice == "3":
                llm = Ollama("gemma3:12b", base_url="http://10.22.208.138:11434")
            else:
                llm = Gemini("gemini-2.0-flash-lite", config.GOOGLE_API_KEY)
            
            chunker = AgenticChunkerV2(llm, cache_dir=cache_dir)
            chunker_name = "agentic_v2"
        else:
            print("Invalid choice, using Recursive Chunker")
            chunker = RecursiveChunker(cache_dir=cache_dir)
            chunker_name = "recursive"
        
        # Step 4: Process documents
        print("\n" + "="*80)
        print("PROCESSING DOCUMENTS")
        print("="*80)
        
        use_cache = input("\nUse cached chunks if available? (Y/n): ").strip().lower() != 'n'
        
        Logger.log(f"\nProcessing with {chunker_name} chunker...")
        if use_cache:
            Logger.log("(Cached documents will be skipped)")
        
        try:
            chunker.load_data_to_chunks(loader.pages, use_cache=use_cache)
        except (KeyboardInterrupt, Exception) as e:
            Logger.log(f"\nProcessing interrupted: {type(e).__name__}")
            Logger.log(f"Saving chunks processed so far...")
        
        # Step 5: Display results
        print("\n" + "="*80)
        print("CHUNKING COMPLETED")
        print("="*80)
        
        Logger.log(f"\n✓ Total chunks created: {len(chunker.chunks)}")
        Logger.log(f"✓ Chunks cached in: {cache_dir}")
        
        # Display sample chunks
        print("\nSample chunks (first 3):")
        print("-" * 80)
        
        for i, (chunk_id, chunk) in enumerate(list(chunker.chunks.items())[:3]):
            print(f"\nChunk {i+1} (ID: {chunk_id}):")
            content = chunk.get_context()
            if len(content) > 200:
                print(f"  Content: {content[:200]}...")
            else:
                print(f"  Content: {content}")
            
            if hasattr(chunk, 'source'):
                print(f"  Source: {chunk.source}")
            if hasattr(chunk, 'page'):
                print(f"  Page: {chunk.page}")
        
        # Step 6: Export to consolidated JSON file
        print("\n" + "="*80)
        print("EXPORTING CHUNKS")
        print("="*80)
        
        import json
        from datetime import datetime
        
        # Ensure chunk_cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        output_file = os.path.join(cache_dir, f"{chunker_name}_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        chunks_data = []
        for chunk_id, chunk in chunker.chunks.items():
            chunk_dict = {
                "id": chunk_id,
                "content": chunk.get_context(),
            }
            
            # Add all available fields based on chunk type
            if hasattr(chunk, 'source'):
                chunk_dict["source"] = chunk.source
            if hasattr(chunk, 'page'):
                chunk_dict["page"] = chunk.page
            if hasattr(chunk, 'total_pages'):
                chunk_dict["total_pages"] = chunk.total_pages
            if hasattr(chunk, 'page_label'):
                chunk_dict["page_label"] = chunk.page_label
            if hasattr(chunk, 'semantic_score'):
                chunk_dict["semantic_score"] = chunk.semantic_score
            if hasattr(chunk, 'boundary_type'):
                chunk_dict["boundary_type"] = chunk.boundary_type
            if hasattr(chunk, 'title'):
                chunk_dict["title"] = chunk.title
            if hasattr(chunk, 'summary'):
                chunk_dict["summary"] = chunk.summary
            if hasattr(chunk, 'propositions'):
                chunk_dict["propositions"] = chunk.propositions
            if hasattr(chunk, 'index'):
                chunk_dict["index"] = chunk.index
            if hasattr(chunk, 'metadata'):
                chunk_dict["metadata"] = chunk.metadata
            
            chunks_data.append(chunk_dict)
        
        # Save with metadata
        export_data = {
            "chunker_type": chunker_name,
            "total_chunks": len(chunks_data),
            "created_at": datetime.now().isoformat(),
            "chunks": chunks_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        Logger.log(f"✓ Chunks exported to: {output_file}")
        Logger.log(f"✓ Use 'python load_chunks_to_db.py' to load these chunks into database")
    
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✓ Documents processed: {len(loader.pages)} pages")
        print(f"✓ Chunks created: {len(chunker.chunks)}")
        print(f"✓ Chunker used: {chunker_name}")
        print(f"✓ Cache directory: {cache_dir}")
        
        # Show additional info for agentic_v2
        if chunker_name == "agentic_v2":
            chunks_with_source = sum(1 for c in chunker.chunks.values() if hasattr(c, 'source') and c.source)
            chunks_with_page = sum(1 for c in chunker.chunks.values() if hasattr(c, 'page') and c.page is not None)
            print(f"✓ Chunks with source info: {chunks_with_source}")
            print(f"✓ Chunks with page info: {chunks_with_page}")
        
        print("\nNext steps:")
        print("- Run this script again with different chunker to compare")
        print("- Use test_all_components.py to evaluate chunks in RAG pipeline")
        print("- Add new PDFs and re-run (only new documents will be processed)")
        if chunker_name == "agentic_v2":
            print("- Use example_source_page_info.py to explore source/page filtering")
        print("="*80)
        
        # Ask if user wants to try another chunker
        print("\n" + "="*80)
        another = input("\nProcess with another chunker? (Y/n): ").strip().lower()
        if another == 'n':
            print("\n" + "="*80)
            print("EXITING DOCUMENT CHUNKING TOOL")
            print("="*80)
            break
        
        # Clear screen for next iteration
        print("\n" * 2)


if __name__ == "__main__":
    asyncio.run(main())
