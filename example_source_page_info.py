"""
Example demonstrating how to use source and page information in AgenticChunkerV2

This example shows:
1. How chunks automatically extract source (filename) and page number
2. How to filter chunks by source
3. How to search chunks and display their source location
4. How to use get_context() which includes source/page information
"""

from loader.local import LocalLoader
from chunker import AgenticChunkerV2
from llm.gemini import GeminiLLM
from logger import Logger

def main():
    Logger.log("=== Source and Page Information Example ===\n")
    
    # Initialize
    llm = GeminiLLM()
    chunker = AgenticChunkerV2(llm=llm, cache_dir="./chunk_cache")
    
    # Load documents
    loader = LocalLoader()
    pages = loader.load_data("./peraturan_pdfs")
    Logger.log(f"Loaded {len(pages)} pages\n")
    
    # Process documents
    chunker.load_data_to_chunks(pages, use_cache=True)
    Logger.log(f"\nTotal chunks: {len(chunker.chunks)}\n")
    
    # Example 1: Show chunks with source and page info
    Logger.log("=== Example 1: Chunks with Source and Page ===")
    for i, (chunk_id, chunk) in enumerate(list(chunker.chunks.items())[:5]):
        Logger.log(f"\nChunk {i+1}:")
        Logger.log(f"  Source: {chunk.source}")
        Logger.log(f"  Page: {chunk.page}")
        Logger.log(f"  Content preview: {chunk.content[:150]}...")
    
    # Example 2: Filter chunks by source
    Logger.log("\n\n=== Example 2: Filter Chunks by Source ===")
    # Get all unique sources
    sources = set(chunk.source for chunk in chunker.chunks.values() if chunk.source)
    Logger.log(f"Found {len(sources)} unique sources:")
    for source in list(sources)[:5]:
        Logger.log(f"  - {source}")
    
    # Get chunks from a specific source
    if sources:
        target_source = list(sources)[0]
        chunks_from_source = [
            chunk for chunk in chunker.chunks.values() 
            if chunk.source == target_source
        ]
        Logger.log(f"\nChunks from '{target_source}': {len(chunks_from_source)}")
        for i, chunk in enumerate(chunks_from_source[:3]):
            Logger.log(f"  Chunk {i+1} - Page {chunk.page}: {chunk.content[:100]}...")
    
    # Example 3: Search chunks by content and show source
    Logger.log("\n\n=== Example 3: Search and Show Source ===")
    search_term = "pekerja"
    matching_chunks = [
        chunk for chunk in chunker.chunks.values()
        if search_term.lower() in chunk.content.lower()
    ]
    Logger.log(f"Found {len(matching_chunks)} chunks containing '{search_term}'")
    for i, chunk in enumerate(matching_chunks[:3]):
        Logger.log(f"\n  Result {i+1}:")
        Logger.log(f"    Source: {chunk.source}")
        Logger.log(f"    Page: {chunk.page}")
        Logger.log(f"    Preview: {chunk.content[:150]}...")
    
    # Example 4: Using get_context() with source and page
    Logger.log("\n\n=== Example 4: get_context() with Source/Page ===")
    sample_chunk = list(chunker.chunks.values())[0]
    context = sample_chunk.get_context()
    Logger.log("Full context (includes source and page):")
    Logger.log(context[:500] + "...")
    
    # Example 5: Group chunks by source and count pages
    Logger.log("\n\n=== Example 5: Statistics by Source ===")
    from collections import defaultdict
    source_stats = defaultdict(lambda: {"chunks": 0, "pages": set()})
    
    for chunk in chunker.chunks.values():
        if chunk.source:
            source_stats[chunk.source]["chunks"] += 1
            if chunk.page is not None:
                source_stats[chunk.source]["pages"].add(chunk.page)
    
    Logger.log("Statistics per source:")
    for source, stats in list(source_stats.items())[:5]:
        Logger.log(f"\n  {source}:")
        Logger.log(f"    Total chunks: {stats['chunks']}")
        Logger.log(f"    Pages covered: {len(stats['pages'])}")
        Logger.log(f"    Page range: {min(stats['pages']) if stats['pages'] else 'N/A'} - {max(stats['pages']) if stats['pages'] else 'N/A'}")
    
    Logger.log("\n\n=== Example Complete ===")

if __name__ == "__main__":
    main()
