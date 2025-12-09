"""
Example usage of AgenticChunkerV2 - Simple agentic chunking without title/summary generation

This version uses a simpler approach:
1. LLM directly splits text into semantic chunks
2. No title or summary generation (faster and more efficient)
3. Uses SimpleChunk model that only stores the text content
4. Perfect for when you just need semantic chunking without extra metadata
"""

from loader.local import LocalLoader
from chunker import AgenticChunkerV2
from llm.gemini import GeminiLLM
from logger import Logger

def main():
    # Initialize components
    Logger.log("=== Agentic Chunking V2 Example ===")
    Logger.log("Using simple semantic chunking without title/summary generation")
    
    # Initialize LLM
    llm = GeminiLLM()
    Logger.log("Initialized Gemini LLM")
    
    # Initialize chunker
    chunker = AgenticChunkerV2(llm=llm, cache_dir="./chunk_cache")
    Logger.log("Initialized AgenticChunkerV2")
    
    # Load documents
    loader = LocalLoader()
    pages = loader.load_data("./peraturan_pdfs")
    Logger.log(f"Loaded {len(pages)} pages from documents")
    
    # Process documents with agentic chunking v2
    Logger.log("\nStarting agentic chunking v2 (simple semantic splitting)...")
    chunker.load_data_to_chunks(pages, use_cache=True)
    
    # Print results
    Logger.log(f"\n=== Results ===")
    Logger.log(f"Total chunks created: {len(chunker.chunks)}")
    
    # Print first few chunks as examples
    Logger.log("\n=== Sample Chunks ===")
    for i, (chunk_id, chunk) in enumerate(list(chunker.chunks.items())[:3]):
        Logger.log(f"\nChunk {i+1}:")
        Logger.log(f"ID: {chunk_id}")
        Logger.log(f"Index: {chunk.index}")
        Logger.log(f"Source: {chunk.source}")
        Logger.log(f"Page: {chunk.page}")
        Logger.log(f"Content length: {len(chunk.content)} characters")
        Logger.log(f"Content preview: {chunk.content[:300]}...")
        Logger.log(f"Metadata: {chunk.metadata}")
    
    Logger.log("\n=== Agentic Chunking V2 Complete ===")

if __name__ == "__main__":
    main()
