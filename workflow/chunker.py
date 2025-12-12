"""
Document Chunker Workflow
Handle document chunking operations
"""

import os
from typing import Any, Tuple

from chunker import AgenticChunker, RecursiveChunker, SemanticChunker, AgenticChunkerV2
from config import Config
from logger import Logger
from loader import LocalPDFLoader
from ui import UserInterface
from factory import LLMFactory


# Constants
CACHE_DIR = "./chunk_cache"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"
DEFAULT_FOLDER = "./test"


class DocumentChunker:
    """Handle document chunking operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ui = UserInterface()
    
    async def run(self):
        """Run document chunking workflow"""
        self.ui.print_header("DOCUMENT CHUNKING")
        
        # Get folder path
        folder_path = input(f"\nEnter PDF folder path (default: {DEFAULT_FOLDER}): ").strip()
        folder_path = folder_path if folder_path else DEFAULT_FOLDER
        
        if not os.path.exists(folder_path):
            Logger.log(f"❌ Folder '{folder_path}' does not exist!")
            return
        
        # Load documents
        self.ui.print_subheader("Loading Documents")
        loader = LocalPDFLoader(folder_path)
        await loader.load_data()
        Logger.log(f"✓ Loaded {len(loader.pages)} pages")
        
        if len(loader.pages) == 0:
            Logger.log("❌ No PDF documents found!")
            return
        
        # Chunking loop
        while True:
            chunker, chunker_name = self._select_chunker()
            use_cache = self.ui.confirm("Use cached chunks if available?")
            
            # Process documents
            self.ui.print_subheader("Processing Documents")
            try:
                chunker.load_data_to_chunks(loader.pages, use_cache=use_cache)
            except (KeyboardInterrupt, Exception) as e:
                Logger.log(f"\n⚠ Processing interrupted: {type(e).__name__}")
            
            # Summary
            self._print_chunk_summary(chunker, chunker_name)
            
            if not self.ui.confirm("Process with another chunker?"):
                break
    
    def _select_chunker(self) -> Tuple[Any, str]:
        """Select and configure chunker"""
        self.ui.print_subheader("Select Chunking Strategy")
        
        options = [
            "Recursive - Fast, rule-based",
            "Semantic - Medium speed, embedding-based",
            "Agentic V1 - Slow, LLM-based with propositions",
            "Agentic V2 - Fast, LLM-based direct splitting"
        ]
        
        choice = self.ui.get_choice("Choose chunker:", options)
        
        if choice == "1":
            return RecursiveChunker(cache_dir=CACHE_DIR), "recursive"
        elif choice == "2":
            return SemanticChunker(embedding_model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR), "semantic"
        elif choice == "3":
            llm = self._select_llm()
            return AgenticChunker(llm, cache_dir=CACHE_DIR), "agentic"
        else:
            llm = self._select_llm()
            return AgenticChunkerV2(llm, cache_dir=CACHE_DIR), "agentic_v2"
    
    def _select_llm(self):
        """Select LLM for agentic chunking"""
        options = [
            "ChatGPT (gpt-5-nano) - Recommended",
            "Gemini (gemini-2.0-flash-lite)",
            "Ollama (gemma3:12b)"
        ]
        
        choice = self.ui.get_choice("Choose LLM:", options)
        return LLMFactory.create_llm(
            {"1": "chatgpt", "2": "gemini", "3": "ollama"}.get(choice, "chatgpt"),
            self.config
        )
    
    def _print_chunk_summary(self, chunker: Any, chunker_name: str):
        """Print chunking summary"""
        self.ui.print_subheader("Summary")
        print(f"✓ Chunks created: {len(chunker.chunks)}")
        print(f"✓ Chunker: {chunker_name}")
        print(f"✓ Cache directory: {CACHE_DIR}")
