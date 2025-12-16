import os
from typing import Any, Tuple
from chunker import RecursiveChunker, SemanticChunker, BaseChunker
from config import Config
from logger import Logger
from loader import LocalPDFLoader
from ui import UserInterface

CACHE_DIR = "./chunk_cache"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"
DEFAULT_FOLDER = "./peraturan_pdfs"


class DocumentChunker:
    
    def __init__(self, config: Config):
        self.config = config
        self.ui = UserInterface()
    
    async def run(self):
        self.ui.print_header("DOCUMENT CHUNKING")
        
        folder_path = input(f"\nEnter PDF folder path (default: {DEFAULT_FOLDER}): ").strip()
        folder_path = folder_path if folder_path else DEFAULT_FOLDER
        
        if not os.path.exists(folder_path):
            Logger.log(f"Folder '{folder_path}' does not exist!")
            return
        
        self.ui.print_subheader("Loading Documents")
        loader = LocalPDFLoader(folder_path)
        await loader.load_data()
        Logger.log(f"Loaded {len(loader.pages)} pages")
        
        if len(loader.pages) == 0:
            Logger.log("No PDF documents found!")
            return
        
        while True:
            chunker, chunker_name = self._select_chunker()
            use_cache = self.ui.confirm("Use cached chunks if available?")
            
            self.ui.print_subheader("Processing Documents")
            try:
                chunker.load_data_to_chunks(loader.pages, use_cache=use_cache)
            except (KeyboardInterrupt, Exception) as e:
                Logger.log(f"\nProcessing interrupted: {type(e).__name__}")
            
            self._print_chunk_summary(chunker, chunker_name)
            
            if not self.ui.confirm("Process with another chunker?"):
                break
    
    def _select_chunker(self) -> Tuple[BaseChunker, str]:
        self.ui.print_subheader("Select Chunking Strategy")
        
        options = [
            "Recursive",
            "Semantic"
        ]
        
        choice = self.ui.get_choice("Choose chunker:", options)
        
        if choice == "1":
            return RecursiveChunker(cache_dir=CACHE_DIR), "recursive"
        else:
            return SemanticChunker(embedding_model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR), "semantic"
    
    def _print_chunk_summary(self, chunker: Any, chunker_name: str):
        self.ui.print_subheader("Summary")
        print(f"✓ Chunks created: {len(chunker.chunks)}")
        print(f"✓ Chunker: {chunker_name}")
        print(f"✓ Cache directory: {CACHE_DIR}")
