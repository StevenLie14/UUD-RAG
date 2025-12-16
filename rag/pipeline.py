from typing import Optional, Dict, Any
from database.base import VectorStore
from generator.base import BaseGenerator
from chunker.base import BaseChunker
from loader.base import BaseLoader
from llm.base import BaseLLM
from rag.search_strategy import SearchStrategy
from logger import Logger


class RAGPipeline:
    def __init__(
        self,
        database: VectorStore,
        llm: BaseLLM,
        search_strategy: SearchStrategy,
        generator: BaseGenerator,
        chunker: Optional[BaseChunker] = None,
        loader: Optional[BaseLoader] = None
    ):
        self.database = database
        self.llm = llm
        self.search_strategy = search_strategy
        self.generator = generator
        self.chunker = chunker
        self.loader = loader
        
        Logger.log(f"RAG Pipeline initialized with:")
        Logger.log(f"Database: {database.__class__.__name__}")
        Logger.log(f"LLM: {llm.__class__.__name__}")
        Logger.log(f"Search Strategy: {search_strategy.__class__.__name__}")
        Logger.log(f"Generator: {generator.__class__.__name__}")
        if chunker:
            Logger.log(f"Chunker: {chunker.__class__.__name__}")
        if loader:
            Logger.log(f"Loader: {loader.__class__.__name__}")
    
    async def ingest_documents(self) -> bool:
        if not self.loader:
            Logger.log("Error: No loader configured for document ingestion")
            return False
        
        if not self.chunker:
            Logger.log("Error: No chunker configured for document ingestion")
            return False
        
        try:
            Logger.log("Starting document ingestion...")
            
            await self.loader.load_data()
            Logger.log(f"Loaded {len(self.loader.pages)} pages")
            
            self.chunker.load_data_to_chunks(self.loader.pages)
            Logger.log(f"Created {len(self.chunker.chunks)} chunks")
            
            self.database.store_chunks(self.chunker.chunks)
            Logger.log(f"Stored chunks in {self.database.__class__.__name__}")
            
            Logger.log("Document ingestion completed successfully!")
            return True
            
        except Exception as e:
            Logger.log(f"Error during document ingestion: {e}")
            return False
    
    def query(self, question: str, limit: int = 5) -> Dict[str, Any]:
        try:
            Logger.log(f"Processing query: {question}")
            result = self.generator.generate_answer(question, limit)
            Logger.log(f"Query processed successfully")
            return result
            
        except Exception as e:
            Logger.log(f"Error processing query: {e}")
            return {
                "answer": f"Maaf, terjadi error saat memproses pertanyaan: {str(e)}",
                "sources": [],
                "query": question,
                "error": str(e)
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        try:
            return self.database.get_info()
        except Exception as e:
            Logger.log(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def close(self):
        try:
            self.database.close()
            Logger.log("RAG Pipeline closed successfully")
        except Exception as e:
            Logger.log(f"Error closing RAG Pipeline: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
