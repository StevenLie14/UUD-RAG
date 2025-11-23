from chunker import AgenticChunker, RecursiveChunker
from config import Config
import asyncio
from logger import Logger
from llm import Gemini, Groq
from database import Qdrant
from generator import RecursiveGenerator
from loader import LocalPDFLoader, HuggingFacePDFLoader
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding



async def main():
    STORE_DATA = True
    config = Config()
    loader = LocalPDFLoader("./test")
    gemini = Gemini("gemini-2.5-pro", config.GOOGLE_API_KEY)
    groq = Groq("meta-llama/llama-guard-4-12b",config.GROQ_API_KEY)
    agentic_chunker = AgenticChunker(groq)
    recursive_chunker = RecursiveChunker()
    # recursive_embedder = RecursiveEmbedder()
    recursive_db = Qdrant(config.QDRANT_HOST, config.QDRANT_API_KEY, "recursive_chunks")

    if STORE_DATA:
        await loader.load_data()
        recursive_chunker.load_data_to_chunks(loader.pages)
        recursive_db.store_chunks(recursive_chunker.chunks)

    


    
    
    # chunker.load_chunks()
    # chunker.print_chunks()
    # db.store_chunks(chunker.chunks)
    # recursive_chunker.load_data_to_chunks(loader.documents)
    # Logger.log("All chunks stored in Qdrant database.")
    # rag = RecursiveGenerator(recursive_db, gemini)
    # answer = rag.generate_answer("Apakah Bank Rakyat Indonesia adalah badan hukum?")
    # Logger.log(f"Answer: {answer}")
    
    
if __name__ == "__main__":
    asyncio.run(main())


