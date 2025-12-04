from chunker import AgenticChunker, RecursiveChunker, SemanticChunker
from config import Config
import asyncio
from logger import Logger
from llm import Gemini, Groq
from database import Qdrant, FAISS
from generator import RecursiveGenerator, FAISSGenerator, SemanticGenerator, AgenticGenerator
from loader import LocalPDFLoader, HuggingFacePDFLoader
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding



def get_user_choice(prompt, choices):
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (number): ")) - 1
            if 0 <= choice < len(choices):
                return choice
            else:
                print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")

async def main():
    print("="*60)
    print("UUD-RAG: Indonesian Legal Document RAG System")
    print("="*60)
    config = Config()
    chunk_choice = get_user_choice(
        "Do you want to use existing chunks or create new chunks?",
        ["Use existing chunks", "Create new chunks from documents"]
    )
    store_data = chunk_choice == 1
    chunker_type = None
    if store_data:
        chunker_choice = get_user_choice(
            "Which chunker do you want to use?",
            ["Recursive Chunker", "Agentic Chunker", "Semantic Chunker"]
        )
        chunker_type = ["recursive", "agentic", "semantic"][chunker_choice]
    db_choice = get_user_choice(
        "Which vector database do you want to use?",
        ["FAISS", "Qdrant"]
    )
    use_faiss = db_choice == 0
    
    loader = LocalPDFLoader("./test")
    gemini = Gemini("gemini-2.5-pro", config.GOOGLE_API_KEY)
    groq = Groq("meta-llama/llama-guard-4-12b", config.GROQ_API_KEY)
    
    recursive_chunker = RecursiveChunker()
    agentic_chunker = AgenticChunker(groq)
    semantic_chunker = SemanticChunker(embedding_model_name="LazarusNLP/all-indo-e5-small-v4")
    
    if store_data:
        if chunker_type == "recursive":
            selected_chunker = recursive_chunker
            collection_name = "recursive_chunks"
            Logger.log("Using Recursive Chunker")
        elif chunker_type == "agentic":
            selected_chunker = agentic_chunker
            collection_name = "agentic_chunks"
            Logger.log("Using Agentic Chunker")
        else:
            selected_chunker = semantic_chunker
            collection_name = "semantic_chunks"
            Logger.log("Using Semantic Chunker")
    else:
        return
    
    if use_faiss:
        Logger.log("Using FAISS database")
        db = FAISS(
            index_path="./faiss_index",
            dense_model_name="LazarusNLP/all-indo-e5-small-v4",
            collection_name=collection_name
        )
        rag_generator = FAISSGenerator(db, gemini)
    else:
        db = Qdrant(config.QDRANT_HOST, config.QDRANT_API_KEY, collection_name)
        if chunker_type == "semantic" and store_data:
            rag_generator = SemanticGenerator(db, gemini)
        elif chunker_type == "agentic" and store_data:
            rag_generator = AgenticGenerator(db, gemini)
        else:
            rag_generator = RecursiveGenerator(db, gemini)

    if store_data:
        await loader.load_data()
        
        Logger.log(f"Processing {len(loader.pages)} pages with {chunker_type} chunker...")
        selected_chunker.load_data_to_chunks(loader.pages)
        
        Logger.log(f"Storing {len(selected_chunker.chunks)} chunks in {db.__class__.__name__} database...")
        db.store_chunks(selected_chunker.chunks)
    else:
        Logger.log("Skipping document loading - using existing chunks")
        
        info = db.get_info()

    print("\n" + "="*60)
    print("RAG System Ready! Enter your questions (type 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            Logger.log("Exiting RAG system...")
            break
        if not query:
            print("Please enter a valid question.")
            continue
        try:
            Logger.log(f"Processing query: {query}")
            answer = rag_generator.generate_answer(query)
            print(f"\n Answer: {answer['answer']}")
            print(f" Sources found: {len(answer['sources'])}")
            if answer['sources']:
                print("\nSource Details:")
                for i, source in enumerate(answer['sources'][:3], 1):
                    if 'source' in source:
                        print(f"   {i}. {source['source']} (Page {source['page']}) - Score: {source['score']:.3f}")
                    elif 'title' in source:
                        print(f"   {i}. {source['title']} - Score: {source['score']:.3f}")
        except Exception as e:
            Logger.log(f"Error processing query: {e}")
            print(f"❌ Error: {str(e)}")
    db.close()
    Logger.log("Session ended successfully!")
    
    
if __name__ == "__main__":
    asyncio.run(main())


