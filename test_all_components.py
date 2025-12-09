"""
Comprehensive Test Script for RAG System
Tests all combinations of chunkers, databases, and search strategies
Evaluates each configuration using RAGAS metrics
"""

import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import RunConfig

from chunker import AgenticChunker, RecursiveChunker, SemanticChunker
from config import Config
from logger import Logger
from llm import Gemini, ChatGPT, Ollama
from database import Qdrant, FAISS
from generator import RecursiveGenerator, SemanticGenerator, AgenticGenerator
from rag.search_strategy import (
    DenseSearchStrategy, 
    SparseSearchStrategy, 
    HybridSearchStrategy,
    HybridColbertSearchStrategy,
    HybridCrossEncoderSearchStrategy
)
from rag.pipeline import RAGPipeline
from loader import LocalPDFLoader


# Constants
CACHE_DIR = "./chunk_cache"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"
EVALUATION_TIMEOUT = 300  # 5 minutes
DEFAULT_RETRIEVAL_LIMIT = 5


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_primary_llm(llm_type: str, config: Config):
        """Create primary LLM for answer generation"""
        llm_configs = {
            "ollama": ("Ollama", lambda: Ollama("qwen3:8b", base_url="https://b84f92e0aabb.ngrok-free.app")),
            "chatgpt": ("ChatGPT", lambda: ChatGPT("gpt-4o-mini", config.OPENAI_API_KEY)),
            "gemini": ("Gemini", lambda: Gemini("gemini-2.0-flash-lite", config.GOOGLE_API_KEY))
        }
        
        name, factory = llm_configs.get(llm_type, llm_configs["gemini"])
        llm = factory()
        Logger.log(f"Using {name} LLM for answer generation")
        return llm
    
    @staticmethod
    def create_evaluation_llm(ragas_llm_type: str, config: Config):
        """Create LLM for RAGAS evaluation"""
        if ragas_llm_type == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model="qwen3:8b",
                base_url="https://b84f92e0aabb.ngrok-free.app",
                temperature=0.0,
                num_predict=2000,
                timeout=EVALUATION_TIMEOUT,
                request_timeout=float(EVALUATION_TIMEOUT)
            )
            Logger.log("Using Ollama LLM for RAGAS evaluation")
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0,
                max_output_tokens=2000
            )
            Logger.log("Using Gemini LLM for RAGAS evaluation")
        
        return LangchainLLMWrapper(llm)
    
    @staticmethod
    def create_evaluation_embeddings(ragas_llm_type: str, config: Config):
        """Create embeddings for RAGAS evaluation"""
        if ragas_llm_type == "ollama":
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                model="qwen3:8b",
                base_url="https://b84f92e0aabb.ngrok-free.app"
            )
            Logger.log("Using Ollama embeddings for RAGAS evaluation")
        else:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GOOGLE_API_KEY
            )
            Logger.log("Using Google embeddings for RAGAS evaluation")
        
        return LangchainEmbeddingsWrapper(embeddings)


class ComponentTester:
    """Test all RAG component combinations and evaluate with RAGAS"""
    
    def __init__(self, testset_path: str, llm_type: str = "gemini", ragas_llm_type: str = "gemini"):
        self.config = Config()
        self.testset_path = testset_path
        self.questions: List[str] = []
        self.ground_truths: List[str] = []
        self.all_results: List[Dict[str, Any]] = []
        
        # Initialize LLMs
        self.primary_llm = LLMFactory.create_primary_llm(llm_type, self.config)
        self.evaluator_llm = LLMFactory.create_evaluation_llm(ragas_llm_type, self.config)
        self.evaluator_embeddings = LLMFactory.create_evaluation_embeddings(ragas_llm_type, self.config)
        
        # Initialize RAGAS metrics
        self._initialize_metrics()
        
        # Load testset
        self._load_testset()
        
        Logger.log("ComponentTester initialized successfully")
    
    def _initialize_metrics(self):
        """Initialize RAGAS evaluation metrics"""
        self.faithfulness_metric = Faithfulness(llm=self.evaluator_llm)
        self.context_recall_metric = LLMContextRecall(llm=self.evaluator_llm)
        self.answer_correctness_metric = AnswerCorrectness(
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings
        )
    
    def _load_testset(self):
        """Load test questions and ground truths"""
        try:
            with open(self.testset_path, 'r', encoding='utf-8') as f:
                testset_data = json.load(f)
            
            for item in testset_data["questions"]:
                self.questions.append(item["question"])
                self.ground_truths.append(item["ground_truth"])
            
            Logger.log(f"Loaded {len(self.questions)} test questions from {self.testset_path}")
            
        except Exception as e:
            Logger.log(f"Error loading testset: {e}")
            raise
    
    def _get_chunker_configs(self) -> List[Tuple[str, Any]]:
        """Get all chunker configurations to test"""
        return [
            ("recursive", RecursiveChunker(cache_dir=CACHE_DIR)),
            ("agentic", AgenticChunker(self.primary_llm, cache_dir=CACHE_DIR)),
            ("semantic", SemanticChunker(embedding_model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR))
        ]
    
    def _create_faiss_db(self, collection_name: str) -> FAISS:
        """Create FAISS database instance"""
        return FAISS(
            index_path=FAISS_INDEX_PATH,
            dense_model_name=EMBEDDING_MODEL,
            collection_name=collection_name
        )
    
    def _create_qdrant_db(self, collection_name: str) -> Optional[Qdrant]:
        """Create Qdrant database instance"""
        try:
            return Qdrant(
                self.config.QDRANT_HOST,
                self.config.QDRANT_API_KEY,
                collection_name
            )
        except Exception as e:
            Logger.log(f"⚠ Failed to create Qdrant instance: {e}")
            return None
    
    def _get_search_strategies(self, db_type: str) -> List[Tuple[str, Any]]:
        """Get compatible search strategies for database type"""
        if db_type == "faiss":
            return [("dense", DenseSearchStrategy())]
        
        return [
            ("dense", DenseSearchStrategy()),
            ("sparse", SparseSearchStrategy()),
            ("hybrid", HybridSearchStrategy()),
            ("hybrid_colbert", HybridColbertSearchStrategy()),
            ("hybrid_crossencoder", HybridCrossEncoderSearchStrategy())
        ]
    
    def _get_generator_class(self, chunker_type: str):
        """Get appropriate generator class based on chunker type"""
        generator_map = {
            "semantic": SemanticGenerator,
            "agentic": AgenticGenerator,
            "recursive": RecursiveGenerator
        }
        return generator_map.get(chunker_type, RecursiveGenerator)
    
    def _process_single_question(
        self, 
        pipeline: RAGPipeline, 
        question: str, 
        ground_truth: str,
        question_num: int,
        total_questions: int
    ) -> Dict[str, Any]:
        """Process a single question and return evaluation data"""
        Logger.log(f"Processing question {question_num}/{total_questions}: {question[:50]}...")
        
        try:
            result = pipeline.query(question)
            contexts = self._extract_contexts(result.get('sources', []))
            
            return {
                "user_input": question,
                "retrieved_contexts": contexts if contexts else ["No relevant context retrieved"],
                "response": result['answer'],
                "reference": ground_truth
            }
        except Exception as e:
            Logger.log(f"Error processing question {question_num}: {e}")
            return {
                "user_input": question,
                "retrieved_contexts": ["Error retrieving context"],
                "response": f"Error: {str(e)}",
                "reference": ground_truth
            }
    
    def _extract_contexts(self, sources: List[Any]) -> List[str]:
        """Extract context texts from sources"""
        contexts = []
        for source in sources:
            if isinstance(source, dict) and source.get('content'):
                context_text = source['content'].strip()
                if len(context_text) > 50:
                    contexts.append(context_text)
        return contexts
    
    def _clean_scores(self, scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Clean scores by removing NaN values"""
        clean_scores = {}
        for key, value in scores.items():
            if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                clean_scores[key] = float(value)
            else:
                clean_scores[key] = None
        return clean_scores
    
    async def test_configuration(
        self,
        chunker_name: str,
        db_name: str,
        search_strategy_name: str,
        database: Any,
        search_strategy: Any,
        generator_class: Any
    ) -> Dict[str, Any]:
        """Test a specific configuration and evaluate with RAGAS"""
        config_name = f"{chunker_name}_{db_name}_{search_strategy_name}"
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Testing Configuration: {config_name}")
        Logger.log(f"{'='*60}")
        
        try:
            generator = generator_class(database, self.primary_llm, search_strategy)
            pipeline = RAGPipeline(
                database=database,
                llm=self.primary_llm,
                search_strategy=search_strategy,
                generator=generator
            )
            
            # Collect evaluation data
            evaluation_data = [
                self._process_single_question(pipeline, q, gt, i+1, len(self.questions))
                for i, (q, gt) in enumerate(zip(self.questions, self.ground_truths))
            ]
            
            # Run RAGAS evaluation
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
            run_config = RunConfig(max_workers=1, timeout=EVALUATION_TIMEOUT)
            
            ragas_result = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    self.context_recall_metric,
                    self.faithfulness_metric,
                    self.answer_correctness_metric
                ],
                llm=self.evaluator_llm,
                run_config=run_config
            )
            
            # Extract and clean scores
            scores = ragas_result.to_dict() if hasattr(ragas_result, 'to_dict') else dict(ragas_result)
            clean_scores = self._clean_scores(scores)
            
            result_data = {
                "configuration": config_name,
                "chunker": chunker_name,
                "database": db_name,
                "search_strategy": search_strategy_name,
                "scores": clean_scores,
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(self.questions)
            }
            
            Logger.log(f"Configuration {config_name} completed!")
            Logger.log(f"Scores: {clean_scores}")
            
            return result_data
            
        except Exception as e:
            Logger.log(f"Error testing configuration {config_name}: {e}")
            return {
                "configuration": config_name,
                "chunker": chunker_name,
                "database": db_name,
                "search_strategy": search_strategy_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _clear_database_collection(self, collection_name: str):
        """Clear a single database collection"""
        # Clear FAISS
        try:
            Logger.log(f"Clearing FAISS collection: {collection_name}")
            faiss_db = self._create_faiss_db(collection_name)
            faiss_db.delete_collection()
            faiss_db.close()
            Logger.log(f"✓ Cleared FAISS collection: {collection_name}")
        except Exception as e:
            Logger.log(f"⚠ Error clearing FAISS collection: {e}")
        
        # Clear Qdrant
        try:
            Logger.log(f"Clearing Qdrant collection: {collection_name}")
            qdrant_db = self._create_qdrant_db(collection_name)
            if qdrant_db:
                qdrant_db.delete_collection()
                qdrant_db.close()
                Logger.log(f"✓ Cleared Qdrant collection: {collection_name}")
        except Exception as e:
            Logger.log(f"⚠ Error clearing Qdrant collection: {e}")
    
    def _clear_all_databases(self, chunker_configs: List[Tuple[str, Any]]):
        """Clear all database collections"""
        Logger.log("\n" + "="*60)
        Logger.log("CLEARING EXISTING DATABASE COLLECTIONS")
        Logger.log("="*60)
        
        for chunker_name, _ in chunker_configs:
            collection_name = f"{chunker_name}_chunks"
            self._clear_database_collection(collection_name)
        
        Logger.log("\n" + "="*60)
        Logger.log("DATABASE CLEARING COMPLETED")
        Logger.log("="*60)
    
    async def _ingest_documents(self, chunker_configs: List[Tuple[str, Any]]):
        """Load and ingest documents into databases"""
        Logger.log("\n" + "="*60)
        Logger.log("LOADING AND CHUNKING DOCUMENTS")
        Logger.log("="*60)
        
        loader = LocalPDFLoader("./test")
        await loader.load_data()
        Logger.log(f"Loaded {len(loader.pages)} pages from test folder")
        
        for chunker_name, chunker in chunker_configs:
            await self._ingest_with_chunker(chunker_name, chunker, loader.pages)
        
        Logger.log("\n" + "="*60)
        Logger.log("DOCUMENT INGESTION COMPLETED")
        Logger.log("="*60)
    
    async def _ingest_with_chunker(self, chunker_name: str, chunker: Any, pages: List[Any]):
        """Ingest documents with a specific chunker"""
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Processing with {chunker_name} chunker...")
        Logger.log(f"{'='*60}")
        
        chunker.load_data_to_chunks(pages, use_cache=True)
        Logger.log(f"✓ {chunker_name} chunker has {len(chunker.chunks)} total chunks")
        
        collection_name = f"{chunker_name}_chunks"
        chunks_for_db = chunker.get_chunks_for_database()
        chunks_dict = {chunk.id: chunk for chunk in chunks_for_db}
        
        # Store in FAISS
        Logger.log(f"Storing {len(chunks_dict)} {chunker_name} chunks in FAISS...")
        faiss_db = self._create_faiss_db(collection_name)
        faiss_db.store_chunks(chunks_dict)
        faiss_db.close()
        Logger.log(f"✓ Stored in FAISS")
        
        # Store in Qdrant
        Logger.log(f"Storing {len(chunks_dict)} {chunker_name} chunks in Qdrant...")
        try:
            qdrant_db = self._create_qdrant_db(collection_name)
            if qdrant_db:
                qdrant_db.store_chunks(chunks_dict)
                qdrant_db.close()
                Logger.log(f"✓ Stored in Qdrant")
        except Exception as e:
            Logger.log(f"⚠ Error storing in Qdrant: {e}")
            Logger.log("Continuing with FAISS only...")
    
    async def test_all_components(self, skip_ingestion: bool = True, clear_db: bool = False):
        """Test all component combinations"""
        Logger.log("\n" + "="*60)
        Logger.log("STARTING COMPREHENSIVE RAG COMPONENT TESTING")
        Logger.log("="*60)
        
        chunker_configs = self._get_chunker_configs()
        
        # Clear databases if requested
        if clear_db and not skip_ingestion:
            self._clear_all_databases(chunker_configs)
        
        # Ingest documents if requested
        if not skip_ingestion:
            await self._ingest_documents(chunker_configs)
        
        # Run tests on all configurations
        await self._run_all_tests(chunker_configs)
        
        Logger.log("\n" + "="*60)
        Logger.log("ALL TESTS COMPLETED")
        Logger.log("="*60)
        
        self._print_summary()
    
    async def _run_all_tests(self, chunker_configs: List[Tuple[str, Any]]):
        """Run tests on all configurations"""
        database_cache = {}
        
        for chunker_name, chunker in chunker_configs:
            collection_name = f"{chunker_name}_chunks"
            
            # Get or create database configs with caching
            if collection_name not in database_cache:
                database_cache[collection_name] = {
                    "faiss": self._create_faiss_db(collection_name),
                    "qdrant": self._create_qdrant_db(collection_name)
                }
            
            database_configs = [("faiss", database_cache[collection_name]["faiss"])]
            
            if database_cache[collection_name]["qdrant"]:
                database_configs.append(("qdrant", database_cache[collection_name]["qdrant"]))
            
            for db_name, database in database_configs:
                search_strategies = self._get_search_strategies(db_name)
                
                for strategy_name, search_strategy in search_strategies:
                    generator_class = self._get_generator_class(chunker_name)
                    
                    result = await self.test_configuration(
                        chunker_name=chunker_name,
                        db_name=db_name,
                        search_strategy_name=strategy_name,
                        database=database,
                        search_strategy=search_strategy,
                        generator_class=generator_class
                    )
                    
                    self.all_results.append(result)
                    self._save_results()
    
    def _save_results(self):
        """Save results to JSON file"""
        output_file = f"component_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_date": datetime.now().isoformat(),
                    "testset": self.testset_path,
                    "num_questions": len(self.questions),
                    "results": self.all_results
                }, f, indent=2, ensure_ascii=False)
            
            Logger.log(f"Results saved to: {output_file}")
            
        except Exception as e:
            Logger.log(f"Error saving results: {e}")
    
    def _calculate_average_score(self, scores: Dict[str, Optional[float]]) -> Optional[float]:
        """Calculate average of valid scores"""
        valid_scores = [v for v in scores.values() if v is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    def _print_configuration_result(self, result: Dict[str, Any]):
        """Print a single configuration result"""
        print(f"\n{result['configuration']}:")
        print(f"  Chunker: {result['chunker']}")
        print(f"  Database: {result['database']}")
        print(f"  Search Strategy: {result['search_strategy']}")
        print(f"  Scores:")
        
        for metric, score in result['scores'].items():
            if score is not None:
                print(f"    - {metric}: {score:.4f}")
            else:
                print(f"    - {metric}: Failed")
        
        avg_score = self._calculate_average_score(result['scores'])
        if avg_score is not None:
            print(f"  Average Score: {avg_score:.4f}")
    
    def _print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        valid_results = [r for r in self.all_results if "error" not in r and r.get("scores")]
        
        for result in valid_results:
            self._print_configuration_result(result)
        
        print("\n" + "="*80)
        
        # Find best configuration
        if valid_results:
            best_result = max(
                valid_results,
                key=lambda r: self._calculate_average_score(r['scores']) or 0
            )
            avg_score = self._calculate_average_score(best_result['scores'])
            print(f"\n🏆 BEST CONFIGURATION: {best_result['configuration']}")
            if avg_score:
                print(f"   Average Score: {avg_score:.4f}")


def get_user_choice(prompt: str, options: List[str], default: str = "1") -> str:
    """Get user choice from list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    choice = input(f"\nEnter your choice (1-{len(options)}): ").strip()
    return choice if choice else default


async def main():
    print("="*80)
    print("RAG SYSTEM COMPREHENSIVE COMPONENT TESTER")
    print("="*80)
    
    # LLM selection for answer generation
    llm_options = [
        "Gemini (Cloud - gemini-2.0-flash-lite)",
        "ChatGPT (Cloud - gpt-4o-mini)",
        "Ollama (Local - qwen3:8b)"
    ]
    llm_choice = get_user_choice("Choose LLM for answer generation:", llm_options)
    llm_type = {"1": "gemini", "2": "chatgpt", "3": "ollama"}.get(llm_choice, "gemini")
    
    # RAGAS evaluation LLM selection
    ragas_options = [
        "Gemini (Cloud - gemini-2.0-flash-lite) - Recommended, reliable",
        "Ollama (Local - qwen3:8b) - Free but may have parsing issues"
    ]
    ragas_choice = get_user_choice("Choose LLM for RAGAS evaluation:", ragas_options)
    ragas_llm_type = "ollama" if ragas_choice == "2" else "gemini"
    
    # Document ingestion
    ingest_options = [
        "Yes - Load from test folder and store in both FAISS and Qdrant",
        "No - Use existing stored chunks"
    ]
    ingest_choice = get_user_choice("Do you want to load and store documents first?", ingest_options)
    skip_ingestion = ingest_choice != "1"
    
    # Database clearing
    clear_db = False
    if not skip_ingestion:
        clear_options = [
            "Yes - Clear all collections in FAISS and Qdrant",
            "No - Keep existing data (may have duplicates)"
        ]
        clear_choice = get_user_choice("Do you want to clear existing database data before ingestion?", clear_options)
        clear_db = clear_choice == "1"
    
    # Testset selection
    testset_options = [
        "Sample testset (ragas/uud_rag_sample_testset.json)",
        "Comprehensive testset (ragas/uud_rag_comprehensive_testset.json)"
    ]
    testset_choice = get_user_choice("Available testsets:", testset_options)
    testset_path = "ragas/uud_rag_sample_testset.json" if testset_choice == "1" else "ragas/uud_rag_comprehensive_testset.json"
    
    # Initialize tester and run
    tester = ComponentTester(testset_path=testset_path, llm_type=llm_type, ragas_llm_type=ragas_llm_type)
    await tester.test_all_components(skip_ingestion=skip_ingestion, clear_db=clear_db)
    
    print("\n✅ Testing completed! Check the results JSON file for detailed output.")


if __name__ == "__main__":
    asyncio.run(main())
