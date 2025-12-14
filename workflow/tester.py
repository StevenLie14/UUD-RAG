"""
Unified Component Testing Workflow
Combines testing logic and user interface for RAG component evaluation
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

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
from evaluator import RAGASEvaluator


# Constants
CACHE_DIR = "./chunk_cache"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"
EVALUATION_TIMEOUT = 300


class RAGComponentTester:
    """Core RAG Component Tester with testing logic"""
    
    def __init__(self, testset_path: str, llm_type: str = "gemini", config: Optional[Config] = None):
        """
        Initialize RAGComponentTester
        
        Args:
            testset_path: Path to the testset JSON file
            llm_type: Type of LLM to use for answer generation ('gemini', 'chatgpt', 'ollama')
            config: Optional Config instance (creates new one if not provided)
        """
        self.config = config or Config()
        self.testset_path = testset_path
        self.all_results: List[Dict[str, Any]] = []
        
        # Initialize primary LLM for answer generation
        self.primary_llm = self._create_primary_llm(llm_type)
        
        # Initialize RAGAS evaluator
        self.evaluator = RAGASEvaluator(testset_path, timeout=EVALUATION_TIMEOUT)
        
        Logger.log("RAGComponentTester initialized successfully")
    
    def _create_primary_llm(self, llm_type: str):
        """Create primary LLM for answer generation"""
        llm_configs = {
            "ollama": ("Ollama", lambda: Ollama("qwen3:8b", base_url="https://b84f92e0aabb.ngrok-free.app")),
            "chatgpt": ("ChatGPT", lambda: ChatGPT("gpt-4.1-mini", self.config.OPENAI_API_KEY)),
            "gemini": ("Gemini", lambda: Gemini("gemini-2.0-flash-lite", self.config.GOOGLE_API_KEY))
        }
        
        name, factory = llm_configs.get(llm_type, llm_configs["gemini"])
        llm = factory()
        Logger.log(f"Using {name} LLM for answer generation")
        return llm
    
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
    
    async def test_configuration(
        self,
        chunker_name: str,
        db_name: str,
        search_strategy_name: str,
        database: Any,
        search_strategy: Any,
        generator_class: Any,
        use_cache: bool = True,
        skip_generation: bool = False
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
            
            # Use RAGAS evaluator to evaluate the pipeline
            result_data = self.evaluator.evaluate_pipeline(
                pipeline, 
                config_name,
                use_cache=use_cache,
                skip_generation=skip_generation
            )
            
            # Add additional metadata
            result_data["chunker"] = chunker_name
            result_data["database"] = db_name
            result_data["search_strategy"] = search_strategy_name
            
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
        try:
            Logger.log(f"Clearing FAISS collection: {collection_name}")
            faiss_db = self._create_faiss_db(collection_name)
            faiss_db.delete_collection()
            faiss_db.close()
            Logger.log(f"✓ Cleared FAISS collection: {collection_name}")
        except Exception as e:
            Logger.log(f"⚠ Error clearing FAISS collection: {e}")
        
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
        
        Logger.log(f"Storing {len(chunks_dict)} {chunker_name} chunks in FAISS...")
        faiss_db = self._create_faiss_db(collection_name)
        faiss_db.store_chunks(chunks_dict)
        faiss_db.close()
        Logger.log(f"✓ Stored in FAISS")
        
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
    
    async def test_all_components(self, skip_ingestion: bool = True, clear_db: bool = False, use_cache: bool = True, skip_generation: bool = False):
        """Test all component combinations"""
        Logger.log("\n" + "="*60)
        Logger.log("STARTING COMPREHENSIVE RAG COMPONENT TESTING")
        Logger.log("="*60)
        Logger.log(f"Use cache: {use_cache}")
        Logger.log(f"Skip generation: {skip_generation}")
        
        chunker_configs = self._get_chunker_configs()
        
        if clear_db and not skip_ingestion:
            self._clear_all_databases(chunker_configs)
        
        if not skip_ingestion:
            await self._ingest_documents(chunker_configs)
        
        await self._run_all_tests(chunker_configs, use_cache, skip_generation)
        
        Logger.log("\n" + "="*60)
        Logger.log("ALL TESTS COMPLETED")
        Logger.log("="*60)
        
        self._print_summary()
    
    async def test_individual_components(
        self, 
        chunkers: List[str], 
        databases: List[str], 
        strategies: List[str],
        skip_ingestion: bool = True, 
        clear_db: bool = False,
        use_cache: bool = True,
        skip_generation: bool = False
    ):
        """Test selected component combinations"""
        Logger.log("\n" + "="*60)
        Logger.log("STARTING INDIVIDUAL COMPONENT TESTING")
        Logger.log("="*60)
        Logger.log(f"Chunkers: {', '.join(chunkers)}")
        Logger.log(f"Databases: {', '.join(databases)}")
        Logger.log(f"Strategies: {', '.join(strategies)}")
        Logger.log(f"Use cache: {use_cache}")
        Logger.log(f"Skip generation: {skip_generation}")
        
        # Filter chunker configs
        all_chunker_configs = self._get_chunker_configs()
        chunker_configs = [(name, chunker) for name, chunker in all_chunker_configs if name in chunkers]
        
        if clear_db and not skip_ingestion:
            self._clear_all_databases(chunker_configs)
        
        if not skip_ingestion:
            await self._ingest_documents(chunker_configs)
        
        await self._run_selected_tests(chunker_configs, databases, strategies, use_cache, skip_generation)
        
        Logger.log("\n" + "="*60)
        Logger.log("INDIVIDUAL TESTS COMPLETED")
        Logger.log("="*60)
        
        self._print_summary()
    
    async def _run_selected_tests(
        self, 
        chunker_configs: List[Tuple[str, Any]], 
        selected_databases: List[str],
        selected_strategies: List[str],
        use_cache: bool = True,
        skip_generation: bool = False
    ):
        """Run tests on selected configurations"""
        database_cache = {}
        
        # Map strategy names to search strategy objects
        strategy_map = {
            "dense": DenseSearchStrategy(),
            "sparse": SparseSearchStrategy(),
            "hybrid": HybridSearchStrategy(),
            "hybrid_colbert": HybridColbertSearchStrategy(),
            "hybrid_crossencoder": HybridCrossEncoderSearchStrategy()
        }
        
        for chunker_name, chunker in chunker_configs:
            collection_name = f"{chunker_name}_chunks"
            
            if collection_name not in database_cache:
                database_cache[collection_name] = {
                    "faiss": self._create_faiss_db(collection_name),
                    "qdrant": self._create_qdrant_db(collection_name)
                }
            
            for db_name in selected_databases:
                database = database_cache[collection_name].get(db_name)
                if not database:
                    Logger.log(f"⚠ Skipping {db_name} - not available")
                    continue
                
                # Filter strategies based on database type
                if db_name == "faiss":
                    available_strategies = ["dense"]
                else:
                    available_strategies = selected_strategies
                
                for strategy_name in available_strategies:
                    if strategy_name not in selected_strategies:
                        continue
                    
                    search_strategy = strategy_map.get(strategy_name)
                    if not search_strategy:
                        Logger.log(f"⚠ Unknown strategy: {strategy_name}")
                        continue
                    
                    generator_class = self._get_generator_class(chunker_name)
                    
                    result = await self.test_configuration(
                        chunker_name=chunker_name,
                        db_name=db_name,
                        search_strategy_name=strategy_name,
                        database=database,
                        search_strategy=search_strategy,
                        generator_class=generator_class,
                        use_cache=use_cache,
                        skip_generation=skip_generation
                    )
                    
                    self.all_results.append(result)
                    self._save_results()
    
    async def _run_all_tests(self, chunker_configs: List[Tuple[str, Any]], use_cache: bool = True, skip_generation: bool = False):
        """Run tests on all configurations"""
        database_cache = {}
        
        for chunker_name, chunker in chunker_configs:
            collection_name = f"{chunker_name}_chunks"
            
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
                        generator_class=generator_class,
                        use_cache=use_cache,
                        skip_generation=skip_generation
                    )
                    
                    self.all_results.append(result)
                    self._save_results()
    
    def _save_results(self):
        """Save results to JSON file"""
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, f"component_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_date": datetime.now().isoformat(),
                    "testset": self.testset_path,
                    "num_questions": len(self.evaluator.questions),
                    "results": self.all_results
                }, f, indent=2, ensure_ascii=False)
            
            Logger.log(f"Results saved to: {output_file}")
            
        except Exception as e:
            Logger.log(f"Error saving results: {e}")
    
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
        
        avg_score = self.evaluator.calculate_average_score(result['scores'])
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
        
        if valid_results:
            best_result = max(
                valid_results,
                key=lambda r: self.evaluator.calculate_average_score(r['scores']) or 0
            )
            avg_score = self.evaluator.calculate_average_score(best_result['scores'])
            print(f"\n🏆 BEST CONFIGURATION: {best_result['configuration']}")
            if avg_score:
                print(f"   Average Score: {avg_score:.4f}")


class ComponentTester:
    """UI wrapper for RAG component testing workflow"""
    
    def __init__(self, config: Config):
        """Initialize ComponentTester with config only"""
        self.config = config
    
    async def run(self):
        """Run component testing workflow with user interaction"""
        print("\n" + "="*80)
        print("RAG COMPONENT TESTING".center(80))
        print("="*80)
        
        # Select test mode
        print("\nSelect test mode:")
        print("  1. Test all components")
        print("  2. Test individual components")
        test_mode = input("\nEnter choice (1-2): ").strip()
        
        # Get configuration
        print("\nChoose LLM for answers:")
        print("  1. Gemini")
        print("  2. ChatGPT")
        print("  3. Ollama")
        llm_choice = input("\nEnter choice (1-3): ").strip()
        llm_type = {"1": "gemini", "2": "chatgpt", "3": "ollama"}.get(llm_choice, "gemini")
        
        Logger.log("ℹ️ Evaluation will use ChatGPT (gpt-4.1-mini) by default")
        
        # Ingestion options
        skip_ingestion_input = input("\nLoad and store documents first? (y/N): ").strip().lower()
        skip_ingestion = skip_ingestion_input != 'y'
        
        clear_db = False
        if not skip_ingestion:
            clear_db_input = input("Clear existing database data? (y/N): ").strip().lower()
            clear_db = clear_db_input == 'y'
        
        # Select testset file
        testset_path = self._select_testset_file()
        if not testset_path:
            Logger.log("❌ No testset file selected!")
            return
        
        # Initialize core tester
        tester = RAGComponentTester(
            testset_path=testset_path,
            llm_type=llm_type,
            config=self.config
        )
        
        # Run based on test mode
        if test_mode == "1":
            await tester.test_all_components(skip_ingestion=skip_ingestion, clear_db=clear_db)
        else:
            selected_chunkers = self._select_chunkers()
            if not selected_chunkers:
                Logger.log("❌ No chunkers selected!")
                return
            
            selected_databases = self._select_databases()
            if not selected_databases:
                Logger.log("❌ No databases selected!")
                return
            
            selected_strategies = self._select_search_strategies()
            if not selected_strategies:
                Logger.log("❌ No search strategies selected!")
                return
            
            await tester.test_individual_components(
                chunkers=selected_chunkers,
                databases=selected_databases,
                strategies=selected_strategies,
                skip_ingestion=skip_ingestion,
                clear_db=clear_db
            )
        
        Logger.log("✅ Testing completed!")
    
    def _select_testset_file(self) -> Optional[str]:
        """Select testset file from test_set folder"""
        testset_dir = "./test_set"
        
        if not os.path.exists(testset_dir):
            Logger.log(f"⚠ Testset directory '{testset_dir}' does not exist!")
            return None
        
        json_files = [f for f in os.listdir(testset_dir) if f.endswith('.json')]
        
        if not json_files:
            Logger.log(f"⚠ No JSON files found in '{testset_dir}'!")
            return None
        
        print("\nAvailable testset files:")
        for i, file in enumerate(json_files, 1):
            print(f"  {i}. {file}")
        
        choice = input(f"\nEnter file number (1-{len(json_files)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            return os.path.join(testset_dir, json_files[int(choice) - 1])
        
        Logger.log("❌ Invalid selection!")
        return None
    
    def _select_chunkers(self) -> Optional[List[str]]:
        """Select which chunkers to test"""
        print("\nSelect chunkers to test:")
        print("  1. Recursive")
        print("  2. Semantic")
        print("  3. Agentic")
        print("  4. All chunkers")
        
        choice = input("\nEnter choice (1-4, or comma-separated like '1,2'): ").strip()
        
        if choice == "4":
            return ["recursive", "semantic", "agentic"]
        
        chunker_map = {"1": "recursive", "2": "semantic", "3": "agentic"}
        selected = []
        
        for c in choice.split(','):
            c = c.strip()
            if c in chunker_map:
                selected.append(chunker_map[c])
        
        return selected if selected else None
    
    def _select_databases(self) -> Optional[List[str]]:
        """Select which databases to test"""
        print("\nSelect databases to test:")
        print("  1. FAISS")
        print("  2. Qdrant")
        print("  3. Both")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            return ["faiss"]
        elif choice == "2":
            return ["qdrant"]
        elif choice == "3":
            return ["faiss", "qdrant"]
        
        return None
    
    def _select_search_strategies(self) -> Optional[List[str]]:
        """Select which search strategies to test"""
        print("\nSelect search strategies to test:")
        print("  1. Dense")
        print("  2. Sparse")
        print("  3. Hybrid")
        print("  4. Hybrid Colbert")
        print("  5. Hybrid CrossEncoder")
        print("  6. All strategies")
        
        choice = input("\nEnter choice (1-6, or comma-separated like '1,3'): ").strip()
        
        if choice == "6":
            return ["dense", "sparse", "hybrid", "hybrid_colbert", "hybrid_crossencoder"]
        
        strategy_map = {
            "1": "dense",
            "2": "sparse",
            "3": "hybrid",
            "4": "hybrid_colbert",
            "5": "hybrid_crossencoder"
        }
        
        selected = []
        for c in choice.split(','):
            c = c.strip()
            if c in strategy_map:
                selected.append(strategy_map[c])
        
        return selected if selected else None


# Convenience function for standalone usage
async def run_component_testing(testset_path: Optional[str] = None, llm_type: str = "gemini"):
    """
    Run component testing workflow
    
    Args:
        testset_path: Optional path to testset file (will prompt if not provided)
        llm_type: LLM type to use ('gemini', 'chatgpt', 'ollama')
    """
    config = Config()
    tester = ComponentTester(config)
    await tester.run()


if __name__ == "__main__":
    # Run standalone
    asyncio.run(run_component_testing())
