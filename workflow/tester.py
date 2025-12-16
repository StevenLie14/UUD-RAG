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


CACHE_DIR = "./chunk_cache"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "LazarusNLP/all-indo-e5-small-v4"
EVALUATION_TIMEOUT = 300


class RAGComponentTester:
    
    def __init__(self, testset_path: str, llm_type: str = "gemini", config: Optional[Config] = None):
        self.config = config or Config()
        self.testset_path = testset_path
        self.all_results: List[Dict[str, Any]] = []
        
        self.primary_llm = self._create_primary_llm(llm_type)
        self.evaluator = RAGASEvaluator(testset_path, timeout=EVALUATION_TIMEOUT)
    
    def _create_primary_llm(self, llm_type: str):
        llm_configs = {
            "ollama": ("Ollama", lambda: Ollama("qwen3:8b", base_url=self.config.OLLAMA_BASE_URL)),
            "chatgpt": ("ChatGPT", lambda: ChatGPT("gpt-4o-mini", self.config.OPENAI_API_KEY)),
            "gemini": ("Gemini", lambda: Gemini("gemini-2.0-flash-lite", self.config.GOOGLE_API_KEY))
        }
        
        name, factory = llm_configs.get(llm_type, llm_configs["gemini"])
        llm = factory()
        Logger.log(f"Using {name} LLM for answer generation")
        return llm
    
    def _get_chunker_configs(self) -> List[Tuple[str, Any]]:
        return [
            ("recursive", RecursiveChunker(cache_dir=CACHE_DIR)),
            ("agentic", AgenticChunker(self.primary_llm, cache_dir=CACHE_DIR)),
            ("semantic", SemanticChunker(embedding_model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR))
        ]
    
    def _create_faiss_db(self, collection_name: str) -> FAISS:
        return FAISS(
            index_path=FAISS_INDEX_PATH,
            dense_model_name=EMBEDDING_MODEL,
            collection_name=collection_name
        )
    
    def _create_qdrant_db(self, collection_name: str) -> Optional[Qdrant]:
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
            
            result_data = self.evaluator.evaluate_pipeline(
                pipeline, 
                config_name,
                use_cache=use_cache,
                skip_generation=skip_generation
            )
            
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
    
    async def test_all_components(self, use_cache: bool = True, skip_generation: bool = False):
        """Test all component combinations"""
        Logger.log("\n" + "="*60)
        Logger.log("STARTING COMPREHENSIVE RAG COMPONENT TESTING")
        Logger.log("="*60)
        Logger.log(f"Use cache: {use_cache}")
        Logger.log(f"Skip generation: {skip_generation}")
        
        chunker_configs = self._get_chunker_configs()
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
        
        all_chunker_configs = self._get_chunker_configs()
        chunker_configs = [(name, chunker) for name, chunker in all_chunker_configs if name in chunkers]
        
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
        database_cache = {}
        
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
                    Logger.log(f"Skipping {db_name} - not available")
                    continue
                
                if db_name == "faiss":
                    available_strategies = ["dense"]
                else:
                    available_strategies = selected_strategies
                
                for strategy_name in available_strategies:
                    if strategy_name not in selected_strategies:
                        continue
                    
                    search_strategy = strategy_map.get(strategy_name)
                    if not search_strategy:
                        Logger.log(f"Unknown strategy: {strategy_name}")
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
    
    async def _run_all_tests(self, chunker_configs: List[Tuple[str, Any]], use_cache: bool = True, skip_generation: bool = False):
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
            print(f"\nBEST CONFIGURATION: {best_result['configuration']}")
            if avg_score:
                print(f"Average Score: {avg_score:.4f}")


class ComponentTester:
    
    def __init__(self, config: Config):
        self.config = config
    
    async def run(self):
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
        
        Logger.log("Evaluation will use ChatGPT (gpt-4o-mini) by default")
        
        testset_path = self._select_testset_file()
        if not testset_path:
            Logger.log("No testset file selected!")
            return
        
        tester = RAGComponentTester(
            testset_path=testset_path,
            llm_type=llm_type,
            config=self.config
        )
        
        if test_mode == "1":
            await tester.test_all_components()
        else:
            selected_chunkers = self._select_chunkers()
            if not selected_chunkers:
                Logger.log("No chunkers selected!")
                return
            
            selected_databases = self._select_databases()
            if not selected_databases:
                Logger.log("No databases selected!")
                return
            
            selected_strategies = self._select_search_strategies()
            if not selected_strategies:
                Logger.log("No search strategies selected!")
                return
            
            await tester.test_individual_components(
                chunkers=selected_chunkers,
                databases=selected_databases,
                strategies=selected_strategies
            )
    
    def _select_testset_file(self) -> Optional[str]:
        testset_dir = "./test_set"
        
        if not os.path.exists(testset_dir):
            Logger.log(f"Testset directory '{testset_dir}' does not exist!")
            return None
        
        json_files = [f for f in os.listdir(testset_dir) if f.endswith('.json')]
        
        if not json_files:
            Logger.log(f"No JSON files found in '{testset_dir}'!")
            return None
        
        print("\nAvailable testset files:")
        for i, file in enumerate(json_files, 1):
            print(f"  {i}. {file}")
        
        choice = input(f"\nEnter file number (1-{len(json_files)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            return os.path.join(testset_dir, json_files[int(choice) - 1])
        
        Logger.log("Invalid selection!")
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
