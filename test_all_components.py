"""
Comprehensive Test Script for RAG System
Tests all combinations of chunkers, databases, and search strategies
Evaluates each configuration using RAGAS metrics
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Tuple
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
from llm import Gemini, Groq, Ollama, GeminiLive
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


class ComponentTester:
    """Test all RAG component combinations and evaluate with RAGAS"""
    
    def __init__(self, testset_path: str = "ragas/uud_rag_sample_testset.json", llm_type: str = "gemini", ragas_llm_type: str = "gemini"):
        self.config = Config()
        self.testset_path = testset_path
        self.questions = []
        self.ground_truths = []
        self.llm_type = llm_type
        self.ragas_llm_type = ragas_llm_type
        
        # Initialize LLMs
        if llm_type == "ollama":
            self.primary_llm = Ollama("qwen3:8b", base_url="https://b84f92e0aabb.ngrok-free.app")
            Logger.log("Using Ollama LLM for answer generation")
        elif llm_type == "gemini_live":
            self.primary_llm = GeminiLive("gemini-2.0-flash-exp", self.config.GOOGLE_API_KEY)
            Logger.log("Using GeminiLive LLM for answer generation")
        else:  # default to gemini
            self.primary_llm = Gemini("gemini-2.0-flash-lite", self.config.GOOGLE_API_KEY)
            Logger.log("Using Gemini LLM for answer generation")
        
        self.groq = Groq("llama-3.3-70b-versatile", self.config.GROQ_API_KEY)
        
        # Setup RAGAS evaluator
        self._setup_ragas_evaluator()
        
        # Load testset
        self._load_testset()
        
        # Results storage
        self.all_results = []
        
    def _setup_ragas_evaluator(self):
        """Setup RAGAS evaluation components"""
        if self.ragas_llm_type == "ollama":
            # Use Ollama for RAGAS evaluation
            from langchain_ollama import ChatOllama
            llm_for_eval = ChatOllama(
                model="qwen3:8b",
                base_url="https://b84f92e0aabb.ngrok-free.app",
                temperature=0.0,
                num_predict=2000,
                timeout=300,  # 5 minutes timeout
                request_timeout=300.0  # 5 minutes request timeout
            )
            Logger.log("Using Ollama LLM for RAGAS evaluation")
        else:
            # Use Gemini for RAGAS evaluation (both gemini and gemini_live use same eval model)
            llm_for_eval = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.0,
                max_output_tokens=2000
            )
            Logger.log("Using Gemini LLM for RAGAS evaluation")
        
        self.evaluator_llm = LangchainLLMWrapper(llm_for_eval)
        
        # Setup embeddings based on ragas_llm_type
        if self.ragas_llm_type == "ollama":
            # Use Ollama embeddings with a commonly available model
            from langchain_ollama import OllamaEmbeddings
            ollama_embeddings = OllamaEmbeddings(
                model="qwen3:8b",  # Use the same model that's already loaded
                base_url="https://b84f92e0aabb.ngrok-free.app"
            )
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
            Logger.log("Using Ollama embeddings (qwen3:8b) for RAGAS evaluation")
        else:
            # Use Google embeddings
            google_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.config.GOOGLE_API_KEY
            )
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(google_embeddings)
            Logger.log("Using Google embeddings for RAGAS evaluation")
        
        # Initialize metrics
        self.faithfulness_metric = Faithfulness(llm=self.evaluator_llm)
        self.context_recall_metric = LLMContextRecall(llm=self.evaluator_llm)
        self.answer_correctness_metric = AnswerCorrectness(
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings
        )
        
        Logger.log("RAGAS evaluator initialized successfully")
    
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
        cache_dir = "./chunk_cache"
        return [
            ("recursive", RecursiveChunker(cache_dir=cache_dir)),
            ("agentic", AgenticChunker(self.primary_llm, cache_dir=cache_dir)),
            ("semantic", SemanticChunker(embedding_model_name="LazarusNLP/all-indo-e5-small-v4", cache_dir=cache_dir))
        ]
    
    def _get_database_configs(self, collection_name: str) -> List[Tuple[str, Any]]:
        """Get all database configurations to test"""
        faiss_db = FAISS(
            index_path="./faiss_index",
            dense_model_name="LazarusNLP/all-indo-e5-small-v4",
            collection_name=collection_name
        )
        
        qdrant_db = Qdrant(
            self.config.QDRANT_HOST,
            self.config.QDRANT_API_KEY,
            collection_name
        )
        
        return [
            ("faiss", faiss_db),
            ("qdrant", qdrant_db)
        ]
    
    def _get_search_strategies(self, db_type: str) -> List[Tuple[str, Any]]:
        """Get compatible search strategies for database type"""
        if db_type == "faiss":
            # FAISS only supports dense search
            return [("dense", DenseSearchStrategy())]
        else:
            # Qdrant supports all strategies
            return [
                ("dense", DenseSearchStrategy()),
                ("sparse", SparseSearchStrategy()),
                ("hybrid", HybridSearchStrategy()),
                ("hybrid_colbert", HybridColbertSearchStrategy()),
                ("hybrid_crossencoder", HybridCrossEncoderSearchStrategy())
            ]
    
    def _get_generator_class(self, chunker_type: str):
        """Get appropriate generator class based on chunker type"""
        if chunker_type == "semantic":
            return SemanticGenerator
        elif chunker_type == "agentic":
            return AgenticGenerator
        else:
            return RecursiveGenerator
    
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
            # Create generator
            generator = generator_class(database, self.primary_llm, search_strategy)
            
            # Create pipeline
            pipeline = RAGPipeline(
                database=database,
                llm=self.primary_llm,
                search_strategy=search_strategy,
                generator=generator
            )
            
            # Collect evaluation data
            evaluation_data = []
            
            for i, (question, ground_truth) in enumerate(zip(self.questions, self.ground_truths)):
                Logger.log(f"Processing question {i+1}/{len(self.questions)}: {question[:50]}...")
                
                try:
                    result = pipeline.query(question)
                    answer = result['answer']
                    
                    # Extract contexts
                    contexts = []
                    if result.get('sources'):
                        for source in result['sources']:
                            if isinstance(source, dict) and source.get('content'):
                                context_text = source['content'].strip()
                                if len(context_text) > 50:
                                    contexts.append(context_text)
                    
                    if not contexts:
                        contexts = ["No relevant context retrieved"]
                    
                    evaluation_data.append({
                        "user_input": question,
                        "retrieved_contexts": contexts,
                        "response": answer,
                        "reference": ground_truth
                    })
                    
                except Exception as e:
                    Logger.log(f"Error processing question {i+1}: {e}")
                    evaluation_data.append({
                        "user_input": question,
                        "retrieved_contexts": ["Error retrieving context"],
                        "response": f"Error: {str(e)}",
                        "reference": ground_truth
                    })
            
            # Create RAGAS dataset
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
            
            # Run RAGAS evaluation
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            run_config = RunConfig(max_workers=1, timeout=300)  # 5 minutes timeout
            
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
            
            # Extract scores
            if hasattr(ragas_result, 'to_dict'):
                scores = ragas_result.to_dict()
            else:
                scores = dict(ragas_result)
            
            # Clean scores (remove NaN)
            clean_scores = {}
            for key, value in scores.items():
                if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                    clean_scores[key] = float(value)
                else:
                    clean_scores[key] = None
            
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
    
    async def test_all_components(self, skip_ingestion: bool = True, clear_db: bool = False):
        """Test all component combinations"""
        
        Logger.log("\n" + "="*60)
        Logger.log("STARTING COMPREHENSIVE RAG COMPONENT TESTING")
        Logger.log("="*60)
        
        chunker_configs = self._get_chunker_configs()
        
        # Clear database collections if requested
        if clear_db and not skip_ingestion:
            Logger.log("\n" + "="*60)
            Logger.log("CLEARING EXISTING DATABASE COLLECTIONS")
            Logger.log("="*60)
            
            for chunker_name, _ in chunker_configs:
                collection_name = f"{chunker_name}_chunks"
                
                # Clear FAISS
                try:
                    Logger.log(f"Clearing FAISS collection: {collection_name}")
                    faiss_db = FAISS(
                        index_path="./faiss_index",
                        dense_model_name="LazarusNLP/all-indo-e5-small-v4",
                        collection_name=collection_name
                    )
                    faiss_db.delete_collection()
                    faiss_db.close()
                    Logger.log(f"✓ Cleared FAISS collection: {collection_name}")
                except Exception as e:
                    Logger.log(f"⚠ Error clearing FAISS collection: {e}")
                
                # Clear Qdrant
                try:
                    Logger.log(f"Clearing Qdrant collection: {collection_name}")
                    qdrant_db = Qdrant(
                        self.config.QDRANT_HOST,
                        self.config.QDRANT_API_KEY,
                        collection_name
                    )
                    qdrant_db.delete_collection()
                    qdrant_db.close()
                    Logger.log(f"✓ Cleared Qdrant collection: {collection_name}")
                except Exception as e:
                    Logger.log(f"⚠ Error clearing Qdrant collection: {e}")
            
            Logger.log("\n" + "="*60)
            Logger.log("DATABASE CLEARING COMPLETED")
            Logger.log("="*60)
        
        # If not skipping ingestion, load and chunk documents first
        if not skip_ingestion:
            Logger.log("\n" + "="*60)
            Logger.log("LOADING AND CHUNKING DOCUMENTS")
            Logger.log("="*60)
            
            loader = LocalPDFLoader("./test")
            await loader.load_data()
            Logger.log(f"Loaded {len(loader.pages)} pages from test folder")
            
            # Process each chunker with per-document caching
            for chunker_name, chunker in chunker_configs:
                Logger.log(f"\n{'='*60}")
                Logger.log(f"Processing with {chunker_name} chunker...")
                Logger.log(f"{'='*60}")
                
                # Load data to chunks (with automatic per-document caching)
                chunker.load_data_to_chunks(loader.pages, use_cache=True)
                Logger.log(f"✓ {chunker_name} chunker has {len(chunker.chunks)} total chunks")
                
                collection_name = f"{chunker_name}_chunks"
                
                # Get chunks ready for database
                chunks_for_db = chunker.get_chunks_for_database()
                
                # Convert list to dict format for database storage
                chunks_dict = {chunk.id: chunk for chunk in chunks_for_db}
                
                # Store in FAISS
                Logger.log(f"Storing {len(chunks_dict)} {chunker_name} chunks in FAISS...")
                faiss_db = FAISS(
                    index_path="./faiss_index",
                    dense_model_name="LazarusNLP/all-indo-e5-small-v4",
                    collection_name=collection_name
                )
                faiss_db.store_chunks(chunks_dict)
                faiss_db.close()
                Logger.log(f"✓ Stored in FAISS")
                
                # Store in Qdrant
                Logger.log(f"Storing {len(chunks_dict)} {chunker_name} chunks in Qdrant...")
                try:
                    qdrant_db = Qdrant(
                        self.config.QDRANT_HOST,
                        self.config.QDRANT_API_KEY,
                        collection_name
                    )
                    qdrant_db.store_chunks(chunks_dict)
                    qdrant_db.close()
                    Logger.log(f"✓ Stored in Qdrant")
                except Exception as e:
                    Logger.log(f"⚠ Error storing in Qdrant: {e}")
                    Logger.log("Continuing with FAISS only...")
            
            Logger.log("\n" + "="*60)
            Logger.log("DOCUMENT INGESTION COMPLETED")
            Logger.log("="*60)
        
        # Now run tests on all configurations
        # Cache database instances to avoid reloading heavy models
        database_cache = {}
        
        for chunker_name, chunker in chunker_configs:
            collection_name = f"{chunker_name}_chunks"
            
            # Get or create database configs with caching
            if collection_name not in database_cache:
                database_cache[collection_name] = {
                    "faiss": FAISS(
                        index_path="./faiss_index",
                        dense_model_name="LazarusNLP/all-indo-e5-small-v4",
                        collection_name=collection_name
                    ),
                    "qdrant": None  # Will be created on-demand
                }
            
            database_configs = [
                ("faiss", database_cache[collection_name]["faiss"])
            ]
            
            # Add Qdrant if not already failed
            try:
                if database_cache[collection_name]["qdrant"] is None:
                    database_cache[collection_name]["qdrant"] = Qdrant(
                        self.config.QDRANT_HOST,
                        self.config.QDRANT_API_KEY,
                        collection_name
                    )
                database_configs.append(("qdrant", database_cache[collection_name]["qdrant"]))
            except Exception as e:
                Logger.log(f"⚠ Skipping Qdrant for {collection_name}: {e}")
            
            for db_name, database in database_configs:
                
                # Get compatible search strategies
                search_strategies = self._get_search_strategies(db_name)
                
                for strategy_name, search_strategy in search_strategies:
                    
                    # Get appropriate generator class
                    generator_class = self._get_generator_class(chunker_name)
                    
                    # Test this configuration
                    result = await self.test_configuration(
                        chunker_name=chunker_name,
                        db_name=db_name,
                        search_strategy_name=strategy_name,
                        database=database,
                        search_strategy=search_strategy,
                        generator_class=generator_class
                    )
                    
                    self.all_results.append(result)
                    
                    # Save intermediate results
                    self._save_results()
        
        Logger.log("\n" + "="*60)
        Logger.log("ALL TESTS COMPLETED")
        Logger.log("="*60)
        
        self._print_summary()
    
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
    
    def _print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        # Sort by average score
        valid_results = [r for r in self.all_results if "error" not in r and r.get("scores")]
        
        for result in valid_results:
            config = result['configuration']
            scores = result['scores']
            
            print(f"\n{config}:")
            print(f"  Chunker: {result['chunker']}")
            print(f"  Database: {result['database']}")
            print(f"  Search Strategy: {result['search_strategy']}")
            print(f"  Scores:")
            
            for metric, score in scores.items():
                if score is not None:
                    print(f"    - {metric}: {score:.4f}")
                else:
                    print(f"    - {metric}: Failed")
            
            # Calculate average
            valid_scores = [v for v in scores.values() if v is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                print(f"  Average Score: {avg_score:.4f}")
        
        print("\n" + "="*80)
        
        # Find best configuration
        if valid_results:
            best_result = max(valid_results, key=lambda r: sum([v for v in r['scores'].values() if v is not None]))
            print(f"\n🏆 BEST CONFIGURATION: {best_result['configuration']}")
            print(f"   Average Score: {sum([v for v in best_result['scores'].values() if v is not None]) / len([v for v in best_result['scores'].values() if v is not None]):.4f}")


async def main():
    print("="*80)
    print("RAG SYSTEM COMPREHENSIVE COMPONENT TESTER")
    print("="*80)
    
    # Ask user for LLM choice
    print("\nChoose LLM for answer generation:")
    print("1. Gemini (Cloud - gemini-2.0-flash-lite)")
    print("2. GeminiLive (Cloud - gemini-2.0-flash-exp)")
    print("3. Ollama (Local - qwen3:8b)")
    
    llm_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if llm_choice == "2":
        llm_type = "gemini_live"
    elif llm_choice == "3":
        llm_type = "ollama"
    else:
        llm_type = "gemini"
    
    # Ask user for RAGAS evaluation LLM choice
    print("\nChoose LLM for RAGAS evaluation:")
    print("1. Gemini (Cloud - gemini-2.0-flash-lite) - Recommended, reliable")
    print("2. Ollama (Local - qwen3:8b) - Free but may have parsing issues")
    print("   Note: Ollama may fail to parse structured outputs required by RAGAS")
    
    ragas_choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if ragas_choice == "2":
        ragas_llm_type = "ollama"
    else:
        ragas_llm_type = "gemini"
    
    # Ask about document ingestion
    print("\nDo you want to load and store documents first?")
    print("1. Yes - Load from test folder and store in both FAISS and Qdrant")
    print("2. No - Use existing stored chunks")
    
    ingest_choice = input("\nEnter your choice (1 or 2): ").strip()
    skip_ingestion = ingest_choice != "1"
    
    # Ask about clearing database
    clear_db = False
    if not skip_ingestion:
        print("\nDo you want to clear existing database data before ingestion?")
        print("1. Yes - Clear all collections in FAISS and Qdrant")
        print("2. No - Keep existing data (may have duplicates)")
        
        clear_choice = input("\nEnter your choice (1 or 2): ").strip()
        clear_db = clear_choice == "1"
    
    # Ask user for testset
    print("\nAvailable testsets:")
    print("1. Sample testset (ragas/uud_rag_sample_testset.json)")
    print("2. Comprehensive testset (ragas/uud_rag_comprehensive_testset.json)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        testset_path = "ragas/uud_rag_sample_testset.json"
    elif choice == "2":
        testset_path = "ragas/uud_rag_comprehensive_testset.json"
    else:
        print("Invalid choice, using sample testset")
        testset_path = "ragas/uud_rag_sample_testset.json"
    
    # Initialize tester
    tester = ComponentTester(testset_path=testset_path, llm_type=llm_type, ragas_llm_type=ragas_llm_type)
    
    # Run all tests
    await tester.test_all_components(skip_ingestion=skip_ingestion, clear_db=clear_db)
    
    print("\n✅ Testing completed! Check the results JSON file for detailed output.")


if __name__ == "__main__":
    asyncio.run(main())
