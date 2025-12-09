"""
RAGAS Evaluator
Handles evaluation of RAG systems using RAGAS metrics
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Config
from logger import Logger
from rag.pipeline import RAGPipeline


class RAGASEvaluator:
    """Evaluate RAG systems using RAGAS metrics"""
    
    def __init__(self, testset_path: str, timeout: int = 300):
        """
        Initialize RAGAS evaluator
        
        Args:
            testset_path: Path to testset JSON file
            timeout: Timeout for evaluation in seconds
        """
        self.config = Config()
        self.testset_path = testset_path
        self.timeout = timeout
        self.questions: List[str] = []
        self.ground_truths: List[str] = []
        
        # Initialize evaluation LLM and embeddings (always ChatGPT)
        self.evaluator_llm = self._create_evaluation_llm()
        self.evaluator_embeddings = self._create_evaluation_embeddings()
        
        # Initialize RAGAS metrics
        self._initialize_metrics()
        
        # Load testset
        self._load_testset()
        
        Logger.log("RAGASEvaluator initialized successfully")
    
    def _create_evaluation_llm(self):
        """Create LLM for RAGAS evaluation - Always uses ChatGPT"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.config.OPENAI_API_KEY,
            temperature=0.1,
        )
        
        Logger.log("Using ChatGPT (gpt-4o-mini) for RAGAS evaluation")
        return LangchainLLMWrapper(llm)
    
    def _create_evaluation_embeddings(self):
        """Create embeddings for evaluation - Always uses OpenAI"""
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.config.OPENAI_API_KEY
        )
        
        Logger.log("Using OpenAI embeddings for RAGAS evaluation")
        return LangchainEmbeddingsWrapper(embeddings)
    
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
    
    def _process_single_question(
        self, 
        pipeline: RAGPipeline, 
        question: str, 
        ground_truth: str,
        question_num: int,
        total_questions: int
    ) -> Dict[str, Any]:
        """
        Process a single question and return evaluation data
        
        Args:
            pipeline: RAG pipeline to use
            question: Question to process
            ground_truth: Ground truth answer
            question_num: Current question number
            total_questions: Total number of questions
            
        Returns:
            Dictionary with evaluation data
        """
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
        """Clean scores by removing NaN values and handling list/dict values"""
        clean_scores = {}
        for key, value in scores.items():
            try:
                # Log the value type for debugging
                Logger.log(f"Processing score '{key}': type={type(value)}, value={value}")
                
                # Handle dict values (RAGAS sometimes returns nested structures)
                if isinstance(value, dict):
                    # Try common keys that might contain the actual score
                    if 'score' in value:
                        value = value['score']
                    elif 'value' in value:
                        value = value['value']
                    elif 'mean' in value:
                        value = value['mean']
                    elif 'average' in value:
                        value = value['average']
                    else:
                        # If no known key, log and skip
                        Logger.log(f"Dict score '{key}' has unknown structure: {value}")
                        clean_scores[key] = None
                        continue
                
                # Handle list values (RAGAS sometimes returns scores as lists)
                if isinstance(value, list):
                    if len(value) > 0:
                        # Take the first element if single item, or average if multiple
                        if len(value) == 1:
                            value = value[0]
                        else:
                            value = sum(float(v) for v in value if v is not None) / len([v for v in value if v is not None])
                    else:
                        value = None
                
                # Check for valid numeric value
                if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                    clean_scores[key] = float(value)
                else:
                    clean_scores[key] = None
                    
            except (ValueError, TypeError) as e:
                Logger.log(f"Error processing score '{key}': {e}")
                clean_scores[key] = None
                
        return clean_scores
    
    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        config_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline using RAGAS metrics
        
        Args:
            pipeline: RAG pipeline to evaluate
            config_name: Name of the configuration being tested
            
        Returns:
            Dictionary with evaluation results
        """
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Evaluating Configuration: {config_name}")
        Logger.log(f"{'='*60}")
        
        try:
            # Collect evaluation data
            evaluation_data = [
                self._process_single_question(pipeline, q, gt, i+1, len(self.questions))
                for i, (q, gt) in enumerate(zip(self.questions, self.ground_truths))
            ]
            
            # Run RAGAS evaluation
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
            run_config = RunConfig(max_workers=1, timeout=self.timeout)
            
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
            try:
                # Try different methods to extract scores
                if hasattr(ragas_result, 'to_dict'):
                    scores = ragas_result.to_dict()
                elif hasattr(ragas_result, '__dict__'):
                    scores = ragas_result.__dict__
                else:
                    scores = dict(ragas_result)
                
                Logger.log(f"Raw scores type: {type(scores)}")
                Logger.log(f"Raw scores: {scores}")
                
                clean_scores = self._clean_scores(scores)
                
                result_data = {
                    "configuration": config_name,
                    "scores": clean_scores,
                    "timestamp": datetime.now().isoformat(),
                    "num_questions": len(self.questions)
                }
                
                Logger.log(f"Configuration {config_name} completed!")
                Logger.log(f"Clean scores: {clean_scores}")
                
                return result_data
                
            except Exception as score_error:
                Logger.log(f"Error extracting scores: {score_error}")
                Logger.log(f"Result object type: {type(ragas_result)}")
                Logger.log(f"Result object attributes: {dir(ragas_result)}")
                raise score_error
            
        except Exception as e:
            Logger.log(f"Error evaluating configuration {config_name}: {e}")
            import traceback
            Logger.log(f"Traceback: {traceback.format_exc()}")
            return {
                "configuration": config_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_average_score(self, scores: Dict[str, Optional[float]]) -> Optional[float]:
        """
        Calculate average of valid scores
        
        Args:
            scores: Dictionary of metric scores
            
        Returns:
            Average score or None if no valid scores
        """
        valid_scores = [v for v in scores.values() if v is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
