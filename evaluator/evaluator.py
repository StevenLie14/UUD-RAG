"""
RAGAS Evaluator
Handles evaluation of RAG systems using RAGAS metrics
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

from config import Config
from logger import Logger
from rag.pipeline import RAGPipeline
from llm import ChatGPT
from model import EvaluationItem
import os 

class RAGASEvaluator:
    
    def __init__(self, testset_path: str, timeout: int = 300, cache_dir: str = "./evaluation_cache"):
        self.config = Config()
        self.testset_path = testset_path
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.questions: List[str] = []
        self.ground_truths: List[str] = []
        
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        self.evaluator_llm = self._create_evaluation_llm()
        self.evaluator_embeddings = self._create_evaluation_embeddings()
        
        self._initialize_metrics()
        
        self._load_testset()
        
    
    def _create_evaluation_llm(self):
        chatgpt = ChatGPT(
            model_name="gpt-4o-mini",
            api_key=self.config.OPENAI_API_KEY
        )
        
        Logger.log("Using ChatGPT (gpt-4o-mini) for RAGAS evaluation")
        return chatgpt.get_ragas_llm()
    
    def _create_evaluation_embeddings(self):
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
        try:
            with open(self.testset_path, 'r', encoding='utf-8') as f:
                testset_data = json.load(f)
            
            for item in testset_data:
                self.questions.append(item["question"])
                self.ground_truths.append(item["ground_truth"])
            
            Logger.log(f"Loaded {len(self.questions)} test questions from {self.testset_path}")
            
        except Exception as e:
            Logger.log(f"Error loading testset: {e}")
            raise
    
    def _save_evaluation_results(self, config_name: str, result_data: Dict[str, Any]):
        technique_dir = os.path.join("./results", config_name)
        os.makedirs(technique_dir, exist_ok=True)
        
        results_file = os.path.join(technique_dir, "evaluation_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            Logger.log(f"Saved evaluation results to {results_file}")
        except Exception as e:
            Logger.log(f"Failed to save evaluation results: {e}")
    
    def _process_single_question(
        self, 
        pipeline: RAGPipeline, 
        question: str, 
        ground_truth: str,
        question_num: int,
        total_questions: int
    ) -> EvaluationItem:
        Logger.log(f"Processing question {question_num}/{total_questions}: {question[:50]}...")
        
        try:
            max_retries = 5
            for attempt in range(max_retries):
                result = pipeline.query(question)

                err_text = result.get('error') or result.get('answer', '')
                is_rate_limit = isinstance(err_text, str) and (
                    'rate_limit_exceeded' in err_text.lower() or
                    'rate limit reached' in err_text.lower() or
                    'error code: 429' in err_text.lower()
                )

                if not is_rate_limit:
                    contexts = self._extract_contexts(result.get('sources', []))
                    return EvaluationItem(
                        user_input=question,
                        retrieved_contexts=contexts if contexts else ["No relevant context retrieved"],
                        response=result['answer'],
                        reference=ground_truth
                    )

                wait_time = min(2 ** attempt, 30)
                Logger.log(f"Rate limit encountered, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)

            contexts = self._extract_contexts(result.get('sources', [])) if 'result' in locals() else []
            return EvaluationItem(
                user_input=question,
                retrieved_contexts=contexts if contexts else ["Error retrieving context"],
                response=result.get('answer', 'Error: Rate limit and retries exhausted') if 'result' in locals() else 'Error: Rate limit and retries exhausted',
                reference=ground_truth
            )
        except Exception as e:
            Logger.log(f"Error processing question {question_num}: {e}")
            return EvaluationItem(
                user_input=question,
                retrieved_contexts=["Error retrieving context"],
                response=f"Error: {str(e)}",
                reference=ground_truth
            )
    
    def _extract_contexts(self, sources: List[Any]) -> List[str]:
        """Extract context texts from sources"""
        contexts = []
        for source in sources:
            if isinstance(source, dict) and source.get('content'):
                context_text = source['content'].strip()
                if len(context_text) > 50:
                    contexts.append(context_text)
        return contexts
    
    def _get_cache_path(self, config_name: str) -> str:
        import os
        return os.path.join("./results", config_name, "evaluation_payload.json")
    
    def _save_payload_cache(self, config_name: str, evaluation_data: List[EvaluationItem]):
        cache_path = self._get_cache_path(config_name)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            payload = [
                {
                    "question": item.user_input,
                    "retrieved_contexts": item.retrieved_contexts,
                    "response": item.response,
                    "ground_truth": item.reference
                }
                for item in evaluation_data
            ]
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            Logger.log(f"Saved payload to {cache_path}")
        except Exception as e:
            Logger.log(f"Error saving payload: {e}")

    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        config_name: str
    ) -> Dict[str, Any]:
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Evaluating Configuration: {config_name}")
        Logger.log(f"{'='*60}")
        
        try:
            selected_questions = self.questions
            selected_ground_truths = self.ground_truths
            total_selected = len(selected_questions)
            Logger.log(f"Using {total_selected} questions for evaluation")

            Logger.log("Generating responses...")
            evaluation_data = [
                self._process_single_question(pipeline, q, gt, i+1, total_selected)
                for i, (q, gt) in enumerate(zip(selected_questions, selected_ground_truths))
            ]
            
            self._save_payload_cache(config_name, evaluation_data)
            
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            evaluation_dict_list = [item.model_dump() for item in evaluation_data]
            evaluation_dataset = EvaluationDataset.from_list(evaluation_dict_list)
            run_config = RunConfig(max_workers=1, timeout=self.timeout)
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
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
                    break
                except Exception as e_eval:
                    msg = str(e_eval)
                    is_rate_limit = ('rate_limit_exceeded' in msg.lower() or
                                     'rate limit reached' in msg.lower() or
                                     'error code: 429' in msg.lower())
                    if not is_rate_limit or attempt >= max_retries - 1:
                        Logger.log(f"RAGAS evaluation error: {e_eval}")
                        raise e_eval
                    wait_time = min(2 ** attempt, 30)
                    Logger.log(f"RAGAS rate limit, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
            
            try:
                final_scores = ragas_result.get('scores', [])
                
                result_data = {
                    "configuration": config_name,
                    "scores": {
                        "context_recall": final_scores.get('context_recall', None),
                        "faithfulness": final_scores.get('faithfulness', None),
                        "answer_correctness": final_scores.get('answer_correctness', None)
                    },
                    "timestamp": datetime.now().isoformat(),
                    "num_questions": total_selected
                }
                
                Logger.log(f"Configuration {config_name} completed!")
                Logger.log(f"Scores: {final_scores}")
                
                self._save_evaluation_results(config_name, result_data)
                
                return result_data
                
            except Exception as score_error:
                Logger.log(f"Error extracting scores: {score_error}")
                Logger.log(f"Result object type: {type(ragas_result)}")
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
        valid_scores = [v for v in scores.values() if v is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None