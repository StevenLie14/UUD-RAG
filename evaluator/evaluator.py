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
            Logger.log(f"✓ Saved evaluation results to {results_file}")
        except Exception as e:
            Logger.log(f"⚠ Failed to save evaluation results: {e}")
    
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
    
    def _is_error_response(self, response: str) -> bool:
        """Check if a response contains an error message"""
        if not response or not isinstance(response, str):
            return False
        
        error_indicators = [
            'rate_limit_exceeded',
            'rate limit reached',
            'error code: 429',
            'error code: 500',
            'error code: 503',
            'maaf, terjadi error',
            'error:',
            'exception:',
            'failed to',
            'cannot process',
            'please try again'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in error_indicators)
    
    def _save_payload_cache(self, config_name: str, evaluation_data: List[EvaluationItem]):
        """Save generation payload to results folder"""
        cache_path = self._get_cache_path(config_name)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Transform to evaluation payload format
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
            Logger.log(f"✓ Saved payload to {cache_path}")
        except Exception as e:
            Logger.log(f"Error saving payload: {e}")
    
    def _clean_scores(self, scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Clean scores by removing NaN values and handling list/dict values"""
        # Known metric names we want to keep
        metric_names = {'context_recall', 'faithfulness', 'answer_correctness', 'context_precision', 
                       'answer_relevancy', 'answer_similarity', 'context_entity_recall'}
        
        clean_scores = {}
        for key, value in scores.items():
            try:
                # Log the value type for debugging
                Logger.log(f"Processing score '{key}': type={type(value)}, value={value}")
                
                # Handle dict values (RAGAS sometimes returns ested structures)
                if isinstance(value, dict):
                    # Special handling for _repr_dict and _scores_dict - these contain the actual metrics
                    if key in ['_repr_dict', '_scores_dict']:
                        # Extract the individual metrics from the dict
                        for metric_name, metric_value in value.items():
                            # Handle list values within the dict
                            if isinstance(metric_value, list) and len(metric_value) > 0:
                                metric_value = metric_value[0] if len(metric_value) == 1 else sum(metric_value) / len(metric_value)
                            if metric_value is not None:
                                clean_scores[metric_name] = float(metric_value)
                        continue
                    
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
                        # If no known key, skip this entry
                        Logger.log(f"Dict score '{key}' has unknown structure: {value}")
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
                        continue
                
                # Only keep known metric names or valid numeric values
                if key in metric_names:
                    if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                        clean_scores[key] = float(value)
                elif value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                    # Skip metadata fields (scores, dataset, binary_columns, etc.)
                    if key not in ['scores', 'dataset', 'binary_columns', 'cost_cb', 'traces', 
                                   'ragas_traces', 'run_id', '_repr_dict', '_scores_dict']:
                        clean_scores[key] = float(value)
                    
            except (ValueError, TypeError) as e:
                Logger.log(f"Error processing score '{key}': {e}")
                clean_scores[key] = None
                
        return clean_scores
    
    def _regenerate_empty_responses(self, pipeline: RAGPipeline, evaluation_data: List[EvaluationItem]) -> List[EvaluationItem]:
        """Regenerate empty responses using already retrieved contexts if available.
        If the pipeline supports answering from provided contexts (e.g., answer_with_contexts(question, contexts)), use it.
        Otherwise, fallback to a normal query.
        """
        updated = False
        for idx, item in enumerate(evaluation_data):
            try:
                if isinstance(item.response, str) and item.response.strip() == "":
                    Logger.log(f"⚠ Empty response detected for question {idx+1}, regenerating using existing contexts...")
                    # Prefer using a pipeline method that accepts contexts if available
                    if hasattr(pipeline, "answer_with_contexts") and callable(getattr(pipeline, "answer_with_contexts")):
                        new_answer = pipeline.answer_with_contexts(item.user_input, item.retrieved_contexts)
                    else:
                        # Fallback to normal query
                        result = pipeline.query(item.user_input)
                        new_answer = result.get("answer", "")
                        # If contexts are present in cache, keep them; otherwise extract
                        if not item.retrieved_contexts:
                            contexts = self._extract_contexts(result.get("sources", []))
                            item.retrieved_contexts = contexts
                    # Update the item
                    item.response = new_answer if isinstance(new_answer, str) else str(new_answer)
                    if not item.retrieved_contexts:
                        item.retrieved_contexts = ["No relevant context retrieved"]
                    updated = True
            except Exception as regen_err:
                Logger.log(f"⚠ Failed to regenerate empty response for item {idx+1}: {regen_err}")
        if updated:
            Logger.log("✓ Regenerated empty responses using existing contexts")
        return evaluation_data

    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        config_name: str,
        use_cache: bool = True,
        skip_generation: bool = False
    ) -> Dict[str, Any]:
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Evaluating Configuration: {config_name}")
        Logger.log(f"{'='*60}")
        
        try:
            # Use all questions from testset
            selected_questions = self.questions
            selected_ground_truths = self.ground_truths
            total_selected = len(selected_questions)
            Logger.log(f"Using {total_selected} questions for evaluation")

            # Try to load cached payload
            evaluation_data = None
            
            # Load cache if available
            if evaluation_data is None and use_cache:
                import os
                cache_path = self._get_cache_path(config_name)
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)

                    # Validate cache
                    if not isinstance(cached_data, list):
                        Logger.log("⚠ Cache is invalid; regenerating.")
                        cached_data = None
                    elif len(cached_data) != total_selected:
                        Logger.log(f"⚠ Cache has {len(cached_data)} items but expected {total_selected}; regenerating.")
                        cached_data = None
                    
                    if cached_data is not None:
                        # Transform cached data to EvaluationItem objects
                        evaluation_data = []
                        items_to_regenerate = []
                        
                        for idx, cached_item in enumerate(cached_data):
                            response = cached_item.get('response', '')
                            
                            # Check if response needs regeneration
                            if self._is_error_response(response) or not response:
                                items_to_regenerate.append(idx)
                            
                            # Create EvaluationItem from cached data
                            eval_item = EvaluationItem(
                                user_input=cached_item.get('question', ''),
                                retrieved_contexts=cached_item.get('retrieved_contexts', []),
                                response=response,
                                reference=cached_item.get('ground_truth', '')
                            )
                            evaluation_data.append(eval_item)
                        
                        if items_to_regenerate:
                            Logger.log(f"⚠ Will regenerate {len(items_to_regenerate)} questions")
                            
                            if not skip_generation:
                                Logger.log("Regenerating responses from cached contexts...")
                                for idx in items_to_regenerate:
                                    question = selected_questions[idx]
                                    ground_truth = selected_ground_truths[idx]
                                    
                                    # Use cached contexts, generate new response
                                    result = pipeline.query(question)
                                    
                                    # Update the evaluation item
                                    evaluation_data[idx].response = result.get('answer', '')
                                
                                Logger.log("✓ Successfully regenerated responses from cached contexts")
                            else:
                                Logger.log("⚠ Skip generation enabled but using cached data with errors")
                        else:
                            Logger.log("✓ Using cached payload")
            
            # Generate new data if not using cache or cache not found
            if evaluation_data is None:
                if skip_generation:
                    Logger.log(f"⚠ Cache not found for {config_name} but skip_generation=True")
                    return {
                        "configuration": config_name,
                        "error": "Cache not found and skip_generation enabled",
                        "timestamp": datetime.now().isoformat()
                    }
                
                Logger.log("Generating responses...")
                evaluation_data = [
                    self._process_single_question(pipeline, q, gt, i+1, total_selected)
                    for i, (q, gt) in enumerate(zip(selected_questions, selected_ground_truths))
                ]
                
                # Save to cache
                if use_cache:
                    self._save_payload_cache(config_name, evaluation_data)
            
            # After evaluation_data is prepared, regenerate any empty responses using existing contexts
            evaluation_data = self._regenerate_empty_responses(pipeline, evaluation_data)
            # Persist possible updates back to cache
            if use_cache:
                self._save_payload_cache(config_name, evaluation_data)
            
            # Run RAGAS evaluation
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            # Convert EvaluationItem objects to dicts for RAGAS
            evaluation_dict_list = [item.model_dump() for item in evaluation_data]
            evaluation_dataset = EvaluationDataset.from_list(evaluation_dict_list)
            run_config = RunConfig(max_workers=1, timeout=self.timeout)
            
            # Retry loop for rate-limit errors during RAGAS evaluation
            max_retries = 5
            last_error: Optional[Exception] = None
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
                    last_error = None
                    break
                except Exception as e_eval:
                    last_error = e_eval
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
                    "num_questions": total_selected
                }
                
                Logger.log(f"Configuration {config_name} completed!")
                Logger.log(f"Clean scores: {clean_scores}")
                
                # Save evaluation results to last_test_set
                self._save_evaluation_results(config_name, result_data)
                
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