"""
RAGAS Evaluator
Handles evaluation of RAG systems using RAGAS metrics
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

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
    
    def __init__(self, testset_path: str, timeout: int = 300, cache_dir: str = "./evaluation_cache"):
        """
        Initialize RAGAS evaluator
        
        Args:
            testset_path: Path to testset JSON file
            timeout: Timeout for evaluation in seconds
            cache_dir: Directory to cache generation payloads
        """
        self.config = Config()
        self.testset_path = testset_path
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.questions: List[str] = []
        self.ground_truths: List[str] = []
        
        # Create cache directory if it doesn't exist
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
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
            model="gpt-4.1-mini",
            api_key=self.config.OPENAI_API_KEY,
            temperature=0.1,
        )
        
        Logger.log("Using ChatGPT (gpt-4.1-mini) for RAGAS evaluation")
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

    def _load_questions_from_reference(self, reference_name: str) -> Optional[Tuple[List[str], List[str]]]:
        """Load questions and ground truths from a reference cache file"""
        import os
        reference_path = self._get_cache_path(f"{reference_name}_questions")
        
        if not os.path.exists(reference_path):
            Logger.log(f"⚠ Reference questions file not found: {reference_path}")
            return None
        
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            
            if not isinstance(reference_data, list):
                Logger.log(f"⚠ Invalid reference questions format in {reference_path}")
                return None
            
            questions = [item['question'] for item in reference_data]
            ground_truths = [item['ground_truth'] for item in reference_data]
            
            Logger.log(f"✓ Loaded {len(questions)} questions from reference: {reference_name}")
            return questions, ground_truths
            
        except Exception as e:
            Logger.log(f"⚠ Error loading reference questions: {e}")
            return None
    
    def _match_reference_with_cache(self, config_name: str, reference_questions: List[str], reference_ground_truths: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Match reference questions with cached responses from technique's cache"""
        import os
        cache_path = self._get_cache_path(config_name)
        
        if not os.path.exists(cache_path):
            Logger.log(f"⚠ Technique cache not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            if not isinstance(cached_data, list):
                Logger.log(f"⚠ Invalid cache format: {cache_path}")
                return None
            
            # Create a mapping of questions to cached responses
            cache_map = {}
            for item in cached_data:
                question = item.get('user_input', '')
                if question:
                    cache_map[question] = item
            
            # Match reference questions with cached data
            matched_data = []
            missing_count = 0
            
            for ref_q, ref_gt in zip(reference_questions, reference_ground_truths):
                if ref_q in cache_map:
                    # Use cached contexts and response, but ensure ground truth matches reference
                    matched_item = cache_map[ref_q].copy()
                    matched_item['reference'] = ref_gt  # Use reference ground truth
                    matched_data.append(matched_item)
                else:
                    missing_count += 1
                    Logger.log(f"⚠ Question not found in cache: {ref_q[:50]}...")
            
            if missing_count > 0:
                Logger.log(f"⚠ {missing_count}/{len(reference_questions)} questions not found in technique cache")
                return None
            
            Logger.log(f"✓ Successfully matched {len(matched_data)} questions with cached responses")
            return matched_data
            
        except Exception as e:
            Logger.log(f"⚠ Error matching reference with cache: {e}")
            return None
    
    def _select_questions(self, max_questions: Optional[int], random_seed: Optional[int]) -> Tuple[List[str], List[str]]:
        """Return a (optionally randomized) subset of questions and ground truths limited by max_questions"""
        if max_questions is None or max_questions >= len(self.questions):
            return self.questions, self.ground_truths

        limit = max_questions
        import random
        rng = random.Random(random_seed)
        indices = rng.sample(range(len(self.questions)), limit)
        indices.sort()  # preserve original order for stability/logging
        return [self.questions[i] for i in indices], [self.ground_truths[i] for i in indices]

    def _save_selected_questions(self, config_name: str, questions: List[str], ground_truths: List[str]):
        """Persist the selected questions/ground truths used for evaluation"""
        payload = [
            {"question": q, "ground_truth": gt}
            for q, gt in zip(questions, ground_truths)
        ]
        cache_path = self._get_cache_path(f"{config_name}_questions")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            Logger.log(f"✓ Saved selected questions to {cache_path}")
        except Exception as e:
            Logger.log(f"⚠ Failed to save selected questions: {e}")
    
    def _save_to_last_test_set(self, questions: List[str], ground_truths: List[str]):
        """Save the evaluation testset to last_test_set folder"""
        import os
        last_test_set_dir = "./last_test_set"
        os.makedirs(last_test_set_dir, exist_ok=True)
        
        payload = [
            {"question": q, "ground_truth": gt}
            for q, gt in zip(questions, ground_truths)
        ]
        
        # Save with timestamp for traceability
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(last_test_set_dir, f"last_evaluation_testset_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            Logger.log(f"✓ Saved evaluation testset to {filename}")
        except Exception as e:
            Logger.log(f"⚠ Failed to save evaluation testset: {e}")
    
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
            # Try querying with simple retry on rate-limit errors
            max_retries = 5
            for attempt in range(max_retries):
                result = pipeline.query(question)

                # Detect rate limit error either in explicit error or answer text
                err_text = result.get('error') or result.get('answer', '')
                is_rate_limit = isinstance(err_text, str) and (
                    'rate_limit_exceeded' in err_text.lower() or
                    'rate limit reached' in err_text.lower() or
                    'error code: 429' in err_text.lower()
                )

                if not is_rate_limit:
                    contexts = self._extract_contexts(result.get('sources', []))
                    return {
                        "user_input": question,
                        "retrieved_contexts": contexts if contexts else ["No relevant context retrieved"],
                        "response": result['answer'],
                        "reference": ground_truth
                    }

                # Backoff and retry on rate-limit
                wait_time = min(2 ** attempt, 30)
                Logger.log(f"Rate limit encountered, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)

            # If all retries exhausted, return the last error response
            contexts = self._extract_contexts(result.get('sources', [])) if 'result' in locals() else []
            return {
                "user_input": question,
                "retrieved_contexts": contexts if contexts else ["Error retrieving context"],
                "response": result.get('answer', 'Error: Rate limit and retries exhausted') if 'result' in locals() else 'Error: Rate limit and retries exhausted',
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
    
    def _get_cache_path(self, config_name: str) -> str:
        """Get cache file path for a configuration"""
        import os
        return os.path.join(self.cache_dir, f"{config_name}_payload.json")
    
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
    
    def _load_cached_payload(self, config_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached generation payload for a configuration, filtering out error responses"""
        import os
        cache_path = self._get_cache_path(config_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check for error responses in cached data
                error_count = 0
                valid_items = []
                error_questions = []
                
                for item in cached_data:
                    response = item.get('response', '')
                    if self._is_error_response(response):
                        error_count += 1
                        error_questions.append(item.get('user_input', 'Unknown question'))
                        Logger.log(f"⚠ Found error response in cache: {response[:100]}...")
                    else:
                        valid_items.append(item)
                
                if error_count > 0:
                    Logger.log(f"⚠ Found {error_count} error responses in cache. These will be regenerated.")
                    # Return None to trigger regeneration of error items
                    # Store the valid items and error questions for partial regeneration
                    return None
                
                Logger.log(f"✓ Loaded cached payload from {cache_path}")
                return cached_data
            except Exception as e:
                Logger.log(f"Error loading cache: {e}")
        return None
    
    def _save_payload_cache(self, config_name: str, evaluation_data: List[Dict[str, Any]]):
        """Save generation payload to cache"""
        cache_path = self._get_cache_path(config_name)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            Logger.log(f"✓ Saved payload cache to {cache_path}")
        except Exception as e:
            Logger.log(f"Error saving cache: {e}")
    
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
                
                # Handle dict values (RAGAS sometimes returns nested structures)
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
    
    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        config_name: str,
        use_cache: bool = True,
        skip_generation: bool = False,
        max_questions: Optional[int] = 100,
        random_seed: Optional[int] = None,
        use_reference_questions: str = "recursive_faiss_dense"
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline using RAGAS metrics
        
        Args:
            pipeline: RAG pipeline to evaluate
            config_name: Name of the configuration being tested
            use_cache: Whether to use cached payloads if available
            skip_generation: If True and cache exists, skip generation and only evaluate
            max_questions: Maximum number of questions to evaluate (None for all)
            random_seed: Seed for randomized question selection (None for non-deterministic)
            use_reference_questions: Config name to load questions from (e.g., 'recursive_faiss_dense')
                                    When set, uses those questions but generates fresh contexts/answers
            
        Returns:
            Dictionary with evaluation results
        """
        Logger.log(f"\n{'='*60}")
        Logger.log(f"Evaluating Configuration: {config_name}")
        Logger.log(f"{'='*60}")
        
        try:
            # Select subset of questions to evaluate
            if use_reference_questions:
                # Load questions from reference cache
                ref_result = self._load_questions_from_reference(use_reference_questions)
                if ref_result:
                    selected_questions, selected_ground_truths = ref_result
                    Logger.log(f"✓ Using {len(selected_questions)} questions from reference: {use_reference_questions}")
                else:
                    Logger.log(f"⚠ Failed to load reference questions, falling back to normal selection")
                    selected_questions, selected_ground_truths = self._select_questions(max_questions, random_seed)
            else:
                selected_questions, selected_ground_truths = self._select_questions(max_questions, random_seed)
            
            total_selected = len(selected_questions)
            Logger.log(f"Using {total_selected} questions for evaluation")
            # Persist the selected subset for traceability
            self._save_selected_questions(config_name, selected_questions, selected_ground_truths)
            # Save to last_test_set folder
            self._save_to_last_test_set(selected_questions, selected_ground_truths)

            # Try to load cached payload
            evaluation_data = None
            
            # If using reference questions, match them with technique's cached responses
            if use_reference_questions and (use_cache or skip_generation):
                Logger.log(f"Matching reference questions with {config_name} cached responses...")
                evaluation_data = self._match_reference_with_cache(config_name, selected_questions, selected_ground_truths)
                if evaluation_data:
                    Logger.log("✓ Using reference questions with technique's cached contexts/responses")
                else:
                    Logger.log(f"⚠ Failed to match reference with cache, will generate fresh responses")
            
            # Normal cache loading (when not using reference questions)
            if evaluation_data is None and use_cache and not use_reference_questions:
                import os
                cache_path = self._get_cache_path(config_name)
                selection_cache_path = self._get_cache_path(f"{config_name}_questions")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)

                    # Ensure question selection matches cache
                    selection_matches = False
                    if os.path.exists(selection_cache_path):
                        try:
                            with open(selection_cache_path, 'r', encoding='utf-8') as sf:
                                cached_selection = json.load(sf)
                            if isinstance(cached_selection, list):
                                cached_pairs = [
                                    (item.get('question'), item.get('ground_truth'))
                                    for item in cached_selection
                                ]
                                current_pairs = list(zip(selected_questions, selected_ground_truths))
                                if len(cached_pairs) >= total_selected and cached_pairs[:total_selected] == current_pairs:
                                    selection_matches = True
                                else:
                                    Logger.log("⚠ Cached question subset differs from current selection; regenerating.")
                            else:
                                Logger.log("⚠ Cached question subset invalid; regenerating.")
                        except Exception as sel_err:
                            Logger.log(f"⚠ Failed to read cached question subset: {sel_err}; regenerating.")
                    else:
                        Logger.log("⚠ Cached question subset missing; regenerating.")
                    
                    # Ensure cache size aligns with requested subset
                    if not isinstance(cached_data, list):
                        Logger.log("⚠ Cache is invalid; regenerating.")
                        cached_data = None
                    elif len(cached_data) < total_selected:
                        Logger.log(
                            f"⚠ Cache has only {len(cached_data)} items (< {total_selected}); regenerating."
                        )
                        cached_data = None
                    elif len(cached_data) > total_selected:
                        Logger.log(
                            f"⚠ Cache has {len(cached_data)} items (> {total_selected}); using first {total_selected}."
                        )
                        cached_data = cached_data[:total_selected]

                    # If selection does not match, force regeneration
                    if cached_data is not None and not selection_matches:
                        cached_data = None
                    
                    if cached_data is not None:
                        # Check for error responses
                        items_to_regenerate = []
                    
                        for idx, item in enumerate(cached_data):
                            response = item.get('response', '')
                            if self._is_error_response(response):
                                items_to_regenerate.append(idx)
                                Logger.log(f"⚠ Question {idx+1} has error response, will regenerate")
                            
                        if items_to_regenerate:
                            Logger.log(f"⚠ Will regenerate {len(items_to_regenerate)} questions with errors")
                            
                            # Regenerate only the questions with errors
                            if not skip_generation:
                                Logger.log("Regenerating error responses...")
                                regenerated_items = []
                                for idx in items_to_regenerate:
                                    question = selected_questions[idx]
                                    ground_truth = selected_ground_truths[idx]
                                    regenerated = self._process_single_question(
                                        pipeline, question, ground_truth, idx+1, total_selected
                                    )
                                    regenerated_items.append((idx, regenerated))
                                
                                # Merge valid cached items with regenerated items
                                evaluation_data = cached_data.copy()
                                for idx, regenerated in regenerated_items:
                                    evaluation_data[idx] = regenerated
                                
                                Logger.log("✓ Successfully regenerated error responses")
                                # Save updated cache
                                self._save_payload_cache(config_name, evaluation_data)
                            else:
                                evaluation_data = cached_data
                        else:
                            evaluation_data = cached_data
                            Logger.log("✓ Using cached payload for generation phase")
            
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
            
            # Run RAGAS evaluation
            Logger.log(f"Running RAGAS evaluation for {config_name}...")
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
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
