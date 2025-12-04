import os
import sys
import json
import asyncio
sys.path.append('..')

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from chunker import AgenticChunker, RecursiveChunker, SemanticChunker
from config import Config
from logger import Logger
from llm import Gemini, Groq
from database import Qdrant, FAISS
from generator import RecursiveGenerator, FAISSGenerator, SemanticGenerator, AgenticGenerator
from ragas import RunConfig

from langchain_google_genai import ChatGoogleGenerativeAI
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

# def load_testset(file_path="uud_rag_comprehensive_testset.json"):
def load_testset(file_path="uud_rag_sample_testset.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            testset_data = json.load(f)
        
        questions = []
        expected_responses = []
        
        for item in testset_data["questions"]:
            questions.append(item["question"])
            expected_responses.append(item["ground_truth"])
        
        return questions, expected_responses
    
    except Exception as e:
        print(f"Error loading testset: {e}")
        return None, None


async def setup_rag_system():
    config = Config()
    chunker_choice = get_user_choice(
        "Which chunker did you want to test?",
        ["Recursive Chunker", "Agentic Chunker", "Semantic Chunker"]
    )
    chunker_type = ["recursive", "agentic", "semantic"][chunker_choice]
    db_choice = get_user_choice(
        "Which vector database did you use with the chunker?",
        ["FAISS", "Qdrant"]
    )
    use_faiss = db_choice == 0
    gemini = Gemini("gemini-2.0-flash", config.GOOGLE_API_KEY)
    groq = Groq("meta-llama/llama-guard-4-12b", config.GROQ_API_KEY)
    recursive_chunker = RecursiveChunker()
    agentic_chunker = AgenticChunker(groq)
    semantic_chunker = SemanticChunker(embedding_model_name="LazarusNLP/all-indo-e5-small-v4")
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
        if chunker_type == "semantic":
            rag_generator = SemanticGenerator(db, gemini)
        elif chunker_type == "agentic":
            rag_generator = AgenticGenerator(db, gemini)
        else:
            rag_generator = RecursiveGenerator(db, gemini)
    return rag_generator, config


def create_ragas_compatible_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=2000
    )


async def run_ragas_evaluation():
    questions, expected_responses = load_testset()
    if questions is None:
        return
    rag_generator, config = await setup_rag_system()
    num_questions = len(questions)
    
    evaluation_data = []
    for i, (question, ground_truth) in enumerate(zip(questions[:num_questions], expected_responses[:num_questions])):
        print(f"\n📋 Processing Question {i+1}/{num_questions}")
        print(f"Q: {question[:80]}...")
        
        try:
            result = rag_generator.generate_answer(question)
            answer = result['answer']
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
            
            print(f"Generated answer: {answer[:60]}...")
            print(f"Contexts: {len(contexts)} chunks")
            
        except Exception as e:
            Logger.log(f"Error processing question {i+1}: {e}")
            print(f"Error: {e}")
            
            evaluation_data.append({
                "user_input": question,
                "retrieved_contexts": ["Error retrieving context"],
                "response": f"Error generating response: {e}",
                "reference": ground_truth
            })
    if not evaluation_data:
        print("No evaluation data generated!")
        return
    try:
        evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
        print("Created RAGAS EvaluationDataset")
    except Exception as e:
        print(f"Error creating RAGAS dataset: {e}")
        return
    
    llm_for_eval = create_ragas_compatible_llm(config.GOOGLE_API_KEY)
    evaluator_llm = LangchainLLMWrapper(llm_for_eval)
    google_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=config.GOOGLE_API_KEY
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(google_embeddings)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    answer_correctness_metric = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    run_config = RunConfig(max_workers=1, timeout=180)
    
    try:
        result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        context_recall_metric,
        faithfulness_metric,
        answer_correctness_metric
    ],
    llm=evaluator_llm,
    run_config=run_config
)
        
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        print(f"Result: {result}")
        try:
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = dict(result)
                
            for metric_name, score in result_dict.items():
                if score is not None and not (isinstance(score, float) and str(score) == 'nan'):
                    print(f"{metric_name}: {score:.4f}")
                else:
                    print(f"{metric_name}: Failed (parsing issues)")
                    
        except Exception as extract_error:
            print(f"Could not extract individual scores: {extract_error}")
            print(f"Raw result: {result}")
        
        results_file = "ragas_evaluation_results.json"
        try:
            if hasattr(result, 'to_dict'):
                save_dict = result.to_dict()
            else:
                save_dict = dict(result)
            clean_dict = {}
            for key, value in save_dict.items():
                if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                    clean_dict[key] = float(value)
                else:
                    clean_dict[key] = None
                    
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(clean_dict, f, indent=2, ensure_ascii=False)
                
            print(f"\nResults saved to: {results_file}")
        except Exception as save_error:
            print(f"Could not save results: {save_error}")
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(str(result), f, indent=2, ensure_ascii=False)
                print(f"Results saved as string to: {results_file}")
            except:
                print("Could not save results at all")
        return result
        
    except Exception as e:
        Logger.log(f"RAGAS evaluation failed: {e}")
        print(f"RAGAS evaluation failed: {e}")
        print("\nDebugging Info:")
        print(f"   - Dataset size: {len(evaluation_data)}")
        print(f"   - Sample data keys: {evaluation_data[0].keys()}")
        return None


if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation())