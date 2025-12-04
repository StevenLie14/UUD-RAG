import os
import sys
sys.path.append('..')

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from llm import Gemini, Groq
from config import Config
from logger import Logger
import pandas as pd
from typing import List, Optional, Dict, Any
import json
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from loader.base import BaseLoader
class RAGTestsetGenerator(BaseLoader):
    def __init__(self, llm_type: str = "gemini", model_name: Optional[str] = None):
        self.config = Config()
        if llm_type.lower() == "gemini":
            self.llm = Gemini(
                model_name or "gemini-1.5-flash-latest", 
                self.config.GOOGLE_API_KEY
            )
        elif llm_type.lower() == "groq":
            self.llm = Groq(
                model_name or "llama-3.1-8b-instant",
                self.config.GROQ_API_KEY
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        self.ragas_llm = self.llm.get_ragas_llm()
        try:

            hf_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        except ImportError as e:
            Logger.log(f"Embedding initialization failed: {e}")
            raise
        Logger.log(f"Initialized testset generator with {type(self.llm).__name__}")

    def load_documents(self, 
                      doc_path: str = "../peraturan_pdfs/", 
                      max_docs: Optional[int] = None,
                      specific_files: Optional[List[str]] = None) -> List[Document]:
        if not specific_files:
            Logger.log("No File Found ")
            return
        
        docs = []
        for filename in specific_files:
            file_path = os.path.join(doc_path, filename)
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size < 100:  
                        Logger.log(f"Skipping tiny file: {filename} ({file_size} bytes)")
                        continue
                        
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        if not header.startswith(b'%PDF'):
                            Logger.log(f"Skipping non-PDF file: {filename}")
                            continue
                    
                    loader = PyPDFLoader(file_path)
                    file_docs = loader.load()
                    
                    for doc in file_docs:
                        if hasattr(doc, 'page_content'):
                            doc.page_content = self._clean_text(doc.page_content)
                    
                    docs.extend(file_docs)
                    Logger.log(f"Successfully loaded {len(file_docs)} pages from {filename}")
                    
                except Exception as e:
                    Logger.log(f"Error loading {filename}: {e}")
                    continue
            else:
                Logger.log(f"File not found: {file_path}")
        
        if max_docs and len(docs) > max_docs:
            docs = docs[:max_docs]
            Logger.log(f"Limited to {max_docs} documents")
            
        return docs
    
    def generate_direct_with_groq(self, 
                                 doc_content: str,
                                 num_questions: int = 5) -> Dict[str, Any]:
        cleaned_content = self._clean_text(doc_content)
        if len(cleaned_content) < 100:
            return []
        current_context = cleaned_content
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", """Anda adalah ahli hukum. Buatlah 2 pertanyaan dan jawaban (Q&A) berdasarkan teks yang diberikan.
            
Format:
Q: [Pertanyaan]
A: [Jawaban]
"""),
            ("human", "Teks: {content}")
        ])
        questions = []
        try:
            response = self.llm.answer(question_prompt, {"content": current_context})
            
            lines = response.split('\n')
            current_q = ""
            current_a = ""

            for line in lines:
                line = line.strip()
                if line.startswith('Q') and ':' in line:
                    if current_q and current_a:
                        questions.append({
                            "question": current_q,
                            "ground_truth": current_a,
                            # "contexts": [current_context] 
                        })
                    current_q = line.split(':', 1)[1].strip()
                    current_a = ""
                elif line.startswith('A') and ':' in line:
                    current_a = line.split(':', 1)[1].strip()
            
            if current_q and current_a:
                questions.append({
                    "question": current_q,
                    "ground_truth": current_a,
                    # "contexts": [current_context]
                })
                
            return questions

        except Exception as e:
            Logger.log(f"Error generating from chunk: {e}")
            return []
    
    def extract_meaningful_context(self, content: str) -> str:
        lines = content.split('\n')
        meaningful_lines = []
        
        found_content = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(skip in line.upper() for skip in ['UNDANG-UNDANG REPUBLIK', 'TENTANG', 'NOMOR', 'TAHUN']):
                if not found_content:
                    meaningful_lines.append(line)
                continue
            
            if line.startswith('Pasal') or 'Pasal' in line or len(line) > 20:
                found_content = True
                meaningful_lines.append(line)
            elif found_content:
                meaningful_lines.append(line)
            
            if len(' '.join(meaningful_lines)) > 800:
                break
        
        return ' '.join(meaningful_lines)[:1000]

    def generate_testset(self, docs: List[Document], testset_size: int = 20, **kwargs):
        all_questions = []
        import random
        selected_docs = docs
        if len(docs) > testset_size:
             random.shuffle(selected_docs)

        for i, doc in enumerate(selected_docs):
            if len(all_questions) >= testset_size:
                break
                
            Logger.log(f"Processing page {i+1}...")
            
            new_questions = self.generate_direct_with_groq(doc.page_content)
            
            all_questions.extend(new_questions)
            Logger.log(f"Generated {len(new_questions)} questions from page {i+1}")

        return {"questions": all_questions[:testset_size]}
    def save_testset(self, 
                    testset, 
                    output_path: str = "uud_rag_testset.json") -> str:
        try:
            if isinstance(testset, dict) and 'questions' in testset:
                Logger.log("Saving direct generation format...")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(testset, f, ensure_ascii=False, indent=2)
                
                csv_path = output_path.replace('.json', '.csv')
                
                rows = []
                for q in testset['questions']:
                    answer = q.get('answer', q.get('ground_truth', ''))
                    context = q.get('context', q.get('contexts', []))
                    
                    rows.append({
                        'question': q['question'],
                        'ground_truth': answer,
                        # 'contexts': [context] if isinstance(context, str) else context
                    })
                
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                
                
            else:
                Logger.log("Saving RAGAS format...")
                
                df = testset.to_pandas()
                testset_dict = {
                    'questions': df['question'].tolist(),
                    'ground_truths': df['ground_truth'].tolist(), 
                    # 'contexts': df['contexts'].tolist(),
                    'evolution_types': df['evolution_type'].tolist() if 'evolution_type' in df.columns else [],
                    'metadata': df['metadata'].tolist() if 'metadata' in df.columns else []
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(testset_dict, f, ensure_ascii=False, indent=2)
                
                csv_path = output_path.replace('.json', '.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')
                
            
            
            return output_path
            
        except Exception as e:
            Logger.log(f"Error saving testset: {e}")
            raise

    def load_testset(self, testset_path: str) -> Dict[str, List]:
        try:
            with open(testset_path, 'r', encoding='utf-8') as f:
                testset = json.load(f)
            
            Logger.log(f"Loaded testset with {len(testset['questions'])} questions from {testset_path}")
            return testset
            
        except Exception as e:
            Logger.log(f"Error loading testset: {e}")
            raise

def generate_sample_testset():
    generator = RAGTestsetGenerator(llm_type="groq")
    key_documents = [
    #     "uu1-1945.pdf",
    #     "uu-no-1-tahun-2024.pdf",
          "51uu012.pdf"
    ]
    docs = generator.load_documents(
        doc_path="../test/",
        specific_files=key_documents
    )
    if not docs:
        Logger.log("No documents loaded. Please check file paths.")
        return
    
    testset = generator.generate_testset(
        docs=docs,
        testset_size=15, # berapa questions
        use_direct=True
    )
    
    output_path = generator.save_testset(testset, "uud_rag_sample_testset.json")
    
    
    if isinstance(testset, dict) and 'questions' in testset:
        print(f"Generated {len(testset['questions'])} questions using direct Groq method")
        print(f"\nPreview of generated questions:")
        for i, q in enumerate(testset['questions'][:3], 1):
            print(f"\nQuestion {i}: {q['question']}")
            answer = q.get('answer', q.get('ground_truth', ''))
            print(f"Answer: {answer[:100]}...")
            print("-" * 50)
    else:
        df = testset.to_pandas()
        print(f"Generated {len(df)} questions using RAGAS method")
        print(f"\nPreview:")
        for i, row in df.head(3).iterrows():
            print(f"\nQuestion {i+1}: {row['question']}")
            print(f"Answer: {row['ground_truth'][:100]}...")
            print("-" * 50)

def generate_comprehensive_testset():
    generator = RAGTestsetGenerator(llm_type="groq")
    docs = generator.load_documents(
        doc_path="../peraturan_pdfs/",
        max_docs=50 # ganti sini untuk lebih banyak
    )
    if not docs:
        Logger.log("No documents loaded. Please check file paths.")
        return
    testset = generator.generate_testset(
        docs=docs,
        testset_size=50,  # ganti sini untuk lebih banyak
        use_direct=True
    )
    output_path = generator.save_testset(testset, "uud_rag_comprehensive_testset.json")
    
    print(f"\n✅ Comprehensive testset generated: {output_path}")
    
    if isinstance(testset, dict) and 'questions' in testset:
        print(f"Generated {len(testset['questions'])} questions from {len(docs)} documents")
    else:
        print(f"Generated testset from {len(docs)} documents")
    df = testset.to_pandas()
    print(f"\nPreview of generated questions:")
    for i, row in df.head(3).iterrows():
        print(f"\nQuestion {i+1}: {row['question']}")
        print(f"Ground Truth: {row['ground_truth'][:100]}...")
        if 'evolution_type' in row:
            print(f"Evolution Type: {row['evolution_type']}")
        print("-" * 50)
    

if __name__ == "__main__":
    
    print("UUD-RAG Testset Generator")
    print("=" * 40)
    print("1. Generate sample testset (15 questions)")
    print("2. Generate comprehensive testset (50 questions)")
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    try:
        if choice == "1":
            generate_sample_testset()
        elif choice == "2":
            generate_comprehensive_testset()
        else:
            print("kocak")
            
    except Exception as e:
        Logger.log(f"Error during testset generation: {e}")
        print(f"\nError: {e}")