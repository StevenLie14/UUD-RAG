from database import Qdrant
from logger import Logger
from langchain_core.prompts import ChatPromptTemplate
from llm import BaseLLM
from .base import BaseGenerator

class AgenticGenerator(BaseGenerator):
    def __init__(self, database : Qdrant,llm: BaseLLM):
        super().__init__(database, llm)
        
    def generate_answer(self, query: str, limit: int = 5):
        relevant_chunks =  self.database.search(query, limit)
        if not relevant_chunks:
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen hukum.",
                "sources": []
            }
        
        Logger.log(f"Found {len(relevant_chunks.points)} relevant chunks for query: '{query}'")
        Logger.log(f"Relevant chunks: {relevant_chunks.points}")
        
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(relevant_chunks.points):
            payload = chunk.payload
            context_parts.append(
                f"[Sumber {idx}] {payload['title']}\n"
                f"Ringkasan: {payload['summary']}\n"
                f"Konten:\n{payload['full_text']}\n"
            )
            sources.append({
                "index": idx,
                "title": payload['title'],
                "summary": payload['summary'],
                "score": chunk.score,
                "content": payload['full_text']
            }) 
        
        context = "\n---\n".join(context_parts)
        prompt = self.generate_prompt(context, query)
        
        try:
            answer = self.llm.answer(prompt, {"context": context, "question": query})
            payload = {
                "answer": answer,
                "sources": sources,
                "query": query
            }
            Logger.log(f"Jawaban berhasil di-generate untuk query: {payload}")
            
            return payload
        
        except Exception as e:
            Logger.log(f"Error generating answer: {e}")
            return {
                "answer": f"Maaf, terjadi error saat memproses pertanyaan: {str(e)}",
                "sources": sources
            }
        
        