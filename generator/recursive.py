from database import Qdrant
from logger import Logger
from llm import BaseLLM
from .base import BaseGenerator
from qdrant_client.models import ScoredPoint

class RecursiveGenerator(BaseGenerator):
    def __init__(self, database : Qdrant,llm: BaseLLM):
        super().__init__(database, llm)
        
    def generate_answer(self, query: str, limit: int = 5):
        relevant_chunks : list[ScoredPoint] =  self.database.hybrid_search_with_colbert(query, limit)
        if not relevant_chunks:
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen hukum.",
                "sources": []
            }
        
        Logger.log(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
        Logger.log(f"Relevant chunks: {relevant_chunks}")
        
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(relevant_chunks):
            payload = chunk.payload
            context_parts.append(payload['full_text'])
            sources.append({
                "source": payload['source'],
                "page": payload['page'],
                "page_label": payload['page_label'],
                "total_pages": payload['total_pages'],
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
        
        