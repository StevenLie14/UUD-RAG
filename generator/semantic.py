from database import Qdrant
from logger import Logger
from llm import BaseLLM
from .base import BaseGenerator
from qdrant_client.models import ScoredPoint

class SemanticGenerator(BaseGenerator):
    def __init__(self, database: Qdrant, llm: BaseLLM):
        super().__init__(database, llm)
        
    def generate_answer(self, query: str, limit: int = 5):
        try:
            relevant_chunks: list[ScoredPoint] = self.database.hybrid_search_with_colbert(query, limit)
            
            if not relevant_chunks:
                return {
                    "answer": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen hukum.",
                    "sources": []
                }
            
            Logger.log(f"Found {len(relevant_chunks)} relevant semantic chunks for query: '{query}'")
            Logger.log(f"Semantic chunks scores: {[chunk.score for chunk in relevant_chunks]}")
            
            context_parts = []
            sources = []
            
            for idx, chunk in enumerate(relevant_chunks):
                payload = chunk.payload
                context_parts.append(payload['full_text'])
                
                # Build source info with semantic-specific metadaxta
                source_info = {
                    "source": payload.get('source', 'Unknown'),
                    "page": payload.get('page', 0),
                    "page_label": payload.get('page_label', ''),
                    "total_pages": payload.get('total_pages', 0),
                    "score": chunk.score,
                    "chunk_id": payload.get('chunk_id', chunk.id),
                    "chunk_type": payload.get('chunk_type', 'semantic'),
                    "content": payload['full_text']
                }
                
                if 'semantic_score' in payload:
                    source_info['semantic_score'] = payload['semantic_score']
                if 'boundary_type' in payload:
                    source_info['boundary_type'] = payload['boundary_type']
                
                sources.append(source_info)
            
            context = "\n---\n".join(context_parts)
            
            prompt = self._generate_semantic_prompt(context, query)
            
            answer = self.llm.answer(prompt, {"context": context, "question": query})
            
            result = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "retrieval_method": "semantic_hybrid_colbert",
                "chunking_method": "semantic"
            }
            
            Logger.log(f"Successfully generated answer using semantic chunks")
            return result
        
        except Exception as e:
            Logger.log(f"Error generating answer with semantic chunks: {e}")
            return {
                "answer": f"Maaf, terjadi error saat memproses pertanyaan: {str(e)}",
                "sources": sources if 'sources' in locals() else [],
                "query": query,
                "retrieval_method": "semantic_hybrid_colbert",
                "chunking_method": "semantic"
            }
    
    def _generate_semantic_prompt(self, context: str, question: str):
        """
        Generate enhanced prompt specifically for semantic chunks
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Anda adalah asisten hukum AI yang membantu menjawab pertanyaan tentang dokumen hukum Indonesia.
                Anda bekerja dengan chunk semantik yang telah dikelompokkan berdasarkan makna dan konteks.
                
                Keunggulan chunk semantik:
                - Chunk dibuat berdasarkan batas-batas makna alami dalam teks
                - Setiap chunk memiliki koherensi konteks yang kuat
                - Informasi terkait dikelompokkan dalam chunk yang sama
                
                Tugas Anda:
                1. Berikan jawaban yang akurat berdasarkan konteks dokumen hukum yang diberikan
                2. Manfaatkan koherensi semantik dalam chunk untuk memberikan jawaban yang lebih utuh
                3. Jika informasi tidak ada dalam konteks, katakan dengan jelas
                4. Gunakan bahasa formal dan profesional
                5. Sebutkan pasal, undang-undang, atau referensi hukum yang spesifik jika ada
                6. Jelaskan hubungan antar konsep hukum jika relevan
                
                Format jawaban:
                - Mulai dengan jawaban langsung
                - Berikan penjelasan detail dengan memanfaatkan konteks semantik
                - Sebutkan dasar hukum atau sumber yang relevan
                - Jika ada ketidakpastian, sebutkan dengan jelas
                
                PENTING: Gunakan informasi dari konteks semantik yang diberikan dan manfaatkan 
                keterkaitan makna antar bagian teks untuk memberikan jawaban yang komprehensif.
                """
            ),
            (
                "user",
                """
                Konteks dari dokumen hukum (chunk semantik):
                
                {context}
                
                ---
                
                Pertanyaan: {question}
                
                Berikan jawaban yang lengkap dan akurat berdasarkan konteks semantik di atas.
                """
            )
        ])

        return prompt