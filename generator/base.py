from database.base import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from llm import BaseLLM
from rag.search_strategy import SearchStrategy

class BaseGenerator:
    def __init__(self, database: VectorStore, llm: BaseLLM, search_strategy: SearchStrategy):
        self.database = database
        self.llm = llm
        self.search_strategy = search_strategy
    
    def generate_prompt(self, context: str, question: str):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Anda adalah asisten hukum AI yang membantu menjawab pertanyaan tentang dokumen hukum Indonesia.
                
                Tugas Anda:
                1. Berikan jawaban yang akurat berdasarkan konteks dokumen hukum yang diberikan
                2. Jika informasi tidak ada dalam konteks, katakan dengan jelas
                3. Gunakan bahasa formal dan profesional
                4. Sebutkan pasal, undang-undang, atau referensi hukum yang spesifik jika ada
                5. **HINDARI penggunaan frasa eksplisit seperti 'Menurut teks,' 'Berdasarkan konteks,' atau 'Dalam dokumen yang diberikan,' kecuali jika Anda secara spesifik menyoroti ketidakpastian sumber.**
                6. Jika ada interpretasi, jelaskan dengan jelas bahwa itu adalah interpretasi
                
                Format jawaban:
                - Mulai dengan jawaban langsung
                - Berikan penjelasan detail jika diperlukan
                - Sebutkan dasar hukum atau sumber yang relevan
                - Jika ada ketidakpastian, sebutkan dengan jelas
                
                PENTING: Hanya gunakan informasi dari konteks yang diberikan. Jangan menambahkan informasi dari 
                pengetahuan umum Anda tanpa menyebutkan bahwa itu adalah informasi tambahan.
                """
            ),
            (
                "user",
                """
                Konteks dari dokumen hukum:
                
                {context}
                
                ---
                
                Pertanyaan: {question}
                
                Berikan jawaban yang lengkap dan akurat berdasarkan konteks di atas.
                """
            )
        ])

        return prompt
        
    def generate_answer(self, query: str, limit: int = 5):
        raise NotImplementedError
        
        