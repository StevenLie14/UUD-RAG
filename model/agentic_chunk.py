from .base_chunk import BaseChunk

class AgenticChunk(BaseChunk):
    title : str
    summary : str
    propositions : list[str]
    index : int
    
    def get_context(self) -> str:
        content = "\n".join(self.propositions)
        return f"Judul: {self.title}\nRingkasan: {self.summary}\n\nKonten:\n{content}"
    
    def get_payload(self) -> dict:
        return {
            "chunk_id": self.id,
            "title": self.title,
            "summary": self.summary,
            "propositions": self.propositions,
            "full_text": self.get_context(),
            "index": self.index
        }
    