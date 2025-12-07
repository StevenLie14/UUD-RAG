from logger import Logger
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.documents import Document
import os
from llm.base import BaseLLM
from .base import BaseChunker
from typing import Dict
from model.chunk.simple_chunk import SimpleChunk


class AgenticChunkerV2(BaseChunker):
    def __init__(self, llm: BaseLLM, cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir, chunker_name="agentic_v2")
        self.llm = llm
        self.current_page = None
        self.current_doc_chunk_ids = None
    
    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        # Load existing cache
        if use_cache:
            self._load_consolidated_cache()
            if len(self.chunks) > 0:
                Logger.log(f"Loaded {len(self.chunks)} chunks from cache")
        
        # Filter uncached documents
        uncached_pages = self.get_uncached_documents(pages)
        if len(uncached_pages) < len(pages):
            Logger.log(f"Skipping {len(pages) - len(uncached_pages)} already processed documents")
        
        if not uncached_pages:
            Logger.log("All documents already processed")
            return
        
        Logger.log(f"Processing {len(uncached_pages)} new documents with agentic chunking v2...")
        
        checkpoint_interval = 25  # Save every 25 documents
        processed_count = 0
        
        try:
            for idx, page in enumerate(uncached_pages, 1):
                try:
                    self.current_doc_chunk_ids = []
                    self.current_page = page
                    
                    # Use the new agentic chunking method
                    self._agentic_chunking(page)
                    
                    # Mark document as processed
                    self.mark_document_processed(page)
                    self.current_doc_chunk_ids = None
                    processed_count += 1
                    
                    # Checkpoint save every N documents
                    if idx % checkpoint_interval == 0:
                        Logger.log(f"Checkpoint: Saving progress ({idx}/{len(uncached_pages)} documents)...")
                        self._save_consolidated_cache()
                        Logger.log(f"Checkpoint saved. Total chunks so far: {len(self.chunks)}")
                    
                except Exception as e:
                    Logger.log(f"Error processing document {idx}/{len(uncached_pages)}: {e}")
                    Logger.log(f"Saving progress before continuing...")
                    self._save_consolidated_cache()
                    Logger.log(f"Progress saved. Skipping problematic document.")
                    continue
        
        except KeyboardInterrupt:
            Logger.log(f"\nInterrupted by user. Saving progress...")
            self._save_consolidated_cache()
            Logger.log(f"Progress saved: {processed_count}/{len(uncached_pages)} documents processed")
            raise
        
        Logger.log(f"Total chunks: {len(self.chunks)} (added {processed_count} documents)")
        
        # Final save
        self._save_consolidated_cache()
    
    def _agentic_chunking(self, page: Document) -> List[str]:
        """
        Dynamically splits text into meaningful chunks using LLM.
        This is a simpler approach - just split text into semantic chunks.
        """
        text = page.page_content

        # Get response from LLM
        response = self.llm.answer(ChatPromptTemplate.from_messages([
            ("system", """Anda adalah asisten AI yang sangat teliti yang membantu membagi teks dokumen hukum Indonesia (Undang-Undang, Peraturan, dll.) menjadi potongan (chunk) yang bermakna berdasarkan struktur dan topik, BUKAN per kalimat.

TUJUAN UTAMA:
- Setiap chunk mewakili satu unit makna yang utuh: bisa berupa satu atau beberapa pasal, ayat, atau bagian yang membahas satu topik/subtopik yang saling berkaitan.
- SEBISA MUNGKIN, satu chunk berisi SATU PASAL utuh beserta seluruh ayatnya.
- Hanya jika pasal tersebut sangat panjang atau jelas terbagi menjadi beberapa sub-topik berbeda, Anda boleh memecahnya menjadi 2–3 chunk.
- Chunk TIDAK BOLEH terlalu pendek seperti hanya satu kalimat jika kalimat itu masih terkait kuat dengan kalimat-kalimat setelah/sebelumnya.
- Utamakan koherensi dan kelengkapan konteks ketimbang pemotongan terlalu sering.

PEDOMAN PEMBAGIAN:
1. Perhatikan struktur hukum:
   - Usahakan membagi berdasarkan BAGIAN/BAB/PARAGRAF/PASAL bila terlihat jelas.
   - Satu chunk boleh berisi beberapa ayat dalam satu pasal jika masih satu topik.
   - Jika satu pasal/ayat terlalu panjang tapi jelas terbagi menjadi beberapa sub-topik, boleh dipecah menjadi 2–3 chunk yang tetap utuh secara makna.

2. JANGAN memotong per kalimat:
   - SATU chunk boleh berisi beberapa kalimat yang saling mendukung.
   - Jangan buat chunk hanya 1–2 kalimat kecuali benar-benar berdiri sendiri dan tidak butuh kalimat lain untuk dipahami.

3. Jaga koherensi dan konteks:
   - Jika ada definisi yang dijelaskan di beberapa kalimat berurutan, satukan dalam satu chunk.
   - Pastikan setiap chunk cukup konteks: sertakan teks "Pasal X." beserta ayat-ayat di bawahnya bila ada di teks.

4. Panjang chunk (kira-kira):
   - Usahakan panjang chunk di kisaran 300–1500 karakter (bukan batas keras, hanya panduan).
   - Jika sebuah bagian sangat penting dan panjang, lebih baik dipecah menjadi 2–3 chunk besar yang sama-sama koheren daripada banyak chunk kecil.

FORMAT OUTPUT:
- Kembalikan setiap chunk yang dipisahkan dengan string khusus: ---SPLIT--- (tanpa spasi di kiri/kanan).
- Setiap chunk harus berupa teks murni tanpa penomoran, bullet points, heading, atau label tambahan.
- JANGAN menambahkan penjelasan di luar isi teks asli.
- JANGAN menambahkan judul baru atau ringkasan; cukup gunakan teks asli yang telah dikelompokkan.

CONTOH FORMAT YANG DIINGINKAN:
Pasal 3.\n(1) Milik-milik Algemcene Volkscredietbank menjadi milik Bank Rakyat Indonesia.\n(2) Hutang-hutang Algemeene Volkscredietbank dioper oleh Bank Rakyat Indonesia.\n(3) Jumlah hutang-hutang yang melebihi jumlah harga sebenarnya dari piutang-piutang, uang kas, saldo pada bank-bank lain dan efek-efek ditutup oleh Pemerintah.

---SPLIT---

[Pasal lain atau kelompok ayat lain yang membahas topik berbeda, juga terdiri dari beberapa kalimat yang saling berkaitan, menjadi chunk berikutnya]"""),
            ("user", f"Silakan bagi teks dokumen hukum berikut menjadi chunk yang terpisah secara semantik dan bermakna. Usahakan setiap chunk mengikuti format seperti contoh Pasal 3 (satu pasal utuh beserta ayat-ayatnya bila memungkinkan):\n\n{text}")
        ]), {})

        # Split based on the delimiter
        chunks_text = response.split("---SPLIT---")

        # Filter out empty chunks
        chunks_text = [chunk.strip() for chunk in chunks_text if chunk.strip()]

        Logger.log(f"Generated {len(chunks_text)} chunks from document")

        # Handle page metadata
        if isinstance(page, tuple):
            page_document = page[1]
        else:
            page_document = page

        # Create SimpleChunk objects for each chunk
        for chunk_text in chunks_text:
            self._create_chunk_from_text(chunk_text, page_document)

        return chunks_text
    
    def _create_chunk_from_text(self, text: str, page: Document) -> str:
        """
        Create a simple chunk from the provided text.
        No title or summary generation - just store the text directly.
        Extracts source and page information from metadata.
        """
        chunk_id = str(uuid.uuid4())
        
        # Extract source and page from metadata
        metadata = page.metadata
        source = metadata.get('source', None)
        page_number = metadata.get('page', None)
        
        # Create SimpleChunk object
        new_chunk = SimpleChunk(
            id=chunk_id,
            content=text,
            index=len(self.chunks),
            metadata=metadata,
            source=source,
            page=page_number
        )
        
        self.chunks[chunk_id] = new_chunk
        
        # Track this chunk for current document
        if self.current_doc_chunk_ids is not None:
            self.current_doc_chunk_ids.append(chunk_id)
        
        Logger.log(f"Created chunk with ID: {chunk_id}, source: {source}, page: {page_number}, length: {len(text)} chars")
        return chunk_id
    
    def print_chunks(self):
        """Print all chunks for debugging"""
        for chunk in self.chunks.values():
            print(f"Chunk ID: {chunk.id}")
            print(f"Index: {chunk.index}")
            print(f"Source: {chunk.source}")
            print(f"Page: {chunk.page}")
            print(f"Content: {chunk.content[:200]}...")  # Show first 200 chars
            print("")
    
    def _get_chunk_type(self) -> str:
        return 'agentic_v2'
    
    def _reconstruct_chunk(self, chunk_dict: dict, chunk_type: str) -> SimpleChunk:
        """Reconstruct SimpleChunk object from dict"""
        return SimpleChunk(**chunk_dict)
