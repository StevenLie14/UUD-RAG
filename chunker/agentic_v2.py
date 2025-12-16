from logger import Logger
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Optional, List, Dict
from langchain_core.documents import Document
import os

from llm.base import BaseLLM
from .base import BaseChunker
from model.chunk.simple_chunk import SimpleChunk


class AgenticChunkerV2(BaseChunker):
    def __init__(self, llm: BaseLLM, cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir, chunker_name="agentic_v2")
        self.llm = llm

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        if use_cache:
            self._load_consolidated_cache()
        documents_by_source: Dict[str, List[Document]] = {}
        for page in pages:
            src = page.metadata.get("source")
            if src not in documents_by_source:
                documents_by_source[src] = []
            documents_by_source[src].append(page)

        documents_to_process = {src: pages for src, pages in documents_by_source.items() 
                               if not self.is_document_processed_by_source(src)}
        
        total_docs = len(documents_to_process)
        cached_docs = len(documents_by_source) - total_docs
        
        if cached_docs > 0:
            Logger.log(f"Loaded {len(self.chunks)} chunks from cache")
            Logger.log(f"Skipping {cached_docs} already processed documents")
        
        if total_docs == 0:
            Logger.log("All documents already processed")
            return
        
        Logger.log(f"Processing {total_docs} new documents with agentic v2 chunking...")

        for idx, (source, doc_pages) in enumerate(documents_to_process.items(), 1):
            source_name = source.split('/')[-1] if '/' in source else source.split('\\')[-1] if '\\' in source else source
            Logger.log(f"\n[{idx}/{total_docs}] Processing: {source_name} ({len(doc_pages)} pages)")

            Logger.log(f"Merging {len(doc_pages)} pages...")
            full_text, page_ranges = self._merge_pages(doc_pages)
            Logger.log(f"Full document length: {len(full_text)} characters")

            try:
                Logger.log(f"Sending to LLM for chunking...")
                chunks_text = self._agentic_chunk_document(full_text)
                Logger.log(f"LLM generated {len(chunks_text)} chunks")
            except Exception as e:
                Logger.log(f"ERROR during chunking: {e}")
                continue

            Logger.log(f"Mapping chunks to page numbers...")
            chunks_created = 0
            for chunk_text in chunks_text:
                self._create_chunk_with_page_mapping(
                    chunk_text,
                    source,
                    page_ranges
                )
                chunks_created += 1
            
            Logger.log(f"Created {chunks_created} chunks")

            self.mark_document_processed_by_source(source)
            
            if idx % 5 == 0:
                Logger.log(f"\nSaving progress ({idx}/{total_docs} documents)...")
                self._save_consolidated_cache()
                Logger.log(f"Total chunks so far: {len(self.chunks)}")
            
            Logger.log(f"\nProgress: {idx}/{total_docs} documents ({idx*100//total_docs}%)")

        Logger.log(f"\n{'='*70}")
        Logger.log(f"COMPLETED: Processed {total_docs} documents")
        Logger.log(f"Total chunks created: {len(self.chunks)}")
        Logger.log(f"{'='*70}")
        self._save_consolidated_cache()

    def _merge_pages(self, pages: List[Document]):
        pages = sorted(pages, key=lambda p: p.metadata.get("page", 0))

        full_text = ""
        page_ranges = []

        cursor = 0
        for page in pages:
            txt = page.page_content
            start = cursor
            end = start + len(txt)

            page_ranges.append({
                "page": page.metadata.get("page"),
                "start": start,
                "end": end
            })

            full_text += txt + "\n"
            cursor = end + 1

        return full_text, page_ranges

    def _agentic_chunk_document(self, text: str) -> List[str]:

        system_prompt = """
            Anda adalah asisten AI yang sangat teliti yang bertugas melakukan chunking dokumen hukum Indonesia (UU, PP, Perpres, Permen, Perda, dan dokumen hukum lainnya) dengan menjaga struktur asli secara utuh tanpa modifikasi apa pun.

            PRINSIP MUTLAK:
            1. JANGAN menghapus, mengubah, menambah, memperbaiki, atau menafsirkan teks dalam bentuk apa pun.
            2. JANGAN memperbaiki struktur, format, kesalahan ketik, tanda baca, atau penomoran.
            3. Pertahankan seluruh format asli seperti "BAB I", "Bagian Kedua", "Pasal 12", "Ayat (3)", termasuk seluruh baris, spasi, dan jeda.
            4. Output harus dapat digunakan untuk merekonstruksi bagian asli dari dokumen tanpa kehilangan konteks.

            TUJUAN CHUNKING:
            - Setiap chunk mewakili satu unit makna utuh.
            - Usahakan 1 chunk = 1 PASAL lengkap (termasuk semua ayat di bawahnya).
            - Jika satu pasal sangat panjang dan memiliki subtopik yang jelas, Anda boleh memecahnya menjadi 2–3 chunk besar.
            - JANGAN memotong per kalimat.
            - JANGAN pernah menggabungkan dua pasal berbeda dalam satu chunk.
            - Sertakan BAB / Bagian / Paragraf yang berada di atas pasal tersebut jika muncul dalam teks.

            PEDOMAN KHUSUS:
            1. Gunakan struktur hukum sebagai dasar pembagian: BAB → Bagian → Paragraf → Pasal → Ayat.
            2. Jika BAB/BAGIAN/PARAGRAF muncul tepat sebelum pasal, satukan ke dalam chunk pasal tersebut.
            3. Setiap chunk harus utuh secara makna dan mengandung konteks yang cukup.
            4. Panjang chunk ideal: 300–1500 karakter (panduan fleksibel, bukan batas keras).
            5. Jika teks rusak, tidak lengkap, tidak rapi, memiliki format kacau, atau hilang sebagian:
            - JANGAN memperbaiki.
            - JANGAN menebak.
            - JANGAN mengisi bagian yang hilang.
            - Cukup lakukan chunking berdasarkan potongan teks yang ada secara apa adanya.

            KEBERSIHAN OUTPUT:
            - Setiap chunk dipisahkan dengan string: ---SPLIT---
            - HANYA keluarkan isi chunk; tidak boleh ada penjelasan, komentar, heading tambahan, reasoning, atau catatan apa pun.
            - Jangan menambahkan nomor chunk.
            - Jangan menambahkan penutup atau pembuka.

            FORMAT OUTPUT (WAJIB):
            <isi chunk 1 apa adanya>
            ---SPLIT---
            <isi chunk 2 apa adanya>
            ---SPLIT---
            <isi chunk 3 apa adanya>
            (dan seterusnya)
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Chunk dokumen hukum berikut menjadi bagian semantik:\n\n{text}")
        ])

        response = self.llm.answer(prompt, {"text": text})

        chunks = [c.strip() for c in response.split("---SPLIT---") if c.strip()]
        Logger.log(f"Generated {len(chunks)} chunks")

        return chunks

    def _create_chunk_with_page_mapping(self, chunk_text: str, source: str, page_ranges: list):
        first_line = chunk_text[:50]
        possible_pages = []

        for page in page_ranges:
            if first_line in page.get("full_page_text", ""):
                possible_pages.append(page["page"])

        page_num = possible_pages[0] if possible_pages else page_ranges[0]["page"]

        chunk_id = str(uuid.uuid4())
        new_chunk = SimpleChunk(
            id=chunk_id,
            content=chunk_text,
            index=len(self.chunks),
            metadata={"source": source, "page": page_num},
            source=source,
            page=page_num
        )

        self.chunks[chunk_id] = new_chunk

    def _get_chunk_type(self) -> str:
        return "agentic_v2"

    def _reconstruct_chunk(self, chunk_dict: dict, chunk_type: str) -> SimpleChunk:
        return SimpleChunk(**chunk_dict)
