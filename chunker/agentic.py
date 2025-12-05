from logger import Logger
import uuid
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.documents import Document
import os
from llm.base import BaseLLM
from .base import BaseChunker
from typing import Dict
from model.chunk.agentic_chunk import AgenticChunk
from utils import json_parser as utils

class AgenticChunker(BaseChunker):
    def __init__(self, llm: BaseLLM, cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir)
        self.llm = llm
        self.current_doc_hash = None
        self.current_doc_chunk_ids = []
    
    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        if use_cache:
            uncached_pages = self.get_uncached_documents(pages)
            if len(uncached_pages) < len(pages):
                Logger.log(f"Loaded {len(pages) - len(uncached_pages)} documents from cache")
            pages = uncached_pages
        
        if not pages:
            Logger.log("All documents already cached")
            return
        
        Logger.log(f"Processing {len(pages)} uncached documents with agentic chunking...")
        
        for page in pages:
            doc_hash = self._get_document_hash(page)
            self.current_doc_hash = doc_hash
            self.current_doc_chunk_ids = []
            
            self._generate_propositions(page)
            
            # Cache chunks for this document
            self.document_chunks[doc_hash] = self.current_doc_chunk_ids
            if use_cache:
                self._save_chunks_to_cache(doc_hash, self.current_doc_chunk_ids)
        
        Logger.log(f"Generated propositions for {len(pages)} pages. Total chunks: {len(self.chunks)}")
    
    def _generate_propositions(self, page: Document) -> list[str]:
        text = page.page_content
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Uraikan "Konten" dokumen hukum menjadi proposisi yang jelas dan sederhana, pastikan proposisi tersebut 
                    dapat dipahami di luar konteks aslinya.
                    
                    1. Pisahkan kalimat majemuk menjadi kalimat-kalimat sederhana. Pertahankan frasa asli dari input 
                    sebisa mungkin.
                    
                    2. Untuk setiap entitas bernama (nama orang, lembaga, peraturan, pasal, dll.) yang disertai informasi 
                    deskriptif tambahan, pisahkan informasi ini menjadi proposisi tersendiri.
                    
                    3. Decontekstualisasikan proposisi dengan menambahkan modifier yang diperlukan pada kata benda atau 
                    seluruh kalimat, dan ganti kata ganti (misalnya, "ini", "itu", "ia", "mereka", "tersebut") dengan 
                    nama lengkap entitas yang dirujuk.
                    
                    4. Kembalikan output dalam format ini:
                    {{
                        "propositions": [
                        "proposisi 1",
                        "proposisi 2",
                        ...
                        ]
                    }}

                    JANGAN sertakan penjelasan apapun, markdown code fences, atau teks di luar JSON.

                    Contoh:

                    Input: Judul: Undang-Undang Cipta Kerja. Bagian: Ketentuan Umum, Definisi Pekerja. Konten:
                    Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan pada tanggal 2 November 2020 
                    oleh Presiden Joko Widodo. Undang-undang ini mengatur berbagai aspek ketenagakerjaan termasuk 
                    definisi pekerja. Dalam undang-undang ini, pekerja didefinisikan sebagai setiap orang yang bekerja 
                    dengan menerima upah atau imbalan dalam bentuk lain. Pekerja dapat bekerja berdasarkan perjanjian 
                    kerja waktu tertentu atau perjanjian kerja waktu tidak tertentu. Pengusaha wajib memberikan jaminan 
                    sosial kepada pekerja sesuai ketentuan peraturan perundang-undangan.
                    
                    Output: [
                    "Undang-Undang Nomor 11 Tahun 2020 adalah tentang Cipta Kerja.",
                    "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan pada tanggal 2 November 2020.",
                    "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan oleh Presiden Joko Widodo.",
                    "Undang-Undang Cipta Kerja mengatur berbagai aspek ketenagakerjaan.",
                    "Undang-Undang Cipta Kerja mengatur definisi pekerja.",
                    "Dalam Undang-Undang Cipta Kerja, pekerja didefinisikan sebagai setiap orang yang bekerja dengan menerima upah atau imbalan dalam bentuk lain.",
                    "Pekerja dapat bekerja berdasarkan perjanjian kerja waktu tertentu.",
                    "Pekerja dapat bekerja berdasarkan perjanjian kerja waktu tidak tertentu.",
                    "Pengusaha wajib memberikan jaminan sosial kepada pekerja.",
                    "Kewajiban pengusaha memberikan jaminan sosial sesuai ketentuan peraturan perundang-undangan."
                    ]
                    """
                ),
                ("user", "Uraikan konten hukum berikut:: {input}"),
            ]
        )
        result = self.llm.answer(PROMPT, {"input": text})
        
        class Sentences (BaseModel):
            propositions: list[str]
        sentences = utils.parse_json_response(result, Sentences)
        
        if sentences is None:
            Logger.log("No propositions generated.")
            return []
        
        # `page` dalam konteks ini adalah (index, Document). Kita perlu mengambil Document
        # Jika Anda memperbaiki `load_data_to_chunks` (seperti di atas), ini akan menjadi objek Document
        if isinstance(page, tuple):
            page_document = page[1]
        else:
            page_document = page

        for proposition in sentences.propositions:
            self.add_proposition(proposition, page_document)
            
        return sentences.propositions

    
    def _update_chunk_summary(self, chunk: AgenticChunk) -> str: # Menerima AgenticChunk
        """If you add a new proposition to a chunk, you may want to update the summary or else they could get stale"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Sebuah proposisi baru saja ditambahkan ke salah satu chunk Anda. Anda harus membuat ringkasan sangat 
                    singkat dalam 1 kalimat yang menginformasikan pembaca tentang isi kelompok chunk tersebut.

                    Ringkasan yang baik akan menjelaskan tentang apa chunk tersebut, dan memberikan instruksi klarifikasi 
                    tentang apa yang harus ditambahkan ke chunk tersebut.

                    Anda akan diberikan sekelompok proposisi yang ada dalam chunk dan ringkasan chunk saat ini.

                    Ringkasan Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang Pasal 1, 
                    generalisasikan ke "ketentuan pasal". Atau jika tentang hak pekerja, generalisasikan ke "hak dan 
                    kewajiban tenaga kerja".

                    Contoh:
                    Input: Proposisi: Pasal 27 UUD 1945 menjamin hak setiap warga negara untuk mendapatkan pekerjaan
                    Output: Chunk ini berisi informasi tentang ketentuan hak warga negara terkait pekerjaan dalam UUD 1945.

                    Hanya berikan ringkasan chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Proposisi dalam Chunk:\n{proposition}\n\nRingkasan chunk saat ini:\n{current_summary}"),
            ]
        )

        new_chunk_summary = self.llm.answer(PROMPT, {
            "proposition": "\n".join(chunk.propositions),
            "current_summary" : chunk.summary
        })

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk: AgenticChunk) -> str: # Menerima AgenticChunk
        """If you add a new proposition to a chunk, you may want to update the title or else it can get stale"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Sebuah proposisi baru saja ditambahkan ke salah satu chunk Anda. Anda harus membuat judul chunk yang 
                    sangat singkat dan diperbarui yang menginformasikan pembaca tentang isi kelompok chunk tersebut.

                    Judul yang baik akan menjelaskan tentang apa chunk tersebut.

                    Anda akan diberikan sekelompok proposisi yang ada dalam chunk, ringkasan chunk, dan judul chunk.

                    Judul Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang Pasal 1, 
                    generalisasikan ke "Ketentuan Pasal". Atau jika tentang putusan pengadilan, generalisasikan ke 
                    "Yurisprudensi & Putusan".

                    Contoh:
                    Input: Ringkasan: Chunk ini berisi informasi tentang tanggal dan waktu pemberlakuan peraturan
                    Output: Tanggal Berlaku Peraturan

                    Hanya berikan judul chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Proposisi dalam Chunk:\n{proposition}\n\nRingkasan chunk:\n{current_summary}\n\nJudul chunk saat ini:\n{current_title}"),
            ]
        )
        
        new_chunk_title = self.llm.answer(PROMPT, {
            "proposition": "\n".join(chunk.propositions), # Mengakses properti
            "current_summary" : chunk.summary, # Mengakses properti
            "current_title" : chunk.title # Mengakses properti
        })

        return new_chunk_title
    
    # ... Metode _get_new_chunk_summary (tidak berubah) ...
    def _get_new_chunk_summary(self, proposition: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Anda harus membuat ringkasan sangat singkat dalam 1 kalimat yang menginformasikan pembaca tentang 
                    isi kelompok chunk tersebut.

                    Ringkasan yang baik akan menjelaskan tentang apa chunk tersebut, dan memberikan instruksi klarifikasi 
                    tentang apa yang harus ditambahkan ke chunk tersebut.

                    Anda akan diberikan proposisi yang akan masuk ke chunk baru. Chunk baru ini memerlukan ringkasan.

                    Ringkasan Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang sanksi pidana, 
                    generalisasikan ke "ketentuan sanksi". Atau jika tentang januari 2020, generalisasikan ke "tanggal 
                    dan waktu".

                    Contoh:
                    Input: Proposisi: Pasal 88 mengatur tentang sanksi pidana penjara minimal 5 tahun
                    Output: Chunk ini berisi informasi tentang ketentuan sanksi pidana dalam peraturan.

                    Hanya berikan ringkasan chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Tentukan ringkasan chunk baru dimana proposisi ini akan masuk:\n{proposition}"),
            ]
        )

        
        new_chunk_summary = self.llm.answer(PROMPT, {
            "proposition": proposition
        })

        return new_chunk_summary
    
    # ... Metode _get_new_chunk_title (tidak berubah) ...
    def _get_new_chunk_title(self, summary) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Anda harus membuat judul chunk yang sangat singkat (beberapa kata) yang menginformasikan pembaca 
                    tentang isi kelompok chunk tersebut.

                    Judul chunk yang baik adalah singkat tetapi mencakup tentang apa chunk tersebut.

                    Anda akan diberikan ringkasan dari chunk yang memerlukan judul.

                    Judul Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang UU Ketenagakerjaan, 
                    generalisasikan ke "Peraturan Ketenagakerjaan". Atau jika tentang Maret 2020, generalisasikan ke 
                    "Tanggal & Waktu".

                    Contoh:
                    Input: Ringkasan: Chunk ini berisi informasi tentang tanggal dan waktu pemberlakuan undang-undang
                    Output: Tanggal & Waktu Pemberlakuan

                    Hanya berikan judul chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Tentukan judul chunk dimana ringkasan ini termasuk:\n{summary}"),
            ]
        )

        new_chunk_title = self.llm.answer(PROMPT, {
            "summary": summary
        })
        return new_chunk_title
    
    def _create_chunk(self, proposition: str, page: Document) -> str:
        id = str(uuid.uuid4())
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)
        
        # MEMBUAT OBJEK AgenticChunk BARU
        new_chunk = AgenticChunk(
            id=id,
            title=title,
            summary=summary,
            propositions=[proposition],
            index=len(self.chunks),
            metadata=page.metadata
        )

        self.chunks[id] = new_chunk
        
        # Track this chunk for current document
        if self.current_doc_chunk_ids is not None:
            self.current_doc_chunk_ids.append(id)
        
        Logger.log(f"Created new chunk with ID: {id}, Title: {title}, Summary: {summary}")
        return id # Mengembalikan ID untuk konsistensi
    
    def get_chunks(self) -> str:
        # Menggunakan properti dari objek AgenticChunk
        chunks = "\n".join([f"Chunk ID: {chunk.id}, Judul: {chunk.title}, Ringkasan: {chunk.summary}" for chunk in self.chunks.values()])
        return chunks
    
    def _find_similar_chunk(self, proposition: str):
        chunks = self.get_chunks()
        
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Tentukan apakah "Proposisi" hukum harus masuk ke salah satu chunk yang sudah ada atau tidak.
                    
                    Chunk merepresentasikan kelompok proposisi yang membahas topik serupa.
                    
                    Input:
                    1. Proposisi baru
                    2. Kumpulan chunk yang ada (ID, Judul, dan Ringkasan)
                    
                    Output harus dalam format JSON:
                    {{
                        "chunk_id": "id-chunk-yang-cocok-atau-null"
                    }}
                    
                    Jika proposisi cocok dengan salah satu chunk yang ada, kembalikan ID chunk tersebut.
                    Jika proposisi TIDAK cocok dengan chunk manapun, kembalikan null.
                    
                    JANGAN sertakan penjelasan, alasan, atau teks apapun di luar JSON.
                    JANGAN gunakan markdown code fences seperti ```json.
                    
                    Contoh Output 1 (cocok):
                    {{"chunk_id": "91c6850b-7fed-456b-bb98-656b552b8e68"}}
                    
                    Contoh Output 2 (tidak cocok):
                    {{"chunk_id": null}}
                    """,
                ),
                ("user", "Chunk Saat Ini:\n--Awal chunk saat ini--\n{current_chunk_outline}\n--Akhir chunk saat ini--"),
                ("user", "Tentukan apakah proposisi berikut harus masuk ke salah satu chunk yang diuraikan:\n{proposition}"),
            ]
        )
        
        chunk_found = self.llm.answer(PROMPT, {
            "proposition": proposition,
            "current_chunk_outline": chunks
        })
        
        Logger.log(f"Chunk found: {chunk_found}")
        
        class ChunkMatch(BaseModel):
            chunk_id: Optional[str] = None
            
        chunk_found = utils.parse_json_response(chunk_found, ChunkMatch)
        
        if chunk_found is None or chunk_found.chunk_id not in self.chunks:
            return None
        return chunk_found.chunk_id
    
    def add_proposition_to_chunk(self, chunk_id: str, proposition: str):
        chunk = self.chunks[chunk_id] # Mengambil objek AgenticChunk
        
        # MEMANIPULASI OBJEK AgenticChunk
        chunk.propositions.append(proposition)
        
        # MENGUPDATE properti objek AgenticChunk
        chunk.title = self._update_chunk_title(chunk)
        chunk.summary = self._update_chunk_summary(chunk)
        
    def add_proposition(self, proposition: str, page: Document):
        Logger.log(f"Adding proposition to chunker: {proposition}")
        
        if len(self.chunks) == 0:
            self._create_chunk(proposition, page)
            return
        
        # Find the most similar chunk
        most_similar_chunk_id = self._find_similar_chunk(proposition)
        if most_similar_chunk_id is None:
            self._create_chunk(proposition, page)
            return
        
        self.add_proposition_to_chunk(most_similar_chunk_id, proposition)
        
    def print_chunks(self):
        # Menggunakan properti dari objek AgenticChunk
        for chunk in self.chunks.values():
            print(f"Chunk ID: {chunk.id}")
            print(f"Judul Chunk: {chunk.title}")
            print(f"Ringkasan Chunk: {chunk.summary}")
            print(f"Proposisi Chunk: {chunk.propositions}")
            print("")