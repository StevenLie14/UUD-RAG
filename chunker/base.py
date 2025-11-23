from langchain_core.documents import Document
from model import BaseChunk
from typing import Dict

class BaseChunker:
    def __init__(self):
        self.chunks : Dict[str, BaseChunk] = {}

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        raise NotImplementedError