from abc import ABC, abstractmethod
from typing import Dict, List
from model import BaseChunk, SearchResult

class VectorStore(ABC):

    @abstractmethod
    def delete_collection(self):
        pass

    @abstractmethod
    def store_chunks(self, chunks: Dict[str, BaseChunk]):
        pass

    @abstractmethod
    def close(self):
        pass


class DenseSearchable(ABC):
    @abstractmethod
    def dense_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class SparseSearchable(ABC):
    @abstractmethod
    def sparse_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class HybridSearchable(ABC):
    @abstractmethod
    def hybrid_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class ColbertSearchable(ABC):
    @abstractmethod
    def hybrid_search_with_colbert(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class CrossEncoderSearchable(ABC):
    @abstractmethod
    def hybrid_search_with_crossencoder(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass