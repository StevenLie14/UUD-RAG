from abc import ABC, abstractmethod
from typing import List
from database.base import VectorStore, DenseSearchable, SparseSearchable, HybridSearchable, ColbertSearchable, CrossEncoderSearchable
from logger import Logger
from model import SearchResult


class SearchStrategy(ABC):
    @abstractmethod
    def search(self, database: VectorStore, query: str, limit: int) -> List[SearchResult]:
        pass


class DenseSearchStrategy(SearchStrategy):
    def search(self, database: DenseSearchable, query: str, limit: int) -> List[SearchResult]:
        return database.dense_search(query, limit)


class SparseSearchStrategy(SearchStrategy):
    def search(self, database: SparseSearchable, query: str, limit: int) -> List[SearchResult]:
        return database.sparse_search(query, limit)


class HybridSearchStrategy(SearchStrategy):
    def search(self, database: HybridSearchable, query: str, limit: int) -> List[SearchResult]:
        return database.hybrid_search(query, limit)


class HybridColbertSearchStrategy(SearchStrategy):
    def search(self, database: ColbertSearchable, query: str, limit: int) -> List[SearchResult]:
        return database.hybrid_search_with_colbert(query, limit)


class HybridCrossEncoderSearchStrategy(SearchStrategy):
    
    def search(self, database: CrossEncoderSearchable, query: str, limit: int) -> List[SearchResult]:
        return database.hybrid_search_with_crossencoder(query, limit)