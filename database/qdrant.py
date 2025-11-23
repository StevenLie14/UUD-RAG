from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import Logger
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector,Document, ScoredPoint, Prefetch, FusionQuery, Fusion, MultiVectorConfig, MultiVectorComparator, HnswConfigDiff
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from sentence_transformers import SentenceTransformer
from model import BaseChunk
from typing import Dict

class Qdrant:
    def __init__(self, qdrant_url : str = "http://localhost:6333", qdrant_api_key: str = None, collection_name: str = "documents", late_interaction_model_name: str = "jinaai/jina-colbert-v2", sparse_model_name: str = "Qdrant/bm25", dense_model_name: str = "LazarusNLP/all-indo-e5-small-v4"):
        Logger.log("qdrant url" + qdrant_url)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="gemini-embedding-001",
        #     google_api_key=google_api_key
        # )
        # model_name = "indobenchmark/indolegalbert-base"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        self.sparse_model = SparseTextEmbedding(sparse_model_name)
        # self.dense_model = TextEmbedding(dense_model)
        self.dense_model = SentenceTransformer(dense_model_name)
        self.late_interaction_model = LateInteractionTextEmbedding(late_interaction_model_name)
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.late_interaction_model_name = late_interaction_model_name
        self._create_collection_if_not_exists()
        
    def _create_collection_if_not_exists(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    ),
                    "late_interaction": VectorParams(
                        size=LateInteractionTextEmbedding.get_embedding_size(self.late_interaction_model_name),
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=HnswConfigDiff(m=0)
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            Logger.log(f"Collection created: {self.collection_name}")
            return
        
    def add_documents(self, documents):
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=documents
            )
        except Exception as e:
            Logger.log(f"Error adding documents: {e}")
            raise
        
    def delete_collection(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            Logger.log(f"Error deleting collection: {e}")
            raise
    
    def get_info(self):
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            Logger.log(f"Error mendapatkan info collection: {e}")
            return {}
        
    def dense_search(self, query: str, limit: int = 5) -> list[ScoredPoint]:
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=self.dense_model.encode(query),
                limit=limit,
                using="dense"
            )
            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def sparse_search(self, query: str, limit: int = 5) -> list[ScoredPoint]:
        try:
            sparse_vecs = next(self.sparse_model.embed(query))
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(**sparse_vecs.as_object()),
                limit=limit,
                using="sparse"
            )
            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def hybrid_search(self, query: str, limit: int = 5) -> list[ScoredPoint]:
        try:
            sparse_vecs = next(self.sparse_model.embed(query))

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=SparseVector(**sparse_vecs.as_object()),
                        using="sparse",
                        limit=limit
                    ),
                    Prefetch(
                        query=self.dense_model.encode(query),
                        using="dense",
                        limit=limit
                    )
                ],
                limit=limit,
                query= FusionQuery(fusion=Fusion.RRF)
            )
            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def hybrid_search_with_colbert(self, query: str, limit: int = 5) -> list[ScoredPoint]:
        try:
            sparse_vecs = next(self.sparse_model.embed(query))
            rrf_prefetch = Prefetch(
                prefetch=[
                    Prefetch(
                        query=SparseVector(**sparse_vecs.as_object()),
                        using="sparse",
                        limit=limit
                    ),
                    Prefetch(
                        query=self.dense_model.encode(query),
                        using="dense",
                        limit=limit
                    )
                ],
                limit=limit,
                query= FusionQuery(fusion=Fusion.RRF)
            )
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    rrf_prefetch
                ],
                limit=limit,
                query= next(self.late_interaction_model.embed(query)),
                using="late_interaction"
            )

            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
        
    def store_chunks(self, chunks : Dict[str, BaseChunk]):
        points = []
        dense_embeddings = list(self.dense_model.encode([chunk.get_context() for chunk in chunks.values()]))
        sparse_embeddings = list(self.sparse_model.embed([chunk.get_context() for chunk in chunks.values()]))
        late_interaction_embeddings = list(self.late_interaction_model.embed([chunk.get_context() for chunk in chunks.values()]))
        keys = list(chunks.keys())
        values = list(chunks.values())
        
        for dense_vec, sparse_vec, late_interaction_vec, key, value in zip(dense_embeddings, sparse_embeddings, late_interaction_embeddings, keys, values):
            point = PointStruct(
                id = str(key),
                vector={
                    "dense" : dense_vec,
                    "sparse" : sparse_vec.as_object(),
                    "late_interaction" : late_interaction_vec
                },
                payload=value.get_payload()
            )
            points.append(point)

        try:
            self.add_documents(points)
        except Exception as e:
            Logger.log(f"Error adding documents: {e}")
            raise
        
            
        
    def close(self):
        self.client.close()
    
        

        
        