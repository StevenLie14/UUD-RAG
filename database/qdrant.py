from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import Logger
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector,Document, ScoredPoint, Prefetch, FusionQuery, Fusion, MultiVectorConfig, MultiVectorComparator, HnswConfigDiff
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from sentence_transformers import SentenceTransformer
from model import BaseChunk
from typing import Dict
from database.base import VectorStore, DenseSearchable, SparseSearchable, HybridSearchable, ColbertSearchable, CrossEncoderSearchable
import json
import os

class Qdrant(VectorStore, DenseSearchable, SparseSearchable, HybridSearchable, ColbertSearchable, CrossEncoderSearchable):
    def __init__(self, qdrant_url : str = "http://localhost:6333", qdrant_api_key: str = None, collection_name: str = "documents", late_interaction_model_name: str = "jinaai/jina-colbert-v2", sparse_model_name: str = "Qdrant/bm25", dense_model_name: str = "LazarusNLP/all-indo-e5-small-v4", reranker_model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
        Logger.log("qdrant url" + qdrant_url)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        
        self.sparse_model = SparseTextEmbedding(sparse_model_name)
        self.dense_model = SentenceTransformer(dense_model_name)
        self.late_interaction_model = LateInteractionTextEmbedding(late_interaction_model_name)
        self.reranker = TextCrossEncoder(reranker_model_name)
        
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.late_interaction_model_name = late_interaction_model_name
        self.reranker_model_name = reranker_model_name
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
        
    def hybrid_search(self, query: str, limit: int = 5, multiplier: int = 3) -> list[ScoredPoint]:
        initial_limit = limit * multiplier
        try:
            sparse_vecs = next(self.sparse_model.embed(query))

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=SparseVector(**sparse_vecs.as_object()),
                        using="sparse",
                        limit=initial_limit
                    ),
                    Prefetch(
                        query=self.dense_model.encode(query),
                        using="dense",
                        limit=initial_limit
                    )
                ],
                limit=limit,
                query= FusionQuery(fusion=Fusion.RRF)
            )
            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def hybrid_search_with_colbert(self, query: str, limit: int = 5,initial_limit_multiplier: int = 5) -> list[ScoredPoint]:
        initial_limit = limit * initial_limit_multiplier

        try:
            sparse_vecs = next(self.sparse_model.embed(query))
            
            rrf_prefetch = Prefetch(
                prefetch=[
                    Prefetch(
                        query=SparseVector(**sparse_vecs.as_object()),
                        using="sparse",
                        limit=initial_limit
                    ),
                    Prefetch(
                        query=self.dense_model.encode(query),
                        using="dense",
                        limit=initial_limit
                    )
                ],
                limit=initial_limit,
                query=FusionQuery(fusion=Fusion.RRF)
            )

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    rrf_prefetch
                ],
                limit=limit, 
                query=next(self.late_interaction_model.embed(query)),
                using="late_interaction"
            )

            return search_result.points
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def hybrid_search_with_crossencoder(self, query: str, limit: int = 5, initial_limit_multiplier: int = 5) -> list[ScoredPoint]:
        try:
            initial_limit = limit * initial_limit_multiplier
            initial_results = self.hybrid_search(query, limit=initial_limit)
            
            if not initial_results:
                return []
            
            documents = [hit.payload.get('full_text', '') for hit in initial_results]
            rerank_scores = list(self.reranker.rerank(query, documents))
            
            ranking = [(i, score) for i, score in enumerate(rerank_scores)]
            ranking.sort(key=lambda x: x[1], reverse=True)
            
            reranked_results = []
            for i in range(min(limit, len(ranking))):
                idx, new_score = ranking[i]
                result = initial_results[idx]
                result.score = float(new_score)
                reranked_results.append(result)
            
            Logger.log(f"Reranked {len(initial_results)} results to top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            Logger.log(f"Error in hybrid search with reranking: {e}")
            return []
        
        
    def store_chunks(self, chunks: Dict[str, BaseChunk], batch_size: int = 8, resume: bool = True):
        keys = list(chunks.keys())
        total = len(keys)
        progress_file = f".qdrant_progress_{self.collection_name}.json"
        
        completed_batches = set()
        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    completed_batches = set(progress.get('completed_batches', []))
                    if completed_batches:
                        Logger.log(f"Resuming: {len(completed_batches)} batches already completed")
            except Exception as e:
                Logger.log(f"Could not load progress file, starting fresh: {e}")
        
        Logger.log(f"Storing {total} chunks to Qdrant in batches of {batch_size}")
        
        batch_num = 0
        for start in range(0, total, batch_size):
            if batch_num in completed_batches:
                batch_num += 1
                continue
            
            end = min(start + batch_size, total)
            batch_keys = keys[start:end]
            batch_values = [chunks[k] for k in batch_keys]
            contexts = [chunk.get_context() for chunk in batch_values]

            dense_embeddings = list(self.dense_model.encode(contexts, batch_size=min(batch_size, len(contexts))))
            sparse_embeddings = list(self.sparse_model.embed(contexts))
            late_interaction_embeddings = list(self.late_interaction_model.embed(contexts))

            points = []
            for dense_vec, sparse_vec, late_interaction_vec, key, value in zip(
                dense_embeddings, sparse_embeddings, late_interaction_embeddings, batch_keys, batch_values
            ):
                point = PointStruct(
                    id=str(key),
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec.as_object(),
                        "late_interaction": late_interaction_vec
                    },
                    payload=value.get_payload()
                )
                points.append(point)

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                completed_batches.add(batch_num)
                
                with open(progress_file, 'w') as f:
                    json.dump({'completed_batches': list(completed_batches)}, f)
                
                Logger.log(f"✓ Stored batch {batch_num} ({start}-{end}, {len(points)} chunks)")
            except Exception as e:
                Logger.log(f"Error adding documents for batch {batch_num} ({start}-{end}): {e}")
                Logger.log(f"Progress saved to {progress_file}. Run again to resume.")
                raise
            
            batch_num += 1
        
        if os.path.exists(progress_file):
            os.remove(progress_file)
        Logger.log(f"✓ All chunks stored successfully!")
        
            
        
    def close(self):
        self.client.close()
    
        

        
        