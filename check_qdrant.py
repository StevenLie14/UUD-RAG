"""
Check Qdrant collection for duplicates and missing chunks
Compares chunks in cache files against points stored in Qdrant
"""
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from collections import Counter
from typing import Set, Dict, List
from logger import Logger
from database.qdrant import Qdrant
from model.chunk.semantic_chunk import SemanticChunk
from model.chunk.recursive_chunk import RecursiveChunk
from model.chunk.agentic_chunk import AgenticChunk
from model.chunk.simple_chunk import SimpleChunk

class QdrantChecker:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents", qdrant_api_key: str | None = None):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        
    def get_all_point_ids(self) -> Set[str]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            Logger.log(f"Total points in Qdrant collection '{self.collection_name}': {total_points}")
            
            point_ids = set()
            offset = None
            batch_size = 100
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                for point in points:
                    point_ids.add(str(point.id))
                
                if next_offset is None:
                    break
                    
                offset = next_offset
                Logger.log(f"  Fetched {len(point_ids)} point IDs...")
            
            Logger.log(f"Retrieved {len(point_ids)} unique point IDs from Qdrant")
            return point_ids
            
        except Exception as e:
            Logger.log(f"Error fetching points from Qdrant: {e}")
            return set()
    
    def _build_cache_file_map(self, cache_dir: str, cache_file_args: List[str] | None) -> Dict[str, str]:
        if cache_file_args:
            cache_files: Dict[str, str] = {}
            for raw_path in cache_file_args:
                cache_path = raw_path if os.path.isabs(raw_path) else os.path.join(cache_dir, raw_path)
                chunker_name = os.path.splitext(os.path.basename(cache_path))[0]
                cache_files[chunker_name] = cache_path
            return cache_files
        return {
            "recursive": os.path.join(cache_dir, "recursive_cache.json"),
            "semantic": os.path.join(cache_dir, "semantic_cache.json"),
            "agentic_v2": os.path.join(cache_dir, "agentic_v2_cache.json")
        }

    def _reconstruct_chunk(self, chunk_dict: dict, chunk_type: str):
        type_map = {
            "semantic": SemanticChunk,
            "recursive": RecursiveChunk,
            "agentic": AgenticChunk,
            "agentic_v2": SimpleChunk,
            "simple": SimpleChunk,
        }
        cls = type_map.get(chunk_type)
        if not cls:
            Logger.log(f"Unknown chunk type '{chunk_type}', skipping reconstruction")
            return None
        try:
            return cls(**chunk_dict)
        except Exception as e:
            Logger.log(f"Failed to reconstruct chunk of type {chunk_type}: {e}")
            return None

    def _load_missing_chunks_from_cache(self, cache_path: str, chunk_type_hint: str, missing_ids: Set[str]) -> Dict[str, object]:
        loaded: Dict[str, object] = {}
        if not missing_ids:
            return loaded
        if not os.path.exists(cache_path):
            Logger.log(f"Cache file not found when loading missing chunks: {cache_path}")
            return loaded
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunk_type = data.get('chunk_type', chunk_type_hint)
            chunks = data.get('chunks', [])
            for chunk_dict in chunks:
                cid = str(chunk_dict.get('id'))
                if cid not in missing_ids:
                    continue
                chunk_obj = self._reconstruct_chunk(chunk_dict)
                if chunk_obj:
                    loaded[cid] = chunk_obj
        except Exception as e:
            Logger.log(f"Error loading missing chunks from {cache_path}: {e}")
        return loaded

    def _insert_missing_chunks(self, cache_files_map: Dict[str, str], missing_by_chunker: Dict[str, Set[str]]):
        try:
            qdrant_db = Qdrant(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name
            )
        except Exception as e:
            Logger.log(f"Failed to initialize Qdrant helper for insertion: {e}")
            return

        total_loaded = 0
        for chunker_name, missing_ids in missing_by_chunker.items():
            if not missing_ids:
                continue
            cache_path = cache_files_map.get(chunker_name)
            if not cache_path:
                Logger.log(f"No cache path found for chunker {chunker_name}, skipping insertion")
                continue

            Logger.log(f"\nInserting {len(missing_ids)} missing chunks for '{chunker_name}' from {cache_path}")
            chunks_dict = self._load_missing_chunks_from_cache(cache_path, chunker_name, missing_ids)
            if not chunks_dict:
                Logger.log(f"No chunks loaded for {chunker_name}; nothing to insert")
                continue

            try:
                qdrant_db.store_chunks(chunks_dict, resume=False)
                total_loaded += len(chunks_dict)
                Logger.log(f"Inserted {len(chunks_dict)} chunks for {chunker_name}")
            except Exception as e:
                Logger.log(f"Error inserting chunks for {chunker_name}: {e}")

        try:
            qdrant_db.close()
        except Exception:
            pass

        if total_loaded == 0:
            Logger.log("No missing chunks were inserted.")
        else:
            Logger.log(f"\nInserted {total_loaded} missing chunks in total.")

    def _delete_duplicate_points(self, duplicate_ids: Set[str]):
        if not duplicate_ids:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=list(duplicate_ids))
            )
            Logger.log(f"Deleted {len(duplicate_ids)} duplicate point IDs")
        except Exception as e:
            Logger.log(f"Error deleting duplicate points: {e}")

    def _delete_extra_points(self, extra_ids: Set[str]):
        if not extra_ids:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=list(extra_ids))
            )
            Logger.log(f"Deleted {len(extra_ids)} extra point IDs (not in cache)")
        except Exception as e:
            Logger.log(f"Error deleting extra points: {e}")

    def get_cache_chunk_ids(self, cache_dir: str = "./chunk_cache", cache_file_args: List[str] | None = None, cache_files_map: Dict[str, str] | None = None) -> Dict[str, Set[str]]:
        cache_files = cache_files_map or self._build_cache_file_map(cache_dir, cache_file_args)
        
        chunk_ids_by_chunker = {}
        
        for chunker_name, cache_path in cache_files.items():
            if not os.path.exists(cache_path):
                Logger.log(f"Cache file not found: {cache_path}")
                continue
            
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get('chunks', [])
                chunk_ids = {chunk['id'] for chunk in chunks}
                chunk_ids_by_chunker[chunker_name] = chunk_ids
                Logger.log(f"Loaded {len(chunk_ids)} chunk IDs from {chunker_name} cache")
                
            except Exception as e:
                Logger.log(f"Error reading cache file {cache_path}: {e}")
        
        return chunk_ids_by_chunker
    
    def check_duplicates_in_qdrant(self, point_ids: Set[str], delete_duplicates: bool = False, cache_files_map: Dict[str, str] | None = None) -> Set[str]:
        Logger.log(f"\n{'='*70}")
        Logger.log(f"CHECKING DUPLICATES IN QDRANT")
        Logger.log(f"{'='*70}")
        
        try:
            all_ids = []
            offset = None
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                for point in points:
                    all_ids.append(str(point.id))
                
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            duplicate_counts = Counter(all_ids)
            duplicates = {id: count for id, count in duplicate_counts.items() if count > 1}
            duplicate_ids = set(duplicates.keys())
            
            if duplicates:
                Logger.log(f"WARNING: Found {len(duplicates)} duplicate point IDs in Qdrant!")
                Logger.log("First 5 duplicates:")
                for id, count in list(duplicates.items())[:5]:
                    Logger.log(f"  - {id}: appears {count} times")

                if delete_duplicates:
                    self._delete_duplicate_points(duplicate_ids)
            else:
                Logger.log("No duplicate point IDs found in Qdrant")
            
            return duplicate_ids
                
        except Exception as e:
            Logger.log(f"Error checking duplicates: {e}")
            return set()
    
    def compare_cache_to_qdrant(self, chunker_name: str, cache_chunk_ids: Set[str], qdrant_point_ids: Set[str]):
        Logger.log(f"\n{'='*70}")
        Logger.log(f"COMPARISON: {chunker_name.upper()} CACHE vs QDRANT")
        Logger.log(f"{'='*70}")
        
        Logger.log(f"Chunks in {chunker_name} cache: {len(cache_chunk_ids)}")
        Logger.log(f"Points in Qdrant: {len(qdrant_point_ids)}")
        
        missing_in_qdrant = cache_chunk_ids - qdrant_point_ids
        
        extra_in_qdrant = qdrant_point_ids - cache_chunk_ids
        
        in_both = cache_chunk_ids & qdrant_point_ids
        
        Logger.log(f"\n{'─'*70}")
        Logger.log(f"RESULTS")
        Logger.log(f"{'─'*70}")
        Logger.log(f"Chunks in both cache and Qdrant: {len(in_both)}")
        Logger.log(f"Missing from Qdrant (in cache only): {len(missing_in_qdrant)}")
        Logger.log(f"Extra in Qdrant (not in cache): {len(extra_in_qdrant)}")
        
        if missing_in_qdrant:
            Logger.log(f"\nWARNING: {len(missing_in_qdrant)} chunks are in cache but missing from Qdrant!")
            Logger.log("First 10 missing chunk IDs:")
            for chunk_id in list(missing_in_qdrant)[:10]:
                Logger.log(f"  - {chunk_id}")
        
        if extra_in_qdrant:
            Logger.log(f"\nWARNING: {len(extra_in_qdrant)} points in Qdrant are not in cache!")
            Logger.log("First 10 extra point IDs:")
            for point_id in list(extra_in_qdrant)[:10]:
                Logger.log(f"  - {point_id}")
        
        if len(cache_chunk_ids) > 0:
            sync_percentage = (len(in_both) / len(cache_chunk_ids)) * 100
            Logger.log(f"\nSync rate: {sync_percentage:.2f}% of cache chunks are in Qdrant")
        
        stats = {
            "chunker": chunker_name,
            "cache_total": len(cache_chunk_ids),
            "qdrant_total": len(qdrant_point_ids),
            "in_both": len(in_both),
            "missing_from_qdrant": len(missing_in_qdrant),
            "extra_in_qdrant": len(extra_in_qdrant),
            "missing_ids": list(missing_in_qdrant)[:100],  # Save first 100
            "extra_ids": list(extra_in_qdrant)[:100]
        }
        return stats, missing_in_qdrant, extra_in_qdrant
    
    def run_full_check(self, cache_dir: str = "./chunk_cache", save_report: bool = True, cache_files: List[str] | None = None, insert_missing: bool = False, delete_duplicates: bool = False, delete_extra: bool = False):
        Logger.log(f"\n{'='*70}")
        Logger.log(f"QDRANT COLLECTION CHECK")
        Logger.log(f"{'='*70}")
        Logger.log(f"Collection: {self.collection_name}")
        Logger.log(f"Cache directory: {cache_dir}")
        if cache_files:
            Logger.log(f"Cache files: {', '.join(cache_files)}")
        if insert_missing:
            Logger.log("Insert-missing mode: ON (will upsert missing cache chunks into Qdrant)")
        if delete_duplicates:
            Logger.log("Delete-duplicates mode: ON (will delete duplicate point IDs)")
        if delete_extra:
            Logger.log("Delete-extra mode: ON (will delete points not present in cache)")

        cache_files_map = self._build_cache_file_map(cache_dir, cache_files)
        
        Logger.log(f"\n{'='*70}")
        Logger.log(f"FETCHING DATA FROM QDRANT")
        Logger.log(f"{'='*70}")
        qdrant_point_ids = self.get_all_point_ids()
        
        if not qdrant_point_ids:
            Logger.log("No points found in Qdrant or collection doesn't exist!")
            return
        
        duplicate_ids = self.check_duplicates_in_qdrant(qdrant_point_ids, delete_duplicates=delete_duplicates, cache_files_map=cache_files_map)
        if delete_duplicates and duplicate_ids:
            qdrant_point_ids = qdrant_point_ids - duplicate_ids
        
        Logger.log(f"\n{'='*70}")
        Logger.log(f"LOADING CACHE FILES")
        Logger.log(f"{'='*70}")
        chunk_ids_by_chunker = self.get_cache_chunk_ids(cache_dir, cache_files, cache_files_map)
        
        if not chunk_ids_by_chunker:
            Logger.log("No cache files found!")
            return
        
        results = []
        missing_by_chunker: Dict[str, Set[str]] = {}
        extra_by_chunker: Dict[str, Set[str]] = {}
        for chunker_name, cache_chunk_ids in chunk_ids_by_chunker.items():
            stats, missing_set, extra_set = self.compare_cache_to_qdrant(chunker_name, cache_chunk_ids, qdrant_point_ids)
            results.append(stats)
            missing_by_chunker[chunker_name] = missing_set
            extra_by_chunker[chunker_name] = extra_set

        if delete_extra:
            extra_union = set().union(*extra_by_chunker.values()) if extra_by_chunker else set()
            if extra_union:
                self._delete_extra_points(extra_union)
                qdrant_point_ids = qdrant_point_ids - extra_union
            else:
                Logger.log("No extra points to delete.")

        if insert_missing:
            self._insert_missing_chunks(cache_files_map, missing_by_chunker)
        
        Logger.log(f"\n{'='*70}")
        Logger.log(f"SUMMARY")
        Logger.log(f"{'='*70}")
        
        total_cache_chunks = sum(r['cache_total'] for r in results)
        total_missing = sum(r['missing_from_qdrant'] for r in results)
        total_in_both = sum(r['in_both'] for r in results)
        
        Logger.log(f"Total chunks in all caches: {total_cache_chunks}")
        Logger.log(f"Total points in Qdrant: {len(qdrant_point_ids)}")
        Logger.log(f"Total chunks successfully stored: {total_in_both}")
        Logger.log(f"Total chunks missing from Qdrant: {total_missing}")
        
        if total_missing > 0:
            Logger.log(f"\nACTION REQUIRED: Re-run store_chunks for missing data!")
        else:
            Logger.log(f"\nAll chunks are properly stored in Qdrant!")
        
        if save_report:
            report_path = "qdrant_check_report.json"
            report = {
                "collection_name": self.collection_name,
                "qdrant_total_points": len(qdrant_point_ids),
                "cache_total_chunks": total_cache_chunks,
                "results_by_chunker": results,
                "summary": {
                    "total_in_both": total_in_both,
                    "total_missing": total_missing,
                    "sync_complete": total_missing == 0
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            Logger.log(f"\n✓ Report saved to: {report_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Qdrant collection for duplicates and missing chunks")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="documents", help="Collection name")
    parser.add_argument("--cache-dir", default="./chunk_cache", help="Cache directory")
    parser.add_argument("--cache-file", action="append", help="Cache file name or path; repeat to check multiple files")
    parser.add_argument("--api-key", help="Qdrant API key (optional)")
    parser.add_argument("--insert-missing", action="store_true", help="Insert missing cache chunks into Qdrant")
    parser.add_argument("--delete-duplicates", action="store_true", help="Delete duplicate point IDs in Qdrant")
    parser.add_argument("--delete-extra", action="store_true", help="Delete points present in Qdrant but not in cache")
    parser.add_argument("--no-report", action="store_true", help="Don't save report file")
    
    args = parser.parse_args()
    
    checker = QdrantChecker(qdrant_url=args.url, collection_name=args.collection, qdrant_api_key=args.api_key)
    checker.run_full_check(
        cache_dir=args.cache_dir,
        save_report=not args.no_report,
        cache_files=args.cache_file,
        insert_missing=args.insert_missing,
        delete_duplicates=args.delete_duplicates,
        delete_extra=args.delete_extra,
    )

if __name__ == "__main__":
    main()
