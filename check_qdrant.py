"""
Check Qdrant collection for duplicates and missing chunks
Compares chunks in cache files against points stored in Qdrant
"""
import json
import os
from qdrant_client import QdrantClient
from collections import Counter
from typing import Set, Dict, List
from logger import Logger

class QdrantChecker:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
    def get_all_point_ids(self) -> Set[str]:
        """Get all point IDs from Qdrant collection"""
        try:
            # Get collection info to check size
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            Logger.log(f"Total points in Qdrant collection '{self.collection_name}': {total_points}")
            
            # Scroll through all points to get IDs
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
            
            Logger.log(f"✓ Retrieved {len(point_ids)} unique point IDs from Qdrant")
            return point_ids
            
        except Exception as e:
            Logger.log(f"Error fetching points from Qdrant: {e}")
            return set()
    
    def get_cache_chunk_ids(self, cache_dir: str = "./chunk_cache") -> Dict[str, Set[str]]:
        """Get all chunk IDs from cache files"""
        cache_files = {
            "recursive": os.path.join(cache_dir, "recursive_cache.json"),
            "semantic": os.path.join(cache_dir, "semantic_cache.json"),
            "agentic_v2": os.path.join(cache_dir, "agentic_v2_cache.json")
        }
        
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
                Logger.log(f"✓ Loaded {len(chunk_ids)} chunk IDs from {chunker_name} cache")
                
            except Exception as e:
                Logger.log(f"Error reading cache file {cache_path}: {e}")
        
        return chunk_ids_by_chunker
    
    def check_duplicates_in_qdrant(self, point_ids: Set[str]):
        """Check for duplicate point IDs in Qdrant (shouldn't happen but good to verify)"""
        Logger.log(f"\n{'='*70}")
        Logger.log(f"CHECKING DUPLICATES IN QDRANT")
        Logger.log(f"{'='*70}")
        
        # This is actually redundant since we use a set, but let's verify
        # by fetching again and counting
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
            
            if duplicates:
                Logger.log(f"⚠ WARNING: Found {len(duplicates)} duplicate point IDs in Qdrant!")
                Logger.log("First 5 duplicates:")
                for id, count in list(duplicates.items())[:5]:
                    Logger.log(f"  - {id}: appears {count} times")
            else:
                Logger.log("✓ No duplicate point IDs found in Qdrant")
                
        except Exception as e:
            Logger.log(f"Error checking duplicates: {e}")
    
    def compare_cache_to_qdrant(self, chunker_name: str, cache_chunk_ids: Set[str], qdrant_point_ids: Set[str]):
        """Compare cache chunks against Qdrant points"""
        Logger.log(f"\n{'='*70}")
        Logger.log(f"COMPARISON: {chunker_name.upper()} CACHE vs QDRANT")
        Logger.log(f"{'='*70}")
        
        # Chunks in cache
        Logger.log(f"Chunks in {chunker_name} cache: {len(cache_chunk_ids)}")
        Logger.log(f"Points in Qdrant: {len(qdrant_point_ids)}")
        
        # Missing from Qdrant (in cache but not in Qdrant)
        missing_in_qdrant = cache_chunk_ids - qdrant_point_ids
        
        # Extra in Qdrant (in Qdrant but not in cache)
        extra_in_qdrant = qdrant_point_ids - cache_chunk_ids
        
        # In both
        in_both = cache_chunk_ids & qdrant_point_ids
        
        Logger.log(f"\n{'─'*70}")
        Logger.log(f"RESULTS")
        Logger.log(f"{'─'*70}")
        Logger.log(f"✓ Chunks in both cache and Qdrant: {len(in_both)}")
        Logger.log(f"⚠ Missing from Qdrant (in cache only): {len(missing_in_qdrant)}")
        Logger.log(f"⚠ Extra in Qdrant (not in cache): {len(extra_in_qdrant)}")
        
        if missing_in_qdrant:
            Logger.log(f"\n⚠ WARNING: {len(missing_in_qdrant)} chunks are in cache but missing from Qdrant!")
            Logger.log("First 10 missing chunk IDs:")
            for chunk_id in list(missing_in_qdrant)[:10]:
                Logger.log(f"  - {chunk_id}")
        
        if extra_in_qdrant:
            Logger.log(f"\n⚠ WARNING: {len(extra_in_qdrant)} points in Qdrant are not in cache!")
            Logger.log("First 10 extra point IDs:")
            for point_id in list(extra_in_qdrant)[:10]:
                Logger.log(f"  - {point_id}")
        
        # Calculate sync percentage
        if len(cache_chunk_ids) > 0:
            sync_percentage = (len(in_both) / len(cache_chunk_ids)) * 100
            Logger.log(f"\nSync rate: {sync_percentage:.2f}% of cache chunks are in Qdrant")
        
        return {
            "chunker": chunker_name,
            "cache_total": len(cache_chunk_ids),
            "qdrant_total": len(qdrant_point_ids),
            "in_both": len(in_both),
            "missing_from_qdrant": len(missing_in_qdrant),
            "extra_in_qdrant": len(extra_in_qdrant),
            "missing_ids": list(missing_in_qdrant)[:100],  # Save first 100
            "extra_ids": list(extra_in_qdrant)[:100]
        }
    
    def run_full_check(self, cache_dir: str = "./chunk_cache", save_report: bool = True):
        """Run complete check and generate report"""
        Logger.log(f"\n{'='*70}")
        Logger.log(f"QDRANT COLLECTION CHECK")
        Logger.log(f"{'='*70}")
        Logger.log(f"Collection: {self.collection_name}")
        Logger.log(f"Cache directory: {cache_dir}")
        
        # Get all point IDs from Qdrant
        Logger.log(f"\n{'='*70}")
        Logger.log(f"FETCHING DATA FROM QDRANT")
        Logger.log(f"{'='*70}")
        qdrant_point_ids = self.get_all_point_ids()
        
        if not qdrant_point_ids:
            Logger.log("⚠ No points found in Qdrant or collection doesn't exist!")
            return
        
        # Check for duplicates in Qdrant
        self.check_duplicates_in_qdrant(qdrant_point_ids)
        
        # Get chunk IDs from cache files
        Logger.log(f"\n{'='*70}")
        Logger.log(f"LOADING CACHE FILES")
        Logger.log(f"{'='*70}")
        chunk_ids_by_chunker = self.get_cache_chunk_ids(cache_dir)
        
        if not chunk_ids_by_chunker:
            Logger.log("⚠ No cache files found!")
            return
        
        # Compare each chunker's cache against Qdrant
        results = []
        for chunker_name, cache_chunk_ids in chunk_ids_by_chunker.items():
            result = self.compare_cache_to_qdrant(chunker_name, cache_chunk_ids, qdrant_point_ids)
            results.append(result)
        
        # Summary
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
            Logger.log(f"\n⚠ ACTION REQUIRED: Re-run store_chunks for missing data!")
        else:
            Logger.log(f"\n✓ All chunks are properly stored in Qdrant!")
        
        # Save report
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
    parser.add_argument("--no-report", action="store_true", help="Don't save report file")
    
    args = parser.parse_args()
    
    checker = QdrantChecker(qdrant_url=args.url, collection_name=args.collection)
    checker.run_full_check(cache_dir=args.cache_dir, save_report=not args.no_report)

if __name__ == "__main__":
    main()
