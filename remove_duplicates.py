"""
Remove duplicate chunks with the same source from semantic_cache.json
"""
import json
import shutil
from datetime import datetime

def remove_duplicates():
    cache_path = "./chunk_cache/semantic_cache.json"
    backup_path = f"./chunk_cache/semantic_cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print("Loading cache file...")
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    processed_docs = data.get('processed_docs', [])
    
    print(f"\nOriginal statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Processed document hashes: {len(processed_docs)}")
    print(f"  Unique document hashes: {len(set(processed_docs))}")
    
    # Create backup
    print(f"\nCreating backup: {backup_path}")
    shutil.copy(cache_path, backup_path)
    
    # Remove duplicate document hashes (keep only unique)
    unique_doc_hashes = list(set(processed_docs))
    removed_doc_hashes = len(processed_docs) - len(unique_doc_hashes)
    
    # Remove duplicate chunks based on content + source
    seen = set()
    unique_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        # Create a unique key based on content (first 200 chars) and source
        content_key = chunk.get('content', '')[:200]
        source = chunk.get('source', 'unknown')
        unique_key = f"{source}::{content_key}"
        
        if unique_key not in seen:
            seen.add(unique_key)
            unique_chunks.append(chunk)
        else:
            duplicate_count += 1
    
    print(f"\nRemoval results:")
    print(f"  Duplicate document hashes removed: {removed_doc_hashes}")
    print(f"  Duplicate chunks removed (same source + content): {duplicate_count}")
    print(f"  Chunks remaining: {len(unique_chunks)}")
    
    # Update data
    data['chunks'] = unique_chunks
    data['processed_docs'] = unique_doc_hashes
    data['total_chunks'] = len(unique_chunks)
    data['last_updated'] = datetime.now().isoformat()
    
    # Save cleaned cache
    print(f"\nSaving cleaned cache to: {cache_path}")
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*70}")
    print(f"Original chunks: {len(chunks)}")
    print(f"Cleaned chunks: {len(unique_chunks)}")
    print(f"Removed: {len(chunks) - len(unique_chunks)} duplicate chunks")
    print(f"\nBackup saved to: {backup_path}")
    print(f"Cleaned cache saved to: {cache_path}")

if __name__ == "__main__":
    try:
        remove_duplicates()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
