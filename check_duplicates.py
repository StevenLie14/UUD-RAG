"""
Check for duplicate chunks in semantic_cache.json
"""
import json
from collections import Counter

def check_duplicates():
    cache_path = "./chunk_cache/semantic_cache.json"
    
    print("Loading cache file...")
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    processed_docs = data.get('processed_docs', [])
    
    print(f"\n{'='*70}")
    print(f"CACHE STATISTICS")
    print(f"{'='*70}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Processed documents (hashes): {len(processed_docs)}")
    print(f"Unique document hashes: {len(set(processed_docs))}")
    
    # Check for duplicate chunk IDs
    chunk_ids = [chunk['id'] for chunk in chunks]
    duplicate_ids = [id for id, count in Counter(chunk_ids).items() if count > 1]
    
    print(f"\n{'='*70}")
    print(f"DUPLICATE CHECK - Chunk IDs")
    print(f"{'='*70}")
    print(f"Total chunk IDs: {len(chunk_ids)}")
    print(f"Unique chunk IDs: {len(set(chunk_ids))}")
    print(f"Duplicate chunk IDs: {len(duplicate_ids)}")
    
    if duplicate_ids:
        print(f"\nWARNING: Found {len(duplicate_ids)} duplicate chunk IDs!")
        print("First 5 duplicates:")
        for id in duplicate_ids[:5]:
            count = chunk_ids.count(id)
            print(f"  - {id}: appears {count} times")
    else:
        print("✓ No duplicate chunk IDs found")
    
    # Check for duplicate document hashes
    duplicate_hashes = [h for h, count in Counter(processed_docs).items() if count > 1]
    
    print(f"\n{'='*70}")
    print(f"DUPLICATE CHECK - Document Hashes")
    print(f"{'='*70}")
    print(f"Total document hashes: {len(processed_docs)}")
    print(f"Unique document hashes: {len(set(processed_docs))}")
    print(f"Duplicate document hashes: {len(duplicate_hashes)}")
    
    if duplicate_hashes:
        print(f"\nWARNING: Found {len(duplicate_hashes)} duplicate document hashes!")
        print("First 5 duplicates:")
        for h in duplicate_hashes[:5]:
            count = processed_docs.count(h)
            print(f"  - {h[:16]}...: appears {count} times")
    else:
        print("✓ No duplicate document hashes found")
    
    # Check for duplicate content
    print(f"\n{'='*70}")
    print(f"DUPLICATE CHECK - Chunk Content")
    print(f"{'='*70}")
    
    content_counter = Counter([chunk['content'][:100] for chunk in chunks])  # First 100 chars
    duplicate_content = [content for content, count in content_counter.items() if count > 1]
    
    print(f"Chunks with duplicate content (first 100 chars): {len(duplicate_content)}")
    
    if duplicate_content:
        print("\nFirst 3 examples of duplicate content:")
        for i, content in enumerate(duplicate_content[:3], 1):
            count = content_counter[content]
            print(f"\n{i}. Appears {count} times:")
            print(f"   {content[:80]}...")
    
    # Check source distribution
    print(f"\n{'='*70}")
    print(f"SOURCE DISTRIBUTION")
    print(f"{'='*70}")
    
    sources = [chunk.get('source', 'unknown') for chunk in chunks]
    source_counts = Counter(sources)
    
    print(f"Total unique sources: {len(source_counts)}")
    print("\nTop 10 sources by chunk count:")
    for source, count in source_counts.most_common(10):
        if source:
            source_name = source.split('/')[-1] if '/' in source else source.split('\\')[-1] if '\\' in source else source
            print(f"  {source_name}: {count} chunks")

if __name__ == "__main__":
    check_duplicates()
