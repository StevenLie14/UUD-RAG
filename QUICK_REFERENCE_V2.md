# Quick Reference: AgenticChunkerV2 dengan Source & Page Info

## Model SimpleChunk

```python
chunk.id          # UUID chunk
chunk.content     # Teks konten chunk
chunk.index       # Index chunk (urutan pembuatan)
chunk.source      # Nama file sumber (e.g., "peraturan_pdfs/UU_No_13.pdf")
chunk.page        # Nomor halaman (e.g., 5)
chunk.metadata    # Dict metadata lengkap dari document loader
```

## Methods

```python
# Get context dengan format lengkap (termasuk source & page)
context = chunk.get_context()
# Output:
# Sumber: peraturan_pdfs/UU_No_13.pdf
# Halaman: 5
#
# [content text...]

# Get payload untuk storage
payload = chunk.get_payload()
# Returns dict dengan semua fields
```

## Common Use Cases

### 1. Filter by Source
```python
chunks_from_file = [
    chunk for chunk in chunker.chunks.values()
    if chunk.source == "peraturan_pdfs/UU_No_13.pdf"
]
```

### 2. Filter by Page
```python
chunks_page_5 = [
    chunk for chunk in chunker.chunks.values()
    if chunk.page == 5
]
```

### 3. Filter by Source Pattern
```python
uu_chunks = [
    chunk for chunk in chunker.chunks.values()
    if chunk.source and "UU" in chunk.source
]
```

### 4. Search with Source Info
```python
search_term = "pekerja"
results = []
for chunk in chunker.chunks.values():
    if search_term.lower() in chunk.content.lower():
        results.append({
            "source": chunk.source,
            "page": chunk.page,
            "content": chunk.content,
            "preview": chunk.content[:200]
        })
```

### 5. Group by Source
```python
from collections import defaultdict

by_source = defaultdict(list)
for chunk in chunker.chunks.values():
    if chunk.source:
        by_source[chunk.source].append(chunk)

for source, chunks in by_source.items():
    print(f"{source}: {len(chunks)} chunks")
```

### 6. Get Page Range per Source
```python
source_pages = defaultdict(set)
for chunk in chunker.chunks.values():
    if chunk.source and chunk.page is not None:
        source_pages[chunk.source].add(chunk.page)

for source, pages in source_pages.items():
    print(f"{source}: pages {min(pages)}-{max(pages)}")
```

### 7. Sort Chunks by Source and Page
```python
sorted_chunks = sorted(
    chunker.chunks.values(),
    key=lambda c: (c.source or "", c.page or 0)
)
```

### 8. RAG with Source Citation
```python
def retrieve_with_citation(query, top_k=5):
    # Assume we have similarity scores
    results = search_chunks(query, top_k)
    
    citations = []
    for chunk, score in results:
        citations.append({
            "content": chunk.content,
            "source": chunk.source,
            "page": chunk.page,
            "relevance": score
        })
    
    return citations

# Use in response
results = retrieve_with_citation("hak pekerja")
response = "Berdasarkan:\n"
for r in results:
    response += f"- {r['source']} (hal. {r['page']}): {r['content'][:100]}...\n"
```

## Cache Format

```json
{
  "chunk_uuid": {
    "id": "chunk_uuid",
    "content": "Full chunk text...",
    "index": 0,
    "source": "peraturan_pdfs/UU_No_13.pdf",
    "page": 5,
    "metadata": {
      "source": "peraturan_pdfs/UU_No_13.pdf",
      "page": 5,
      "total_pages": 150,
      ...
    }
  }
}
```

## Examples

Run these examples:
```bash
# Basic usage
python example_agentic_v2.py

# Source and page filtering examples
python example_source_page_info.py
```
