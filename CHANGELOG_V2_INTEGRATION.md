# Changelog: AgenticChunkerV2 Integration

## Updates to `chunk_documents.py`

### What's New

Added **Agentic Chunker V2** as option 4 in the chunking strategy menu.

### Changes Made

1. **Import Statement**
   ```python
   from chunker import RecursiveChunker, SemanticChunker, AgenticChunker, AgenticChunkerV2
   ```

2. **Menu Options Updated**
   - Option 1: Recursive Chunker (unchanged)
   - Option 2: Semantic Chunker (unchanged)
   - Option 3: Agentic Chunker V1 (renamed from "Agentic Chunker")
   - **Option 4: Agentic Chunker V2** (NEW)

3. **V2 Configuration**
   - Same LLM options as V1 (Gemini, GeminiLive, Ollama)
   - Uses `AgenticChunkerV2` class
   - Cache name: `agentic_v2`

4. **Enhanced Summary Display**
   - For V2, shows count of chunks with source info
   - For V2, shows count of chunks with page info
   - Suggests `example_source_page_info.py` for V2 users

### Usage

```bash
python chunk_documents.py
```

Then select:
- **Option 4** for Agentic Chunker V2
- Choose your preferred LLM
- Process documents (with or without cache)

### Output Format

V2 chunks are exported with additional fields:
```json
{
  "chunker_type": "agentic_v2",
  "chunks": [
    {
      "id": "uuid",
      "content": "...",
      "source": "peraturan_pdfs/UU_No_13.pdf",
      "page": 5,
      "index": 0,
      "metadata": {...}
    }
  ]
}
```

### Benefits of V2

1. **~3x faster** than V1 (no title/summary generation)
2. **Source tracking** - know which file each chunk came from
3. **Page tracking** - know which page each chunk is on
4. **Simpler model** - just content, no propositions
5. **Better for citations** - easy to reference back to source

### Comparison

| Feature | V1 (Agentic) | V2 (Agentic V2) |
|---------|--------------|-----------------|
| Speed | Slow | Fast (~3x) |
| LLM calls per doc | Many (3+ per chunk) | One |
| Title generation | ✅ Yes | ❌ No |
| Summary generation | ✅ Yes | ❌ No |
| Source tracking | ❌ No | ✅ Yes |
| Page tracking | ❌ No | ✅ Yes |
| Best for | Complex analysis | Fast processing + citations |

### Next Steps

After chunking with V2:
1. Load chunks to database: `python load_chunks_to_db.py`
2. Explore filtering: `python example_source_page_info.py`
3. Test RAG pipeline: `python test_all_components.py`
