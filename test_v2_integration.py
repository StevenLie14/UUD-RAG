"""
Quick Test: Verify AgenticChunkerV2 Integration

This script tests that AgenticChunkerV2 can be imported and used correctly.
"""

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        from chunker import RecursiveChunker, SemanticChunker, AgenticChunker, AgenticChunkerV2
        print("✅ All chunkers imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_v2_initialization():
    """Test that AgenticChunkerV2 can be initialized"""
    print("\nTesting AgenticChunkerV2 initialization...")
    try:
        from chunker import AgenticChunkerV2
        from llm import Gemini
        from config import Config
        
        config = Config()
        llm = Gemini("gemini-2.0-flash-lite", config.GOOGLE_API_KEY)
        chunker = AgenticChunkerV2(llm=llm, cache_dir="./chunk_cache")
        
        print("✅ AgenticChunkerV2 initialized successfully")
        print(f"   - Chunker name: {chunker.chunker_name}")
        print(f"   - Cache dir: {chunker.cache_dir}")
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_simple_chunk_model():
    """Test that SimpleChunk model works"""
    print("\nTesting SimpleChunk model...")
    try:
        from model.chunk import SimpleChunk
        
        chunk = SimpleChunk(
            id="test-id",
            content="Test content",
            index=0,
            source="test.pdf",
            page=1,
            metadata={"test": "data"}
        )
        
        print("✅ SimpleChunk created successfully")
        print(f"   - ID: {chunk.id}")
        print(f"   - Source: {chunk.source}")
        print(f"   - Page: {chunk.page}")
        print(f"   - Content: {chunk.content}")
        
        # Test get_context
        context = chunk.get_context()
        assert "Sumber: test.pdf" in context
        assert "Halaman: 1" in context
        print("✅ get_context() works correctly")
        
        # Test get_payload
        payload = chunk.get_payload()
        assert payload["source"] == "test.pdf"
        assert payload["page"] == 1
        print("✅ get_payload() works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("AGENTICCHUNKERV2 INTEGRATION TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("V2 Initialization", test_v2_initialization()))
    results.append(("SimpleChunk Model", test_simple_chunk_model()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AgenticChunkerV2 is ready to use.")
        print("\nNext steps:")
        print("- Run: python chunk_documents.py")
        print("- Select option 4 (Agentic Chunker V2)")
        print("- Process your documents!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60)

if __name__ == "__main__":
    main()
