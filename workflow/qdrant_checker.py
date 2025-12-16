"""
Workflow wrapper for Qdrant collection checker
Provides interactive UI for checking and fixing Qdrant collections
"""
import os
from logger import Logger
from check_qdrant import QdrantChecker
from ui import UserInterface


class QdrantCheckerWorkflow:
    """Workflow for checking and maintaining Qdrant collections"""
    
    def __init__(self, config):
        self.config = config
        self.ui = UserInterface()
        
    async def run(self):
        """Run the Qdrant checker workflow"""
        self.ui.print_header("QDRANT COLLECTION CHECKER")
        
        # Get collection name
        print("\nAvailable collections:")
        print("1. documents (default)")
        print("2. Custom collection name")
        
        collection_choice = input("\nSelect collection (1-2, default: 1): ").strip()
        
        if collection_choice == "2":
            collection_name = input("Enter collection name: ").strip()
        else:
            collection_name = "documents"
        
        # Get Qdrant connection details
        qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")
        qdrant_api_key = self.config.get("qdrant_api_key")
        
        print(f"\nQdrant URL: {qdrant_url}")
        if qdrant_api_key:
            print("API Key: [configured]")
        
        # Get cache directory and files
        cache_dir = input("\nCache directory (default: ./chunk_cache): ").strip() or "./chunk_cache"
        
        print("\nSelect cache files to check:")
        print("1. All cache files (recursive, semantic, agentic_v2)")
        print("2. Specific cache file(s)")
        
        cache_choice = input("Select option (1-2, default: 1): ").strip()
        
        cache_files = None
        if cache_choice == "2":
            cache_files = []
            print("\nEnter cache filenames (e.g., recursive_cache.json)")
            print("Press Enter without input when done")
            while True:
                cache_file = input("Cache file: ").strip()
                if not cache_file:
                    break
                cache_files.append(cache_file)
        
        # Get operation mode
        print("\nSelect operations to perform:")
        
        insert_missing = self.ui.get_yes_no("Insert missing chunks from cache into Qdrant?", default=False)
        delete_duplicates = self.ui.get_yes_no("Delete duplicate point IDs in Qdrant?", default=False)
        delete_extra = self.ui.get_yes_no("Delete points not in cache from Qdrant?", default=False)
        save_report = self.ui.get_yes_no("Save report to JSON file?", default=True)
        
        # Confirm before proceeding
        print("\n" + "="*70)
        print("OPERATION SUMMARY")
        print("="*70)
        print(f"Collection: {collection_name}")
        print(f"Cache directory: {cache_dir}")
        if cache_files:
            print(f"Cache files: {', '.join(cache_files)}")
        else:
            print("Cache files: All")
        print(f"Insert missing: {'YES' if insert_missing else 'NO'}")
        print(f"Delete duplicates: {'YES' if delete_duplicates else 'NO'}")
        print(f"Delete extra: {'YES' if delete_extra else 'NO'}")
        print(f"Save report: {'YES' if save_report else 'NO'}")
        print("="*70)
        
        if not self.ui.get_yes_no("\nProceed with check?", default=True):
            print("Operation cancelled")
            return
        
        # Create checker and run
        try:
            checker = QdrantChecker(
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                qdrant_api_key=qdrant_api_key
            )
            
            checker.run_full_check(
                cache_dir=cache_dir,
                save_report=save_report,
                cache_files=cache_files,
                insert_missing=insert_missing,
                delete_duplicates=delete_duplicates,
                delete_extra=delete_extra
            )
            
            print("\n" + "="*70)
            print("CHECK COMPLETED")
            print("="*70)
            
            if save_report:
                print("\nReport saved to: qdrant_check_report.json")
            
        except Exception as e:
            Logger.log(f"Error during Qdrant check: {e}")
            print(f"\n❌ Error: {e}")
