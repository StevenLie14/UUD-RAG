import asyncio
from config import Config
from ui import UserInterface
from workflow import DocumentChunker, DatabaseLoader, ComponentTester


class RAGManager:
    
    def __init__(self):
        self.config = Config()
        self.ui = UserInterface()
        self.chunker = DocumentChunker(self.config)
        self.loader = DatabaseLoader(self.config)
        self.tester = ComponentTester(self.config)
    
    async def run(self):
        self.ui.print_header("RAG SYSTEM MANAGER")
        
        while True:
            options = [
                "Document Chunking - Process PDFs into chunks",
                "Database Loading - Load chunks into FAISS/Qdrant",
                "Component Testing - Test and evaluate RAG configurations",
                "Exit"
            ]
            
            choice = self.ui.get_choice("\nMain Menu:", options)
            
            if choice == "1":
                await self.chunker.run()
            elif choice == "2":
                await self.loader.run()
            elif choice == "3":
                await self.tester.run()
            elif choice == "4":
                self.ui.print_header("GOODBYE")
                print("Thank you for using RAG System Manager!\n")
                break
            else:
                print("Invalid choice, please try again")
            
            input("\nPress Enter to continue...")


async def main():
    manager = RAGManager()
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
