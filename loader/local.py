
import os
from logger import Logger
from .base import BaseLoader


class LocalPDFLoader(BaseLoader):
    def __init__(self, source: str):
        super().__init__(source)
    
    async def load_data(self):
        Logger.log(f"Loading PDFs from local folder: {self.source}")

        for file_name in os.listdir(self.source):
            file_path = os.path.join(self.source, file_name)
            if not file_name.lower().endswith(".pdf"):
                continue

            await self._load_single_pdf(file_path, file_name)
    
    
        
    
        
    