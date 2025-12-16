import re
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
from logger import Logger


class BaseLoader:
    def __init__(self, source: str):
        self.source = source
        self.pages = []

    def _clean_text(self, text: str) -> str:
        text = text.replace('PRESIDEN REPUBLIK INDONESIA', '')
        
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def load_data(self):
        raise NotImplementedError

    async def _load_single_pdf(self, file_path: str, file_name: str):
        """Load a single PDF file"""
        try:
            Logger.log(f"Loading: {file_name}")
            
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    return
            
            loader = PyPDFLoader(file_path)
            pages_loaded = []
            page_count = 0
            
            try:
                async for page in loader.alazy_load():
                    page_count += 1
                    Logger.log(f"Page {page_count} loaded, cleaning text...")
                    page.page_content = self._clean_text(page.page_content)
                    Logger.log(f"Page {page_count} cleaned")
                    pages_loaded.append(page)
                self.pages.extend(pages_loaded)
                Logger.log(f"Loaded {len(pages_loaded)} pages from {file_name}")
                
            except Exception as async_error:
                Logger.log(f"Async loading failed for {file_name}: {async_error}")
                raise async_error
                
        except PdfStreamError as pdf_error:
            Logger.log(f"PDF stream error for {file_name}: {pdf_error}")
            
        except Exception as e:
            Logger.log(f"Unexpected error on {file_name}: {e}")