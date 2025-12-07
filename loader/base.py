import re
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
from logger import Logger


class BaseLoader:
    def __init__(self, source: str):
        self.source = source
        self.pages = []

    def _clean_text(self,text: str) -> str:
        # Remove common headers/footers and artifacts
        text = re.sub(r'PRESIDEN\s+REPUBLIK\s+INDONESIA', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SK\s+No\s+\d+\s*[A-Z]*', '', text)
        # Remove page number patterns like "- 12 -" or "(12)"
        text = re.sub(r'(?:^|\n)\s*[-–—]?\s*\(?\d+\)?\s*[-–—]?\s*(?:$|\n)', '\n', text)
        # Remove dot/bullet-only lines (leaders)
        text = '\n'.join(
            ln for ln in text.splitlines()
            if not re.match(r"^\s*(?:[\.·•]+\s*)+$", ln)
        )
        # Remove leading dot leaders at start of lines
        text = re.sub(r"(^|\n)\s*[\.·•]{2,}\s*", "\n", text)
        # Normalize whitespace: collapse multiple spaces and excessive newlines
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove stray non-word artifacts but keep basic punctuation
        text = re.sub(r"[^\w\s.,;:()'\-]", "", text)
        # Trim
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
            
            try:
                async for page in loader.alazy_load():
                    page.page_content = self._clean_text(page.page_content)
                    pages_loaded.append(page)
                self.pages.extend(pages_loaded)
                
            except Exception as async_error:
                Logger.log(f"Async loading failed for {file_name}: {async_error}")
                raise async_error
                
        except PdfStreamError as pdf_error:
            Logger.log(f"PDF stream error for {file_name}: {pdf_error}")
            self._try_pymupdf_fallback(file_path, file_name)
            
        except Exception as e:
            Logger.log(f"Unexpected error on {file_name}: {e}")