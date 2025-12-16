from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
import os
import tempfile
import shutil
import asyncio
from huggingface_hub import list_repo_files, hf_hub_download
from logger import Logger
from .base import BaseLoader


class HuggingFacePDFLoader(BaseLoader):
    def __init__(self, source: str, hf_token: str = None):
        super().__init__(source)
        self.hf_token = hf_token
    
    async def load_data(self):
        Logger.log(f"Loading PDFs from Hugging Face repo: {self.source}")
        
        repo_type_to_try = "dataset" 

        try:
            Logger.log(f"Attempting to list files from repo as type: '{repo_type_to_try}'")
            files = list_repo_files(self.source, repo_type=repo_type_to_try, token=self.hf_token)
            
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            Logger.log(f"Found {len(pdf_files)} PDF files in repository")
            
            for pdf_file in pdf_files:
                temp_path = None
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=self.source,
                        filename=pdf_file,
                        token=self.hf_token,
                        repo_type=repo_type_to_try 
                    )
                    
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    shutil.copy2(downloaded_path, temp_path)
                    
                    await self._load_single_pdf(temp_path, pdf_file)
                    
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    Logger.log(f"Failed to download or process {pdf_file}: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except PermissionError:
                            await asyncio.sleep(0.1)
                            try:
                                os.unlink(temp_path)
                            except Exception as cleanup_error:
                                Logger.log(f"Warning: Could not clean up temp file {temp_path}: {cleanup_error}")
                    
        except Exception as e:
            Logger.log(f"[ERROR]: Failed to access Hugging Face repository: {e}")
            if "404" in str(e):
                 Logger.log("Got a 404 error. Please check your `repo_id` and `repo_type`.")
            if self.hf_token is None and "private" in str(e).lower():
                 Logger.log("Repository might be private. Please provide a Hugging Face token.")
            raise
        
    
        
    