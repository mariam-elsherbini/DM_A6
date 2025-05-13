"""
Document loader module for loading documents from different file formats.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
)
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Class for loading documents from different file formats."""
    
    def __init__(self, docs_dir: str = "documents"):
        """
        Initialize the DocumentLoader with a directory containing documents.
        
        Args:
            docs_dir: Path to the directory containing documents
        """
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            logger.warning(f"Directory {docs_dir} does not exist. Creating it now.")
            self.docs_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_loader_for_file(self, file_path: str) -> Optional[Any]:
        """
        Get the appropriate loader based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            A document loader instance or None if the file type is unsupported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return PyPDFLoader(file_path)
            elif file_extension == '.txt':
                return TextLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                return Docx2txtLoader(file_path)
            else:
                # Fallback to unstructured loader for other file types
                logger.warning(f"Using generic loader for unsupported file type: {file_extension}")
                return UnstructuredFileLoader(file_path)
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document and return it as a list of LangChain Document objects.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        loader = self._get_loader_for_file(file_path)
        if not loader:
            logger.error(f"Could not load document: {file_path}")
            return []
        
        try:
            documents = loader.load()
            
            # Add additional metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_type': os.path.splitext(file_path)[1][1:],  # Remove the dot
                    'loader_type': loader.__class__.__name__
                })
            
            logger.info(f"Successfully loaded document: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load multiple documents from a list of file paths or all documents in the docs_dir.
        
        Args:
            file_paths: Optional list of file paths to load. If None, load all files in docs_dir.
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if file_paths is None:
            # If no specific files provided, load all from the docs_dir
            file_paths = [str(f) for f in self.docs_dir.glob('**/*') if f.is_file()]
        
        total_files = len(file_paths)
        logger.info(f"Loading {total_files} documents...")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Loading document {i}/{total_files}: {file_path}")
            docs = self.load_single_document(file_path)
            documents.extend(docs)
        
        logger.info(f"Successfully loaded {len(documents)} document chunks from {total_files} files")
        return documents


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader("documents")
    docs = loader.load_documents()
    print(f"Loaded {len(docs)} document chunks")
    
    if docs:
        print("\nSample document:")
        print(f"Content (first 100 chars): {docs[0].page_content[:100]}...")
        print(f"Metadata: {docs[0].metadata}")
