"""
Document processor module for splitting documents into chunks.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get chunk size and overlap from environment or use defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


class DocumentProcessor:
    """Class for processing and chunking documents."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the DocumentProcessor with chunk size and overlap.
        
        Args:
            chunk_size: Size of each chunk in characters or tokens
            chunk_overlap: Overlap between chunks in characters or tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def create_text_splitter(self, splitter_type: str = "recursive"):
        """
        Create a text splitter of the specified type.
        
        Args:
            splitter_type: Type of text splitter to create (recursive, character, token)
            
        Returns:
            A text splitter instance
        """
        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            logger.warning(f"Unknown splitter type: {splitter_type}. Using recursive splitter.")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def split_documents(self, documents: List[Document], splitter_type: str = "recursive") -> List[Document]:
        """
        Split documents into chunks using the specified text splitter.
        
        Args:
            documents: List of documents to split
            splitter_type: Type of text splitter to use
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        text_splitter = self.create_text_splitter(splitter_type)
        
        try:
            logger.info(f"Splitting {len(documents)} documents using {splitter_type} splitter")
            split_docs = text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["chunk_size"] = self.chunk_size
                doc.metadata["chunk_overlap"] = self.chunk_overlap
                doc.metadata["splitter_type"] = splitter_type
            
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return documents  # Return original documents on error
    
    def compare_splitter_types(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Compare different text splitter types on the same documents.
        
        Args:
            documents: List of documents to split
            
        Returns:
            Dictionary mapping splitter types to lists of document chunks
        """
        splitter_types = ["recursive", "character", "token"]
        results = {}
        
        for splitter_type in splitter_types:
            split_docs = self.split_documents(documents, splitter_type)
            results[splitter_type] = split_docs
            logger.info(f"{splitter_type.capitalize()} splitter created {len(split_docs)} chunks")
        
        return results
    
    def analyze_chunk_distribution(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze the distribution of chunk lengths.
        
        Args:
            documents: List of document chunks
            
        Returns:
            Dictionary with chunk length statistics
        """
        if not documents:
            return {"min": 0, "max": 0, "avg": 0, "total": 0}
        
        chunk_lengths = [len(doc.page_content) for doc in documents]
        return {
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "avg": sum(chunk_lengths) / len(chunk_lengths),
            "total": len(documents)
        }


if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentLoader
    
    loader = DocumentLoader("documents")
    docs = loader.load_documents()
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    split_docs = processor.split_documents(docs)
    
    print(f"Original documents: {len(docs)}")
    print(f"Split chunks: {len(split_docs)}")
    
    stats = processor.analyze_chunk_distribution(split_docs)
    print(f"Chunk statistics: {stats}")
    
    # Compare different splitter types
    comparison = processor.compare_splitter_types(docs)
    for splitter_type, chunks in comparison.items():
        stats = processor.analyze_chunk_distribution(chunks)
        print(f"{splitter_type.capitalize()} splitter stats: {stats}")
