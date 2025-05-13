"""
Vector store module for storing and retrieving document embeddings.
"""
import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

from embeddings import EmbeddingFactory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directory for storing vector indexes
VECTOR_STORE_DIR = "vector_store"


class VectorStoreManager:
    """Class for managing vector stores and document retrieval."""
    
    def __init__(self, vector_store_dir: str = VECTOR_STORE_DIR):
        """
        Initialize the vector store manager.
        
        Args:
            vector_store_dir: Directory to store vector indexes
        """
        self.vector_store_dir = Path(vector_store_dir)
        if not self.vector_store_dir.exists():
            logger.info(f"Creating vector store directory: {vector_store_dir}")
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_factory = EmbeddingFactory()
        logger.info(f"Initialized VectorStoreManager with store directory: {vector_store_dir}")
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        embedding_model_name: str = None,
        store_name: str = "default"
    ) -> FAISS:
        """
        Create a vector store from a list of documents.
        
        Args:
            documents: List of documents to embed
            embedding_model_name: Name of the embedding model to use
            store_name: Name for the vector store
            
        Returns:
            FAISS vector store instance
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return None
        
        try:
            # Get embedding model
            embeddings = self.embedding_factory.get_embedding_model(embedding_model_name)
            
            # Create FAISS index
            logger.info(f"Creating FAISS vector store with {len(documents)} documents")
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # Add metadata to vector store (for saving)
            vector_store.embeddings = embeddings
            vector_store.store_name = store_name
            
            # Save the vector store
            self.save_vector_store(vector_store, store_name)
            
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def save_vector_store(self, vector_store: FAISS, store_name: str = "default") -> bool:
        """
        Save a vector store to disk.
        
        Args:
            vector_store: FAISS vector store to save
            store_name: Name for the vector store
            
        Returns:
            Boolean indicating success
        """
        try:
            store_path = self.vector_store_dir / store_name
            
            # Create directory if it doesn't exist
            if not store_path.exists():
                store_path.mkdir(parents=True)
            
            # Save the index
            vector_store.save_local(str(store_path))
            logger.info(f"Saved vector store to {store_path}")
            
            # Save additional metadata
            metadata = {
                "document_count": len(vector_store.docstore._dict),
                "embedding_model": getattr(vector_store.embeddings, "model_name", str(vector_store.embeddings)),
                "store_name": store_name
            }
            
            metadata_path = store_path / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vector_store(self, store_name: str = "default", embedding_model_name: str = None) -> Optional[FAISS]:
        """
        Load a vector store from disk.
        
        Args:
            store_name: Name of the vector store to load
            embedding_model_name: Name of the embedding model to use (if different from saved)
            
        Returns:
            FAISS vector store instance or None if not found
        """
        store_path = self.vector_store_dir / store_name
        
        if not store_path.exists():
            logger.warning(f"Vector store {store_name} not found at {store_path}")
            return None
        
        try:
            # Get embedding model (use saved model if embedding_model_name not provided)
            embeddings = None
            metadata_path = store_path / "metadata.pkl"
            
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                
                if not embedding_model_name and "embedding_model" in metadata:
                    embedding_model_name = metadata["embedding_model"]
            
            embeddings = self.embedding_factory.get_embedding_model(embedding_model_name)
            
            # Load the index
            vector_store = FAISS.load_local(str(store_path), embeddings)
            logger.info(f"Loaded vector store from {store_path}")
            
            # Add metadata to vector store (for saving)
            vector_store.embeddings = embeddings
            vector_store.store_name = store_name
            
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def list_vector_stores(self) -> List[Dict[str, Any]]:
        """
        List all available vector stores.
        
        Returns:
            List of dictionaries with vector store metadata
        """
        stores = []
        
        for path in self.vector_store_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "metadata.pkl"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "rb") as f:
                            metadata = pickle.load(f)
                        stores.append(metadata)
                    except Exception as e:
                        logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
                        stores.append({"store_name": path.name, "error": str(e)})
                else:
                    stores.append({"store_name": path.name})