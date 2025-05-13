"""
Embeddings module for creating and using different embedding models.
"""
import os
import logging
import time
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory class for creating and managing different embedding models."""
    
    def __init__(self):
        """Initialize the embedding factory with supported embedding models."""
        self.embedding_models = {
            "sentence-transformers/all-MiniLM-L6-v2": None,  # Small, fast model
            "sentence-transformers/all-mpnet-base-v2": None,  # More powerful model
            "openai": None,  # OpenAI embeddings (requires API key)
        }
        
        self.default_model = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Initialized EmbeddingFactory")
    
    def get_embedding_model(self, model_name: str = None):
        """
        Get or create an embedding model by name.
        
        Args:
            model_name: Name of the embedding model to use
            
        Returns:
            An embedding model instance
        """
        if model_name is None:
            model_name = self.default_model
        
        # Check if we already have this model instantiated
        if model_name in self.embedding_models and self.embedding_models[model_name] is not None:
            return self.embedding_models[model_name]
        
        # Create the model
        try:
            if model_name == "openai":
                if os.getenv("OPENAI_API_KEY"):
                    model = OpenAIEmbeddings(model="text-embedding-3-small")
                    logger.info("Created OpenAI embeddings model")
                else:
                    logger.warning("OpenAI API key not found. Using default model instead.")
                    return self.get_embedding_model(self.default_model)
            else:
                # Using HuggingFace embeddings
                model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "mps"}  # Use Metal Performance Shaders on Mac Silicon
                )
                logger.info(f"Created HuggingFace embeddings model: {model_name}")
            
            # Cache the model
            self.embedding_models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error creating embedding model {model_name}: {str(e)}")
            if model_name != self.default_model:
                logger.info(f"Falling back to default model: {self.default_model}")
                return self.get_embedding_model(self.default_model)
            else:
                raise e
    
    def list_available_models(self) -> List[str]:
        """
        List all available embedding models.
        
        Returns:
            List of model names
        """
        return list(self.embedding_models.keys())
    
    def evaluate_embedding_model(self, model_name: str, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate an embedding model's performance on a set of texts.
        
        Args:
            model_name: Name of the embedding model to evaluate
            texts: List of texts to embed
            
        Returns:
            Dictionary with performance metrics
        """
        model = self.get_embedding_model(model_name)
        
        start_time = time.time()
        embeddings = model.embed_documents(texts)
        end_time = time.time()
        
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        return {
            "model_name": model_name,
            "embedding_dimension": embedding_dim,
            "embedding_time": end_time - start_time,
            "texts_processed": len(texts),
            "average_time_per_text": (end_time - start_time) / max(len(texts), 1)
        }
    
    def compare_embedding_models(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Compare the performance of different embedding models.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of dictionaries with performance metrics for each model
        """
        results = []
        
        for model_name in self.list_available_models():
            try:
                metrics = self.evaluate_embedding_model(model_name, texts)
                results.append(metrics)
                logger.info(f"Model {model_name}: {metrics['embedding_dimension']} dimensions, "
                          f"{metrics['average_time_per_text']:.4f} seconds per text")
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
        
        return results


def get_document_embeddings(documents: List[Document], embedding_model_name: str = None) -> Dict[str, Any]:
    """
    Create embeddings for a list of documents.
    
    Args:
        documents: List of documents to embed
        embedding_model_name: Name of the embedding model to use
        
    Returns:
        Dictionary with embedded documents and metadata
    """
    if not documents:
        logger.warning("No documents to embed")
        return {"documents": [], "model": None, "embeddings": []}
    
    factory = EmbeddingFactory()
    model = factory.get_embedding_model(embedding_model_name)
    
    texts = [doc.page_content for doc in documents]
    
    start_time = time.time()
    embeddings = model.embed_documents(texts)
    end_time = time.time()
    
    logger.info(f"Created {len(embeddings)} embeddings using {getattr(model, 'model_name', str(model))}")
    logger.info(f"Embedding time: {end_time - start_time:.2f} seconds")
    
    return {
        "documents": documents,
        "model": getattr(model, "model_name", str(model)),
        "embeddings": embeddings,
        "embedding_time": end_time - start_time
    }


if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentLoader
    from document_processor import DocumentProcessor
    
    # Load and process documents
    loader = DocumentLoader("documents")
    docs = loader.load_documents()
    
    processor = DocumentProcessor()
    split_docs = processor.split_documents(docs)
    
    # Create a factory and list available models
    factory = EmbeddingFactory()
    models = factory.list_available_models()
    print(f"Available embedding models: {models}")
    
    # Create embeddings using the default model
    default_model = factory.default_model
    print(f"Using default model: {default_model}")
    
    # Get sample texts for comparison
    sample_texts = [doc.page_content for doc in split_docs[:5]]
    
    # Compare different embedding models
    comparison = factory.compare_embedding_models(sample_texts)
    for result in comparison:
        print(f"Model: {result['model_name']}")
        print(f"  Dimension: {result['embedding_dimension']}")
        print(f"  Time per text: {result['average_time_per_text']:.4f} seconds")