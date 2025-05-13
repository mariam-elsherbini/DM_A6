"""
Main RAG system implementation for CSAI 422: Laboratory Assignment 6.
This script integrates document loading, processing, embedding, retrieval, and generation.
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from document_loader import DocumentLoader
from document_processor import DocumentProcessor
from embeddings import EmbeddingFactory, get_document_embeddings
from vector_store import VectorStoreManager

# LangChain imports for RAG
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.retrievers import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main class for the RAG system implementation."""
    
    def __init__(
        self,
        docs_dir: str = "documents",
        vector_store_dir: str = "vector_store",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        store_name: str = "default",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG system with configuration parameters.
        
        Args:
            docs_dir: Directory containing documents
            vector_store_dir: Directory for storing vector indexes
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the LLM to use for generation
            store_name: Name for the vector store
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between document chunks
        """
        self.docs_dir = docs_dir
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.store_name = store_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.document_loader = DocumentLoader(docs_dir)
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedding_factory = EmbeddingFactory()
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        
        # Placeholders for later initialization
        self.documents = None
        self.chunks = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        
        logger.info(f"Initialized RAG system with embedding model: {embedding_model_name}")
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from the specified paths or the docs_dir.
        
        Args:
            file_paths: Optional list of specific file paths to load
            
        Returns:
            List of loaded documents
        """
        self.documents = self.document_loader.load_documents(file_paths)
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def process_documents(self, splitter_type: str = "recursive") -> List[Document]:
        """
        Process documents into chunks.
        
        Args:
            splitter_type: Type of text splitter to use
            
        Returns:
            List of document chunks
        """
        if not self.documents:
            logger.warning("No documents loaded. Loading documents first.")
            self.load_documents()
        
        self.chunks = self.document_processor.split_documents(self.documents, splitter_type)
        logger.info(f"Split documents into {len(self.chunks)} chunks")
        return self.chunks
    
    def create_vector_store(self, force_recreate: bool = False) -> FAISS:
        """
        Create or load a vector store for document embeddings.
        
        Args:
            force_recreate: Whether to force recreation of the vector store
            
        Returns:
            FAISS vector store
        """
        # Check if vector store already exists
        if not force_recreate:
            vector_store = self.vector_store_manager.load_vector_store(self.store_name, self.embedding_model_name)
            if vector_store:
                self.vector_store = vector_store
                logger.info(f"Loaded existing vector store: {self.store_name}")
                return self.vector_store
        
        # Process documents if needed
        if not self.chunks:
            self.process_documents()
        
        # Create new vector store
        self.vector_store = self.vector_store_manager.create_vector_store(
            self.chunks, 
            self.embedding_model_name,
            self.store_name
        )
        logger.info(f"Created new vector store: {self.store_name}")
        return self.vector_store
    
    def initialize_retriever(self, retrieval_type: str = "similarity", **kwargs) -> Any:
        """
        Initialize the retriever with the specified strategy.
        
        Args:
            retrieval_type: Type of retrieval strategy ("similarity", "mmr", "filter", "hybrid")
            **kwargs: Additional parameters for the retriever
            
        Returns:
            Retriever instance
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized. Creating vector store first.")
            self.create_vector_store()
        
        if retrieval_type == "similarity":
            k = kwargs.get("k", 4)
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            logger.info(f"Initialized similarity retriever with k={k}")
        
        elif retrieval_type == "mmr":
            k = kwargs.get("k", 4)
            fetch_k = kwargs.get("fetch_k", 20)
            lambda_mult = kwargs.get("lambda_mult", 0.5)
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
            )
            logger.info(f"Initialized MMR retriever with k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        elif retrieval_type == "filter":
            k = kwargs.get("k", 4)
            filter_threshold = kwargs.get("filter_threshold", 0.7)
            embedding_model = self.embedding_factory.get_embedding_model(self.embedding_model_name)
            
            # Create base retriever
            base_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k * 2})
            
            # Create embeddings filter
            embeddings_filter = EmbeddingsFilter(
                embeddings=embedding_model,
                similarity_threshold=filter_threshold
            )
            
            # Create compression retriever
            self.retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever
            )
            logger.info(f"Initialized filter retriever with threshold={filter_threshold}")
        
        elif retrieval_type == "hybrid":
            # Hybrid search combining keyword and semantic search
            # This is a simplified implementation that gives weight to both keyword and semantic matches
            k = kwargs.get("k", 4)
            keyword_weight = kwargs.get("keyword_weight", 0.3)
            semantic_weight = kwargs.get("semantic_weight", 0.7)
            
            # Since FAISS doesn't have native hybrid search, we implement a simple version
            # using a custom retriever function
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                k=k
            )
            logger.info(f"Initialized hybrid retriever with keyword_weight={keyword_weight}, semantic_weight={semantic_weight}")
        
        else:
            logger.warning(f"Unknown retrieval type: {retrieval_type}. Using default similarity search.")
            self.retriever = self.vector_store.as_retriever()
        
        return self.retriever
    
    def initialize_llm(self):
        """
        Initialize the LLM for generation.
        
        Returns:
            LLM instance
        """
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=0.2,
            max_tokens=1000
        )
        logger.info(f"Initialized LLM: {self.llm_model_name}")
        return self.llm
    
    def create_qa_chain(self, chain_type: str = "stuff"):
        """
        Create the QA chain for RAG.
        
        Args:
            chain_type: Type of chain to use ("stuff", "map_reduce", "refine")
            
        Returns:
            QA chain instance
        """
        if not self.retriever:
            logger.warning("Retriever not initialized. Initializing default retriever.")
            self.initialize_retriever()
        
        if not self.llm:
            logger.warning("LLM not initialized. Initializing default LLM.")
            self.initialize_llm()
        
        # Create the prompt template
        template = """You are an AI assistant providing helpful information based on the context provided.
        Please answer the question based only on the given context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info(f"Created {chain_type} QA chain")
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.qa_chain:
            logger.warning("QA chain not initialized. Creating QA chain.")
            self.create_qa_chain()
        
        start_time = time.time()
        response = self.qa_chain.invoke(question)
        end_time = time.time()
        
        # Get retrieved documents
        retrieved_docs = self.get_retrieved_documents(question)
        
        result = {
            "question": question,
            "answer": response["result"],
            "retrieved_documents": retrieved_docs,
            "response_time": end_time - start_time
        }
        
        logger.info(f"Query: {question}")
        logger.info(f"Response time: {result['response_time']:.2f} seconds")
        
        return result
    
    def get_retrieved_documents(self, question: str) -> List[Dict[str, Any]]:
        """
        Get the documents retrieved for a question.
        
        Args:
            question: The question to retrieve documents for
            
        Returns:
            List of retrieved documents
        """
        if not self.retriever:
            logger.warning("Retriever not initialized. Initializing default retriever.")
            self.initialize_retriever()
        
        docs = self.retriever.get_relevant_documents(question)
        
        # Convert to simpler format for display
        retrieved_docs = []
        for i, doc in enumerate(docs):
            retrieved_docs.append({
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "metadata": {k: v for k, v in doc.metadata.items() if k != "source"}
            })
        
        return retrieved_docs
    
    def evaluate_retrieval(self, questions: List[str], retrieval_types: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate different retrieval strategies on a set of questions.
        
        Args:
            questions: List of questions to evaluate
            retrieval_types: List of retrieval types to compare
            
        Returns:
            Dictionary with evaluation results
        """
        if retrieval_types is None:
            retrieval_types = ["similarity", "mmr", "filter", "hybrid"]
        
        results = {}
        
        for retrieval_type in retrieval_types:
            # Initialize retriever
            self.initialize_retriever(retrieval_type)
            
            # Evaluate for each question
            question_results = []
            for question in questions:
                start_time = time.time()
                retrieved_docs = self.get_retrieved_documents(question)
                end_time = time.time()
                
                question_results.append({
                    "question": question,
                    "num_docs": len(retrieved_docs),
                    "retrieval_time": end_time - start_time,
                    "sources": [doc["source"] for doc in retrieved_docs]
                })
            
            # Aggregate results
            avg_time = sum(result["retrieval_time"] for result in question_results) / len(question_results)
            avg_docs = sum(result["num_docs"] for result in question_results) / len(question_results)
            
            results[retrieval_type] = {
                "avg_retrieval_time": avg_time,
                "avg_num_docs": avg_docs,
                "questions": question_results
            }
            
            logger.info(f"Evaluated {retrieval_type} retriever: avg_time={avg_time:.4f}s, avg_docs={avg_docs:.2f}")
        
        return results
    
    def evaluate_qa(self, test_cases: List[Dict[str, str]], retrieval_types: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the QA performance on test cases with different retrieval strategies.
        
        Args:
            test_cases: List of dictionaries with "question" and "expected_answer" keys
            retrieval_types: List of retrieval types to compare
            
        Returns:
            Dictionary with evaluation results
        """
        if retrieval_types is None:
            retrieval_types = ["similarity", "mmr", "filter"]
        
        results = {}
        
        for retrieval_type in retrieval_types:
            # Initialize retriever and QA chain
            self.initialize_retriever(retrieval_type)
            self.create_qa_chain()
            
            # Evaluate for each test case
            case_results = []
            for test_case in test_cases:
                question = test_case["question"]
                expected_answer = test_case["expected_answer"]
                
                start_time = time.time()
                response = self.query(question)
                end_time = time.time()
                
                # Simple exact match evaluation
                match = expected_answer.lower() in response["answer"].lower()
                
                case_results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": response["answer"],
                    "match": match,
                    "response_time": end_time - start_time
                })
            
            # Aggregate results
            matches = sum(1 for result in case_results if result["match"])
            accuracy = matches / len(case_results) if case_results else 0
            avg_time = sum(result["response_time"] for result in case_results) / len(case_results) if case_results else 0
            
            results[retrieval_type] = {
                "accuracy": accuracy,
                "avg_response_time": avg_time,
                "cases": case_results
            }
            
            logger.info(f"Evaluated {retrieval_type} QA: accuracy={accuracy:.2f}, avg_time={avg_time:.4f}s")
        
        return results


class HybridRetriever:
    """Custom retriever that combines keyword and semantic search."""
    
    def __init__(self, vector_store, keyword_weight=0.3, semantic_weight=0.7, k=4):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector store to use
            keyword_weight: Weight for keyword search results
            semantic_weight: Weight for semantic search results
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using both keyword and semantic search.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
        """
        # Get semantic search results
        semantic_docs = self.vector_store.similarity_search(query, k=self.k * 2)
        
        # Get keyword search results (simple implementation)
        # In a more advanced implementation, you might use BM25 or another keyword search algorithm
        keyword_docs = self._keyword_search(query, k=self.k * 2)
        
        # Combine results with weights
        combined_docs = self._combine_results(semantic_docs, keyword_docs)
        
        # Return top k results
        return combined_docs[:self.k]
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """
        Simple keyword search implementation.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Extract all documents from the vector store
        all_docs = list(self.vector_store.docstore._dict.values())
        
        # Score documents based on keyword matches
        scored_docs = []
        query_terms = query.lower().split()
        
        for doc in all_docs:
            score = 0
            content = doc.page_content.lower()
            
            for term in query_terms:
                if term in content:
                    score += content.count(term)
            
            scored_docs.append((doc, score))
        
        # Sort by score (descending) and take top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def _combine_results(self, semantic_docs: List[Document], keyword_docs: List[Document]) -> List[Document]:
        """
        Combine semantic and keyword search results with weights.
        
        Args:
            semantic_docs: Documents from semantic search
            keyword_docs: Documents from keyword search
            
        Returns:
            Combined list of documents
        """
        # Create a map of document ID to score
        doc_scores = {}
        
        # Add semantic search results with weight
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.get("chunk_id", i)
            doc_scores[doc_id] = self.semantic_weight * (1.0 - i / len(semantic_docs))
        
        # Add keyword search results with weight
        for i, doc in enumerate(keyword_docs):
            doc_id = doc.metadata.get("chunk_id", i)
            existing_score = doc_scores.get(doc_id, 0)
            doc_scores[doc_id] = existing_score + self.keyword_weight * (1.0 - i / len(keyword_docs))
        
        # Create a map of document ID to document
        doc_map = {}
        for doc in semantic_docs + keyword_docs:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            doc_map[doc_id] = doc
        
        # Sort documents by score
        sorted_ids = sorted(doc_scores.keys(), key=lambda doc_id: doc_scores[doc_id], reverse=True)
        
        # Return sorted documents
        return [doc_map[doc_id] for doc_id in sorted_ids if doc_id in doc_map]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of get_relevant_documents.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
        """
        # For simplicity, just call the sync version
        return self.get_relevant_documents(query)


def main():
    """Main function to demonstrate RAG system functionality."""
    # Create RAG system
    rag = RAGSystem(
        docs_dir="documents",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Load and process documents
    rag.load_documents()
    rag.process_documents()
    
    # Create vector store
    rag.create_vector_store(force_recreate=True)
    
    # Example queries
    questions = [
        "What is RAG?",
        "How does document chunking work?",
        "What are the different retrieval strategies?",
        "How are embeddings used in RAG systems?",
        "What evaluation metrics can be used for RAG systems?"
    ]
    
    # Initialize retriever and LLM
    rag.initialize_retriever(retrieval_type="similarity", k=4)
    rag.initialize_llm()
    rag.create_qa_chain()
    
    # Run queries
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['retrieved_documents'])} documents")
        print(f"Response time: {result['response_time']:.2f} seconds")
    
    # Compare retrieval strategies
    print("\nComparing retrieval strategies...")
    retrieval_results = rag.evaluate_retrieval(questions[:2])
    for retrieval_type, results in retrieval_results.items():
        print(f"\n{retrieval_type.capitalize()} retriever:")
        print(f"  Average retrieval time: {results['avg_retrieval_time']:.4f} seconds")
        print(f"  Average documents retrieved: {results['avg_num_docs']:.2f}")


if __name__ == "__main__":
    main()