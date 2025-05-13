# RAG System with Open-Source Tools and Local Vector Storage

This repository contains a complete implementation of a Retrieval-Augmented Generation (RAG) system using open-source components with local vector storage. The system enhances Large Language Model (LLM) capabilities by retrieving relevant information from external knowledge sources before generating responses.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [RAG Pipeline Components](#rag-pipeline-components)
- [Advanced Features](#advanced-features)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Retrieval-Augmented Generation (RAG) enhances LLM responses by integrating external knowledge retrieval into the generation process. This project implements a complete RAG pipeline using LangChain for orchestration, Sentence Transformers for embeddings, FAISS for vector storage, and integrates with various LLM APIs for the generation component.

## Features

- **Document Processing:** Load and process documents from various formats (PDF, DOCX, TXT)
- **Intelligent Chunking:** Split documents into appropriate chunks with customizable parameters
- **Multiple Embedding Models:** Support for different embedding models from Sentence Transformers and OpenAI
- **Local Vector Storage:** Efficient storage and retrieval using FAISS vector database
- **Advanced Retrieval Strategies:** Basic similarity search, MMR, metadata filtering, and hybrid search
- **Flexible LLM Integration:** Compatible with various LLM providers (OpenAI, Anthropic, etc.)
- **Evaluation Framework:** Built-in metrics for RAG performance analysis

## System Architecture

The RAG system follows a modular architecture with these main components:

1. **Document Loading:** Processes various document formats while preserving metadata
2. **Document Processing:** Chunks documents into appropriate segments for indexing
3. **Embedding Generation:** Converts text chunks into vector embeddings
4. **Vector Storage:** Efficiently stores and retrieves embeddings using FAISS
5. **Retrieval Strategies:** Implements various algorithms for finding relevant information
6. **Generation:** Integrates with LLMs for producing final responses
7. **Evaluation:** Assesses the performance of different RAG configurations

```
User Query → [Query Processing] → [Retrieval] → [Context Integration] → [LLM Generation] → Response
                                      ↑
                            [Vector Store (FAISS)]
                                      ↑
                            [Document Embeddings]
                                      ↑
                            [Document Processing]
                                      ↑
                          [Document Loading/Corpus]
```

## Installation

### Prerequisites
- Python 3.10+
- Access to an LLM API (OpenAI, Anthropic, etc.)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   ```

5. Create a `documents` directory and add your corpus:
   ```bash
   mkdir -p documents
   # Add your PDF, DOCX, and TXT files to this directory
   ```

## Usage

### Basic Usage

```python
from document_loader import DocumentLoader
from document_processor import DocumentProcessor
from embeddings import EmbeddingFactory
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline

# Load and process documents
loader = DocumentLoader("documents")
docs = loader.load_documents()

# Process documents
processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
chunks = processor.split_documents(docs)

# Create vector store
vs_manager = VectorStoreManager()
vector_store = vs_manager.create_vector_store(
    chunks, 
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    store_name="my_knowledge_base"
)

# Initialize RAG pipeline
rag = RAGPipeline(
    vector_store=vector_store,
    llm_provider="openai",
    model_name="gpt-3.5-turbo"
)

# Query the system
response = rag.query("What is retrieval-augmented generation?")
print(response)
```

### Command-Line Interface

The project includes a CLI for common operations:

```bash
# Index documents
python rag_cli.py index --docs_dir documents --store_name my_knowledge_base

# Query the system
python rag_cli.py query --store_name my_knowledge_base --query "What is RAG?"

# Evaluate performance
python rag_cli.py evaluate --store_name my_knowledge_base --test_file test_queries.json
```

## Project Structure

```
rag-system/
├── documents/                # Document corpus directory
├── vector_store/            # Vector store indexes
├── document_loader.py       # Document loading module
├── document_processor.py    # Document chunking module
├── embeddings.py            # Embedding models module
├── vector_store.py          # Vector storage module
├── retrieval.py             # Retrieval strategies module
├── rag_pipeline.py          # Main RAG pipeline
├── evaluation.py            # Evaluation metrics and tools
├── rag_cli.py               # Command-line interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Configuration

The system can be configured using environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE` | Size of document chunks | 512 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `VECTOR_STORE_DIR` | Directory for vector indexes | "vector_store" |

## RAG Pipeline Components

### Document Loader

The `DocumentLoader` class supports various document formats:
- PDF files using PyPDFLoader
- Text files using TextLoader
- Word documents using Docx2txtLoader
- Fallback to UnstructuredFileLoader for other formats

### Document Processor

The `DocumentProcessor` provides document chunking with different strategies:
- Recursive character splitting (default)
- Character-based splitting
- Token-based splitting

You can configure chunk size and overlap for optimal retrieval.

### Embedding Models

The `EmbeddingFactory` supports multiple embedding models:
- sentence-transformers/all-MiniLM-L6-v2 (default)
- sentence-transformers/all-mpnet-base-v2
- OpenAI's text-embedding-3-small

### Vector Store

The `VectorStoreManager` handles:
- Creating FAISS vector stores
- Saving and loading stores to/from disk
- Managing multiple vector stores

### Retrieval Strategies

The system implements various retrieval strategies:
- Basic similarity search
- Maximum Marginal Relevance (MMR)
- Metadata filtering
- Hybrid semantic and keyword search

## Advanced Features

### Query Rewriting

The system can optionally rewrite user queries using an LLM to improve retrieval performance.

### Self-Checking Generation

The RAG pipeline can assess its own answers and regenerate responses if they don't meet quality thresholds.

### Hierarchical Retrieval

For large document collections, the system supports two-stage retrieval with coarse-grained and fine-grained passes.

## Evaluation

The system includes tools for evaluating RAG performance:

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate_pipeline(
    rag_pipeline,
    test_queries=["What is RAG?", "How does FAISS work?"],
    ground_truth=["RAG is...", "FAISS is..."]
)

evaluator.print_results(results)
```

Performance metrics include:
- Retrieval precision, recall, and F1
- Answer relevance and correctness
- Generation quality
- Query processing time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
