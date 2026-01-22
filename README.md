# aeroRAG - NASA STI Repository RAG System

An offline Retrieval-Augmented Generation (RAG) system for NASA Scientific and Technical Information (STI) Repository data, designed to work with locally deployed LLaMA4 models via Ollama.

## Project Timeline

### September (20 hours)
- ✅ Environment Setup & Architecture Design
- ✅ Data Acquisition & Exploration
- ✅ Model Research & Selection

### October (20 hours)
- ✅ Data Processing Pipeline
- ✅ Chunking Strategy Implementation
- ✅ Pipeline Documentation

### November (20 hours)
- ✅ Embedding System
- ✅ Embedding Optimization
- ✅ Initial Dataset Processing

### December (20 hours)
- ✅ Vector Database Creation (FAISS/ChromaDB)
- ✅ Similarity Search Development
- ✅ Database Performance Tuning

### January (20 hours)
- ✅ LLaMA4-Ollama Integration
- ✅ RAG Pipeline Assembly
- ✅ Hallucination Prevention

## Features

- **Data Acquisition**: Download and process NASA STI Repository NDJSON datasets
- **Intelligent Chunking**: Optimized chunking strategy for titles, abstracts, and keywords
- **Embedding Generation**: Generate vector embeddings using sentence transformers
- **Vector Database**: FAISS and ChromaDB support with persistent storage
- **Similarity Search**: Efficient retrieval with optional reranking and hybrid search
- **Performance Tuning**: Benchmarking and optimization utilities
- **LLM Integration**: Ollama/LLaMA4 integration for offline operation
- **RAG Pipeline**: Complete retrieval-augmented generation workflow
- **Hallucination Prevention**: Validation and filtering to ensure grounded responses
- **Conversational RAG**: Multi-turn conversation support with history

## Installation

### Prerequisites
- Python 3.8+
- Ollama (for LLM inference)
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd aeroRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Ollama and pull LLaMA4 model:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama4  # Replace with actual LLaMA4 variant when available
```

## Quick Start

### Option 1: Complete RAG System (Recommended)

```python
from src.complete_rag_system import create_complete_rag_system

# Create complete RAG system
rag_system = create_complete_rag_system(
    embedding_model="all-MiniLM-L6-v2",
    vector_db_type="faiss",
    llm_model="llama3.2"
)

# Build from data (or load from saved)
rag_system.build_from_data(
    data_source="file",
    file_path="ntrs-public-metadata.json",
    save_path="data/embeddings/ntrs_rag"
)

# Query with hallucination prevention
response = rag_system.query(
    "What propulsion technologies are studied at NASA?",
    k=5,
    use_grounding=True
)

print(response['answer'])
print(f"Sources: {len(response['sources'])} documents")
```

### Option 2: Step-by-Step Pipeline

```python
from src.pipeline import run_complete_pipeline

# Run pipeline from data acquisition to embeddings
df_rag, df_chunks, embedding_system = run_complete_pipeline(
    data_source="url",
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    save_embeddings=True
)

# Test search
results = embedding_system.search("propulsion systems", k=5)
for result in results:
    print(f"{result['title']}: {result['chunk_text'][:200]}...")
```

### Option 2: Step-by-Step

```python
from src.data_acquisition import load_data_from_url
from src.data_processing import run_processing_pipeline
from src.embeddings import process_complete_dataset

# 1. Acquire data
df_raw = load_data_from_url("https://ntrs.staging.sti.appdat.jsc.nasa.gov/api/docs/ntrs-public-metadata.json.gz?attachment=true")

# 2. Process and chunk
df_rag, df_chunks = run_processing_pipeline(df_raw, chunk_size=512, chunk_overlap=50)

# 3. Generate embeddings
embedding_system = process_complete_dataset(df_chunks, save_path="data/embeddings/ntrs")
```

### Option 3: Using the Notebook

Open `script.ipynb` in Jupyter for interactive exploration.

## Project Structure

```
aeroRAG/
├── src/
│   ├── data_acquisition.py      # Data download and loading
│   ├── data_processing.py        # Cleaning, preprocessing, chunking
│   ├── embeddings.py             # Embedding generation & FAISS indexing
│   ├── benchmarking.py           # Optimization utilities
│   └── pipeline.py               # Complete pipeline orchestration
├── data/
│   ├── raw/                      # Raw NASA STI data
│   ├── processed/                # Processed chunks
│   └── embeddings/               # FAISS indices
├── docs/
│   └── PIPELINE_DOCUMENTATION.md # Comprehensive pipeline docs
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

### Chunking Parameters
- **Chunk Size**: 512 characters (default)
- **Chunk Overlap**: 50 characters (default)

### Embedding Model
- **Default**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Alternatives**: `all-mpnet-base-v2` (768 dimensions, higher quality)

### Data Source
- **URL**: NASA STI API endpoint (default)
- **Local**: JSON file path (alternative)

## Benchmarking & Optimization

```python
from src.benchmarking import benchmark_chunk_parameters, benchmark_embedding_models

# Benchmark chunk parameters
results = benchmark_chunk_parameters(df_rag, 
                                    chunk_sizes=[256, 512, 768, 1024],
                                    chunk_overlaps=[0, 25, 50, 100])

# Benchmark embedding models
embedding_results = benchmark_embedding_models(df_chunks,
                                               model_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
```

## Web Interface

Run the Streamlit app:

```bash
streamlit run app.py
```

The interface allows you to:
- Query the RAG system
- Adjust temperature and max tokens
- View conversation history
- See source documents

## Documentation

- **Pipeline Documentation**: [docs/PIPELINE_DOCUMENTATION.md](docs/PIPELINE_DOCUMENTATION.md) - September-November phases
- **December & January Phases**: [docs/DECEMBER_JANUARY_PHASES.md](docs/DECEMBER_JANUARY_PHASES.md) - Vector DB, RAG pipeline, LLM integration

## Development Status

- ✅ Data acquisition pipeline
- ✅ Data processing and chunking
- ✅ Embedding system
- ✅ Vector database (FAISS/ChromaDB)
- ✅ Similarity search and retrieval
- ✅ Performance tuning utilities
- ✅ LLM integration (Ollama)
- ✅ Complete RAG pipeline
- ✅ Hallucination prevention
- ✅ Conversational RAG
- ⏳ Evaluation metrics
- ⏳ Advanced reranking

## License

[Add your license here]

## Acknowledgments

- NASA STI Repository for providing the data
- LangChain for text splitting utilities
- Sentence Transformers for embedding models
- FAISS for vector search
