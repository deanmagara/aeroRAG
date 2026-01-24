# December & January Phases Documentation

## Overview

This document describes the December and January phase implementations for the aeroRAG system, covering vector database creation, similarity search, performance tuning, LLM integration, RAG pipeline assembly, and hallucination prevention.

---

## December Phase (20 hours)

### Vector Database Creation

**Module**: `src/vector_db.py`

#### FAISS Implementation (`FAISSVectorDB`)

- **Features**:
  - Multiple index types: cosine similarity, L2 distance, inner product
  - GPU acceleration support
  - Advanced index factories (IVF, PQ) for large datasets
  - Persistent storage with metadata
  - Training support for approximate indexes

- **Usage**:
```python
from src.vector_db import FAISSVectorDB

db = FAISSVectorDB(
    embedding_dim=384,
    index_type="cosine",
    use_gpu=False,
    index_factory="IVF1024,Flat"  # Optional for large datasets
)

# Add vectors
db.add_vectors(embeddings, metadata)

# Search
results = db.search(query_embedding, k=5)

# Save/Load
db.save("path/to/db")
db.load("path/to/db")
```

#### ChromaDB Implementation (`ChromaVectorDB`)

- **Features**:
  - Automatic persistence
  - Built-in metadata management
  - Collection-based organization
  - Simple API

- **Usage**:
```python
from src.vector_db import ChromaVectorDB

db = ChromaVectorDB(
    collection_name="ntrs_documents",
    persist_directory="data/chroma_db"
)
```

#### Factory Function

```python
from src.vector_db import create_vector_db

db = create_vector_db(
    db_type="faiss",  # or "chroma"
    embedding_dim=384,
    index_type="cosine"
)
```

---

### Similarity Search Development

**Module**: `src/similarity_search.py`

#### SimilaritySearchEngine

- **Features**:
  - Query embedding generation
  - Similarity search with thresholds
  - Optional reranking with cross-encoders
  - Metadata filtering
  - Hybrid search (vector + keyword)
  - Batch search

- **Usage**:
```python
from src.similarity_search import SimilaritySearchEngine
from src.vector_db import create_vector_db
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_db = create_vector_db("faiss", embedding_dim=384)

search_engine = SimilaritySearchEngine(
    embedding_model=embedding_model,
    vector_db=vector_db,
    rerank=True,  # Optional reranking
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Search
results = search_engine.search(
    "propulsion systems",
    k=5,
    min_similarity=0.3,
    rerank_top_k=10
)
```

#### RetrievalSystem

- **Features**:
  - Context assembly
  - Result formatting
  - Deduplication by document
  - Source tracking

- **Usage**:
```python
from src.similarity_search import RetrievalSystem

retrieval = RetrievalSystem(search_engine)

# Retrieve with context
result = retrieval.retrieve(
    "aerospace materials",
    k=5,
    format_context=True
)

# Deduplicated retrieval
result = retrieval.retrieve_with_deduplication(
    "propulsion",
    k=5
)
```

---

### Database Performance Tuning

**Module**: `src/performance_tuning.py`

#### PerformanceProfiler

- **Features**:
  - Search performance profiling
  - Batch addition benchmarking
  - Performance recommendations

- **Usage**:
```python
from src.performance_tuning import PerformanceProfiler

profiler = PerformanceProfiler(vector_db)

# Profile search
metrics = profiler.profile_search(query_vectors, k=5, num_runs=10)
print(f"Throughput: {metrics['throughput_queries_per_sec']} queries/sec")

# Get recommendations
recommendations = profiler.get_recommendations()
```

#### Index Optimization

```python
from src.performance_tuning import optimize_faiss_index, benchmark_index_types

# Get optimal index type
optimal_index = optimize_faiss_index(
    embedding_dim=384,
    num_vectors=100000,
    index_type="cosine"
)

# Benchmark different index types
results_df = benchmark_index_types(vectors, query_vectors, embedding_dim=384)
```

#### Batch Size Optimization

```python
from src.performance_tuning import optimize_batch_size

result = optimize_batch_size(
    embedding_model,
    chunks_df,
    test_sizes=[16, 32, 64, 128]
)

print(f"Optimal batch size: {result['optimal_batch_size']}")
```

---

## January Phase (20 hours)

### LLaMA4-Ollama Integration

**Module**: `src/llm_integration.py`

#### OllamaLLM

- **Features**:
  - Direct Ollama API integration
  - Streaming support
  - Chat completion interface
  - Configurable temperature and max tokens
  - Connection testing

- **Usage**:
```python
from src.llm_integration import OllamaLLM

llm = OllamaLLM(
    model_name="llama3.2:latest",  # or "llama4" when available
    base_url="http://localhost:11434",
    temperature=0.2,
    max_tokens=512
)

# Generate
response = llm.generate("What is NASA's research focus?")

# Chat
messages = [
    {"role": "user", "content": "Hello"}
]
response = llm.chat(messages)
```

#### Connection Testing

```python
from src.llm_integration import test_ollama_connection, list_available_models

# Test connection
if test_ollama_connection():
    print("Ollama is running")
    
# List models
models = list_available_models()
print(f"Available models: {models}")
```

---

### RAG Pipeline Assembly

**Module**: `src/rag_pipeline.py`

#### RAGPipeline

- **Features**:
  - Complete RAG workflow
  - Context formatting
  - Prompt templating
  - Response generation

- **Usage**:
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    retrieval_system=retrieval_system,
    llm=llm,
    max_context_length=2000
)

# Query
response = rag.query(
    "What propulsion technologies are studied?",
    k=5,
    temperature=0.2
)

print(response['answer'])
print(f"Sources: {response['sources']}")
```

#### ConversationalRAG

- **Features**:
  - Conversation history
  - Context-aware queries
  - History management

- **Usage**:
```python
from src.rag_pipeline import ConversationalRAG

conversational_rag = ConversationalRAG(rag_pipeline, max_history=5)

# Chat
response1 = conversational_rag.chat("What is NASA's research focus?")
response2 = conversational_rag.chat("Tell me more about that")

# Clear history
conversational_rag.clear_history()
```

---

### Hallucination Prevention

**Module**: `src/hallucination_prevention.py`

#### HallucinationPrevention

- **Features**:
  - Response validation
  - Citation checking
  - Source overlap analysis
  - Uncertainty detection
  - Response filtering

- **Usage**:
```python
from src.hallucination_prevention import HallucinationPrevention

prevention = HallucinationPrevention(
    retrieval_system=retrieval_system,
    similarity_threshold=0.3,
    require_citation=True
)

# Validate response
validation = prevention.validate_response(
    query="What is propulsion?",
    answer=generated_answer,
    retrieved_sources=sources
)

if validation['confidence'] < 0.5:
    # Filter response
    filtered_answer, info = prevention.filter_response(
        query, answer, sources, min_confidence=0.5
    )
```

#### GroundedRAGPipeline

- **Features**:
  - Built-in hallucination prevention
  - Automatic validation
  - Confidence-based filtering
  - Source requirement enforcement

- **Usage**:
```python
from src.hallucination_prevention import GroundedRAGPipeline

grounded_rag = GroundedRAGPipeline(
    rag_pipeline=rag_pipeline,
    hallucination_prevention=prevention,
    enforce_validation=True
)

# Query with automatic validation
response = grounded_rag.query(
    "What is NASA's research?",
    k=5,
    min_confidence=0.5
)

# Response includes validation info
print(f"Confidence: {response['validation']['confidence']}")
print(f"Warnings: {response['validation']['warnings']}")
```

---

## Complete RAG System

**Module**: `src/complete_rag_system.py`

The `CompleteRAGSystem` class orchestrates all components:

```python
from src.complete_rag_system import create_complete_rag_system

# Create system
rag_system = create_complete_rag_system(
    embedding_model="all-MiniLM-L6-v2",
    vector_db_type="faiss",
    llm_model="llama3.2:latest"
)

# Build from data
rag_system.build_from_data(
    data_source="file",
    file_path="ntrs-public-metadata.json",
    save_path="data/embeddings/ntrs_rag"
)

# Or load from saved
rag_system.load_from_saved("data/embeddings/ntrs_rag")

# Query
response = rag_system.query(
    "What propulsion technologies are studied?",
    k=5,
    use_grounding=True
)
```

---

## Performance Considerations

### Vector Database

- **Small datasets (<10K)**: Use flat index
- **Medium datasets (10K-100K)**: Use IVF with moderate clusters
- **Large datasets (>100K)**: Use IVF with quantization (PQ)

### Embedding Generation

- Optimal batch size: 32-64 (depends on memory)
- GPU acceleration recommended for large datasets

### Search Performance

- Flat index: O(n) exact search
- IVF index: O(log n) approximate search
- GPU acceleration: 10-100x speedup

### LLM Generation

- Temperature: 0.2-0.3 for factual responses
- Max tokens: 512-1024 for most queries
- Streaming: Recommended for better UX

---

## Best Practices

1. **Index Selection**: Choose index type based on dataset size
2. **Batch Processing**: Use appropriate batch sizes for embeddings
3. **Similarity Thresholds**: Set min_similarity to filter low-quality results
4. **Context Length**: Limit context to avoid token limits
5. **Validation**: Always use hallucination prevention for production
6. **Monitoring**: Track confidence scores and warnings

---

## References

- FAISS Documentation: https://github.com/facebookresearch/faiss
- ChromaDB Documentation: https://docs.trychroma.com/
- Ollama Documentation: https://github.com/ollama/ollama
- Sentence Transformers: https://www.sbert.net/

