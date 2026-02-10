# NASA STI Repository RAG System - Pipeline Documentation

## Overview

This document provides comprehensive documentation of the data flow and processing architecture for the aeroRAG system, which processes NASA Scientific and Technical Information (STI) Repository data for offline Retrieval-Augmented Generation (RAG).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│  (September Phase)                                           │
│  - Download from NASA STI API                               │
│  - Decompress gzip files                                    │
│  - Parse JSON/NDJSON                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA PROCESSING PIPELINE                       │
│  (October Phase)                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Data Cleaning & Preprocessing                     │   │
│  │    - Fill missing values                             │   │
│  │    - Flatten nested structures                       │   │
│  │    - Validate records                                │   │
│  │    - Generate text_source field                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Intelligent Chunking                              │   │
│  │    - RecursiveCharacterTextSplitter                  │   │
│  │    - Optimal chunk size/overlap                      │   │
│  │    - Preserve semantic boundaries                    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDING GENERATION                            │
│  (November Phase)                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Model Selection & Initialization                  │   │
│  │    - Sentence Transformers                           │   │
│  │    - Model: all-MiniLM-L6-v2 (default)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Batch Embedding Generation                       │   │
│  │    - Process chunks in batches                       │   │
│  │    - Generate vector embeddings                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Vector Index Building                             │   │
│  │    - FAISS index for similarity search               │   │
│  │    - Cosine similarity (default)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RAG QUERY SYSTEM                               │
│  - Query embedding                                          │
│  - Similarity search                                        │
│  - Context retrieval                                        │
│  - LLM generation (Ollama/Llama4)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Acquisition (September)

### Source
- **URL**: `https://ntrs.staging.sti.appdat.jsc.nasa.gov/api/docs/ntrs-public-metadata.json.gz?attachment=true`
- **Format**: Compressed JSON (gzip)
- **Structure**: Dictionary where keys are document IDs and values are document metadata

### Implementation
- **Module**: `src/data_acquisition.py`
- **Functions**:
  - `load_data_from_url(url)`: Downloads and decompresses data from URL
  - `load_data_from_file(file_path)`: Loads data from local JSON file

### Data Quality Assessment
- Total records: Variable (typically 100,000+ documents)
- Fields include: `title`, `abstract`, `keywords`, `authorAffiliations`, etc.
- Validation: Records with missing titles are filtered out

---

## Phase 2: Data Processing Pipeline (October)

### 2.1 Data Cleaning & Preprocessing

**Module**: `src/data_processing.py` → `preprocess_data()`

**Steps**:
1. **Missing Value Handling**:
   - `abstract`: Fill with empty string
   - `keywords`: Convert to empty list if not already a list

2. **Data Validation**:
   - Filter records with empty or missing titles
   - Log number of filtered records

3. **Feature Engineering**:
   - **Author Flattening**: Extract author names from nested `authorAffiliations` structure
   - **Keyword Flattening**: Convert keyword lists to pipe-separated strings

4. **Text Source Generation**:
   - Create `text_source` field by concatenating:
     ```
     TITLE: {title}
     ABSTRACT: {abstract}
     AUTHORS: {authors_flat}
     KEYWORDS: {keywords_flat}
     ```

**Output**: DataFrame with columns: `document_id`, `title`, `abstract`, `text_source`

### 2.2 Chunking Strategy

**Module**: `src/data_processing.py` → `chunk_data()`

**Tool**: `RecursiveCharacterTextSplitter` from LangChain

**Parameters**:
- **Chunk Size**: 512 characters (default, configurable)
- **Chunk Overlap**: 50 characters (default, configurable)
- **Separators**: `["\n\n", "\n", ".", " ", ""]` (prioritizes semantic boundaries)

**Strategy**:
- Maintains semantic boundaries by prioritizing splits at:
  1. Paragraph breaks (`\n\n`)
  2. Line breaks (`\n`)
  3. Sentence endings (`.`)
  4. Word boundaries (` `)
  5. Character boundaries (fallback)

**Output**: DataFrame with columns:
- `document_id`: Original document ID
- `title`: Document title
- `chunk_id`: Unique chunk identifier (`{document_id}-{chunk_number}`)
- `chunk_text`: Text content of the chunk
- `chunk_size`: Character count of the chunk

**Performance**:
- Average chunks per document: ~2-4 (depends on document length)
- Processing speed: ~1000-5000 chunks/second (depends on hardware)

---

## Phase 3: Embedding System (November)

### 3.1 Model Selection

**Default Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: Fast (good for large datasets)
- **Quality**: Good balance between speed and quality

**Alternative Models**:
- `all-mpnet-base-v2`: 768 dimensions, higher quality, slower
- `paraphrase-MiniLM-L6-v2`: Optimized for semantic similarity

**Module**: `src/embeddings.py` → `EmbeddingSystem`

### 3.2 Embedding Generation

**Process**:
1. Initialize SentenceTransformer model
2. Process chunks in batches (default: 32 chunks per batch)
3. Generate vector embeddings for each chunk
4. Store embeddings as numpy array

**Output**: 
- Embedding matrix: `(n_chunks, embedding_dim)`
- Example: 100,000 chunks × 384 dimensions = 38.4M float32 values (~147 MB)

### 3.3 Vector Index Building

**Tool**: FAISS (Facebook AI Similarity Search)

**Index Type**: 
- **Cosine Similarity** (default): Normalized L2 + Inner Product
- **L2 Distance**: Euclidean distance
- **Inner Product**: Dot product

**Implementation**:
- `IndexFlatIP` or `IndexFlatL2` for exact search
- Fast similarity search: O(n) for exact, O(log n) for approximate

**Storage**:
- Index file: `{path}.index` (FAISS binary format)
- Metadata file: `{path}.pkl` (pickle with model info and chunks DataFrame)

### 3.4 Search Functionality

**Query Process**:
1. Generate embedding for query text
2. Search FAISS index for k nearest neighbors
3. Return chunk information with similarity scores

**API**:
```python
results = embedding_system.search("propulsion systems", k=5)
```

**Output**: List of dictionaries with:
- `chunk_id`: Unique chunk identifier
- `document_id`: Source document ID
- `title`: Document title
- `chunk_text`: Chunk content
- `rank`: Search rank (1-based)
- `distance`: Similarity distance (optional)

---

## Data Flow Summary

```
Raw JSON/NDJSON
    ↓
DataFrame (raw)
    ↓ [preprocess_data]
DataFrame (cleaned, with text_source)
    ↓ [chunk_data]
DataFrame (chunks)
    ↓ [generate_embeddings]
Numpy Array (embeddings)
    ↓ [build_faiss_index]
FAISS Index + Metadata
    ↓ [search]
Query Results
```

---

## File Structure

```
aeroRAG/
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py      # September: Data download/loading
│   ├── data_processing.py        # October: Cleaning, preprocessing, chunking
│   ├── embeddings.py             # November: Embedding generation & indexing
│   ├── benchmarking.py           # November: Optimization utilities
│   └── pipeline.py               # Complete pipeline orchestration
├── data/
│   ├── raw/                      # Raw NASA STI data
│   ├── processed/                # Processed chunks (CSV)
│   └── embeddings/               # FAISS indices and metadata
├── docs/
│   └── PIPELINE_DOCUMENTATION.md # This file
├── app.py                        # Streamlit interface
├── script.py                     # Legacy script
├── script.ipynb                  # Jupyter notebook
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

---

## Configuration Parameters

### Chunking Parameters
- `CHUNK_SIZE`: 512 (characters)
- `CHUNK_OVERLAP`: 50 (characters)

### Embedding Parameters
- `MODEL_NAME`: "all-MiniLM-L6-v2"
- `BATCH_SIZE`: 32
- `INDEX_TYPE`: "cosine"

### Performance Tuning
- Adjust `CHUNK_SIZE` based on document length distribution
- Adjust `BATCH_SIZE` based on available memory
- Choose embedding model based on quality vs. speed tradeoff

---

## Benchmarking & Optimization

**Module**: `src/benchmarking.py`

**Utilities**:
1. `benchmark_chunk_parameters()`: Test different chunk sizes/overlaps
2. `benchmark_embedding_models()`: Compare embedding models
3. `find_optimal_parameters()`: Find optimal chunk parameters

**Metrics Tracked**:
- Total chunks generated
- Chunks per document
- Processing time
- Embedding throughput (chunks/second)
- Memory usage

---

## Next Steps (Future Phases)

1. **LLM Integration**: Connect to Ollama/Llama4 for generation
2. **Query Orchestration**: Build RAG query system
3. **Evaluation**: Implement retrieval quality metrics
4. **Deployment**: Package for offline deployment

---

## References

- NASA STI Repository: https://ntrs.nasa.gov/
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss

