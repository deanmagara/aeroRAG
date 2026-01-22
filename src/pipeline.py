"""
Complete Pipeline Orchestration
Combines all phases: data acquisition, processing, chunking, and embedding
"""

import pandas as pd
from typing import Optional, Tuple
from .data_acquisition import load_data_from_url, load_data_from_file
from .data_processing import run_processing_pipeline
from .embeddings import process_complete_dataset, EmbeddingSystem


# Configuration
NASA_STI_URL = "https://ntrs.staging.sti.appdat.jsc.nasa.gov/api/docs/ntrs-public-metadata.json.gz?attachment=true"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def run_complete_pipeline(
    data_source: str = "url",
    file_path: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    save_embeddings: bool = True,
    embeddings_path: str = "data/embeddings/ntrs_embeddings",
    save_chunks: bool = False,
    chunks_path: str = "data/processed/ntrs_chunks.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, EmbeddingSystem]:
    """
    Run the complete RAG pipeline from data acquisition to embeddings.
    
    Args:
        data_source: "url" or "file"
        file_path: Path to local file if data_source is "file"
        chunk_size: Chunk size parameter
        chunk_overlap: Chunk overlap parameter
        embedding_model: Embedding model name
        save_embeddings: Whether to save embeddings to disk
        embeddings_path: Path to save embeddings
        save_chunks: Whether to save chunks to CSV
        chunks_path: Path to save chunks CSV
        
    Returns:
        Tuple of (preprocessed_df, chunks_df, embedding_system)
    """
    print("\n" + "="*70)
    print("AERORAG - COMPLETE PIPELINE EXECUTION")
    print("="*70)
    
    # Phase 1: Data Acquisition (September)
    print("\n📥 PHASE 1: DATA ACQUISITION")
    print("-" * 70)
    if data_source == "url":
        df_raw = load_data_from_url(NASA_STI_URL)
    else:
        if file_path is None:
            raise ValueError("file_path must be provided when data_source is 'file'")
        df_raw = load_data_from_file(file_path)
    
    if df_raw is None:
        raise ValueError("Data acquisition failed. Cannot continue pipeline.")
    
    # Phase 2: Data Processing & Chunking (October)
    print("\n🔧 PHASE 2: DATA PROCESSING & CHUNKING")
    print("-" * 70)
    df_rag, df_chunks = run_processing_pipeline(
        df_raw, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Save chunks if requested
    if save_chunks:
        import os
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        df_chunks.to_csv(chunks_path, index=False)
        print(f"\n💾 Chunks saved to {chunks_path}")
    
    # Phase 3: Embedding Generation (November)
    print("\n🧠 PHASE 3: EMBEDDING GENERATION")
    print("-" * 70)
    
    if save_embeddings:
        import os
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    
    embedding_system = process_complete_dataset(
        df_chunks,
        model_name=embedding_model,
        index_type="cosine",
        batch_size=32,
        save_path=embeddings_path if save_embeddings else None
    )
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"   Documents processed: {len(df_rag):,}")
    print(f"   Total chunks: {len(df_chunks):,}")
    print(f"   Embeddings generated: {embedding_system.index.ntotal:,}")
    print(f"   Embedding dimension: {embedding_system.embedding_dim}")
    
    return df_rag, df_chunks, embedding_system


if __name__ == "__main__":
    # Example usage
    df_rag, df_chunks, embedding_system = run_complete_pipeline(
        data_source="file",  # or "url"
        file_path="ntrs-public-metadata.json",
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        save_embeddings=True
    )
    
    # Test search
    print("\n🔍 Testing search functionality...")
    results = embedding_system.search("propulsion systems", k=3)
    for result in results:
        print(f"\nRank {result['rank']}: {result['title']}")
        print(f"  {result['chunk_text'][:200]}...")

