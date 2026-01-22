"""
Benchmarking and Optimization Utilities - November Phase
Fine-tune chunk parameters and benchmark embedding generation performance
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .embeddings import EmbeddingSystem
from .data_processing import chunk_data


def benchmark_chunk_parameters(df_rag: pd.DataFrame, 
                               chunk_sizes: List[int] = [256, 512, 768, 1024],
                               chunk_overlaps: List[int] = [0, 25, 50, 100]) -> pd.DataFrame:
    """
    Benchmark different chunk size and overlap combinations.
    
    Args:
        df_rag: Preprocessed DataFrame
        chunk_sizes: List of chunk sizes to test
        chunk_overlaps: List of chunk overlaps to test
        
    Returns:
        DataFrame with benchmarking results
    """
    print("\n" + "="*60)
    print("CHUNK PARAMETER BENCHMARKING")
    print("="*60)
    
    results = []
    
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            if chunk_overlap >= chunk_size:
                continue  # Skip invalid combinations
                
            print(f"\n📊 Testing: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
            start_time = time.time()
            df_chunks = chunk_data(df_rag, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunking_time = time.time() - start_time
            
            # Calculate statistics
            avg_chunk_size = df_chunks['chunk_size'].mean()
            total_chunks = len(df_chunks)
            chunks_per_doc = total_chunks / len(df_rag)
            
            results.append({
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'total_chunks': total_chunks,
                'chunks_per_doc': chunks_per_doc,
                'avg_chunk_size': avg_chunk_size,
                'chunking_time_sec': chunking_time,
                'chunks_per_sec': total_chunks / chunking_time if chunking_time > 0 else 0
            })
            
            print(f"   ✅ Total chunks: {total_chunks:,}, Time: {chunking_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


def benchmark_embedding_models(chunks_df: pd.DataFrame,
                               model_names: List[str] = [
                                   "all-MiniLM-L6-v2",
                                   "all-mpnet-base-v2",
                                   "paraphrase-MiniLM-L6-v2"
                               ],
                               batch_sizes: List[int] = [16, 32, 64]) -> pd.DataFrame:
    """
    Benchmark different embedding models and batch sizes.
    
    Args:
        chunks_df: DataFrame with chunks
        model_names: List of model names to test
        batch_sizes: List of batch sizes to test
        
    Returns:
        DataFrame with benchmarking results
    """
    print("\n" + "="*60)
    print("EMBEDDING MODEL BENCHMARKING")
    print("="*60)
    
    results = []
    sample_size = min(1000, len(chunks_df))  # Use sample for faster benchmarking
    sample_chunks = chunks_df.sample(n=sample_size, random_state=42)
    
    for model_name in model_names:
        print(f"\n🤖 Testing model: {model_name}")
        
        try:
            embedding_system = EmbeddingSystem(model_name=model_name)
            
            for batch_size in batch_sizes:
                print(f"   📦 Batch size: {batch_size}")
                
                start_time = time.time()
                embeddings = embedding_system.generate_embeddings(
                    sample_chunks, 
                    batch_size=batch_size,
                    show_progress=False
                )
                embedding_time = time.time() - start_time
                
                embedding_dim = embeddings.shape[1]
                throughput = sample_size / embedding_time
                
                results.append({
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'embedding_dim': embedding_dim,
                    'samples_processed': sample_size,
                    'time_sec': embedding_time,
                    'throughput_chunks_per_sec': throughput,
                    'memory_mb': embeddings.nbytes / (1024 * 1024)  # Approximate
                })
                
                print(f"      ✅ Time: {embedding_time:.2f}s, Throughput: {throughput:.1f} chunks/sec")
                
        except Exception as e:
            print(f"   ❌ Error with model {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'batch_size': batch_size,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("EMBEDDING BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


def find_optimal_parameters(df_rag: pd.DataFrame, 
                           target_chunks_per_doc: float = 3.0,
                           min_chunk_size: int = 256,
                           max_chunk_size: int = 1024) -> Dict:
    """
    Find optimal chunk parameters based on target chunks per document.
    
    Args:
        df_rag: Preprocessed DataFrame
        target_chunks_per_doc: Target average chunks per document
        min_chunk_size: Minimum chunk size to consider
        max_chunk_size: Maximum chunk size to consider
        
    Returns:
        Dictionary with optimal parameters
    """
    print("\n" + "="*60)
    print("OPTIMAL PARAMETER FINDING")
    print("="*60)
    
    # Sample a few documents to estimate
    sample_size = min(100, len(df_rag))
    sample_df = df_rag.sample(n=sample_size, random_state=42)
    
    # Test different chunk sizes
    chunk_sizes = range(min_chunk_size, max_chunk_size + 1, 128)
    best_params = None
    best_diff = float('inf')
    
    for chunk_size in chunk_sizes:
        chunk_overlap = chunk_size // 10  # 10% overlap
        df_chunks = chunk_data(sample_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_per_doc = len(df_chunks) / len(sample_df)
        
        diff = abs(chunks_per_doc - target_chunks_per_doc)
        
        if diff < best_diff:
            best_diff = diff
            best_params = {
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'chunks_per_doc': chunks_per_doc
            }
    
    print(f"\n✅ Optimal parameters found:")
    print(f"   Chunk Size: {best_params['chunk_size']}")
    print(f"   Chunk Overlap: {best_params['chunk_overlap']}")
    print(f"   Expected Chunks per Doc: {best_params['chunks_per_doc']:.2f}")
    
    return best_params

