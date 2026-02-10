"""
Database Performance Tuning - December Phase
Optimizes query speed and storage efficiency
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .vector_db import VectorDatabase, FAISSVectorDB
import faiss


class PerformanceProfiler:
    """Profiles and optimizes vector database performance."""
    
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize performance profiler.
        
        Args:
            vector_db: Vector database instance to profile
        """
        self.vector_db = vector_db
        self.profile_data = []
    
    def profile_search(self, query_vectors: np.ndarray, k: int = 5, 
                     num_runs: int = 10) -> Dict:
        """
        Profile search performance.
        
        Args:
            query_vectors: Sample query vectors
            k: Number of results per query
            num_runs: Number of runs for averaging
            
        Returns:
            Performance metrics dictionary
        """
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            for query_vec in query_vectors:
                self.vector_db.search(query_vec, k=k)
            elapsed = time.time() - start
            times.append(elapsed / len(query_vectors))
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        metrics = {
            'avg_query_time_ms': avg_time * 1000,
            'std_query_time_ms': std_time * 1000,
            'throughput_queries_per_sec': throughput,
            'num_runs': num_runs,
            'num_queries': len(query_vectors)
        }
        
        self.profile_data.append(('search', metrics))
        return metrics
    
    def profile_add(self, vectors: np.ndarray, batch_sizes: List[int]) -> Dict:
        """
        Profile vector addition performance for different batch sizes.
        
        Args:
            vectors: Vectors to add
            batch_sizes: List of batch sizes to test
            
        Returns:
            Performance metrics for each batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            for _ in range(3):  # Multiple runs
                # Create test metadata
                metadata = [{'chunk_id': f'test_{i}'} for i in range(batch_size)]
                
                start = time.time()
                self.vector_db.add_vectors(vectors[:batch_size], metadata)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time if avg_time > 0 else 0
            
            results[batch_size] = {
                'avg_time_sec': avg_time,
                'throughput_vectors_per_sec': throughput,
                'time_per_vector_ms': (avg_time / batch_size) * 1000
            }
        
        return results
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        stats = self.vector_db.get_stats()
        
        # Check index type
        if isinstance(self.vector_db, FAISSVectorDB):
            if stats['total_vectors'] > 100000 and not self.vector_db.index_factory:
                recommendations.append(
                    "Consider using approximate index (IVF) for large datasets (>100K vectors)"
                )
            
            if not self.vector_db.use_gpu and stats['total_vectors'] > 50000:
                recommendations.append(
                    "Consider using GPU acceleration for faster search"
                )
        
        # Check batch size
        if len(self.profile_data) > 0:
            last_profile = self.profile_data[-1][1]
            if last_profile.get('throughput_queries_per_sec', 0) < 10:
                recommendations.append(
                    "Search throughput is low. Consider optimizing index type or using GPU"
                )
        
        return recommendations


def optimize_faiss_index(embedding_dim: int, num_vectors: int,
                         index_type: str = "cosine") -> str:
    """
    Recommend optimal FAISS index factory string based on dataset size.
    
    Args:
        embedding_dim: Embedding dimension
        num_vectors: Number of vectors in dataset
        index_type: "cosine", "L2", or "IP"
        
    Returns:
        Recommended index factory string
    """
    if num_vectors < 10000:
        # Small dataset: use flat index
        if index_type == "cosine":
            return "Flat"  # Will normalize separately
        elif index_type == "L2":
            return "Flat"
        else:
            return "Flat"
    
    elif num_vectors < 100000:
        # Medium dataset: use IVF with moderate number of clusters
        nlist = min(1024, num_vectors // 10)
        if index_type == "cosine":
            return f"IVF{nlist},Flat"
        else:
            return f"IVF{nlist},Flat"
    
    else:
        # Large dataset: use IVF with quantization
        nlist = min(4096, num_vectors // 100)
        if index_type == "cosine":
            return f"IVF{nlist},PQ32"  # Product Quantization for compression
        else:
            return f"IVF{nlist},PQ32"


def benchmark_index_types(vectors: np.ndarray, 
                         query_vectors: np.ndarray,
                         embedding_dim: int) -> pd.DataFrame:
    """
    Benchmark different FAISS index types.
    
    Args:
        vectors: Database vectors
        query_vectors: Query vectors for testing
        embedding_dim: Embedding dimension
        
    Returns:
        DataFrame with benchmark results
    """
    index_configs = [
        ("Flat", "cosine", None),
        ("IVF1024,Flat", "cosine", 1024),
        ("IVF4096,Flat", "cosine", 4096),
    ]
    
    results = []
    
    for index_factory, index_type, nlist in index_configs:
        print(f"\nTesting: {index_factory}")
        
        # Create index
        db = FAISSVectorDB(
            embedding_dim=embedding_dim,
            index_type=index_type,
            index_factory=index_factory if index_factory != "Flat" else None
        )
        
        # Train if needed
        if nlist and index_factory != "Flat":
            training_vectors = vectors[:min(10000, len(vectors))]
            db.train_index(training_vectors)
        
        # Add vectors
        metadata = [{'chunk_id': f'v_{i}'} for i in range(len(vectors))]
        start = time.time()
        db.add_vectors(vectors, metadata)
        add_time = time.time() - start
        
        # Search benchmark
        search_times = []
        for qv in query_vectors[:10]:  # Test on 10 queries
            start = time.time()
            db.search(qv, k=5)
            search_times.append(time.time() - start)
        
        avg_search_time = np.mean(search_times)
        
        # Get index size (approximate)
        if hasattr(db.index, 'ntotal'):
            # Rough estimate: ntotal * dim * 4 bytes (float32)
            index_size_mb = (db.index.ntotal * embedding_dim * 4) / (1024 * 1024)
        else:
            index_size_mb = 0
        
        results.append({
            'index_type': index_factory,
            'add_time_sec': add_time,
            'avg_search_time_ms': avg_search_time * 1000,
            'index_size_mb': index_size_mb,
            'throughput_queries_per_sec': 1.0 / avg_search_time if avg_search_time > 0 else 0
        })
    
    return pd.DataFrame(results)


def optimize_batch_size(embedding_model, chunks_df: pd.DataFrame,
                       test_sizes: List[int] = [16, 32, 64, 128]) -> Dict:
    """
    Find optimal batch size for embedding generation.
    
    Args:
        embedding_model: SentenceTransformer model
        chunks_df: DataFrame with chunks
        test_sizes: Batch sizes to test
        
    Returns:
        Dictionary with optimal batch size and performance metrics
    """
    sample_size = min(1000, len(chunks_df))
    sample_chunks = chunks_df.sample(n=sample_size, random_state=42)
    texts = sample_chunks['chunk_text'].tolist()
    
    results = {}
    
    for batch_size in test_sizes:
        times = []
        for _ in range(3):
            start = time.time()
            embeddings = embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = sample_size / avg_time
        
        results[batch_size] = {
            'avg_time_sec': avg_time,
            'throughput_chunks_per_sec': throughput,
            'memory_estimate_mb': (batch_size * embeddings.shape[1] * 4) / (1024 * 1024)
        }
    
    # Find optimal (balance between speed and memory)
    optimal = max(results.items(), 
                  key=lambda x: x[1]['throughput_chunks_per_sec'] / max(x[1]['memory_estimate_mb'], 0.1))
    
    return {
        'optimal_batch_size': optimal[0],
        'results': results,
        'recommendation': f"Optimal batch size: {optimal[0]} (throughput: {optimal[1]['throughput_chunks_per_sec']:.1f} chunks/sec)"
    }

