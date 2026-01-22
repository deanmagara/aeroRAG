"""
Embedding System - November Phase
Selects and implements embedding model; generates document embeddings for NASA STI content
"""

import numpy as np
import pandas as pd
import time
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from tqdm import tqdm


class EmbeddingSystem:
    """
    Manages embedding generation and vector storage for the RAG system.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Name of the sentence transformer model
                       Options: "all-MiniLM-L6-v2" (fast, 384 dims),
                               "all-mpnet-base-v2" (better quality, 768 dims),
                               "sentence-transformers/all-MiniLM-L6-v2"
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        print(f"\n🔧 Initializing Embedding System...")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   ✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        
        self.index = None
        self.chunks_df = None
        
    def generate_embeddings(self, chunks_df: pd.DataFrame, batch_size: int = 32, 
                           show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for all chunks in the DataFrame.
        
        Args:
            chunks_df: DataFrame with 'chunk_text' column
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_chunks, embedding_dim)
        """
        start_time = time.time()
        print(f"\n📊 Generating Embeddings...")
        print(f"   Total chunks: {len(chunks_df):,}")
        print(f"   Batch size: {batch_size}")
        
        texts = chunks_df['chunk_text'].tolist()
        
        if show_progress:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        print(f"   ✅ Embeddings generated. Shape: {embeddings.shape}")
        print(f"   (Embedding generation took {time.time() - start_time:.2f} seconds)")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = "L2") -> faiss.Index:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of index ("L2" for Euclidean distance, "IP" for inner product)
            
        Returns:
            FAISS index object
        """
        start_time = time.time()
        print(f"\n🔍 Building FAISS Index...")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Index type: {index_type}")
        
        # Normalize embeddings for cosine similarity (using inner product)
        if index_type == "cosine":
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "L2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "IP":
            index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        self.index = index
        print(f"   ✅ Index built. Total vectors: {index.ntotal:,}")
        print(f"   (Index building took {time.time() - start_time:.2f} seconds)")
        
        return index
    
    def search(self, query: str, k: int = 5, return_distances: bool = False) -> List[dict]:
        """
        Search for similar chunks given a query.
        
        Args:
            query: Query text
            k: Number of results to return
            return_distances: Whether to return similarity distances
            
        Returns:
            List of dictionaries with chunk information and optionally distances
        """
        if self.index is None or self.chunks_df is None:
            raise ValueError("Index and chunks_df must be set. Call build_faiss_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Normalize if using cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks_df):
                result = {
                    'chunk_id': self.chunks_df.iloc[idx]['chunk_id'],
                    'document_id': self.chunks_df.iloc[idx]['document_id'],
                    'title': self.chunks_df.iloc[idx]['title'],
                    'chunk_text': self.chunks_df.iloc[idx]['chunk_text'],
                    'rank': i + 1
                }
                if return_distances:
                    result['distance'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """
        Save the embedding system (index and metadata) to disk.
        
        Args:
            filepath: Base filepath (will create .index and .pkl files)
        """
        if self.index is None or self.chunks_df is None:
            raise ValueError("Index and chunks_df must be set before saving.")
        
        print(f"\n💾 Saving Embedding System...")
        
        # Save FAISS index
        index_path = f"{filepath}.index"
        faiss.write_index(self.index, index_path)
        print(f"   ✅ Index saved to {index_path}")
        
        # Save metadata
        metadata_path = f"{filepath}.pkl"
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'chunks_df': self.chunks_df
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   ✅ Metadata saved to {metadata_path}")
    
    def load(self, filepath: str):
        """
        Load the embedding system from disk.
        
        Args:
            filepath: Base filepath (will load .index and .pkl files)
        """
        print(f"\n📂 Loading Embedding System...")
        
        # Load FAISS index
        index_path = f"{filepath}.index"
        self.index = faiss.read_index(index_path)
        print(f"   ✅ Index loaded from {index_path}")
        
        # Load metadata
        metadata_path = f"{filepath}.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.embedding_dim = metadata['embedding_dim']
        self.chunks_df = metadata['chunks_df']
        
        # Reload model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        print(f"   ✅ Metadata loaded. Model: {self.model_name}, Dims: {self.embedding_dim}")
        print(f"   ✅ Total vectors in index: {self.index.ntotal:,}")


def process_complete_dataset(chunks_df: pd.DataFrame, 
                            model_name: str = "all-MiniLM-L6-v2",
                            index_type: str = "cosine",
                            batch_size: int = 32,
                            save_path: Optional[str] = None) -> EmbeddingSystem:
    """
    Process the complete NASA STI dataset through the embedding pipeline.
    
    Args:
        chunks_df: DataFrame with chunks ready for embedding
        model_name: Embedding model to use
        index_type: Type of FAISS index
        batch_size: Batch size for embedding generation
        save_path: Optional path to save the embedding system
        
    Returns:
        Configured EmbeddingSystem instance
    """
    print("\n" + "="*60)
    print("COMPLETE DATASET EMBEDDING PROCESSING")
    print("="*60)
    
    # Initialize embedding system
    embedding_system = EmbeddingSystem(model_name=model_name)
    
    # Generate embeddings
    embeddings = embedding_system.generate_embeddings(chunks_df, batch_size=batch_size)
    
    # Store chunks_df for search functionality
    embedding_system.chunks_df = chunks_df
    
    # Build index
    embedding_system.build_faiss_index(embeddings, index_type=index_type)
    
    # Save if path provided
    if save_path:
        embedding_system.save(save_path)
    
    print("\n✅ Complete dataset processing finished!")
    return embedding_system

