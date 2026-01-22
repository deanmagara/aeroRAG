"""
Vector Database - December Phase
Implements vector storage using FAISS/ChromaDB with indexing and persistence mechanisms
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import faiss

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available. Install with: pip install chromadb")


class VectorDatabase(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Add vectors to the database."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the database to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the database from disk."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass


class FAISSVectorDB(VectorDatabase):
    """
    FAISS-based vector database with enhanced persistence and indexing.
    Supports multiple index types for different use cases.
    """
    
    def __init__(self, embedding_dim: int, index_type: str = "cosine", 
                 use_gpu: bool = False, index_factory: Optional[str] = None):
        """
        Initialize FAISS vector database.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: "cosine", "L2", or "IP" (inner product)
            use_gpu: Whether to use GPU (requires faiss-gpu)
            index_factory: Optional FAISS index factory string for advanced indexes
                          e.g., "IVF1024,Flat" for approximate search
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index_factory = index_factory
        
        # Initialize index
        if index_factory:
            self.index = faiss.index_factory(embedding_dim, index_factory)
            if index_type == "cosine":
                # Will normalize before adding/searching
                pass
        else:
            if index_type == "cosine":
                # Use inner product with normalized vectors
                self.index = faiss.IndexFlatIP(embedding_dim)
            elif index_type == "L2":
                self.index = faiss.IndexFlatL2(embedding_dim)
            elif index_type == "IP":
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                raise ValueError(f"Unknown index_type: {index_type}")
        
        # Move to GPU if requested
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("   ✅ Using GPU acceleration")
            except Exception:
                print("   ⚠️  GPU not available, using CPU")
                self.use_gpu = False
        
        # Metadata storage
        self.metadata = []
        self.chunks_df = None
        
        print(f"   ✅ FAISS index initialized: {index_type}, dim={embedding_dim}")
        if index_factory:
            print(f"   Index factory: {index_factory}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add vectors to the database.
        
        Args:
            vectors: Numpy array of shape (n, embedding_dim)
            metadata: List of metadata dictionaries for each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")
        
        # Normalize for cosine similarity
        if self.index_type == "cosine":
            vectors = vectors.astype('float32')
            faiss.normalize_L2(vectors)
        else:
            vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        print(f"   ✅ Added {len(vectors):,} vectors. Total: {self.index.ntotal:,}")
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
               return_distances: bool = True) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector of shape (1, embedding_dim) or (embedding_dim,)
            k: Number of results to return
            return_distances: Whether to return similarity distances
            
        Returns:
            List of result dictionaries with metadata and optionally distances
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        
        # Normalize for cosine similarity
        if self.index_type == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                if return_distances:
                    result['distance'] = float(distances[0][i])
                    # Convert distance to similarity score for cosine
                    if self.index_type == "cosine":
                        result['similarity'] = float(distances[0][i])  # Already similarity
                    else:
                        result['similarity'] = 1.0 / (1.0 + float(distances[0][i]))
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save(self, path: str) -> None:
        """Save the database to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save FAISS index
        index_path = f"{path}.index"
        if isinstance(self.index, faiss.Index):
            # If on GPU, move to CPU for saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
        print(f"   ✅ Index saved to {index_path}")
        
        # Save metadata
        metadata_path = f"{path}.metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'index_factory': self.index_factory,
                'chunks_df': self.chunks_df
            }, f)
        print(f"   ✅ Metadata saved to {metadata_path}")
    
    def load(self, path: str) -> None:
        """Load the database from disk."""
        # Load FAISS index
        index_path = f"{path}.index"
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                self.use_gpu = False
        
        # Load metadata
        metadata_path = f"{path}.metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data.get('index_type', 'cosine')
            self.index_factory = data.get('index_factory')
            self.chunks_df = data.get('chunks_df')
        
        print(f"   ✅ Loaded {self.index.ntotal:,} vectors from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'index_factory': self.index_factory,
            'use_gpu': self.use_gpu,
            'metadata_count': len(self.metadata)
        }
    
    def train_index(self, training_vectors: np.ndarray) -> None:
        """
        Train an approximate index (e.g., IVF) on training data.
        
        Args:
            training_vectors: Sample vectors for training
        """
        if hasattr(self.index, 'train'):
            if self.index_type == "cosine":
                faiss.normalize_L2(training_vectors.astype('float32'))
            self.index.train(training_vectors.astype('float32'))
            print(f"   ✅ Index trained on {len(training_vectors):,} vectors")
        else:
            print("   ⚠️  Index type does not support training")


class ChromaVectorDB(VectorDatabase):
    """
    ChromaDB-based vector database implementation.
    Provides persistent storage with automatic metadata management.
    """
    
    def __init__(self, collection_name: str = "ntrs_documents", 
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector database.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (None for in-memory)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✅ Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"   ✅ Created new collection: {collection_name}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Add vectors to ChromaDB."""
        # Prepare data for ChromaDB
        ids = [m.get('chunk_id', f"chunk_{i}") for i, m in enumerate(metadata)]
        documents = [m.get('chunk_text', '') for m in metadata]
        
        # Convert numpy array to list of lists
        embeddings = vectors.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"   ✅ Added {len(vectors):,} vectors to ChromaDB")
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
               return_distances: bool = True) -> List[Dict]:
        """Search ChromaDB for similar vectors."""
        # Convert to list
        if query_vector.ndim == 1:
            query_embedding = query_vector.tolist()
        else:
            query_embedding = query_vector[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'chunk_id': results['ids'][0][i],
                    'chunk_text': results['documents'][0][i] if results['documents'] else '',
                    'rank': i + 1
                }
                
                # Add metadata
                if results['metadatas'] and results['metadatas'][0]:
                    result.update(results['metadatas'][0][i])
                
                # Add distances
                if return_distances and results['distances']:
                    result['distance'] = float(results['distances'][0][i])
                    result['similarity'] = 1.0 - float(results['distances'][0][i])  # ChromaDB uses distance
                
                formatted_results.append(result)
        
        return formatted_results
    
    def save(self, path: str) -> None:
        """ChromaDB auto-persists if persist_directory is set."""
        if self.persist_directory:
            print(f"   ✅ ChromaDB persisted to {self.persist_directory}")
        else:
            print("   ⚠️  ChromaDB is in-memory. Set persist_directory for persistence.")
    
    def load(self, path: str) -> None:
        """Load ChromaDB collection."""
        self.persist_directory = path
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"   ✅ Loaded ChromaDB collection from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        count = self.collection.count()
        return {
            'total_vectors': count,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }


def create_vector_db(db_type: str = "faiss", **kwargs) -> VectorDatabase:
    """
    Factory function to create a vector database.
    
    Args:
        db_type: "faiss" or "chroma"
        **kwargs: Additional arguments for the database constructor
        
    Returns:
        VectorDatabase instance
    """
    if db_type.lower() == "faiss":
        return FAISSVectorDB(**kwargs)
    elif db_type.lower() == "chroma" or db_type.lower() == "chromadb":
        return ChromaVectorDB(**kwargs)
    else:
        raise ValueError(f"Unknown database type: {db_type}")

