"""
Complete RAG System - December & January Phases
Orchestrates all components into a unified RAG system
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from .data_acquisition import load_data_from_url, load_data_from_file
from .data_processing import run_processing_pipeline
from .embeddings import EmbeddingSystem
from .vector_db import create_vector_db, FAISSVectorDB
from .similarity_search import SimilaritySearchEngine, RetrievalSystem
from .llm_integration import OllamaLLM
from .rag_pipeline import RAGPipeline, ConversationalRAG
from .hallucination_prevention import HallucinationPrevention, GroundedRAGPipeline


class CompleteRAGSystem:
    """
    Complete RAG system integrating all components.
    """
    
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 vector_db_type: str = "faiss",
                 llm_model_name: str = "llama3.2:latest",
                 ollama_base_url: str = "http://localhost:11434",
                 use_gpu: bool = False):
        """
        Initialize complete RAG system.
        
        Args:
            embedding_model_name: Sentence transformer model name
            vector_db_type: "faiss" or "chroma"
            llm_model_name: Ollama model name
            ollama_base_url: Ollama API base URL
            use_gpu: Whether to use GPU for embeddings/FAISS
        """
        print("\n" + "="*70)
        print("INITIALIZING COMPLETE RAG SYSTEM")
        print("="*70)
        
        # 1. Initialize embedding model
        print("\n1. Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"   ✅ Embedding model: {embedding_model_name} ({self.embedding_dim} dims)")
        
        # 2. Initialize vector database (will be populated later)
        print("\n2. Initializing vector database...")
        self.vector_db = create_vector_db(
            db_type=vector_db_type,
            embedding_dim=self.embedding_dim,
            use_gpu=use_gpu
        )
        print(f"   ✅ Vector DB: {vector_db_type}")
        
        # 3. Initialize similarity search engine
        print("\n3. Initializing similarity search...")
        self.search_engine = SimilaritySearchEngine(
            embedding_model=self.embedding_model,
            vector_db=self.vector_db
        )
        print("   ✅ Search engine ready")
        
        # 4. Initialize retrieval system
        print("\n4. Initializing retrieval system...")
        self.retrieval_system = RetrievalSystem(self.search_engine)
        print("   ✅ Retrieval system ready")
        
        # 5. Initialize LLM
        print("\n5. Initializing LLM...")
        try:
            self.llm = OllamaLLM(
                model_name=llm_model_name,
                base_url=ollama_base_url
            )
            print("   ✅ LLM ready")
        except Exception as e:
            print(f"   ⚠️  LLM initialization failed: {e}")
            self.llm = None
        
        # 6. Initialize RAG pipeline
        print("\n6. Initializing RAG pipeline...")
        if self.llm:
            self.rag_pipeline = RAGPipeline(
                retrieval_system=self.retrieval_system,
                llm=self.llm
            )
            print("   ✅ RAG pipeline ready")
            
            # 7. Initialize hallucination prevention
            print("\n7. Initializing hallucination prevention...")
            self.hallucination_prevention = HallucinationPrevention(
                retrieval_system=self.retrieval_system
            )
            print("   ✅ Hallucination prevention ready")
            
            # 8. Initialize grounded RAG pipeline
            self.grounded_rag = GroundedRAGPipeline(
                rag_pipeline=self.rag_pipeline,
                hallucination_prevention=self.hallucination_prevention
            )
            print("   ✅ Grounded RAG pipeline ready")
        else:
            self.rag_pipeline = None
            self.grounded_rag = None
        
        print("\n" + "="*70)
        print("✅ RAG SYSTEM INITIALIZED")
        print("="*70)
    
    def build_from_data(self, data_source: str = "url",
                       file_path: Optional[str] = None,
                       chunk_size: int = 512,
                       chunk_overlap: int = 50,
                       save_path: Optional[str] = None,
                       max_chunks: Optional[int] = None):
        """
        Build the RAG system from raw data.
        
        Args:
            data_source: "url" or "file"
            file_path: Path to local file if data_source is "file"
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            save_path: Path to save vector database
            max_chunks: Optional limit on number of chunks to embed (useful for large corpora)
        """
        print("\n" + "="*70)
        print("BUILDING RAG SYSTEM FROM DATA")
        print("="*70)
        
        # 1. Acquire data
        print("\n📥 Acquiring data...")
        if data_source == "url":
            df_raw = load_data_from_url(
                "https://ntrs.staging.sti.appdat.jsc.nasa.gov/api/docs/ntrs-public-metadata.json.gz?attachment=true"
            )
        else:
            df_raw = load_data_from_file(file_path)
        
        if df_raw is None:
            raise ValueError("Data acquisition failed")
        
        # 2. Process and chunk
        print("\n🔧 Processing and chunking...")
        df_rag, df_chunks = run_processing_pipeline(
            df_raw,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if max_chunks is not None and len(df_chunks) > max_chunks:
            print(f"⚠️  Limiting chunks from {len(df_chunks)} to {max_chunks} for stability") # 
            df_chunks = df_chunks.iloc[:max_chunks].reset_index(drop=True)
        
        # 3. Generate embeddings
        print("\n🧠 Generating embeddings...")
        embeddings = self.embedding_model.encode(
            df_chunks['chunk_text'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 4. Prepare metadata
        metadata = []
        for _, row in df_chunks.iterrows():
            metadata.append({
                'chunk_id': row['chunk_id'],
                'document_id': row['document_id'],
                'title': row['title'],
                'chunk_text': row['chunk_text']
            })
        
        # 5. Add to vector database
        print("\n💾 Adding to vector database...")
        #self.vector_db.add_vectors(embeddings, metadata)  # adds vectors to the DB at once
        # Modified to add in batches to avoid potential memory issues
        batch_size = 10_000

        for i in range(0, len(embeddings), batch_size):
            self.vector_db.add_vectors(
                embeddings[i:i+batch_size],
                metadata[i:i+batch_size]
            )
        self.vector_db.chunks_df = df_chunks
        
        # 6. Save if requested
        if save_path:
            print(f"\n💾 Saving vector database to {save_path}...")
            self.vector_db.save(save_path)
        
        print("\n✅ RAG system built successfully!")
        return df_rag, df_chunks
    
    def load_from_saved(self, vector_db_path: str):
        """
        Load RAG system from saved vector database.
        
        Args:
            vector_db_path: Path to saved vector database
        """
        print(f"\n📂 Loading vector database from {vector_db_path}...")
        self.vector_db.load(vector_db_path)
        print("✅ Vector database loaded")
    
    def query(self, query: str, k: int = 5,
             use_grounding: bool = True,
             **kwargs) -> Dict[str, Any]:
        """
        Execute RAG query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_grounding: Whether to use hallucination prevention
            **kwargs: Additional arguments for RAG pipeline
            
        Returns:
            Response dictionary
        """
        if not self.rag_pipeline:
            raise ValueError("LLM not initialized. Cannot execute queries.")
        
        if use_grounding and self.grounded_rag:
            return self.grounded_rag.query(query, k=k, **kwargs)
        else:
            return self.rag_pipeline.query(query, k=k, **kwargs)
    
    def get_conversational_rag(self) -> ConversationalRAG:
        """Get conversational RAG interface."""
        if not self.rag_pipeline:
            raise ValueError("LLM not initialized.")
        return ConversationalRAG(self.rag_pipeline)


def create_complete_rag_system(
    embedding_model: str = "all-MiniLM-L6-v2",
    vector_db_type: str = "faiss",
    llm_model: str = "  llama3.2:latest",
    use_gpu: bool = False
) -> CompleteRAGSystem:
    """
    Factory function to create a complete RAG system.
    
    Args:
        embedding_model: Embedding model name
        vector_db_type: "faiss" or "chroma"
        llm_model: Ollama model name
        use_gpu: Whether to use GPU
        
    Returns:
        CompleteRAGSystem instance
    """
    return CompleteRAGSystem(
        embedding_model_name=embedding_model,
        vector_db_type=vector_db_type,
        llm_model_name=llm_model,
        use_gpu=use_gpu
    )
