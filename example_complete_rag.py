"""
Complete RAG System Example - December & January Phases
Demonstrates the full RAG pipeline with vector database, retrieval, and LLM integration
"""

from src.complete_rag_system import CompleteRAGSystem, create_complete_rag_system


def example_build_and_query():
    """Example: Build RAG system from scratch and query it."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Building Complete RAG System")
    print("="*70)
    
    # 1. Create RAG system
    rag_system = create_complete_rag_system(
        embedding_model="all-MiniLM-L6-v2",
        vector_db_type="faiss",
        llm_model="llama3.2",
        use_gpu=False
    )
    
    # 2. Build from data (or load from saved)
    # Option A: Build from data
    # rag_system.build_from_data(
    #     data_source="file",
    #     file_path="ntrs-public-metadata.json",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     save_path="data/embeddings/ntrs_rag"
    # )
    
    # Option B: Load from saved
    rag_system.load_from_saved("data/embeddings/ntrs_rag")
    
    # 3. Query the system
    print("\n" + "="*70)
    print("QUERYING RAG SYSTEM")
    print("="*70)
    
    queries = [
        "What propulsion technologies are studied at NASA?",
        "Tell me about aerospace materials research",
        "What are the latest developments in space exploration?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 70)
        
        response = rag_system.query(
            query,
            k=5,
            use_grounding=True,
            min_confidence=0.5
        )
        
        print(f"\n💬 Answer:")
        print(response['answer'])
        
        print(f"\n📚 Sources ({response['num_sources']} documents):")
        for i, source in enumerate(response.get('sources', [])[:3], 1):
            print(f"  {i}. {source.get('title', 'Unknown')}")
            print(f"     Similarity: {source.get('similarity', 0):.3f}")
        
        if 'validation' in response:
            validation = response['validation']
            print(f"\n✅ Validation:")
            print(f"   Confidence: {validation['confidence']:.2f}")
            print(f"   Source Overlap: {validation['source_overlap']:.2f}")
            if validation['warnings']:
                print(f"   Warnings: {', '.join(validation['warnings'])}")
        
        print(f"\n⏱️  Total time: {response['total_time']:.2f}s")
        print("=" * 70)


def example_conversational_rag():
    """Example: Conversational RAG with history."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Conversational RAG")
    print("="*70)
    
    # Create system and load
    rag_system = create_complete_rag_system()
    rag_system.load_from_saved("data/embeddings/ntrs_rag")
    
    # Get conversational interface
    conversational_rag = rag_system.get_conversational_rag()
    
    # Simulate conversation
    conversation = [
        "What is NASA's research focus on propulsion?",
        "Tell me more about the specific technologies mentioned",
        "What are the applications of these technologies?"
    ]
    
    for query in conversation:
        print(f"\n👤 User: {query}")
        response = conversational_rag.chat(query, k=5)
        print(f"\n🤖 Assistant: {response['answer']}")
        print(f"   (Turn {response['conversation_turn']})")


def example_performance_benchmarking():
    """Example: Performance benchmarking."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Performance Benchmarking")
    print("="*70)
    
    from src.performance_tuning import PerformanceProfiler, optimize_faiss_index
    from src.vector_db import FAISSVectorDB
    import numpy as np
    
    # Create test vectors
    embedding_dim = 384
    num_vectors = 10000
    test_vectors = np.random.randn(num_vectors, embedding_dim).astype('float32')
    query_vectors = np.random.randn(10, embedding_dim).astype('float32')
    
    # Test different index types
    print("\n📊 Benchmarking FAISS Index Types...")
    
    index_configs = [
        ("Flat", None),
        ("IVF1024,Flat", 1024),
    ]
    
    for index_factory, nlist in index_configs:
        print(f"\nTesting: {index_factory}")
        db = FAISSVectorDB(
            embedding_dim=embedding_dim,
            index_type="cosine",
            index_factory=index_factory if index_factory != "Flat" else None
        )
        
        # Train if needed
        if nlist:
            training = test_vectors[:1000]
            db.train_index(training)
        
        # Add vectors
        metadata = [{'chunk_id': f'v_{i}'} for i in range(num_vectors)]
        db.add_vectors(test_vectors, metadata)
        
        # Profile search
        profiler = PerformanceProfiler(db)
        metrics = profiler.profile_search(query_vectors, k=5, num_runs=5)
        
        print(f"   Avg query time: {metrics['avg_query_time_ms']:.2f} ms")
        print(f"   Throughput: {metrics['throughput_queries_per_sec']:.1f} queries/sec")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE RAG SYSTEM EXAMPLES")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Build and query
    # example_build_and_query()
    
    # Example 2: Conversational RAG
    # example_conversational_rag()
    
    # Example 3: Performance benchmarking
    # example_performance_benchmarking()
    
    print("\n✅ Examples ready. Uncomment the example you want to run.")

