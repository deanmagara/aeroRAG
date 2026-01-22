"""
Example Usage Script for aeroRAG Pipeline
Demonstrates how to use the complete RAG pipeline
"""

# Example 1: Complete Pipeline from URL
print("="*70)
print("EXAMPLE 1: Complete Pipeline from URL")
print("="*70)

from src.pipeline import run_complete_pipeline

df_rag, df_chunks, embedding_system = run_complete_pipeline(
    data_source="url",
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    save_embeddings=True,
    embeddings_path="data/embeddings/ntrs_embeddings"
)

# Test search
print("\n" + "="*70)
print("Testing Search Functionality")
print("="*70)

query = "propulsion systems"
results = embedding_system.search(query, k=3)

print(f"\nQuery: '{query}'")
print(f"\nTop {len(results)} Results:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Document ID: {result['document_id']}")
    print(f"   Chunk: {result['chunk_text'][:150]}...")
    print()


# Example 2: Load Existing Embeddings
print("\n" + "="*70)
print("EXAMPLE 2: Loading Existing Embeddings")
print("="*70)

from src.embeddings import EmbeddingSystem

# Load previously saved embeddings
embedding_system_loaded = EmbeddingSystem(model_name="all-MiniLM-L6-v2")
embedding_system_loaded.load("data/embeddings/ntrs_embeddings")

# Search with loaded system
results = embedding_system_loaded.search("aerospace materials", k=5)
print(f"\nFound {len(results)} results for 'aerospace materials'")


# Example 3: Benchmarking
print("\n" + "="*70)
print("EXAMPLE 3: Benchmarking Chunk Parameters")
print("="*70)

from src.benchmarking import benchmark_chunk_parameters

# Benchmark different chunk parameters (using sample)
benchmark_results = benchmark_chunk_parameters(
    df_rag.sample(n=min(1000, len(df_rag)), random_state=42),
    chunk_sizes=[256, 512, 768],
    chunk_overlaps=[25, 50, 100]
)

print("\nBenchmarking complete! Check results_df for optimal parameters.")

