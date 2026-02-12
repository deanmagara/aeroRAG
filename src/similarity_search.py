"""
Similarity Search Development - December Phase
Creates efficient similarity search and retrieval functionality
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional, Tuple, Callable
from sentence_transformers import SentenceTransformer
from .vector_db import VectorDatabase, FAISSVectorDB, ChromaVectorDB
from .caching import cached_retrieval  # NEW: Import caching

class SimilaritySearchEngine:
    """
    High-level similarity search engine with query processing and result ranking.
    """
    
    def __init__(self, embedding_model: SentenceTransformer, 
                 vector_db: VectorDatabase,
                 rerank: bool = False,
                 rerank_model: Optional[str] = None):
        """
        Initialize similarity search engine.
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.rerank = rerank
        self.rerank_model_name = rerank_model
        
        # Initialize reranker if requested
        if rerank and rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(rerank_model)
                print(f"   ✅ Reranker loaded: {rerank_model}")
            except ImportError:
                print("   ⚠️  CrossEncoder not available. Install sentence-transformers with [rerank]")
                self.rerank = False
                self.reranker = None
        else:
            self.reranker = None
    
    def search(self, query: str, k: int = 5, 
               min_similarity: float = 0.0,
               filter_metadata: Optional[Dict] = None,
               rerank_top_k: Optional[int] = None) -> List[Dict]:
        """
        Perform similarity search with optional reranking.
        """
        start_time = time.time()
        
        # 1. Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # 2. Initial search (get more results if reranking)
        search_k = rerank_top_k if (self.rerank and rerank_top_k) else k
        results = self.vector_db.search(query_embedding, k=search_k, return_distances=True)
        
        # 3. Apply similarity threshold
        if min_similarity > 0:
            results = [r for r in results if r.get('similarity', 0) >= min_similarity]
        
        # 4. Apply metadata filters
        if filter_metadata:
            results = self._filter_by_metadata(results, filter_metadata)
        
        # 5. Rerank if enabled
        if self.rerank and self.reranker and len(results) > 1:
            results = self._rerank_results(query, results, k)
        
        # 6. Limit to k results
        results = results[:k]
        
        search_time = time.time() - start_time
        
        # Add search metadata
        for result in results:
            result['search_time'] = search_time
        
        return results
    
    def _filter_by_metadata(self, results: List[Dict], 
                           filters: Dict) -> List[Dict]:
        """Filter results by metadata criteria."""
        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if result.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(result)
        return filtered
    
    def _rerank_results(self, query: str, results: List[Dict], 
                       top_k: int) -> List[Dict]:
        """Rerank results using cross-encoder."""
        if not self.reranker:
            return results
        
        # Prepare pairs for reranking
        pairs = [[query, r.get('chunk_text', '')] for r in results]
        
        # Get rerank scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update results with rerank scores
        for i, result in enumerate(results):
            result['rerank_score'] = float(rerank_scores[i])
            result['original_rank'] = result.get('rank', i + 1)
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results[:top_k]
    
    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """Perform batch similarity search for multiple queries."""
        query_embeddings = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        all_results = []
        for query_embedding in query_embeddings:
            results = self.vector_db.search(
                query_embedding.reshape(1, -1), 
                k=k, 
                return_distances=True
            )
            all_results.append(results)
        
        return all_results
    
    def hybrid_search(self, query: str, k: int = 5,
                     keyword_weight: float = 0.3,
                     vector_weight: float = 0.7) -> List[Dict]:
        """Hybrid search combining vector similarity and keyword matching."""
        vector_results = self.search(query, k=k*2)
        query_keywords = set(query.lower().split())
        
        for result in vector_results:
            chunk_text = result.get('chunk_text', '').lower()
            chunk_keywords = set(chunk_text.split())
            keyword_overlap = len(query_keywords & chunk_keywords)
            keyword_score = keyword_overlap / max(len(query_keywords), 1)
            result['keyword_score'] = keyword_score
        
        for result in vector_results:
            vector_score = result.get('similarity', 0)
            keyword_score = result.get('keyword_score', 0)
            result['hybrid_score'] = (
                vector_weight * vector_score + 
                keyword_weight * keyword_score
            )
        
        vector_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        for i, result in enumerate(vector_results[:k]):
            result['rank'] = i + 1
        
        return vector_results[:k]


class RetrievalSystem:
    """
    Complete retrieval system with context assembly and result formatting.
    """
    
    def __init__(self, search_engine: SimilaritySearchEngine):
        self.search_engine = search_engine
    
    @cached_retrieval  # NEW: Cache standard retrieval
    def retrieve(self, query: str, k: int = 5,
                include_metadata: bool = True,
                format_context: bool = True) -> Dict:
        """Retrieve relevant documents and format context."""
        results = self.search_engine.search(query, k=k)
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            chunk_text = result.get('chunk_text', '')
            title = result.get('title', 'Unknown')
            doc_id = result.get('document_id', '')
            
            if format_context:
                context_parts.append(
                    f"SOURCE [{i+1}]\n"
                    f"Document Title: {title}\n"
                    f"Document ID: {doc_id}\n"
                    f"Text: {chunk_text}\n"
                )
            
            if include_metadata:
                sources.append({
                    'title': title,
                    'document_id': doc_id,
                    'chunk_id': result.get('chunk_id'),
                    'similarity': result.get('similarity', 0),
                    'rank': i + 1,
                    'chunk_text': chunk_text # Ensure text is passed for hallucination check
                })
        
        context = "\n".join(context_parts) if format_context else ""
        
        return {
            'query': query,
            'results': results,
            'context': context,
            'sources': sources,
            'num_results': len(results)
        }
    
    @cached_retrieval  # NEW: Cache deduplicated retrieval
    def retrieve_with_deduplication(self, query: str, k: int = 5) -> Dict:
        """
        Retrieve documents with deduplication and numbered formatting.
        """
        # Get more results to account for deduplication
        results = self.search_engine.search(query, k=k*3)
        
        # Deduplicate by document_id
        seen_docs = set()
        unique_results = []
        
        for result in results:
            doc_id = result.get('document_id')
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_results.append(result)
                if len(unique_results) >= k:
                    break
        
        # Format context with numbering
        context_parts = []
        sources = []  # FIX: Initialize sources list
        
        for i, result in enumerate(unique_results):
            context_parts.append(
                f"SOURCE [{i+1}]\n"
                f"Document Title: {result.get('title', 'Unknown')}\n"
                f"Document ID: {result.get('document_id', '')}\n"
                f"Text: {result.get('chunk_text', '')}\n"
            )
            
            # FIX: Populate sources list so HallucinationPrevention can verify them
            sources.append({
                'title': result.get('title', 'Unknown'),
                'document_id': result.get('document_id', ''),
                'chunk_id': result.get('chunk_id'),
                'similarity': result.get('similarity', 0),
                'rank': i + 1,
                'chunk_text': result.get('chunk_text', '') 
            })
        
        return {
            'query': query,
            'results': unique_results,
            'context': "\n".join(context_parts),
            'sources': sources, # FIX: Include sources in return
            'num_unique_documents': len(unique_results),
            'total_results_considered': len(results)
        }