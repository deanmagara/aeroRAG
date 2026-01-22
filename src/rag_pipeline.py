"""
RAG Pipeline Assembly - January Phase
Connects query embedding, retrieval, context injection, and response generation
"""

import time
from typing import Dict, List, Optional, Any
from .similarity_search import RetrievalSystem
from .llm_integration import OllamaLLM


class RAGPipeline:
    """
    Complete RAG pipeline: query → embedding → retrieval → context → generation
    """
    
    def __init__(self, retrieval_system: RetrievalSystem,
                 llm: OllamaLLM,
                 context_template: Optional[str] = None,
                 max_context_length: int = 2000):
        """
        Initialize RAG pipeline.
        
        Args:
            retrieval_system: RetrievalSystem instance
            llm: OllamaLLM instance
            context_template: Template for formatting context (None for default)
            max_context_length: Maximum characters in context
        """
        self.retrieval_system = retrieval_system
        self.llm = llm
        self.max_context_length = max_context_length
        
        # Default context template
        if context_template is None:
            self.context_template = """Use the following NASA STI documents to answer the question. If the answer cannot be found in the documents, say "I don't have information about this in the NASA STI database."

Documents:
{context}

Question: {query}

Answer:"""
        else:
            self.context_template = context_template
    
    def query(self, query: str, 
             k: int = 5,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             include_sources: bool = True,
             deduplicate: bool = True) -> Dict[str, Any]:
        """
        Execute complete RAG query pipeline.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            temperature: LLM temperature (None for default)
            max_tokens: Maximum tokens to generate (None for default)
            include_sources: Whether to include source documents in response
            deduplicate: Whether to deduplicate by document_id
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # 1. Retrieve relevant documents
        if deduplicate:
            retrieval_result = self.retrieval_system.retrieve_with_deduplication(query, k=k)
        else:
            retrieval_result = self.retrieval_system.retrieve(query, k=k)
        
        context = retrieval_result['context']
        
        # 2. Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
            print(f"   ⚠️  Context truncated to {self.max_context_length} characters")
        
        # 3. Format prompt with context
        prompt = self.context_template.format(
            context=context,
            query=query
        )
        
        # 4. Generate response
        answer = self.llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        total_time = time.time() - start_time
        
        # 5. Build response
        response = {
            'query': query,
            'answer': answer.strip(),
            'retrieval_time': retrieval_result.get('search_time', 0),
            'generation_time': total_time - retrieval_result.get('search_time', 0),
            'total_time': total_time,
            'num_sources': retrieval_result.get('num_unique_documents', 
                                               retrieval_result.get('num_results', 0))
        }
        
        if include_sources:
            response['sources'] = retrieval_result.get('sources', [])
            response['context'] = context  # Include for debugging
        
        return response
    
    def query_with_validation(self, query: str,
                             k: int = 5,
                             min_similarity: float = 0.3,
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Query with similarity threshold validation.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            min_similarity: Minimum similarity score for retrieved documents
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary, or None if no relevant documents found
        """
        # Retrieve with similarity threshold
        results = self.retrieval_system.search_engine.search(
            query, 
            k=k, 
            min_similarity=min_similarity
        )
        
        if len(results) == 0:
            return {
                'query': query,
                'answer': "I don't have relevant information about this in the NASA STI database.",
                'sources': [],
                'num_sources': 0,
                'reason': 'no_relevant_documents'
            }
        
        # Continue with normal query
        return self.query(query, k=k, temperature=temperature, max_tokens=max_tokens)
    
    def batch_query(self, queries: List[str],
                   k: int = 5,
                   temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries
            k: Number of documents per query
            temperature: LLM temperature
            
        Returns:
            List of response dictionaries
        """
        results = []
        for query in queries:
            result = self.query(query, k=k, temperature=temperature)
            results.append(result)
        return results


class ConversationalRAG:
    """
    RAG pipeline with conversation history support.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline,
                 max_history: int = 5):
        """
        Initialize conversational RAG.
        
        Args:
            rag_pipeline: RAGPipeline instance
            max_history: Maximum conversation turns to keep in history
        """
        self.rag_pipeline = rag_pipeline
        self.max_history = max_history
        self.conversation_history = []
    
    def chat(self, query: str,
            k: int = 5,
            temperature: Optional[float] = None,
            include_history: bool = True) -> Dict[str, Any]:
        """
        Chat with conversation history.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            temperature: LLM temperature
            include_history: Whether to include conversation history in context
            
        Returns:
            Response dictionary
        """
        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': query
        })
        
        # Build context-aware query
        if include_history and len(self.conversation_history) > 1:
            # Include recent conversation context
            recent_history = self.conversation_history[-(self.max_history*2):-1]
            context_query = self._build_contextual_query(query, recent_history)
        else:
            context_query = query
        
        # Execute RAG query
        response = self.rag_pipeline.query(
            context_query,
            k=k,
            temperature=temperature
        )
        
        # Add assistant response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response['answer']
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history*2):]
        
        response['conversation_turn'] = len(self.conversation_history) // 2
        return response
    
    def _build_contextual_query(self, current_query: str,
                               history: List[Dict]) -> str:
        """Build query with conversation context."""
        context_parts = []
        for msg in history:
            role = msg['role']
            content = msg['content']
            context_parts.append(f"{role.capitalize()}: {content}")
        
        context = "\n".join(context_parts)
        return f"Previous conversation:\n{context}\n\nCurrent question: {current_query}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()

