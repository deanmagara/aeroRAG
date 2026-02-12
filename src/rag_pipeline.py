"""
RAG Pipeline Assembly - January Phase
Connects query embedding, retrieval, context injection, and response generation
"""

import time
from typing import Dict, List, Optional, Any
from .similarity_search import RetrievalSystem
from .llm_integration import OllamaLLM
from .domainConfig import AerospacePrompts  # NEW: Import domain config

class RAGPipeline:
    """
    Complete RAG pipeline: query → embedding → retrieval → context → generation
    """
    
    def __init__(self, retrieval_system: RetrievalSystem,
                 llm: OllamaLLM,
                 context_template: Optional[str] = None,
                 max_context_length: int = 2500): # Increased context window
        """
        Initialize RAG pipeline.
        """
        self.retrieval_system = retrieval_system
        self.llm = llm
        self.max_context_length = max_context_length
        
        # NEW: Use Aerospace System Prompt by default
        if context_template is None:
            self.context_template = AerospacePrompts.DEFAULT_SYSTEM_PROMPT
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
            'retrieval_time': retrieval_result.get('search_time', 0), # Note: caching might remove this key if not careful, handled by decorator logic if needed
            'generation_time': total_time,
            'total_time': total_time,
            'num_sources': len(retrieval_result.get('sources', []))
        }
        
        if include_sources:
            response['sources'] = retrieval_result.get('sources', [])
            response['context'] = context
        
        return response
    
    # ... (Rest of class query_with_validation, batch_query remain same) ...
    def query_with_validation(self, query: str,
                             k: int = 5,
                             min_similarity: float = 0.3,
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Query with similarity threshold validation."""
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
        
        return self.query(query, k=k, temperature=temperature, max_tokens=max_tokens)

    def batch_query(self, queries: List[str], k: int = 5, temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        results = []
        for query in queries:
            result = self.query(query, k=k, temperature=temperature)
            results.append(result)
        return results

class ConversationalRAG:
    """RAG pipeline with conversation history support."""
    def __init__(self, rag_pipeline: RAGPipeline, max_history: int = 5):
        self.rag_pipeline = rag_pipeline
        self.max_history = max_history
        self.conversation_history = []
    
    def chat(self, query: str, k: int = 5, temperature: Optional[float] = None, include_history: bool = True) -> Dict[str, Any]:
        self.conversation_history.append({'role': 'user', 'content': query})
        
        if include_history and len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-(self.max_history*2):-1]
            context_query = self._build_contextual_query(query, recent_history)
        else:
            context_query = query
        
        response = self.rag_pipeline.query(context_query, k=k, temperature=temperature)
        self.conversation_history.append({'role': 'assistant', 'content': response['answer']})
        
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history*2):]
        
        response['conversation_turn'] = len(self.conversation_history) // 2
        return response
    
    def _build_contextual_query(self, current_query: str, history: List[Dict]) -> str:
        context_parts = []
        for msg in history:
            context_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        context = "\n".join(context_parts)
        return f"Previous conversation:\n{context}\n\nCurrent question: {current_query}"
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        return self.conversation_history.copy()