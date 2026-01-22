"""
Hallucination Prevention - January Phase
Implements mechanisms to restrict responses to knowledge base only
"""

import re
from typing import Dict, List, Optional, Tuple
from .similarity_search import RetrievalSystem


class HallucinationPrevention:
    """
    Mechanisms to prevent hallucinations by ensuring responses are grounded in retrieved documents.
    """
    
    def __init__(self, retrieval_system: RetrievalSystem,
                 similarity_threshold: float = 0.3,
                 require_citation: bool = True):
        """
        Initialize hallucination prevention system.
        
        Args:
            retrieval_system: RetrievalSystem instance for validation
            similarity_threshold: Minimum similarity for retrieved documents
            require_citation: Whether to require citations in responses
        """
        self.retrieval_system = retrieval_system
        self.similarity_threshold = similarity_threshold
        self.require_citation = require_citation
    
    def validate_response(self, query: str, answer: str,
                         retrieved_sources: List[Dict]) -> Dict[str, Any]:
        """
        Validate that the answer is grounded in retrieved sources.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_sources: List of retrieved source documents
            
        Returns:
            Validation result with confidence score and warnings
        """
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'warnings': [],
            'citations_found': False,
            'source_overlap': 0.0
        }
        
        # 1. Check for citations
        citations = self._extract_citations(answer)
        validation_result['citations_found'] = len(citations) > 0
        
        if self.require_citation and len(citations) == 0:
            validation_result['is_valid'] = False
            validation_result['warnings'].append(
                "Answer does not contain citations to source documents"
            )
            validation_result['confidence'] *= 0.5
        
        # 2. Check semantic overlap with sources
        source_texts = [s.get('chunk_text', '') for s in retrieved_sources]
        overlap_score = self._calculate_text_overlap(answer, source_texts)
        validation_result['source_overlap'] = overlap_score
        
        if overlap_score < 0.1:
            validation_result['is_valid'] = False
            validation_result['warnings'].append(
                f"Low semantic overlap with sources ({overlap_score:.2f})"
            )
            validation_result['confidence'] *= 0.3
        
        # 3. Check for disclaimers
        has_disclaimer = self._check_disclaimer(answer)
        if not has_disclaimer and overlap_score < 0.3:
            validation_result['warnings'].append(
                "Answer may contain information not in sources"
            )
            validation_result['confidence'] *= 0.7
        
        # 4. Check for uncertainty markers
        uncertainty_markers = self._check_uncertainty(answer)
        if uncertainty_markers:
            validation_result['warnings'].append(
                f"Answer contains uncertainty markers: {', '.join(uncertainty_markers)}"
            )
        
        return validation_result
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text (e.g., [1], [Document: ...], etc.)."""
        # Pattern for [number] citations
        number_citations = re.findall(r'\[(\d+)\]', text)
        
        # Pattern for [Document: ...] citations
        doc_citations = re.findall(r'\[Document:\s*([^\]]+)\]', text)
        
        return number_citations + doc_citations
    
    def _calculate_text_overlap(self, answer: str, source_texts: List[str]) -> float:
        """
        Calculate semantic/text overlap between answer and sources.
        Simple keyword-based overlap (can be enhanced with embeddings).
        """
        if not answer or not source_texts:
            return 0.0
        
        # Extract keywords from answer
        answer_words = set(answer.lower().split())
        answer_words = {w for w in answer_words if len(w) > 3}  # Filter short words
        
        if not answer_words:
            return 0.0
        
        # Calculate overlap with each source
        max_overlap = 0.0
        for source_text in source_texts:
            source_words = set(source_text.lower().split())
            source_words = {w for w in source_words if len(w) > 3}
            
            if source_words:
                overlap = len(answer_words & source_words) / len(answer_words)
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _check_disclaimer(self, text: str) -> bool:
        """Check if answer contains appropriate disclaimers."""
        disclaimer_patterns = [
            r"don't have information",
            r"not in the.*database",
            r"cannot be found",
            r"not available",
            r"based on.*documents",
            r"according to.*sources"
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in disclaimer_patterns)
    
    def _check_uncertainty(self, text: str) -> List[str]:
        """Check for uncertainty markers in the answer."""
        uncertainty_markers = []
        
        uncertainty_patterns = {
            'maybe': r'\b(maybe|perhaps|possibly|might|could)\b',
            'uncertain': r'\b(uncertain|unclear|unknown|unsure)\b',
            'speculation': r'\b(probably|likely|appears|seems)\b'
        }
        
        text_lower = text.lower()
        for marker_type, pattern in uncertainty_patterns.items():
            if re.search(pattern, text_lower):
                uncertainty_markers.append(marker_type)
        
        return uncertainty_markers
    
    def filter_response(self, query: str, answer: str,
                       retrieved_sources: List[Dict],
                       min_confidence: float = 0.5) -> Tuple[str, Dict]:
        """
        Filter or modify response based on validation.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_sources: Retrieved source documents
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_answer, validation_info)
        """
        validation = self.validate_response(query, answer, retrieved_sources)
        
        if validation['confidence'] < min_confidence:
            # Replace with safe response
            filtered_answer = (
                "I don't have sufficient information in the NASA STI database "
                "to provide a reliable answer to this question. "
                "Please try rephrasing your query or asking about a different topic."
            )
            validation['was_filtered'] = True
        else:
            filtered_answer = answer
            validation['was_filtered'] = False
        
        return filtered_answer, validation


class GroundedRAGPipeline:
    """
    RAG pipeline with built-in hallucination prevention.
    """
    
    def __init__(self, rag_pipeline, 
                 hallucination_prevention: HallucinationPrevention,
                 enforce_validation: bool = True):
        """
        Initialize grounded RAG pipeline.
        
        Args:
            rag_pipeline: RAGPipeline instance
            hallucination_prevention: HallucinationPrevention instance
            enforce_validation: Whether to enforce validation (filter low-confidence responses)
        """
        self.rag_pipeline = rag_pipeline
        self.hallucination_prevention = hallucination_prevention
        self.enforce_validation = enforce_validation
    
    def query(self, query: str, k: int = 5,
             min_confidence: float = 0.5,
             **kwargs) -> Dict[str, Any]:
        """
        Execute RAG query with hallucination prevention.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            min_confidence: Minimum confidence for response
            **kwargs: Additional arguments for RAG pipeline
            
        Returns:
            Response dictionary with validation information
        """
        # Execute normal RAG query
        response = self.rag_pipeline.query(query, k=k, **kwargs)
        
        # Validate response
        retrieved_sources = response.get('sources', [])
        validation = self.hallucination_prevention.validate_response(
            query,
            response['answer'],
            retrieved_sources
        )
        
        response['validation'] = validation
        
        # Filter if needed
        if self.enforce_validation and validation['confidence'] < min_confidence:
            filtered_answer, filter_info = self.hallucination_prevention.filter_response(
                query,
                response['answer'],
                retrieved_sources,
                min_confidence
            )
            response['answer'] = filtered_answer
            response['was_filtered'] = True
        else:
            response['was_filtered'] = False
        
        return response
    
    def query_with_sources_required(self, query: str, k: int = 5,
                                   min_similarity: float = 0.3,
                                   **kwargs) -> Dict[str, Any]:
        """
        Query that requires high-quality sources.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            min_similarity: Minimum similarity for sources
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary
        """
        # Use validation-aware query
        response = self.rag_pipeline.query_with_validation(
            query,
            k=k,
            min_similarity=min_similarity,
            **kwargs
        )
        
        # Additional validation
        if response.get('num_sources', 0) > 0:
            validation = self.hallucination_prevention.validate_response(
                query,
                response['answer'],
                response.get('sources', [])
            )
            response['validation'] = validation
        
        return response

