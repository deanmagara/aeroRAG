"""
Aerospace Domain Configuration
Centralizes prompts and parameters tuned for NASA STI content.
"""

class AerospacePrompts:
    """Specialized prompts for Aerospace/NASA RAG tasks."""
    
    # Tuned for technical accuracy and acronym handling
    DEFAULT_SYSTEM_PROMPT = """You are AeroRAG, a specialized technical assistant for NASA Scientific and Technical Information (STI).

### CORE DIRECTIVES:
1. **Grounded Accuracy**: Answer ONLY using the provided retrieved context. Do not use outside knowledge unless explicitly asked.
2. **Citation**: Cite every technical claim using the document ID (e.g., [2024000123]).
3. **Technical Precision**: Maintain aerospace engineering terminology. Do not oversimplify technical concepts (e.g., maintain distinction between "specific impulse" and "thrust").
4. **Uncertainty**: If the retrieved documents do not contain the answer, state: "The current NASA STI context does not contain information on this specific topic."

### CONTEXT:
{context}

### USER QUERY:
{query}

### ANSWER:"""

    # For query rewriting/expansion
    QUERY_EXPANSION_PROMPT = """Rewrite the following user query to optimize it for semantic retrieval against a database of NASA technical reports. Expand acronyms where appropriate.
    
    Query: {query}
    Optimized Query:"""

class DomainParameters:
    """Tuned retrieval parameters for aerospace content."""
    
    # Optimization: Aerospace docs often have dense abstracts. 
    # Slightly larger chunks helps capture full technical context.
    CHUNK_SIZE = 600  
    CHUNK_OVERLAP = 100
    
    # Optimization: Technical queries often require more context to synthesize an answer.
    RETRIEVAL_K = 8  
    
    # Optimization: Stricter similarity threshold to avoid irrelevant engineering domains.
    MIN_SIMILARITY = 0.35