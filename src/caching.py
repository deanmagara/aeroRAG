"""
Caching Mechanism
Implements LRU caching for retrieval and embedding operations.
"""

import functools
import hashlib
import json
import time
from typing import Any, Dict, List

class RAGCache:
    """Simple in-memory LRU cache wrapper."""
    
    def __init__(self, capacity: int = 200):
        self._cache = {}
        self._capacity = capacity
        self._access_order = []
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._capacity:
            # Remove least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            
        self._cache[key] = value
        self._access_order.append(key)

    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate a stable hash key."""
        content = f"{prefix}|{args}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

# Global cache instance
retrieval_cache = RAGCache(capacity=200)

def cached_retrieval(func):
    """Decorator to cache retrieval results."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Assuming first arg is self, second is query
        if len(args) > 0:
            query = args[0]
            # Generate cache key based on query and kwargs
            cache_key = RAGCache.generate_key("retrieval", query, **kwargs)
            
            # Check cache
            cached_result = retrieval_cache.get(cache_key)
            if cached_result:
                # Add a flag to indicate cache hit (optional debug)
                cached_result['_cache_hit'] = True
                return cached_result
            
        # Execute and cache
        result = func(self, *args, **kwargs)
        if len(args) > 0:
             retrieval_cache.set(cache_key, result)
        return result
        
    return wrapper