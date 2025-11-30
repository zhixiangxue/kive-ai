"""
Lightweight memo cache using diskcache

Stores memo_id -> Memo mapping for CRUD operations.
Enables delete/update operations for backends that don't support get-by-id.
"""

from pathlib import Path
from typing import Optional

from diskcache import Cache

from ...models import Memo, BackendType, CogneeBackendData
from ...utils.logger import logger


class MemoCache:
    """Lightweight memo cache using diskcache
    
    Provides persistent storage for memo metadata, enabling CRUD operations
    for backends that don't support direct get-by-id queries (e.g., Cognee).
    
    Storage format: memo_id -> Memo (serialized as JSON dict)
    Backend: SQLite-based diskcache (thread-safe, process-safe)
    """
    
    def __init__(self, cache_dir: str = ".kive/memo_cache"):
        """Initialize memo cache
        
        Args:
            cache_dir: Cache directory path (relative or absolute)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diskcache with SQLite backend
        self._cache = Cache(str(self.cache_dir))
        
        logger.info(f"MemoCache initialized at: {self.cache_dir}")
        logger.info(f"Cache initialized, count={len(self._cache)}")
    
    def save(self, memo: Memo) -> None:
        """Save memo to cache
        
        Args:
            memo: Memo object to save
        """
        # Serialize to dict (JSON-compatible)
        memo_dict = memo.model_dump(mode='json')
        self._cache[memo.id] = memo_dict
        logger.debug(f"Cached memo: {memo.id}")
    
    def get(self, memo_id: str) -> Optional[Memo]:
        """Get memo from cache
        
        Args:
            memo_id: Memo ID
            
        Returns:
            Memo object or None if not found
        """
        memo_dict = self._cache.get(memo_id)
        if memo_dict:
            # Handle backend polymorphism
            backend_data = memo_dict.get('backend', {})
            backend_type = backend_data.get('type')
            
            # Reconstruct correct backend type
            if backend_type == BackendType.COGNEE or backend_type == 'cognee':
                memo_dict['backend'] = CogneeBackendData(**backend_data)
            # Add more backend types here as needed
            # elif backend_type == BackendType.GRAPHITI:
            #     memo_dict['backend'] = GraphitiBackendData(**backend_data)
            
            return Memo(**memo_dict)
        return None
    
    def delete(self, memo_id: str) -> bool:
        """Delete memo from cache
        
        Args:
            memo_id: Memo ID
            
        Returns:
            True if deleted, False if not found
        """
        if memo_id in self._cache:
            del self._cache[memo_id]
            logger.debug(f"Deleted memo from cache: {memo_id}")
            return True
        return False
    
    def exists(self, memo_id: str) -> bool:
        """Check if memo exists in cache
        
        Args:
            memo_id: Memo ID
            
        Returns:
            True if exists
        """
        return memo_id in self._cache
    
    def stats(self) -> dict:
        """Get cache statistics
        
        Returns:
            Statistics dict with hits, misses, size, etc.
        """
        cache_stats = self._cache.stats()
        # diskcache stats() returns a tuple (hits, misses)
        return {
            'hits': cache_stats[0],
            'misses': cache_stats[1],
            'count': len(self._cache),
            'size': self._cache.volume(),
        }
    
    def clear(self) -> None:
        """Clear all cached memos"""
        self._cache.clear()
        logger.info("MemoCache cleared")
    
    def close(self) -> None:
        """Close cache"""
        self._cache.close()
        logger.info("MemoCache closed")
