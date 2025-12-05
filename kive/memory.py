"""Kive Memory - Local memory client (no server needed)

Provides direct access to memory backends without HTTP server.
"""

import asyncio
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from llama_index.core.schema import Document

from .exceptions import KiveError
from .models import AddMemoRequest, Memo, SearchMemoRequest
from .adapters.base import BaseMemoryAdapter
from .utils.logger import logger


def ensure_initialized(func):
    """Decorator to ensure adapter is initialized before calling method
    
    Automatically initializes the adapter on first call to any decorated method.
    Subsequent calls skip initialization (using lock to prevent race conditions).
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        await self._ensure_initialized()
        return await func(self, *args, **kwargs)
    return wrapper


class Memory:
    """Kive memory client for local usage
    
    Usage:
        from kive import engines, Memory
        
        # Create engine
        engine = engines.Mem0(llm_api_key="sk-xxx", llm_model="gpt-4o-mini")
        
        # Create memory client
        memory = Memory(engine=engine)
        await memory.initialize()
        
        # Add memo
        memos = await memory.add(text="Hello world", namespace="personal")
        
        # Search
        results = await memory.search(query="Hello", namespace="personal")
    """
    
    def __init__(self, engine: BaseMemoryAdapter):
        """Initialize memory client with backend adapter
        
        Args:
            engine: Memory adapter instance (Mem0Adapter, CogneeAdapter, GraphitiAdapter)
        
        Example:
            from kive import engines, Memory
            
            # Create engine
            engine = engines.Mem0(llm_api_key="sk-xxx", llm_model="gpt-4o-mini")
            
            # Pass to Memory (initialization happens automatically on first use)
            memory = Memory(engine=engine)
            await memory.add(text="Hello")  # Auto-initializes here
        """
        self.adapter = engine
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def _ensure_initialized(self) -> None:
        """Ensure adapter is initialized (called automatically by decorated methods)
        
        This method is thread-safe and ensures initialization happens only once.
        """
        if self._initialized:
            return
        
        async with self._init_lock:
            # Double-check pattern to prevent race conditions
            if self._initialized:
                return
            
            try:
                await self.adapter.initialize()
                self._initialized = True
                logger.info(f"Memory initialized with backend: {self.adapter.__class__.__name__}")
            except Exception as e:
                raise KiveError(f"Failed to initialize memory: {e}")
    
    @ensure_initialized
    async def add(
        self,
        text: Optional[str] = None,
        file: Optional[str] = None,
        url: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Context fields
        app_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        ai_id: str = "default",
        tenant_id: str = "default",
        session_id: str = "default",
    ) -> List[Memo]:
        """Add memo to memory
        
        Args:
            text: Text content
            file: File path
            url: URL address
            messages: Conversational messages (list of {"role": "user/assistant", "content": "..."})
            metadata: Additional metadata
            
            app_id: Application ID (default: "default")
            user_id: User ID (default: "default")
            namespace: Namespace for data isolation (default: "default")
            ai_id: AI/Agent ID (default: "default")
            tenant_id: Tenant ID (default: "default")
            session_id: Session ID (default: "default")
        
        Returns:
            List of created Memo objects
            
        Raises:
            KiveError: If add operation fails
        """
        try:
            request = AddMemoRequest(
                text=text,
                file=file,
                url=url,
                messages=messages,
                metadata=metadata or {},
                app_id=app_id,
                user_id=user_id,
                namespace=namespace,
                ai_id=ai_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            
            memos = await self.adapter.add(request)
            logger.info(f"Added {len(memos)} memo(s) to {self.adapter.__class__.__name__}")
            return memos
            
        except Exception as e:
            raise KiveError(f"Failed to add memo: {e}")
    
    @ensure_initialized
    async def search(
        self,
        query: str,
        limit: int = 10,
        # Context fields
        app_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        ai_id: str = "default",
        tenant_id: str = "default",
        session_id: str = "default",
    ) -> List[Memo]:
        """Search memories
        
        Args:
            query: Search query text
            limit: Maximum number of results (default: 10)
            
            app_id: Application ID (default: "default")
            user_id: User ID (default: "default")
            namespace: Namespace for data isolation (default: "default")
            ai_id: AI/Agent ID (default: "default")
            tenant_id: Tenant ID (default: "default")
            session_id: Session ID (default: "default")
        
        Returns:
            List of matching Memo objects
            
        Raises:
            KiveError: If search operation fails
        """
        try:
            request = SearchMemoRequest(
                query=query,
                limit=limit,
                app_id=app_id,
                user_id=user_id,
                namespace=namespace,
                ai_id=ai_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            
            results = await self.adapter.search(request)
            logger.info(f"Search returned {len(results)} result(s) from {self.adapter.__class__.__name__}")
            return results
            
        except Exception as e:
            raise KiveError(f"Failed to search: {e}")
    
    @ensure_initialized
    async def get(self, memo_id: str, **kwargs) -> Optional[Memo]:
        """Get single memo by ID
        
        Note: Not all backends support this operation.
        
        Args:
            memo_id: Memo ID
            **kwargs: Backend-specific parameters
        
        Returns:
            Memo object if found, None otherwise
            
        Raises:
            KiveError: If get operation fails
        """
        try:
            memo = await self.adapter.get(memo_id, **kwargs)
            return memo
        except Exception as e:
            raise KiveError(f"Failed to get memo: {e}")
    
    @ensure_initialized
    async def update(
        self,
        memo: Memo,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memo:
        """Update memo
        
        Args:
            memo: Memo object to update
            text: New text content
            metadata: New metadata (merged with existing)
        
        Returns:
            Updated Memo object (may have new ID)
            
        Raises:
            KiveError: If update operation fails
        """
        try:
            # Build document with updated content
            updated_metadata = {**memo.metadata, **(metadata or {})}
            document = Document(
                text=text if text is not None else memo.text,
                metadata=updated_metadata,
            )
            
            updated_memo = await self.adapter.update(memo, document)
            logger.info(f"Updated memo: {memo.id}")
            return updated_memo
            
        except Exception as e:
            raise KiveError(f"Failed to update memo: {e}")
    
    @ensure_initialized
    async def delete(self, memos: Union[Memo, List[Memo]]) -> bool:
        """Delete memo(s)
        
        Args:
            memos: Single Memo or list of Memos to delete
        
        Returns:
            True if at least one deletion succeeded
            
        Raises:
            KiveError: If delete operation fails
        """
        try:
            success = await self.adapter.delete(memos)
            memo_count = 1 if isinstance(memos, Memo) else len(memos)
            logger.info(f"Deleted {memo_count} memo(s) from {self.adapter.__class__.__name__}")
            return success
            
        except Exception as e:
            raise KiveError(f"Failed to delete memo(s): {e}")
    
    @ensure_initialized
    async def process(self, **kwargs) -> Dict[str, Any]:
        """Trigger backend processing (e.g., cognee's cognify)
        
        Note: Not all backends support this operation.
        
        Args:
            **kwargs: Backend-specific parameters
        
        Returns:
            Processing result dict
            
        Raises:
            KiveError: If process operation fails
        """
        try:
            result = await self.adapter.process(**kwargs)
            logger.info(f"Process result: {result}")
            return result
            
        except Exception as e:
            raise KiveError(f"Failed to process: {e}")
    
    async def close(self) -> None:
        """Close adapter connection and cleanup resources"""
        try:
            await self.adapter.close()
            self._initialized = False
            logger.info(f"Memory closed: {self.adapter.__class__.__name__}")
        except Exception as e:
            raise KiveError(f"Failed to close memory: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
