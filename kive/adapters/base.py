"""
Base class for backend adapters

Defines the interface that all memory engine adapters must implement
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from llama_index.core.schema import Document

from ..models import Memo
from ..utils.logger import logger

if TYPE_CHECKING:
    from ..models import AddMemoRequest, SearchMemoRequest


class BaseMemoryAdapter(ABC):
    """Unified base class for memory engine adapters
    
    All backend adapters must inherit from this class and implement the corresponding methods
    """
    
    def __init__(
        self,
        # Auto-process configuration
        auto_process: bool = False,
        process_interval: int = 30,
        process_batch_size: int = 100,
        # LLM configuration
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        # Embedding configuration
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        # Vector DB configuration
        vector_db_provider: Optional[str] = None,
        vector_db_uri: Optional[str] = None,
        vector_db_key: Optional[str] = None,
        # Graph DB configuration
        graph_db_provider: Optional[str] = None,
        graph_db_uri: Optional[str] = None,
        graph_db_username: Optional[str] = None,
        graph_db_password: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            auto_process: Whether to automatically process data
            process_interval: Auto-process interval in seconds
            process_batch_size: Trigger processing immediately when reaching this count
            
            llm_provider: LLM provider (openai/anthropic/gemini/ollama/groq/mistral/custom)
            llm_model: LLM model name
            llm_api_key: LLM API key
            llm_base_url: LLM API base URL (for custom providers)
            
            embedding_provider: Embedding provider (openai/ollama/custom)
            embedding_model: Embedding model name
            embedding_api_key: Embedding API key
            embedding_base_url: Embedding API base URL
            embedding_dimensions: Embedding dimensions
            
            vector_db_provider: Vector database provider (chromadb/lancedb)
            vector_db_uri: Vector database connection URI
            vector_db_key: Vector database authentication key
            
            graph_db_provider: Graph database provider (kuzu/neo4j/falkordb/networkx)
            graph_db_uri: Graph database connection URI
            graph_db_username: Graph database username (for Neo4j)
            graph_db_password: Graph database password (for Neo4j)
            
            **kwargs: Backend-specific configuration parameters
        """
        # Auto-process configuration
        self.auto_process = auto_process
        self.process_interval = process_interval
        self.process_batch_size = process_batch_size
        
        # LLM configuration
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        
        # Embedding configuration
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_dimensions = embedding_dimensions
        
        # Vector DB configuration
        self.vector_db_provider = vector_db_provider
        self.vector_db_uri = vector_db_uri
        self.vector_db_key = vector_db_key
        
        # Graph DB configuration
        self.graph_db_provider = graph_db_provider
        self.graph_db_uri = graph_db_uri
        self.graph_db_username = graph_db_username
        self.graph_db_password = graph_db_password
        
        # Internal state
        self._pending_count = 0  # Pending count
        self._processing = False  # Processing flag
        self._last_process_time: Optional[float] = None
        self._background_task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend connection
        
        Raises:
            ConnectionError: Raised when connection fails
        """
        pass
    
    @abstractmethod
    async def add(self, request: 'AddMemoRequest') -> List[Memo]:
        """Add memories
        
        Args:
            request: AddMemoRequest with text/file/url and context fields
            
        Returns:
            List of Memo objects with backend data and Kive metadata
            
        Raises:
            AdapterError: Raised when adding fails
        """
        pass
    
    @abstractmethod
    async def search(self, request: 'SearchMemoRequest') -> List[Memo]:
        """Search memories
        
        Args:
            request: SearchMemoRequest with query and context fields
            
        Returns:
            List of Memo objects with search results
            
        Raises:
            SearchError: Raised when search fails
        """
        pass
    
    @abstractmethod
    async def get(self, memo_id: str, **kwargs) -> Optional[Memo]:
        """Get a single memory
        
        Args:
            memo_id: Memory ID
            **kwargs: Backend-specific parameters
            
        Returns:
            Memo object, returns None if not found or not supported
            
        Raises:
            AdapterError: Raised when retrieval fails
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        memo: Memo,
        document: Document,
        **kwargs
    ) -> Memo:
        """Update memory
        
        Args:
            memo: Memo to update (contains backend tracking info)
            document: New document content
            **kwargs: Backend-specific parameters
            
        Returns:
            Updated Memo object (may have new backend ID)
            
        Raises:
            AdapterError: Update failed
        """
        pass
    
    @abstractmethod
    async def delete(self, memos: Union[Memo, List[Memo]], **kwargs) -> bool:
        """Delete memories
        
        Args:
            memos: Memo or list of Memos to delete (contains backend tracking info)
            **kwargs: Backend-specific parameters
            
        Returns:
            Whether delete succeeded
            
        Raises:
            AdapterError: Delete failed
        """
        pass
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """Process data (cognee's cognify)
        
        For backends that don't support process, returns not_supported directly
        Subclasses can override this method to implement specific processing logic
        
        Args:
            **kwargs: Backend-specific parameters
            
        Returns:
            Processing result {"status": "success/not_supported/error", "message": "..."}
        """
        return {
            "status": "not_supported",
            "message": f"{self.__class__.__name__} does not support process"
        }
    
    async def close(self) -> None:
        """Close connection and cleanup resources"""
        # Stop background task
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info(f"{self.__class__.__name__} closed")
    
    # ===== Auto-processing related methods =====
    
    def increment_pending(self, count: int = 1) -> None:
        """Increment pending count"""
        self._pending_count += count
        logger.debug(f"Pending count: {self._pending_count}")
    
    def should_trigger_process(self) -> bool:
        """Determine whether to trigger processing"""
        if not self.auto_process:
            return False
        if self._processing:
            return False
        if self._pending_count >= self.process_batch_size:
            return True
        return False
    
    async def trigger_process(self, **kwargs) -> Dict[str, Any]:
        """Trigger processing (can be called externally or internally)"""
        if self._processing:
            logger.warning("Processing already in progress")
            return {"status": "already_processing"}
        
        try:
            self._processing = True
            logger.info(f"Starting process, pending: {self._pending_count}")
            
            result = await self.process(**kwargs)
            
            if result.get("status") != "not_supported":
                self._pending_count = 0
                self._last_process_time = asyncio.get_event_loop().time()
            
            logger.info(f"Process completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Process failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self._processing = False
    
    async def start_background_processor(self) -> None:
        """Start background processing task"""
        if not self.auto_process:
            return
        
        async def _processor():
            while True:
                await asyncio.sleep(self.process_interval)
                if self._pending_count > 0 and not self._processing:
                    await self.trigger_process()
        
        self._background_task = asyncio.create_task(_processor())
        logger.info(
            f"Background processor started: "
            f"interval={self.process_interval}s, "
            f"batch_size={self.process_batch_size}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "adapter": self.__class__.__name__,
            "auto_process": self.auto_process,
            "pending_count": self._pending_count,
            "processing": self._processing,
            "last_process_time": self._last_process_time,
        }
