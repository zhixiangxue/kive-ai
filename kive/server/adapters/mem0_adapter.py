"""Mem0 backend adapter

Integrates mem0 hybrid memory system (vector + graph storage)
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from llama_index.core.schema import Document

from ...exceptions import AdapterError, ConnectionError, SearchError
from ...models import BackendType, Mem0BackendData, Memo
from .base import BaseMemoryAdapter
from ...utils.logger import logger
from .llm_bridge import LLMConfigBridge, UnifiedLLMConfig, LLMProvider, LLMProviderType


class Mem0Adapter(BaseMemoryAdapter):
    """Mem0 hybrid memory adapter
    
    Supports vector search with optional graph storage for relationship tracking
    """
    
    def __init__(
        self,
        # Vector DB configuration (Chroma embedded)
        vector_db_provider: str = "chroma",
        vector_db_uri: Optional[str] = None,  # HTTP URL or None for embedded
        
        # Graph DB configuration (Kuzu embedded, optional)
        graph_db_provider: Optional[str] = None,  # "kuzu" or None
        graph_db_uri: Optional[str] = None,  # DB path or ":memory:"
        
        # LLM configuration (for entity extraction)
        llm_provider: LLMProviderType = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        
        # Embedding configuration
        embedding_provider: LLMProviderType = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        
        # Reranker configuration (optional)
        reranker_provider: Optional[str] = None,
        reranker_model: Optional[str] = None,
        
        # Mem0 multi-tenancy defaults (暂时不支持多租户，使用固定默认值)
        default_user_id: str = "kive_user",
        default_agent_id: Optional[str] = None,
        default_run_id: Optional[str] = None,
    ):
        """
        Args:
            vector_db_provider: Vector database provider (chroma)
            vector_db_uri: Vector database URI
                - Chroma embedded: None (default to system directory)
                - Chroma HTTP: http://localhost:8000
            
            graph_db_provider: Graph database provider (kuzu/None)
            graph_db_uri: Graph database URI
                - Kuzu: File path (e.g., /path/to/mem0.kuzu) or ":memory:"
            
            llm_provider: LLM provider (openai/anthropic/gemini/ollama/groq)
            llm_model: LLM model name
            llm_api_key: LLM API key
            llm_base_url: LLM API base URL (for custom providers)
            
            embedding_provider: Embedding provider (openai/custom)
            embedding_model: Embedding model name
            embedding_api_key: Embedding API key
            embedding_base_url: Embedding API base URL
            embedding_dimensions: Embedding dimensions
            
            reranker_provider: Reranker provider (optional)
            reranker_model: Reranker model name
            
            default_user_id: Default user ID (mem0 requires user_id parameter)
            default_agent_id: Default agent ID
            default_run_id: Default run ID
        """
        # Mem0 processes in real-time, no batch processing
        super().__init__(
            auto_process=False,
            process_interval=0,
            process_batch_size=0,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_dimensions=embedding_dimensions,
            vector_db_provider=vector_db_provider,
            vector_db_uri=vector_db_uri,
            graph_db_provider=graph_db_provider,
            graph_db_uri=graph_db_uri,
        )
        
        # Reranker configuration
        self.reranker_provider = reranker_provider
        self.reranker_model = reranker_model
        
        # Multi-tenancy defaults
        self.default_user_id = default_user_id
        self.default_agent_id = default_agent_id
        self.default_run_id = default_run_id
        
        self._memory = None  # AsyncMemory instance
    
    async def initialize(self) -> None:
        """Initialize Mem0 connection"""
        try:
            from mem0 import AsyncMemory
            from pathlib import Path
            
            logger.info("Initializing Mem0 adapter...")
            
            # Build mem0 configuration
            config = {}
            
            # 1. Configure Vector Store (Chroma)
            if self.vector_db_provider == "chroma":
                vector_config = {
                    "provider": "chroma",
                    "config": {}
                }
                
                # Chroma embedded or HTTP
                if self.vector_db_uri:
                    # HTTP mode
                    vector_config["config"]["host"] = self.vector_db_uri
                    logger.info(f"Chroma HTTP mode: {self.vector_db_uri}")
                else:
                    # Embedded mode - use .kive directory
                    project_root = Path.cwd()
                    chroma_dir = project_root / ".kive" / "chroma"
                    chroma_dir.mkdir(parents=True, exist_ok=True)
                    vector_config["config"]["path"] = str(chroma_dir)
                    logger.info(f"Chroma embedded mode: {chroma_dir}")
                
                config["vector_store"] = vector_config
            else:
                raise ValueError(
                    f"Unsupported vector_db_provider: {self.vector_db_provider}. "
                    "Currently only 'chroma' is supported."
                )
            
            # 2. Configure Graph Store (Kuzu, optional)
            if self.graph_db_provider:
                if self.graph_db_provider.lower() == "kuzu":
                    graph_config = {
                        "provider": "kuzu",
                        "config": {}
                    }
                    
                    # Kuzu database path
                    if self.graph_db_uri:
                        db_path = self.graph_db_uri
                    else:
                        # Default: .kive/mem0.kuzu
                        project_root = Path.cwd()
                        kive_dir = project_root / ".kive"
                        kive_dir.mkdir(parents=True, exist_ok=True)
                        db_path = str(kive_dir / "mem0.kuzu")
                    
                    # Convert to absolute path if needed
                    if db_path != ":memory:":
                        db_path_obj = Path(db_path)
                        if not db_path_obj.is_absolute():
                            db_path_obj = Path.cwd() / db_path_obj
                        db_path = str(db_path_obj)
                    
                    # Kuzu uses 'db' field for database path (not 'path')
                    graph_config["config"]["db"] = db_path
                    config["graph_store"] = graph_config
                    logger.info(f"Kuzu graph store enabled: {db_path}")
                else:
                    raise ValueError(
                        f"Unsupported graph_db_provider: {self.graph_db_provider}. "
                        "Currently only 'kuzu' is supported."
                    )
            
            # 3. Configure LLM using bridge
            unified_llm_config = UnifiedLLMConfig(
                provider=LLMProvider(self.llm_provider),
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
            
            bridge = LLMConfigBridge()
            llm_config = bridge.to_mem0(unified_llm_config)
            
            config["llm"] = llm_config
            logger.info(f"LLM configured: provider={self.llm_provider}, model={self.llm_model}")
            
            # 4. Configure Embedder using bridge
            unified_embedding_config = UnifiedLLMConfig(
                provider=LLMProvider(self.embedding_provider),
                model=self.embedding_model,
                api_key=self.embedding_api_key or self.llm_api_key,
                base_url=self.embedding_base_url or self.llm_base_url,
            )
            
            embedder_config = bridge.to_mem0(unified_embedding_config)
            
            # Add embedding dimensions if specified
            if self.embedding_dimensions:
                embedder_config["config"]["embedding_dims"] = self.embedding_dimensions
            
            config["embedder"] = embedder_config
            logger.info(f"Embedder configured: provider={self.embedding_provider}, model={self.embedding_model}")
            
            # 5. Configure Reranker (optional)
            if self.reranker_provider:
                reranker_config = {
                    "provider": self.reranker_provider,
                    "config": {}
                }
                
                if self.reranker_model:
                    reranker_config["config"]["model"] = self.reranker_model
                
                config["reranker"] = reranker_config
                logger.info(f"Reranker configured: provider={self.reranker_provider}")
            
            # Initialize AsyncMemory
            self._memory = await AsyncMemory.from_config(config)
            
            config_summary = {
                "vector_db_provider": self.vector_db_provider,
                "vector_db_uri": self.vector_db_uri or "embedded",
                "graph_db_provider": self.graph_db_provider or "disabled",
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "reranker_provider": self.reranker_provider or "disabled",
                "default_user_id": self.default_user_id,
            }
            logger.info(f"Mem0Adapter initialized with config: {config_summary}")
            
        except ImportError:
            raise ConnectionError(
                "mem0ai is not installed. "
                "Please install with: pip install kive[mem0]"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Mem0: {e}")
    
    def _create_memo(
        self,
        memo_id: str,
        text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        relations: Optional[Dict[str, Any]] = None,  # Changed from List to Dict
        score: Optional[float] = None,
    ) -> Memo:
        """Create a Memo object with Mem0BackendData
        
        Args:
            memo_id: Memo ID
            text: Memory text content
            user_id: User ID in mem0
            agent_id: Agent ID in mem0
            run_id: Run ID in mem0
            metadata: User-defined metadata
            relations: Graph relations (if graph store enabled)
            score: Search similarity score (optional)
            
        Returns:
            Memo object with Mem0BackendData
        """
        return Memo(
            id=memo_id,
            text=text,
            metadata=metadata or {},
            backend=Mem0BackendData(
                type=BackendType.MEM0,
                version="0.1.0",
                memory_id=memo_id,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                relations=relations,
            ),
            score=score,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    async def add(self, documents: List[Document], **kwargs) -> List[Memo]:
        """Add documents to Mem0 as memories"""
        try:
            if not self._memory:
                raise AdapterError("Mem0 not initialized, call initialize() first")
            
            logger.info(f"Starting add operation for {len(documents)} documents...")
            
            # Get user_id, agent_id, run_id from kwargs or use defaults
            user_id = kwargs.get("user_id", self.default_user_id)
            agent_id = kwargs.get("agent_id", self.default_agent_id)
            run_id = kwargs.get("run_id", self.default_run_id)
            
            memos = []
            
            for doc in documents:
                # Convert Document to messages format for mem0
                # mem0 expects messages = [{"role": "user", "content": "..."}]
                messages = [
                    {"role": "user", "content": doc.text}
                ]
                
                # Add to mem0
                # Get infer parameter from kwargs, default to True (mem0's default behavior)
                infer = kwargs.get("infer", True)
                
                add_kwargs = {"user_id": user_id, "infer": infer}
                if agent_id:
                    add_kwargs["agent_id"] = agent_id
                if run_id:
                    add_kwargs["run_id"] = run_id
                
                result = await self._memory.add(messages, **add_kwargs)
                
                # Debug: log the actual result structure
                logger.info(f"Mem0 add result type: {type(result)}")
                logger.info(f"Mem0 add result: {result}")
                
                # Extract memory_id and relations from result
                # mem0 returns: {"results": [...], "relations": {...}} or {"results": [...]}
                if isinstance(result, dict):
                    # Get graph relations (shared across all results)
                    graph_relations = result.get("relations")
                    
                    # Process vector store results
                    if "results" in result and result["results"]:
                        logger.info(f"Found {len(result['results'])} vector store results")
                        for item in result["results"]:
                            memory_id = item.get("id", str(item))
                            memory_text = item.get("memory", doc.text)
                            
                            memo = self._create_memo(
                                memo_id=memory_id,
                                text=memory_text,
                                user_id=user_id,
                                agent_id=agent_id,
                                run_id=run_id,
                                metadata=doc.metadata,
                                relations=graph_relations,
                            )
                            memos.append(memo)
                    else:
                        # No vector results - mem0 decided not to store this as a memory
                        # This can happen when content is filtered, deduplicated, or only graph relations are extracted
                        if graph_relations:
                            logger.warning(
                                f"Mem0 did not create a memory (results is empty), only graph relations extracted. "
                                f"Document: '{doc.text[:50]}...'"
                            )
                        else:
                            logger.warning(
                                f"Mem0 did not create a memory or relations. "
                                f"Document: '{doc.text[:50]}...'"
                            )
                else:
                    # Fallback: create memo with generated ID
                    logger.warning(f"Unexpected add result format: {result}")
                    memo = self._create_memo(
                        memo_id=str(result),
                        text=doc.text,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        metadata=doc.metadata,
                    )
                    memos.append(memo)
            
            logger.info(f"Added {len(memos)} memories to Mem0")
            return memos
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise AdapterError(f"Failed to add documents: {e}")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> List[Memo]:
        """Search in Mem0
        
        Args:
            query: Search query text
            limit: Maximum number of results
            **kwargs: Additional parameters (rerank, filters, etc.)
        """
        try:
            if not self._memory:
                raise AdapterError("Mem0 not initialized")
            
            # Get user_id, agent_id, run_id from kwargs or use defaults
            user_id = kwargs.get("user_id", self.default_user_id)
            agent_id = kwargs.get("agent_id", self.default_agent_id)
            run_id = kwargs.get("run_id", self.default_run_id)
            
            # Enable reranking if reranker is configured
            rerank = kwargs.get("rerank", bool(self.reranker_provider))
            
            # Build search kwargs
            search_kwargs = {
                "user_id": user_id,
                "limit": limit,
                "rerank": rerank,
            }
            
            if agent_id:
                search_kwargs["agent_id"] = agent_id
            if run_id:
                search_kwargs["run_id"] = run_id
            
            # Call mem0.search
            search_results = await self._memory.search(query, **search_kwargs)
            
            logger.info(f"Search returned {len(search_results.get('results', [])) if isinstance(search_results, dict) else 0} results")
            
            # Convert mem0 search results to Memos
            memos = []
            
            if isinstance(search_results, dict) and "results" in search_results:
                for i, result in enumerate(search_results["results"][:limit]):
                    memory_id = result.get("id", f"search_result_{i}")
                    memory_text = result.get("memory", "")
                    memory_score = result.get("score", 1.0 - (i * 0.05))
                    relations = result.get("relations")  # Graph relations if enabled
                    
                    # Extract metadata
                    metadata = result.get("metadata", {})
                    
                    memo = self._create_memo(
                        memo_id=memory_id,
                        text=memory_text,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        metadata=metadata,
                        relations=relations,
                        score=memory_score,
                    )
                    memos.append(memo)
            
            logger.info(f"Search completed: query='{query}', results={len(memos)}")
            return memos
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def get(self, memo_id: str, **kwargs) -> Optional[Memo]:
        """Get single memory by ID
        
        Args:
            memo_id: Memory ID
        """
        try:
            if not self._memory:
                raise AdapterError("Mem0 not initialized")
            
            # Get user_id from kwargs or use default
            # Note: mem0's get() requires user_id for permission check
            user_id = kwargs.get("user_id", self.default_user_id)
            
            # Call mem0.get
            result = await self._memory.get(memory_id=memo_id)
            
            if not result:
                logger.warning(f"Memory not found: {memo_id}")
                return None
            
            # Create Memo from result
            memory_text = result.get("memory", "")
            metadata = result.get("metadata", {})
            relations = result.get("relations")
            
            # Extract user_id, agent_id, run_id from result if available
            result_user_id = result.get("user_id", user_id)
            result_agent_id = result.get("agent_id")
            result_run_id = result.get("run_id")
            
            memo = self._create_memo(
                memo_id=memo_id,
                text=memory_text,
                user_id=result_user_id,
                agent_id=result_agent_id,
                run_id=result_run_id,
                metadata=metadata,
                relations=relations,
            )
            
            logger.info(f"Retrieved memory: {memo_id}")
            return memo
            
        except Exception as e:
            logger.error(f"Failed to get memory {memo_id}: {e}")
            raise AdapterError(f"Failed to get memory: {e}")
    
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
        """
        try:
            if not self._memory:
                raise AdapterError("Mem0 not initialized")
            
            # Extract backend data (type-safe)
            if not isinstance(memo.backend, Mem0BackendData):
                raise AdapterError(f"Expected Mem0BackendData, got {type(memo.backend).__name__}")
            
            backend_data: Mem0BackendData = memo.backend
            
            # Call mem0.update (only needs memory_id and data)
            result = await self._memory.update(
                memory_id=backend_data.memory_id,
                data=document.text,
            )
            
            logger.info(f"Mem0 update result: {result}")
            
            # Create updated Memo
            # mem0 update returns the updated memory data
            if isinstance(result, dict):
                memory_text = result.get("memory", document.text)
                relations = result.get("relations")
            else:
                memory_text = document.text
                relations = None
            
            updated_memo = self._create_memo(
                memo_id=backend_data.memory_id,
                text=memory_text,
                user_id=backend_data.user_id,
                agent_id=backend_data.agent_id,
                run_id=backend_data.run_id,
                metadata=document.metadata,
                relations=relations,
            )
            
            logger.info(f"Updated memory: {memo.id}")
            return updated_memo
            
        except Exception as e:
            logger.error(f"Failed to update memo {memo.id}: {e}")
            raise AdapterError(f"Failed to update memo: {e}")
    
    async def delete(self, memos: Union[Memo, List[Memo]], **kwargs) -> bool:
        """Delete memories"""
        if not self._memory:
            raise AdapterError("Mem0 not initialized")
        
        # Normalize to list
        memo_list = [memos] if isinstance(memos, Memo) else memos
        
        success_count = 0
        failed_count = 0
        
        for memo in memo_list:
            try:
                # Extract backend data (type-safe)
                if not isinstance(memo.backend, Mem0BackendData):
                    logger.error(f"Expected Mem0BackendData for memo {memo.id}, got {type(memo.backend).__name__}, skipping")
                    failed_count += 1
                    continue
                
                backend_data: Mem0BackendData = memo.backend
                
                # Call mem0.delete (only needs memory_id)
                await self._memory.delete(
                    memory_id=backend_data.memory_id,
                )
                
                logger.info(f"Deleted memory {memo.id}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to delete memory {memo.id}: {e}")
                failed_count += 1
        
        logger.info(f"Delete completed: success={success_count}, failed={failed_count}")
        return success_count > 0
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """Mem0 processes in real-time, no batch processing needed"""
        return {
            "status": "not_supported",
            "message": "Mem0 processes memories in real-time, no batch processing needed"
        }
    
    async def close(self) -> None:
        """Close Mem0 connection"""
        # AsyncMemory doesn't require explicit cleanup
        # But we can reset the instance
        self._memory = None
        logger.info("Mem0 connection closed")
        await super().close()
