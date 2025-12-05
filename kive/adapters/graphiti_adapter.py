"""Graphiti backend adapter

Integrates graphiti temporal knowledge graph engine
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from llama_index.core.schema import Document

from ..exceptions import AdapterError, ConnectionError, SearchError
from ..models import BackendType, GraphitiBackendData, Memo
from .base import BaseMemoryAdapter
from ..utils.logger import logger
from .llm_bridge import LLMConfigBridge, UnifiedLLMConfig, LLMProvider, LLMProviderType


class GraphitiAdapter(BaseMemoryAdapter):
    """Graphiti temporal knowledge graph adapter
    
    Supports episodic memory with temporal awareness and hybrid search
    """
    
    def __init__(
        self,
        # Graph DB configuration
        graph_db_provider: str = "kuzu",
        graph_db_uri: Optional[str] = None,
        graph_db_username: Optional[str] = None,
        graph_db_password: Optional[str] = None,
        # LLM configuration
        llm_provider: Optional[LLMProviderType] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        # Embedding configuration
        embedding_provider: Optional[LLMProviderType] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        # Other settings
        default_source_description: str = "kive document",
    ):
        """
        Args:
            graph_db_provider: Graph database provider (kuzu/neo4j/falkordb)
            graph_db_uri: Graph database connection URI
                - Kuzu: File path (e.g., /path/to/graphiti.kuzu) or :memory: for in-memory
                - Neo4j: Connection URI (e.g., bolt://localhost:7687)
                - FalkorDB: Redis URI (e.g., redis://localhost:6379)
            graph_db_username: Graph database username (for Neo4j)
            graph_db_password: Graph database password (for Neo4j)
            
            llm_provider: LLM provider (openai/anthropic/gemini/ollama/groq)
            llm_model: LLM model name
            llm_api_key: LLM API key
            llm_base_url: LLM API base URL (for custom providers)
            
            embedding_provider: Embedding provider (openai/custom)
            embedding_model: Embedding model name
            embedding_api_key: Embedding API key
            embedding_base_url: Embedding API base URL
            embedding_dimensions: Embedding dimensions
            
            default_source_description: Default episode source description
        """
        # Graphiti processes in real-time, no batch processing needed
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
            graph_db_provider=graph_db_provider,
            graph_db_uri=graph_db_uri,
            graph_db_username=graph_db_username,
            graph_db_password=graph_db_password,
        )
        
        # Normalize graph_db_provider to lowercase
        if self.graph_db_provider:
            self.graph_db_provider = self.graph_db_provider.lower()
        
        # Other settings
        self.default_source_description = default_source_description
        
        self._graphiti = None
    

    
    async def initialize(self) -> None:
        """Initialize Graphiti connection"""
        try:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from pathlib import Path
            import os
            
            # Disable telemetry
            os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
            logger.info("Graphiti telemetry disabled")
            
            # Initialize graph driver based on provider
            graph_driver = None
            graphiti_kwargs = {}
            
            if self.graph_db_provider == "kuzu":
                from graphiti_core.driver.kuzu_driver import KuzuDriver
                
                # Determine Kuzu database path
                if self.graph_db_uri:
                    db_path = self.graph_db_uri
                    # Convert to absolute path if needed
                    if db_path != ":memory:":
                        db_path_obj = Path(db_path)
                        if not db_path_obj.is_absolute():
                            db_path_obj = Path.cwd() / db_path_obj
                        db_path = str(db_path_obj)
                else:
                    # Default: .kive/graphiti.kuzu
                    project_root = Path.cwd()
                    kive_dir = project_root / ".kive"
                    kive_dir.mkdir(parents=True, exist_ok=True)
                    db_path = str(kive_dir / "graphiti.kuzu")
                
                logger.info(f"Kuzu database path: {db_path}")
                graph_driver = KuzuDriver(db=db_path)
                
                # ============================================================
                # TODO: Remove this workaround when Graphiti fixes KuzuDriver bug
                # ============================================================
                # Issue: Graphiti's add_episode() (line 698/883) checks:
                #        if group_id != self.driver._database:
                #        but KuzuDriver does NOT have `_database` attribute
                # 
                # Root Cause:
                #   - Graphiti assumes all drivers have `_database` for multi-db support
                #   - Neo4j/FalkorDB drivers have this attribute
                #   - KuzuDriver is embedded (one file = one database), doesn't need it
                #   - See: .venv/Lib/site-packages/graphiti_core/driver/kuzu_driver.py:97-107
                # 
                # Error:
                #   AttributeError: 'KuzuDriver' object has no attribute '_database'
                # 
                # Workaround:
                #   Monkey-patch `_database` attribute to None (Kuzu doesn't use it)
                #   This allows Graphiti's check to pass without breaking other drivers
                # 
                # Impact:
                #   - ✅ Fixes Kuzu compatibility
                #   - ✅ Does NOT affect Neo4j/FalkorDB (they have their own _database)
                #   - ✅ group_id still works for namespace isolation
                # 
                # Upstream Issue: https://github.com/getzep/graphiti/issues/XXX
                # Related: kive/server/adapters/request_bridge.py:to_graphiti_add()
                # ============================================================
                if not hasattr(graph_driver, '_database'):
                    graph_driver._database = None  # Kuzu doesn't support multi-database
                    logger.info("Applied KuzuDriver._database workaround for Graphiti compatibility")
                # ============================================================
                
                # Apply Kuzu FTS indices extension immediately after driver creation
                # This must be done before Graphiti initialization
                from .extensions.graphiti.drivers import patch_kuzu_fulltext_indices
                await patch_kuzu_fulltext_indices(graph_driver)
                
                graphiti_kwargs["graph_driver"] = graph_driver
                
            elif self.graph_db_provider == "neo4j":
                # Neo4j requires URI, username, password
                if not all([self.graph_db_uri, self.graph_db_username, self.graph_db_password]):
                    raise ValueError(
                        "Neo4j requires graph_db_uri, graph_db_username, and graph_db_password. "
                        "Please provide all three parameters."
                    )
                
                logger.info(f"Neo4j URI: {self.graph_db_uri}")
                graphiti_kwargs["uri"] = self.graph_db_uri
                graphiti_kwargs["user"] = self.graph_db_username
                graphiti_kwargs["password"] = self.graph_db_password
                
            elif self.graph_db_provider == "falkordb":
                from graphiti_core.driver.falkordb_driver import FalkorDBDriver
                
                # FalkorDB uses Redis protocol (default: redis://localhost:6379)
                redis_uri = self.graph_db_uri or "redis://localhost:6379"
                logger.info(f"FalkorDB URI: {redis_uri}")
                
                graph_driver = FalkorDBDriver(url=redis_uri)
                graphiti_kwargs["graph_driver"] = graph_driver
                
            else:
                raise ValueError(
                    f"Unsupported graph_db_provider: {self.graph_db_provider}. "
                    "Supported providers: kuzu, neo4j, falkordb"
                )
            
            # Configure LLM client using bridge
            unified_llm_config = UnifiedLLMConfig(
                provider=LLMProvider(self.llm_provider) if self.llm_provider else LLMProvider.OPENAI,
                model=self.llm_model or "gpt-4o-mini",
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
            
            bridge = LLMConfigBridge()
            llm_client = bridge.to_graphiti(unified_llm_config)
            
            logger.info(f"LLM client configured: model={unified_llm_config.model}, base_url={unified_llm_config.base_url}")
            
            # Create LLM config for reranker (OpenAI format)
            from graphiti_core.llm_client.config import LLMConfig
            llm_config = LLMConfig(
                api_key=self.llm_api_key or "placeholder",
                model=self.llm_model or "gpt-4o-mini",
                base_url=self.llm_base_url,
            )
            
            # Configure Embedder (OpenAI-compatible, shared by all providers)
            embedder_config = OpenAIEmbedderConfig(
                api_key=self.embedding_api_key or self.llm_api_key or "placeholder",
                embedding_model=self.embedding_model or "text-embedding-3-small",
                base_url=self.embedding_base_url or self.llm_base_url,
                embedding_dim=self.embedding_dimensions or 1536,
            )
            embedder = OpenAIEmbedder(config=embedder_config)
            logger.info(f"Embedder configured: model={embedder_config.embedding_model}, dim={embedder_config.embedding_dim}")
            
            # Configure Cross Encoder (reranker, shared by all providers)
            cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)
            logger.info("Cross encoder (reranker) configured")
            
            # Initialize Graphiti with all components
            self._graphiti = Graphiti(
                **graphiti_kwargs,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
            
            # Build indices and constraints (safe to call multiple times)
            await self._graphiti.build_indices_and_constraints()
            
            config_summary = {
                "graph_db_provider": self.graph_db_provider,
                "graph_db_uri": self.graph_db_uri,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
            }
            logger.info(f"GraphitiAdapter initialized with config: {config_summary}")
            
        except ImportError as e:
            if "kuzu" in str(e).lower():
                raise ConnectionError(
                    "kuzu driver is not installed. "
                    "Please install with: pip install kive[graphiti-kuzu]"
                )
            elif "falkordb" in str(e).lower():
                raise ConnectionError(
                    "falkordb driver is not installed. "
                    "Please install with: pip install kive[graphiti-falkordb]"
                )
            else:
                raise ConnectionError(
                    "graphiti-core is not installed. "
                    "Please install with: pip install kive[graphiti]"
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Graphiti: {e}")
    
    def _create_memo(
        self,
        memo_id: str,
        text: str,
        episode_id: str,
        source: str = "text",
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
    ) -> Memo:
        """Create a Memo object with GraphitiBackendData
        
        Args:
            memo_id: Memo ID
            text: Memory text content
            episode_id: Graphiti episode UUID
            source: Episode source type
            source_description: Episode source description
            metadata: User-defined metadata
            score: Search similarity score (optional)
            
        Returns:
            Memo object with GraphitiBackendData
        """
        return Memo(
            id=memo_id,
            text=text,
            metadata=metadata or {},
            backend=GraphitiBackendData(
                type=BackendType.GRAPHITI,
                version="0.24.1",
                episode_id=episode_id,
                source=source,
                source_description=source_description,
            ),
            score=score,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    async def add(self, request: 'AddMemoRequest') -> List[Memo]:
        """Add documents to Graphiti as episodes
        
        Args:
            request: AddMemoRequest with text and context fields
        """
        try:
            if not self._graphiti:
                raise AdapterError("Graphiti not initialized, call initialize() first")
            
            # Import models
            from .request_bridge import RequestBridge
            from ..models import AddMemoRequest
            from graphiti_core.nodes import EpisodeType
            
            logger.info(f"Starting add operation with request: namespace={request.namespace}, user_id={request.user_id}")
            
            # Use bridge to get Graphiti parameters
            bridge = RequestBridge()
            graphiti_params = bridge.to_graphiti_add(request)
            
            # Add episode to Graphiti
            add_result = await self._graphiti.add_episode(
                name=graphiti_params["name"],
                episode_body=graphiti_params["episode_body"],
                source=EpisodeType.text,
                source_description=graphiti_params["source_description"],
                reference_time=graphiti_params["reference_time"],
                group_id=graphiti_params["group_id"],
            )
            
            # Extract episode from result
            episode = add_result.episode
            
            # Get external metadata (with Kive context)
            external_metadata = bridge.get_external_metadata(request)
            
            # Create Memo
            memo = self._create_memo(
                memo_id=str(episode.uuid),
                text=request.text or "",
                episode_id=str(episode.uuid),
                source=EpisodeType.text.name,
                source_description=graphiti_params["source_description"],
                metadata=external_metadata,
            )
            
            logger.info(f"Added episode: {graphiti_params['name']}, UUID: {episode.uuid}")
            logger.info(f"Added 1 episode to Graphiti namespace '{request.namespace}'")
            
            return [memo]
            
        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            raise AdapterError(f"Failed to add episode: {e}")
    
    async def search(self, request: 'SearchMemoRequest') -> List[Memo]:
        """Search in Graphiti using hybrid search
        
        Args:
            request: SearchMemoRequest with query and context fields
        """
        try:
            if not self._graphiti:
                raise AdapterError("Graphiti not initialized")
            
            # Import models
            from .request_bridge import RequestBridge
            from ..models import SearchMemoRequest
            
            logger.info(f"Starting search with query='{request.query}', namespace={request.namespace}")
            
            # Use bridge to get Graphiti parameters
            bridge = RequestBridge()
            graphiti_params = bridge.to_graphiti_search(request)
            
            # Call graphiti.search
            search_kwargs = {
                "query": request.query,
            }
            
            # Add group_ids if provided
            if graphiti_params.get("group_ids"):
                search_kwargs["group_ids"] = graphiti_params["group_ids"]
            
            search_results = await self._graphiti.search(**search_kwargs)
            
            logger.info(f"Search returned {len(search_results) if search_results else 0} results")
            
            # Convert search results to Memos
            memos = []
            for i, result in enumerate(search_results[:request.limit] if search_results else []):
                # Graphiti returns Edge objects with fact, uuid, etc.
                result_id = str(result.uuid)
                result_text = result.fact
                result_score = 1.0 - (i * 0.05)  # Simple scoring based on rank
                
                # Extract metadata
                result_metadata = {
                    "source_node_uuid": str(result.source_node_uuid) if hasattr(result, "source_node_uuid") else None,
                    "target_node_uuid": str(result.target_node_uuid) if hasattr(result, "target_node_uuid") else None,
                    "valid_at": str(result.valid_at) if hasattr(result, "valid_at") and result.valid_at else None,
                    "invalid_at": str(result.invalid_at) if hasattr(result, "invalid_at") and result.invalid_at else None,
                }
                
                # Get external metadata (with Kive context)
                external_metadata = bridge.get_external_metadata(request)
                # Merge with result metadata
                merged_metadata = {**result_metadata, **external_metadata}
                
                # Create Memo (search results don't have episode_id)
                memo = self._create_memo(
                    memo_id=result_id,
                    text=result_text,
                    episode_id=result_id,  # Use edge UUID as fallback
                    source="search_result",
                    source_description="hybrid search result",
                    metadata=merged_metadata,
                    score=result_score,
                )
                memos.append(memo)
            
            logger.info(f"Search completed: query='{request.query}', results={len(memos)}")
            return memos
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def get(self, memo_id: str, **kwargs) -> Optional[Memo]:
        """Get single memory by episode UUID
        
        Args:
            memo_id: Episode UUID
        """
        try:
            if not self._graphiti:
                raise AdapterError("Graphiti not initialized")
            
            from graphiti_core.nodes import EpisodicNode
            
            # Get episode by UUID
            episode = await EpisodicNode.get_by_uuid(
                self._graphiti.driver,
                memo_id
            )
            
            if not episode:
                logger.warning(f"Episode not found: {memo_id}")
                return None
            
            # Create Memo from episode
            memo = self._create_memo(
                memo_id=str(episode.uuid),
                text=episode.content if hasattr(episode, "content") else episode.name,
                episode_id=str(episode.uuid),
                source="text",
                source_description=None,
                metadata={},
            )
            
            logger.info(f"Retrieved episode: {memo_id}")
            return memo
            
        except Exception as e:
            logger.error(f"Failed to get episode {memo_id}: {e}")
            raise AdapterError(f"Failed to get episode: {e}")
    
    async def update(
        self,
        memo: Memo,
        document: Document,
        **kwargs
    ) -> Memo:
        """Update memory (delete old episode + add new episode)
        
        Graphiti's recommended approach for updates
        """
        try:
            if not self._graphiti:
                raise AdapterError("Graphiti not initialized")
            
            # Extract backend data (type-safe)
            if not isinstance(memo.backend, GraphitiBackendData):
                raise AdapterError(f"Expected GraphitiBackendData, got {type(memo.backend).__name__}")
            
            backend_data: GraphitiBackendData = memo.backend
            
            # Delete old episode first
            await self.delete(memo)
            
            # Reconstruct AddMemoRequest from document + memo context
            from ..models import AddMemoRequest
            
            # Extract Kive context from original memo metadata
            request = AddMemoRequest(
                text=document.text,
                # Restore context from memo metadata (with fallback to defaults)
                app_id=memo.metadata.get("_kive_app_id", "default"),
                user_id=memo.metadata.get("_kive_user_id", "default"),
                namespace=memo.metadata.get("_kive_namespace", "default"),
                ai_id=memo.metadata.get("_kive_ai_id", "default"),
                tenant_id=memo.metadata.get("_kive_tenant_id", "default"),
                session_id=memo.metadata.get("_kive_session_id", "default"),
                # Merge document metadata with original metadata (document takes priority)
                metadata={**memo.metadata, **(document.metadata or {})},
            )
            
            # Add new episode
            new_memos = await self.add(request)
            
            if not new_memos:
                raise AdapterError("Failed to create new episode after update")
            
            new_memo = new_memos[0]
            logger.info(f"Updated memo: old_id={memo.id}, new_id={new_memo.id}")
            return new_memo
            
        except Exception as e:
            logger.error(f"Failed to update memo {memo.id}: {e}")
            raise AdapterError(f"Failed to update memo: {e}")
    
    async def delete(self, memos: Union[Memo, List[Memo]], **kwargs) -> bool:
        """Delete memories (delete episodes)"""
        if not self._graphiti:
            raise AdapterError("Graphiti not initialized")
        
        # Normalize to list
        memo_list = [memos] if isinstance(memos, Memo) else memos
        
        from graphiti_core.nodes import EpisodicNode
        
        success_count = 0
        failed_count = 0
        
        for memo in memo_list:
            try:
                # Extract backend data (type-safe)
                if not isinstance(memo.backend, GraphitiBackendData):
                    logger.error(f"Expected GraphitiBackendData for memo {memo.id}, got {type(memo.backend).__name__}, skipping")
                    failed_count += 1
                    continue
                
                backend_data: GraphitiBackendData = memo.backend
                
                # Get episode and delete
                episode = await EpisodicNode.get_by_uuid(
                    self._graphiti.driver,
                    backend_data.episode_id
                )
                
                if episode:
                    await episode.delete(self._graphiti.driver)
                    logger.info(f"Deleted episode {memo.id}")
                    success_count += 1
                else:
                    logger.warning(f"Episode not found: {memo.id}")
                    failed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to delete episode {memo.id}: {e}")
                failed_count += 1
        
        logger.info(f"Delete completed: success={success_count}, failed={failed_count}")
        return success_count > 0
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """Graphiti processes in real-time, no batch processing needed"""
        return {
            "status": "not_supported",
            "message": "Graphiti processes episodes in real-time, no batch processing needed"
        }
    
    async def close(self) -> None:
        """Close Graphiti connection"""
        if self._graphiti:
            await self._graphiti.close()
            logger.info("Graphiti connection closed")
        await super().close()
