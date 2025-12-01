"""Cognee backend adapter

Integrates cognee memory engine
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from llama_index.core.schema import Document

from ...exceptions import AdapterError, ConnectionError, SearchError
from ...models import BackendType, CogneeBackendData, Memo
from .base import BaseMemoryAdapter
from ...utils.logger import logger
from .llm_bridge import LLMConfigBridge, UnifiedLLMConfig, LLMProvider, LLMProviderType


class CogneeAdapter(BaseMemoryAdapter):
    """Cognee memory engine adapter
    
    Supports full cognee functionality including add -> cognify -> search workflow
    """
    
    def __init__(
        self,
        auto_process: bool = False,
        process_interval: int = 30,
        process_batch_size: int = 100,
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
        huggingface_tokenizer: Optional[str] = None,  # For ollama provider
        # Vector DB configuration
        vector_db_provider: Optional[str] = None,
        vector_db_uri: Optional[str] = None,
        vector_db_key: Optional[str] = None,
        # Graph DB configuration
        graph_db_provider: Optional[str] = None,
        graph_db_uri: Optional[str] = None,
    ):
        """
        Args:
            auto_process: Whether to automatically execute cognify
            process_interval: Auto cognify interval in seconds
            process_batch_size: Trigger cognify immediately when reaching this count
            
            llm_provider: LLM provider (openai/anthropic/gemini/ollama/mistral/custom)
            llm_model: LLM model name
            llm_api_key: LLM API key
            llm_base_url: LLM API base URL (for custom provider, e.g., Bailain)
            
            embedding_provider: Embedding provider (ollama/custom)
            embedding_model: Embedding model name
                - Ollama: e.g., nomic-embed-text:latest
                - Custom: e.g., provider/your-embedding-model
            embedding_api_key: Embedding API key (required for custom provider)
            embedding_base_url: Embedding API base URL
                - Ollama: e.g., http://localhost:11434/api/embed
                - Custom: e.g., https://your-endpoint.example.com/v1
            embedding_dimensions: Embedding dimensions (required for custom provider)
            huggingface_tokenizer: HuggingFace Tokenizer model (required for ollama)
                - Example: nomic-ai/nomic-embed-text-v1.5
            
            vector_db_provider: Vector database provider (chromadb/lancedb)
            vector_db_uri: Vector database connection URI
                - ChromaDB: HTTP service address, e.g., http://localhost:8000
                - LanceDB: File path, e.g., /path/to/cognee.lancedb (optional, defaults to system directory)
            vector_db_key: Vector database authentication key
                - ChromaDB: Requires authentication token
                - LanceDB: Not required
            
            graph_db_provider: Graph database provider (kuzu/neo4j/networkx)
            graph_db_uri: Graph database connection URI
        """
        super().__init__(
            auto_process=auto_process,
            process_interval=process_interval,
            process_batch_size=process_batch_size,
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
            vector_db_key=vector_db_key,
            graph_db_provider=graph_db_provider,
            graph_db_uri=graph_db_uri,
        )
        # Store Cognee-specific configuration
        self.huggingface_tokenizer = huggingface_tokenizer
        
        self._cognee = None
    
    async def initialize(self) -> None:
        """Initialize cognee connection"""
        try:
            import cognee
            from pathlib import Path
            import os
            
            self._cognee = cognee
            
            # Disable telemetry to avoid network issues
            os.environ["TELEMETRY_DISABLED"] = "1"
            logger.info("Cognee telemetry disabled")
            
            # Set data directories
            project_root = Path.cwd()
            data_dir = project_root / ".kive/data"
            system_dir = project_root / ".kive/system"
            
            data_dir.mkdir(parents=True, exist_ok=True)
            system_dir.mkdir(parents=True, exist_ok=True)
            
            cognee.config.data_root_directory(str(data_dir))
            cognee.config.system_root_directory(str(system_dir))
            
            logger.info(f"Data directory: {data_dir}")
            logger.info(f"System directory: {system_dir}")
            
            # Configure using cognee.config API
            # Note: Cognee's embedding configuration is read from environment variables, must be set before importing cognee
            
            # 1. Configure Embedding (via environment variables)
            if self.embedding_provider:
                # Ollama: Local embedding service
                if self.embedding_provider == "ollama":
                    if not self.embedding_model:
                        raise ValueError(
                            "Ollama embedding requires embedding_model (e.g., nomic-embed-text:latest). "
                            "Please provide embedding_model parameter."
                        )
                    if not self.huggingface_tokenizer:
                        raise ValueError(
                            "Ollama embedding requires huggingface_tokenizer (e.g., nomic-ai/nomic-embed-text-v1.5). "
                            "Please provide huggingface_tokenizer parameter."
                        )
                    
                    os.environ["EMBEDDING_PROVIDER"] = self.embedding_provider
                    os.environ["EMBEDDING_MODEL"] = self.embedding_model
                    os.environ["HUGGINGFACE_TOKENIZER"] = self.huggingface_tokenizer
                    
                    if self.embedding_base_url:
                        os.environ["EMBEDDING_ENDPOINT"] = self.embedding_base_url
                    
                    if self.embedding_dimensions:
                        os.environ["EMBEDDING_DIMENSIONS"] = str(self.embedding_dimensions)
                    
                    logger.info(f"Embedding config (ollama): model={self.embedding_model}, tokenizer={self.huggingface_tokenizer}")
                
                # Custom: OpenAI-compatible embedding interface
                elif self.embedding_provider == "custom":
                    if not self.embedding_model:
                        raise ValueError(
                            "Custom embedding requires embedding_model (e.g., provider/your-model). "
                            "Please provide embedding_model parameter."
                        )
                    if not self.embedding_endpoint:
                        raise ValueError(
                            "Custom embedding requires embedding_endpoint. "
                            "Please provide embedding_endpoint parameter."
                        )
                    if not self.embedding_dimensions:
                        raise ValueError(
                            "Custom embedding requires embedding_dimensions. "
                            "Please provide embedding_dimensions parameter."
                        )
                    
                    os.environ["EMBEDDING_PROVIDER"] = self.embedding_provider
                    os.environ["EMBEDDING_MODEL"] = self.embedding_model
                    os.environ["EMBEDDING_ENDPOINT"] = self.embedding_base_url
                    os.environ["EMBEDDING_DIMENSIONS"] = str(self.embedding_dimensions)
                    
                    if self.embedding_api_key:
                        os.environ["EMBEDDING_API_KEY"] = self.embedding_api_key
                    
                    logger.info(f"Embedding config (custom): model={self.embedding_model}, base_url={self.embedding_base_url}")
                
                # Currently only supports Ollama and Custom
                else:
                    raise ValueError(
                        f"Unsupported embedding_provider: {self.embedding_provider}. "
                        "Currently only 'ollama' and 'custom' are supported."
                    )
            
            # 2. Configure LLM using bridge
            if self.llm_api_key or self.llm_provider or self.llm_model:
                unified_llm_config = UnifiedLLMConfig(
                    provider=LLMProvider(self.llm_provider) if self.llm_provider else LLMProvider.OPENAI,
                    model=self.llm_model or "gpt-4o-mini",
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                )
                
                bridge = LLMConfigBridge()
                llm_config = bridge.to_cognee(unified_llm_config)
                
                cognee.config.set_llm_config(llm_config)
                logger.info(f"LLM config set: {llm_config}")
            
            # 3. Configure Vector DB
            if self.vector_db_provider:
                vector_config = {
                    "vector_db_provider": self.vector_db_provider
                }
                
                # ChromaDB: Requires HTTP URL and authentication token
                if self.vector_db_provider == "chromadb":
                    if not self.vector_db_uri:
                        raise ValueError(
                            "ChromaDB requires vector_db_uri (e.g., http://localhost:8000). "
                            "Please provide vector_db_uri parameter."
                        )
                    
                    vector_config["vector_db_url"] = self.vector_db_uri
                    
                    # vector_db_key can be empty string (compatible with unauthenticated ChromaDB)
                    if self.vector_db_key:
                        vector_config["vector_db_key"] = self.vector_db_key
                    else:
                        vector_config["vector_db_key"] = ""
                        logger.warning("ChromaDB vector_db_key not provided, using empty string")
                
                # LanceDB: File-based database, URL is optional file path
                elif self.vector_db_provider == "lancedb":
                    if self.vector_db_uri:
                        # Ensure path is absolute
                        from pathlib import Path
                        lancedb_path = Path(self.vector_db_uri)
                        if not lancedb_path.is_absolute():
                            lancedb_path = Path.cwd() / lancedb_path
                        vector_config["vector_db_url"] = str(lancedb_path)
                    # If URI not provided, LanceDB will use default system directory, no need to set
                
                # Currently only supports ChromaDB and LanceDB
                else:
                    raise ValueError(
                        f"Unsupported vector_db_provider: {self.vector_db_provider}. "
                        "Currently only 'chromadb' and 'lancedb' are supported."
                    )
                
                cognee.config.set_vector_db_config(vector_config)
                logger.info(f"Vector DB config set: {vector_config}")
            
            # 4. Configure Graph DB
            if self.graph_db_provider:
                graph_config = {
                    "graph_database_provider": self.graph_db_provider
                }
                # Kuzu embedded doesn't need URI
                if self.graph_db_uri and self.graph_db_provider not in ["kuzu", "networkx"]:
                    graph_config["graph_database_url"] = self.graph_db_uri
                
                cognee.config.set_graph_db_config(graph_config)
                logger.info(f"Graph DB config set: {graph_config}")
            
            config_summary = {
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "llm_base_url": self.llm_base_url,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "vector_db_provider": self.vector_db_provider,
                "vector_db_uri": self.vector_db_uri,
                "vector_db_key": "***" if self.vector_db_key else None,
                "graph_db_provider": self.graph_db_provider,
            }
            logger.info(f"CogneeAdapter initialized with config: {config_summary}")
            
        except ImportError:
            raise ConnectionError(
                "cognee is not installed. "
                "Please install with: pip install kive[cognee]"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize cognee: {e}")
    
    def _create_memo(
        self,
        memo_id: str,
        text: str,
        data_id: str,
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
    ) -> Memo:
        """Create a Memo object with CogneeBackendData
        
        Args:
            memo_id: Memo ID
            text: Memory text content
            data_id: Cognee data UUID
            dataset_id: Cognee dataset UUID
            metadata: User-defined metadata
            score: Search similarity score (optional)
            
        Returns:
            Memo object with CogneeBackendData
        """
        return Memo(
            id=memo_id,
            text=text,
            metadata=metadata or {},
            backend=CogneeBackendData(
                type=BackendType.COGNEE,
                version="0.4.1",
                data_id=data_id,
                dataset_id=dataset_id,
            ),
            score=score,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    async def add(self, documents: List[Document], **kwargs) -> List[Memo]:
        """Add documents to cognee"""
        try:
            if not self._cognee:
                raise AdapterError("Cognee not initialized, call initialize() first")
            
            logger.info(f"Starting add operation for {len(documents)} documents...")
            
            # Cognee's add method accepts text, file paths, or binary streams
            data_to_add = []
            
            for doc in documents:
                # If Document has file_path metadata, use file path
                if "file_path" in doc.metadata:
                    data_to_add.append(doc.metadata["file_path"])
                else:
                    # Otherwise use text content
                    data_to_add.append(doc.text)
            
            logger.info(f"Prepared {len(data_to_add)} items to add")
            
            # Call cognee.add
            dataset_name = kwargs.get("dataset_name", "kive_dataset")
            result = await self._cognee.add(
                data=data_to_add,
                dataset_name=dataset_name,
            )
            
            logger.info(f"Cognee.add() completed, result: {result}")
            
            # Extract data_id and dataset_id from result, generate Memo list
            memos = []
            dataset_id = str(result.dataset_id) if hasattr(result, 'dataset_id') else None
            
            if hasattr(result, 'data_ingestion_info') and result.data_ingestion_info:
                for i, item in enumerate(result.data_ingestion_info):
                    if isinstance(item, dict) and 'data_id' in item:
                        data_id = str(item['data_id'])
                        doc = documents[i] if i < len(documents) else Document(text="")
                        
                        # Generate Memo with CogneeBackendData
                        memo = self._create_memo(
                            memo_id=data_id,
                            text=doc.text,
                            data_id=data_id,
                            dataset_id=dataset_id,
                            metadata=doc.metadata,
                        )
                        memos.append(memo)
                    else:
                        logger.warning(f"Unexpected item format in data_ingestion_info: {item}")
            
            if not memos:
                # If no data_ingestion_info, the add operation failed
                raise AdapterError("Failed to add documents: no data_id returned from Cognee")
            
            logger.info(f"Generated {len(memos)} memos")
            
            # Increment pending count
            self.increment_pending(len(documents))
            
            logger.info(f"Added {len(documents)} documents to cognee dataset '{dataset_name}'")
            return memos
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise AdapterError(f"Failed to add documents: {e}")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        query_type: str = "CHUNKS",
        **kwargs
    ) -> List[Memo]:
        """Search in cognee
        
        Args:
            query: Search query text
            limit: Maximum number of results
            query_type: Cognee search type (CHUNKS/GRAPH_COMPLETION/INSIGHTS/etc.)
            **kwargs: Additional parameters (datasets, etc.)
        """
        try:
            if not self._cognee:
                raise AdapterError("Cognee not initialized")
            
            # Import SearchType
            from cognee.modules.search.types import SearchType
            
            # Parse query_type
            try:
                search_type = SearchType[query_type]
            except KeyError:
                logger.warning(f"Invalid query_type '{query_type}', using CHUNKS as fallback")
                search_type = SearchType.CHUNKS
            
            # Call cognee.search
            search_results = await self._cognee.search(
                query_text=query,
                query_type=search_type,
                top_k=limit,
                datasets=kwargs.get("datasets"),
            )
            
            logger.info(f"Search returned {len(search_results) if search_results else 0} results")
            
            # Convert cognee search results to Memo list
            memos = []
            for i, result in enumerate(search_results[:limit] if search_results else []):
                # Cognee SearchResult format varies by query_type
                if isinstance(result, dict):
                    # Already a dict
                    result_id = result.get("id", f"search_result_{i}")
                    result_text = result.get("text", "")
                    result_score = result.get("score", 1.0 - (i * 0.05))
                    result_metadata = result.get("metadata", {})
                elif isinstance(result, str):
                    # GRAPH_COMPLETION returns text
                    result_id = f"search_result_{i}"
                    result_text = result
                    result_score = 1.0 - (i * 0.05)
                    result_metadata = {}
                else:
                    # SearchResult object
                    result_id = getattr(result, "id", f"search_result_{i}")
                    result_text = getattr(result, "text", str(result))
                    result_score = getattr(result, "score", 1.0 - (i * 0.05))
                    result_metadata = getattr(result, "metadata", {})
                
                # Create Memo with CogneeBackendData
                # Note: Search results may not have data_id/dataset_id
                memo = self._create_memo(
                    memo_id=result_id,
                    text=result_text,
                    data_id=result_id,  # Use result_id as fallback
                    dataset_id="unknown",  # Search results don't include dataset_id
                    metadata=result_metadata,
                    score=result_score,
                )
                memos.append(memo)
            
            logger.info(f"Search completed: query='{query}', results={len(memos)}")
            return memos
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def get(self, memo_id: str, **kwargs) -> Optional[Memo]:
        """Get single memory
        
        Note: Cognee does not support direct get-by-ID API.
        This method always returns None.
        """
        logger.warning(f"Cognee does not support get by ID: {memo_id}")
        return None
    
    async def update(
        self,
        memo: Memo,
        document: Document,
        **kwargs
    ) -> Memo:
        """Update memory
        
        Note: Cognee's update returns new data_id after delete+add+cognify.
        """
        try:
            if not self._cognee:
                raise AdapterError("Cognee not initialized")
            
            # Extract backend data (type-safe)
            if not isinstance(memo.backend, CogneeBackendData):
                raise AdapterError(f"Expected CogneeBackendData, got {type(memo.backend).__name__}")
            
            backend_data: CogneeBackendData = memo.backend
            
            # Convert to UUID
            from uuid import UUID
            try:
                data_id = UUID(backend_data.data_id)
                dataset_uuid = UUID(backend_data.dataset_id)
            except (ValueError, TypeError) as e:
                raise AdapterError(f"Invalid UUID in memo.backend: {e}")
            
            # Call cognee.update
            result = await self._cognee.update(
                data_id=data_id,
                data=document.text,
                dataset_id=dataset_uuid,
            )
            
            logger.info(f"Cognee update result: {result}")
            
            # Extract new data_id from result
            # result is a dict: {dataset_id: PipelineRunCompleted}
            if isinstance(result, dict):
                for dataset_uuid_key, pipeline_run in result.items():
                    dataset_id_str = str(dataset_uuid_key)
                    
                    if hasattr(pipeline_run, 'data_ingestion_info') and pipeline_run.data_ingestion_info:
                        # Get first data_id (should only be one since we updated one memo)
                        first_item = pipeline_run.data_ingestion_info[0]
                        if isinstance(first_item, dict) and 'data_id' in first_item:
                            new_data_id = str(first_item['data_id'])
                            
                            # Create new Memo with new backend ID
                            new_memo = self._create_memo(
                                memo_id=new_data_id,
                                text=document.text,
                                data_id=new_data_id,
                                dataset_id=dataset_id_str,
                                metadata=document.metadata,
                            )
                            
                            logger.info(f"Updated memo: old_id={memo.id}, new_id={new_memo.id}")
                            return new_memo
            
            raise AdapterError("Failed to extract new data_id from update result")
            
        except Exception as e:
            logger.error(f"Failed to update memo {memo.id}: {e}")
            raise AdapterError(f"Failed to update memo: {e}")
    
    async def delete(self, memos: Union[Memo, List[Memo]], **kwargs) -> bool:
        """Delete memories"""
        if not self._cognee:
            raise AdapterError("Cognee not initialized")
        
        # Normalize to list
        memo_list = [memos] if isinstance(memos, Memo) else memos
        
        from uuid import UUID
        
        success_count = 0
        failed_count = 0
        
        for memo in memo_list:
            try:
                # Extract backend data (type-safe)
                if not isinstance(memo.backend, CogneeBackendData):
                    logger.error(f"Expected CogneeBackendData for memo {memo.id}, got {type(memo.backend).__name__}, skipping")
                    failed_count += 1
                    continue
                
                backend_data: CogneeBackendData = memo.backend
                
                # Convert to UUID
                try:
                    data_id = UUID(backend_data.data_id)
                    dataset_uuid = UUID(backend_data.dataset_id)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid UUID in memo {memo.id}: {e}")
                    failed_count += 1
                    continue
                
                # Call cognee.delete
                await self._cognee.delete(
                    data_id=data_id,
                    dataset_id=dataset_uuid,
                )
                
                logger.info(f"Deleted memo {memo.id}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to delete memo {memo.id}: {e}")
                failed_count += 1
        
        logger.info(f"Delete completed: success={success_count}, failed={failed_count}")
        return success_count > 0  # At least one succeeded
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """Execute cognify processing
        
        This is Cognee's core processing workflow that transforms added data into knowledge graph
        """
        try:
            if not self._cognee:
                raise AdapterError("Cognee not initialized")
            
            logger.info("Starting cognify process...")
            
            # Call cognee.cognify()
            # Reference: examples/database_examples/chromadb_example.py
            # cognify([dataset_name]) - parameter is dataset name list
            datasets = kwargs.get("datasets")
            if datasets:
                result = await self._cognee.cognify(datasets)
            else:
                result = await self._cognee.cognify()
            
            logger.info(f"Cognify completed: {result}")
            
            return {
                "status": "success",
                "message": "Cognify completed successfully",
                "result": str(result)
            }
            
        except Exception as e:
            logger.error(f"Cognify failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
