# Cognee Supported Providers

Based on Cognee source code analysis, here are the supported providers:

## 1. LLM Providers

### Native Support
- **openai** - OpenAI models (GPT-4, GPT-3.5, etc.)
- **anthropic** - Anthropic Claude models
- **gemini** - Google Gemini models
- **mistral** - Mistral AI models
- **ollama** - Local Ollama models
- **custom** - Custom LLM endpoints (Generic OpenAI-compatible API)

### Configuration
```python
llm_provider = "openai"  # or anthropic, gemini, mistral, ollama, custom
llm_model = "gpt-4"
llm_api_key = "your-api-key"
llm_endpoint = ""  # Optional, for custom endpoints
```

## 2. Embedding Providers

### Native Support
- **openai** - OpenAI embeddings (text-embedding-3-large, ada-002, etc.)
- **fastembed** - FastEmbed (local, no API key needed)
- **ollama** - Local Ollama embeddings
- **LiteLLM** - Uses LiteLLM library (supports 100+ providers)

### Configuration
```python
embedding_provider = "openai"  # or fastembed, ollama
embedding_model = "text-embedding-3-large"
embedding_dimensions = 3072
embedding_api_key = "your-api-key"  # Optional for fastembed
```

## 3. Vector Database Providers

### Native Support
- **lancedb** - LanceDB (default, local-first)
- **chromadb** - ChromaDB
- **pgvector** - PostgreSQL with pgvector extension
- **qdrant** - Qdrant (inferred from config, not confirmed)

### Configuration
```python
vector_db_provider = "lancedb"  # or chromadb, pgvector
vector_db_url = "./data/lancedb"  # Local path or remote URL
```

## 4. Graph Database Providers

### Native Support
- **kuzu** - KuzuDB (default, local-first, embedded)
- **neo4j** - Neo4j graph database

### Configuration
```python
graph_database_provider = "kuzu"  # or neo4j
graph_database_url = ""  # For Neo4j
graph_database_username = ""
graph_database_password = ""
```

---

## How to Support Chinese Services (百炼, etc.)

### Option 1: Use Custom LLM Provider (Recommended)

Cognee supports **custom** provider which accepts any OpenAI-compatible API:

```python
adapter = CogneeAdapter(
    # Use custom provider
    llm_provider="custom",
    llm_model="qwen-plus",  # Your model name
    llm_api_key="your-bailain-api-key",
    llm_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 百炼 compatible endpoint
    
    # Embedding - use fastembed (local, no API needed)
    # embedding_provider will be handled separately
)
```

### Option 2: Extend Cognee Adapters

Since Cognee uses LiteLLM, we can create custom adapters:

#### Step 1: Create BaiLianAdapter

```python
# kive/server/backends/extensions/bailain_llm.py
"""
Bailain (百炼) LLM Adapter for Cognee

This extends Cognee to support Alibaba Cloud Bailain service
"""

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import GenericAPIAdapter

class BaiLianLLMAdapter(GenericAPIAdapter):
    """Adapter for Alibaba Cloud Bailain (百炼) service"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus", **kwargs):
        super().__init__(
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model=model,
            name="BaiLian",
            **kwargs
        )
```

#### Step 2: Create BaiLianEmbeddingEngine

```python
# kive/server/backends/extensions/bailain_embedding.py
"""
Bailain Embedding Engine for Cognee
"""

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

class BaiLianEmbeddingEngine(LiteLLMEmbeddingEngine):
    """Embedding engine for Bailain text-embedding-v2"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            provider="openai",  # Use OpenAI-compatible mode
            model="text-embedding-v2",
            api_key=api_key,
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            dimensions=1536,
            **kwargs
        )
```

#### Step 3: Extend CogneeAdapter

```python
# kive/server/backends/cognee_adapter.py

async def initialize(self) -> None:
    """Initialize cognee connection"""
    
    # Support Bailain
    if self.llm_provider == "bailain":
        # Set as custom with Bailain endpoint
        cognee.config.set_llm_provider("custom")
        cognee.config.set_llm_endpoint(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
```

### Option 3: Use Ollama with Local Models

For completely local solution:

```python
adapter = CogneeAdapter(
    llm_provider="ollama",
    llm_model="qwen2",  # or any model you pull in Ollama
    llm_endpoint="http://localhost:11434",
    llm_api_key="ollama",  # Dummy key for Ollama
    
    embedding_provider="fastembed",  # Local embeddings
    vector_db_provider="lancedb",  # Local vector DB
    graph_database_provider="kuzu",  # Local graph DB
)
```

---

## Recommended Setup for Chinese Environment

### Best Practice: Hybrid Approach

```python
from kive.server.backends import CogneeAdapter

adapter = CogneeAdapter(
    # Auto-process config
    auto_process=True,
    process_interval=30,
    process_batch_size=50,
    
    # LLM: Use Bailain via custom provider
    llm_provider="custom",
    llm_model="qwen-plus",
    llm_api_key="your-bailain-api-key",
    llm_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
    
    # Embedding: Use FastEmbed (local, no API needed)
    # This will be set via environment variables or separate config
    # EMBEDDING_PROVIDER=fastembed
    # EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
    
    # Vector DB: LanceDB (local, fast)
    vector_db_provider="lancedb",
    vector_db_url="./data/lancedb",
    
    # Graph DB: Kuzu (local, embedded)
    graph_database_provider="kuzu",
)
```

### Environment Variables (.env)

```bash
# LLM (handled by adapter params)
LLM_PROVIDER=custom
LLM_MODEL=qwen-plus
LLM_API_KEY=your-bailain-api-key
LLM_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1

# Embedding (use local FastEmbed)
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIMENSIONS=512

# Vector DB
VECTOR_DB_PROVIDER=lancedb
VECTOR_DB_URL=./data/lancedb

# Graph DB
GRAPH_DATABASE_PROVIDER=kuzu
```

---

## Next Steps

1. **Test with Bailain**: Verify OpenAI-compatible endpoint works
2. **Benchmark FastEmbed**: Test Chinese text embedding quality
3. **Optional Extension**: If custom provider doesn't work, implement BaiLianAdapter
4. **Documentation**: Add Bailain setup guide to Kive README

## References

- Cognee LLM Config: `cognee/infrastructure/llm/config.py`
- Embedding Engines: `cognee/infrastructure/databases/vector/embeddings/`
- LiteLLM Docs: https://docs.litellm.ai/docs/providers
