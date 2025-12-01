# Kive Examples

This directory contains examples demonstrating how to use Kive for building memory-enabled AI applications.

## Quick Start

### 1. Server Example

**`server_quickstart.py`** - Start Kive server with different memory engines

```bash
python examples/server_quickstart.py
```

Features:
- **3 Memory Engines**: Cognee, Graphiti, Mem0
- **Unified Configuration**: Clean parameter management
- **Easy Switching**: Change `engine_choice` to try different engines

**Supported Engines:**

| Engine | Best For | Features |
|--------|----------|----------|
| **Cognee** | Complex knowledge relationships | Knowledge graph + semantic search |
| **Graphiti** | Time-aware episodic memory | Temporal knowledge graph + hybrid search |
| **Mem0** | Simple setup & fast queries | Vector search + optional graph storage |

### 2. Client Example

**`client_crud.py`** - Complete CRUD operations via HTTP API

```bash
# 1. Start server first
python examples/server_quickstart.py

# 2. Run client test (in another terminal)
python examples/client_crud.py
```

Test coverage:
- ✓ Health Check
- ✓ Add (single + batch)
- ✓ Process (cognify)
- ✓ Search
- ✓ Get
- ✓ Update
- ✓ Delete

## Configuration

### LLM Providers

Edit the configuration in `server_quickstart.py`:

```python
LLM_CONFIG = {
    "llm_provider": "bailian",  # Choose your provider
    "llm_model": "qwen-plus",
    "llm_api_key": "sk-***",    # Your API key
    "llm_base_url": "https://...",
}
```

**Supported providers:**
- Cloud: `openai`, `anthropic`, `bailian`, `moonshot`, `deepseek`, `zhipu`, `doubao`, `groq`
- Local: `ollama`, `lmstudio`, `vllm`

### Embedding Providers

```python
EMBEDDING_CONFIG = {
    "embedding_provider": "ollama",  # Local or cloud
    "embedding_model": "nomic-embed-text:latest",
    "embedding_base_url": "http://localhost:11434/api/embed",
    "embedding_dimensions": 768,
}
```

**Options:**
- **Ollama** (local, free): Fast, privacy-friendly
- **OpenAI/Bailian** (cloud): Higher quality, requires API key

### Database Configuration

```python
DB_CONFIG = {
    "vector_db_provider": "lancedb",  # File-based, no service needed
    "graph_db_provider": "kuzu",      # Embedded, no service needed
}
```

**Options:**
- **Vector DB**: `lancedb` (file) or `chroma` (HTTP service)
- **Graph DB**: `kuzu` (embedded) or `neo4j` (service)

## Installation

### Minimal (client only)
```bash
pip install kive[client]
```

### With specific engine
```bash
pip install kive[cognee]   # or [graphiti] or [mem0]
```

### All engines
```bash
pip install kive[all]
```

## Requirements

### For Local Embedding (Ollama)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

### For Cloud LLM
Get API key from:
- OpenAI: https://platform.openai.com/api-keys
- Bailian (阿里云): https://dashscope.aliyuncs.com/
- Anthropic: https://console.anthropic.com/
- Others: Check provider documentation

## Next Steps

1. **Choose your engine** based on use case
2. **Configure LLM provider** (add your API key)
3. **Start server** with `server_quickstart.py`
4. **Test CRUD** with `client_crud.py`
5. **Build your app** using the Kive client!

## Tips

- **Start simple**: Use Mem0 for prototyping
- **Go deep**: Use Cognee for knowledge-intensive tasks
- **Track time**: Use Graphiti for temporal awareness
- **Mix and match**: Different engines for different use cases

## Need Help?

- 📖 Documentation: (coming soon)
- 💬 GitHub Issues: https://github.com/yourusername/kive-ai/issues
- 🌟 Star us: https://github.com/yourusername/kive-ai
