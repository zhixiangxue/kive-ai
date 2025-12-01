<div align="center">

<a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a>

[![PyPI version](https://badge.fury.io/py/kive.svg)](https://badge.fury.io/py/kive)
[![Python Version](https://img.shields.io/pypi/pyversions/kive)](https://pypi.org/project/kive/)
[![License](https://img.shields.io/github/license/yourusername/kive-ai)](https://github.com/yourusername/kive-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/kive)](https://pypi.org/project/kive/)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/kive-ai?style=social)](https://github.com/yourusername/kive-ai)

[English](README.md) | [中文](docs/README_CN.md)

**A unified memory service for AI applications with pluggable backends.**

Kive is not a memory engine itself, but a universal adapter that lets you switch between different memory backends without changing your application code. Just focus on building your AI application, let Kive handle memory complexity.

</div>

---

## Core Features

### 🌱 Unified Memory API

No matter which backend you choose, the API stays the same:

```python
# Start server - choose your memory engine
from kive.server import Server
from kive.server.engines import Cognee

engine = Cognee(llm_provider="openai", llm_api_key="YOUR_KEY")
server = Server(engine=engine, port=12306)
server.run()
```

```python
# Connect with client - same API for all engines
from kive.client import Client

client = Client("http://localhost:12306")
await client.add(text="AI is fascinating")
results = await client.search("tell me about AI")
```

Start the service once, call from anywhere. Kive keeps things simple.

### 🪴 Pluggable Memory Backends

Kive supports multiple memory engines with different strengths:

```python
# Switch engine with one line
from kive.server.engines import Cognee, Graphiti, Mem0

# Deep knowledge relationships
engine = Cognee(llm_provider="openai", llm_model="gpt-4")

# Temporal awareness
engine = Graphiti(llm_provider="openai", llm_model="gpt-4")

# Fast vector search
engine = Mem0(llm_provider="openai", llm_model="gpt-4")
```

- **Now**: Three production-ready engines (Cognee, Graphiti, Mem0)
- **Planning**: More backends, expanding the ecosystem

No one else provides such a unified abstraction layer. Kive's adapter pattern makes memory backends fully pluggable and swappable.

### 🌻 Simple Integration

Use Kive your way - as library or service:

```python
# As library - direct integration
from kive.server.adapters import CogneeAdapter

adapter = CogneeAdapter(
    llm_provider="openai",
    llm_api_key="YOUR_KEY"
)
await adapter.add(text="Python is great")
results = await adapter.search("programming languages")

# As service - HTTP API
from kive.client import Client

client = Client("http://localhost:12306")
await client.add(text="Python is great")
results = await client.search("programming languages")
```

- **Now**: Both modes work seamlessly
- **Planning**: Streaming responses, multi-tenancy support

---

## Supported Memory Engines (3)

[Cognee](https://github.com/topoteretes/cognee), [Graphiti](https://github.com/getzep/graphiti), [Mem0](https://github.com/mem0ai/mem0)

---

## Quick Start

### Installation

```bash
# Basic installation (client only)
pip install kive[client]

# With specific engine
pip install kive[cognee]  # or [graphiti] or [mem0]

# Install all optional dependencies
pip install kive[all]
```

### Use memory engines in a few lines

```python
from kive.client import Client

client = Client("http://localhost:12306")

# Add data
await client.add(text="Python is a programming language")

# Search semantically
results = await client.search("what is Python?")
for memo in results.memos:
    print(memo.text, memo.score)
```

Kive handles: backend initialization, format conversion, vector embedding, graph building, semantic search... You just need to `add` and `search`.

---

## Switch Memory Backends

Three supported engines with different strengths:

- **Cognee**: Knowledge graph, deep reasoning, complex relationships
- **Graphiti**: Temporal knowledge graph, time-aware episodic memory
- **Mem0**: Fast vector search, optional graph, real-time queries

Quick start:

```python
from kive.server import Server
from kive.server.engines import Cognee

engine = Cognee(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

server = Server(engine=engine, port=12306)
server.run()
```

See full examples:

- **Server quickstart**: [examples/server_quickstart.py](examples/server_quickstart.py)
- **Client CRUD**: [examples/client_crud.py](examples/client_crud.py)
- **Configuration guide**: [examples/README.md](examples/README.md)

---

## Unified Operations

All engines support the same operations:

```python
from kive.client import Client

client = Client("http://localhost:12306")

# Add single or batch
await client.add(text="Knowledge to remember")
await client.add_batch([
    {"text": "First fact"},
    {"text": "Second fact"}
])

# Semantic search
results = await client.search("query", limit=10)

# Get by ID
memo = await client.get(memo_id="uuid-here")

# Update
await client.update(memo_id="uuid-here", text="Updated content")

# Delete
await client.delete(memo_id="uuid-here")

# Process/cognify (if supported)
await client.process()
```

---

## Memory Engine Comparison

Choose the right engine for your use case:

| Use Case | Recommended Engine | Why |
|----------|-------------------|-----|
| RAG chatbot | **Mem0** | Fast vector search, real-time queries |
| Knowledge base | **Cognee** | Deep relationships, knowledge extraction |
| Conversational AI | **Graphiti** | Temporal awareness, episodic memory |
| Document Q&A | **Cognee** | Semantic search, reasoning |
| Personal assistant | **Graphiti** | Time-aware memory |

---

## Is Kive for You?

If you:
- Need to work with multiple memory engines
- Want a unified, simple API across backends
- Want to switch memory strategies without code changes
- Want to focus on building AI applications, not wrestling with memory complexity

Then Kive is made for you.

<div align="right"><a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a></div>