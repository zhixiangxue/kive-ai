<div align="center">

<a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a>

[![PyPI version](https://badge.fury.io/py/kive.svg)](https://badge.fury.io/py/kive)
[![Python Version](https://img.shields.io/pypi/pyversions/kive)](https://pypi.org/project/kive/)
[![License](https://img.shields.io/github/license/zhixiangxue/kive-ai)](https://github.com/zhixiangxue/kive-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/kive)](https://pypi.org/project/kive/)
[![GitHub Stars](https://img.shields.io/github/stars/zhixiangxue/kive-ai?style=social)](https://github.com/zhixiangxue/kive-ai)

[English](README.md) | [中文](docs/README_CN.md)

**A unified memory for AI applications with pluggable backends.**

Kive is not a memory engine itself, but a universal adapter that lets you switch between different memory backends without changing your application code. 

</div>

---

## Core Features

### 🌱 Unified Initialization

One set of parameters to configure any memory backend, No need to learn different initialization patterns for each backend:

```python
from kive import Memory, engines

# Same parameters work across all engines
engine = engines.Mem0(  # or engines.Cognee / engines.Graphiti
    # LLM configuration (for knowledge extraction)
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY",
    llm_base_url="https://api.openai.com/v1",
    
    # Embedding configuration (for vector search)
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    embedding_api_key="YOUR_KEY",
    embedding_base_url="https://api.openai.com/v1",
    embedding_dimensions=1536,

    # Vector DB configuration
    vector_db_provider="chroma",
    vector_db_uri=None,  # will use .kive/chroma by default
    
    # Graph DB configuration (optional)
    graph_db_provider="kuzu",
    graph_db_uri=".kive/memory.kuzu",
    
    # Multi-tenancy defaults
    default_user_id="kive_user",
)

# That's it! Now you can use the memory
memory = Memory(engine=engine)
```



### 🪴 Unified CRUD Operations

One API for all memory operations, Whether you use Cognee, Graphiti, or Mem0 - the API stays clean and simple:

```python
# Same CRUD syntax across all engines
memo  = await memory.add(text="Python is a programming language")
memos = await memory.search("what is Python?", limit=10)
memo  = await memory.get(memo_id="uuid-here")
memo  = await memory.update(memo, text="Updated content")
await memory.delete(memo)
```

### 🌻 Optional HTTP Gateway

Need to call from different languages? Start a local memory gateway:

```python
from kive.server import Server

# Start once, use anywhere
server = Server(engine=engine, port=12306)
server.run()
```

Then call from any language via HTTP:

```bash
curl -X POST http://localhost:12306/add \
  -H "Content-Type: application/json" \
  -d '{"text": "Python is great"}'
```



---

## Supported Memory Engines (3)

| Engine | GitHub | Best For | Key Features |
|--------|--------|----------|-------------|
| **Mem0** | [![GitHub](https://img.shields.io/badge/GitHub-mem0ai/mem0-181717?logo=github)](https://github.com/mem0ai/mem0) | RAG chatbot, Fast queries | Fast vector search, Real-time processing, Optional graph |
| **Cognee** | [![GitHub](https://img.shields.io/badge/GitHub-topoteretes/cognee-181717?logo=github)](https://github.com/topoteretes/cognee) | Knowledge base, Document Q&A | Deep knowledge graph, Batch processing, Complex reasoning |
| **Graphiti** | [![GitHub](https://img.shields.io/badge/GitHub-getzep/graphiti-181717?logo=github)](https://github.com/getzep/graphiti) | Conversational AI, Personal assistant | Temporal awareness, Episodic memory, Time-aware facts |

---

## Quick Start

### Installation

```bash
# Basic installation
pip install kive

# With specific engine
pip install kive[mem0]     # Fast vector search
pip install kive[cognee]   # Knowledge graph
pip install kive[graphiti] # Temporal graph

# Install all engines
pip install kive[all]
```

### Basic Usage

Use memory engines directly in your code:

```python
import asyncio
from kive import Memory, engines

# 1. Choose and configure an engine
engine = engines.Mem0(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
)

# 2. Create memory instance
memory = Memory(engine=engine)

# 3. Use it!
await memory.add(text="Python is a programming language")
results = await memory.search("what is Python?")
for memo in results:
    print(memo.text, memo.score)
```

**See complete examples:**
- [Mem0 example](examples/memory_with_mem0.py)
- [Cognee example](examples/memory_with_cognee.py)
- [Graphiti example](examples/memory_with_graphiti.py)

---

## Switch Memory Backends

Three supported engines with different strengths:

- **Mem0**: Fast vector search, real-time queries, optional graph
- **Cognee**: Deep knowledge graph, complex relationships, batch processing
- **Graphiti**: Temporal knowledge graph, time-aware episodic memory

Switching is as simple as changing the engine:

```python
from kive import Memory, engines

# Use Mem0 for fast search
engine = engines.Mem0(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

# Switch to Cognee for knowledge graph
engine = engines.Cognee(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

# Or Graphiti for temporal awareness
engine = engines.Graphiti(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

memory = Memory(engine=engine)
# Same API for all!
```

**See full examples:**
- [Mem0 example](examples/memory_with_mem0.py) - Fast vector search
- [Cognee example](examples/memory_with_cognee.py) - Knowledge graph
- [Graphiti example](examples/memory_with_graphiti.py) - Temporal graph

---

## Unified Operations

All engines support the same operations with comprehensive multi-tenancy and context isolation:

### Core API Methods

```python
from kive import Memory, engines

# Create memory instance
engine = engines.Mem0(llm_provider="openai", llm_api_key="YOUR_KEY")
memory = Memory(engine=engine)

# Add single memo
await memory.add(text="Knowledge to remember")

# Semantic search
results = await memory.search("query", limit=10)

# Get by ID
memo = await memory.get(memo_id="uuid-here")

# Update
await memory.update(memo, text="Updated content")

# Delete
await memory.delete(memo)

# Process/cognify (if supported)
await memory.process()
```

### Content Input Types

Kive supports multiple input formats for adding memories, giving you flexibility in how content is processed:

```python
# Text content (most common)
await memory.add(
    text="Python is a powerful programming language",
    user_id="user_123"
)

# File content (PDF, DOCX, TXT, etc.)
await memory.add(
    file="/path/to/document.pdf",
    user_id="user_123"
)

# Web page content (automatically fetched and extracted)
await memory.add(
    url="https://example.com/article",
    user_id="user_123"
)

# Conversational messages (chat history)
await memory.add(
    messages=[
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "It's sunny and 25°C."}
    ],
    user_id="user_123"
)

# With additional metadata
await memory.add(
    text="Important meeting notes",
    metadata={
        "category": "work",
        "priority": "high",
        "tags": ["meeting", "project-alpha"],
        "created_by": "user_123"
    },
    user_id="user_123"
)
```

#### Input Format Details

- **`text`**: Plain text content for direct memory storage
- **`file`**: Local file path - supports PDF, DOCX, TXT, MD and other common formats  
- **`url`**: Web URL - automatically fetches and extracts content from web pages
- **`messages`**: Conversation history in OpenAI chat format - preserves dialogue context
- **`metadata`**: Additional structured data - tags, categories, timestamps, etc.

### Context & Multi-Tenancy Parameters

Kive provides comprehensive context isolation through hierarchical ID parameters. These help you organize memories at different scopes and ensure proper data isolation in multi-user, multi-app scenarios.

#### Parameter Hierarchy (from broadest to most specific)

```python
# All add/search operations support these context parameters:
await memory.add(
    text="Your content here",
    
    # Infrastructure & Organization Level(optional)
    tenant_id="acme_corp",      #   Organization/company for B2B SaaS isolation
                                #   • Represents an entire customer/organization
                                #   • Ensures complete data separation between enterprises
                                #   • Optional: Use "default" for single-tenant apps
    
    # Application Level (optional) 
    app_id="healthbot_v2",      #   Specific application or product identifier
                                #   • Distinguishes between different AI products
                                #   • Prevents cross-app data leakage in multi-product platforms
                                #   • Example: "healthbot" vs "financebot" vs "chatbot"
                                #   • Recommended: always set for production apps
    
    # AI Agent Level (optional)
    ai_id="wellness_coach",     #   AI agent or role identifier  
                                #   • Distinguishes different AI personalities/roles
                                #   • Important for user+AI collaborative memories
                                #   • Example: "customer_service" vs "health_coach" vs "tutor"
                                #   • Use "default" if single AI agent
    
    # Group/Project Level (optional)
    namespace="family_2024",    #   Shared memory space identifier
                                #   • Most flexible isolation level
                                #   • Can represent: project_id, workspace, team, family, class
                                #   • Personal memories: namespace = user_id  
                                #   • Shared memories: namespace = "team_123" (multi-user access)
                                #   • Recommended as unified abstraction for group contexts
    
    # User Level (essential)
    user_id="user_10086",       #   End-user identifier (CRITICAL)
                                #   • Final owner of personal memories
                                #   • Required for almost all systems
                                #   • In shared contexts: appears as contributor
    
    # Session Level (optional)
    session_id="chat_abc123",   #   Conversation/session identifier
                                #   • Represents current interaction session
                                #   • Binds short-term memories to specific conversations
                                #   • Useful for audit, debugging, and temporary context
                                #   • Can be None for long-term operations, but recommended
)
```

#### Practical Usage Patterns

```python
# Personal Assistant (single user, single app)
await memory.add(
    text="User prefers morning meetings",
    user_id="user_123",
    namespace="user_123",  # Personal namespace = user_id
    app_id="personal_assistant"
)

# Team Project Memory (shared workspace)
await memory.add(
    text="Project deadline is March 15th",
    user_id="user_123",        # Contributor
    namespace="project_alpha",  # Shared team namespace
    app_id="project_manager",
    tenant_id="acme_corp"
)

# Multi-Product Platform (different AI services)
# Health Bot memory
await memory.add(
    text="User has diabetes and monitors blood sugar",
    user_id="user_123",
    namespace="user_123",
    app_id="healthbot",
    ai_id="health_coach"
)

# Finance Bot memory (same user, different app - isolated!)
await memory.add(
    text="User has $5000 monthly investment budget", 
    user_id="user_123",
    namespace="user_123",
    app_id="financebot",  # Different app = separate memory space
    ai_id="financial_advisor"
)
```

#### Search with Context

All context parameters are available during search to query specific memory scopes:

```python
# Search user's personal memories only
personal_memos = await memory.search(
    query="health preferences",
    user_id="user_123",
    namespace="user_123"
)

# Search team project memories
team_memos = await memory.search(
    query="project deadlines",
    namespace="project_alpha"
)

# Search across entire organization (admin use)
org_memos = await memory.search(
    query="company policies",
    tenant_id="acme_corp"
)
```

#### Data Isolation Guarantees

- **tenant_id**: Complete enterprise-level data separation
- **app_id**: Prevents cross-application data leakage  
- **namespace**: Controls memory sharing scope (personal vs team)
- **user_id**: Personal memory ownership and access control
- **ai_id**: Role-based memory differentiation
- **session_id**: Temporary conversation binding

#### Best Practices

1. **Always set `user_id`** - Required for personal memory ownership
2. **Use `namespace` for shared contexts** - More intuitive than project_id/space_id
3. **Set `app_id` for multi-product platforms** - Prevents accidental data sharing
4. **Consider `tenant_id` for B2B SaaS** - Essential for enterprise customers
5. **Use `ai_id` for multi-agent systems** - Differentiates AI roles and perspectives

---

## Optional: HTTP Gateway

Need to call from different languages? Start a local gateway:

```python
from kive.server import Server
from kive import engines

# Start server
engine = engines.Mem0(llm_provider="openai", llm_api_key="YOUR_KEY")
server = Server(engine=engine, port=12306)
server.run()
```

Then use HTTP client:

```python
from kive.client import Client

client = Client("http://localhost:12306")
await client.add(text="Knowledge to remember")
results = await client.search("query")
```

**See server examples:**
- [Server quickstart](examples/server_quickstart.py)
- [Client usage](examples/client_crud.py)

---

## Is Kive for You?

If you:
- Need to work with multiple memory engines
- Want a unified, simple API across backends
- Want to switch memory strategies without code changes
- Want to focus on building AI applications, not wrestling with memory complexity

Then Kive is made for you.

<div align="right"><a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a></div>