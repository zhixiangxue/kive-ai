<div align="center">

<a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a>

[![PyPI version](https://badge.fury.io/py/kive.svg)](https://badge.fury.io/py/kive)
[![Python Version](https://img.shields.io/pypi/pyversions/kive)](https://pypi.org/project/kive/)
[![License](https://img.shields.io/github/license/zhixiangxue/kive-ai)](https://github.com/zhixiangxue/kive-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/kive)](https://pypi.org/project/kive/)
[![GitHub Stars](https://img.shields.io/github/stars/zhixiangxue/kive-ai?style=social)](https://github.com/zhixiangxue/kive-ai)

[English](README.md) | [中文](docs/README_CN.md)

**一个为 AI 应用提供的统一记忆，支持可插拔的记忆后端。**

Kive 本身不是一个记忆引擎，而是一个通用适配器，让你无需更改应用代码就能在不同的记忆后端之间切换。

</div>

---

## 核心特性

### 🌱统一接入记忆

一套参数配置任何记忆后端，无需为每个后端学习不同的初始化模式：

```python
from kive import Memory, engines

# 同一套参数适用于所有引擎
engine = engines.Mem0(  # 或者 engines.Cognee / engines.Graphiti
    # LLM 配置（用于知识提取）
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY",
    llm_base_url="https://api.openai.com/v1",
    
    # 嵌入模型配置（用于向量搜索）
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    embedding_api_key="YOUR_KEY",
    embedding_base_url="https://api.openai.com/v1",
    embedding_dimensions=1536,

    # 向量数据库配置
    vector_db_provider="chroma",
    vector_db_uri=None,  # 嵌入式模式，将使用 .kive/chroma
    
    # 图数据库配置（可选）
    graph_db_provider="kuzu",
    graph_db_uri=".kive/memory.kuzu",
    
    # 多租户默认设置
    default_user_id="kive_user",
)

# 就这样！现在你可以使用这个记忆了
memory = Memory(engine=engine)
```

### 🪴统一操作记忆

所有记忆操作使用同一套 API，无论你使用 Cognee、Graphiti 还是 Mem0 - API 都保持简洁明了：

```python
# 所有引擎使用相同的 CRUD 语法
memo  = await memory.add(text="Python 是一种编程语言")
memos = await memory.search("Python 是什么？", limit=10)
memo  = await memory.get(memo_id="uuid-here")
memo  = await memory.update(memo, text="更新后的内容")
await memory.delete(memo)
```

### 🌻可选的 HTTP 网关

需要从不同语言调用吗？启动一个本地记忆网关：

```python
from kive.server import Server

# 启动一次，随处使用
server = Server(engine=engine, port=12306)
server.run()
```

然后通过 HTTP 从任何语言调用：

```bash
curl -X POST http://localhost:12306/add \
  -H "Content-Type: application/json" \
  -d '{"text": "Python 很棒"}'
```

---

## 支持的记忆引擎 (3)

| 引擎 | GitHub | 最佳适用场景 | 关键特性 |
|--------|--------|----------|-------------|
| **Mem0** | https://github.com/mem0ai/mem0 | RAG 聊天机器人、快速查询 | 快速向量搜索、实时处理、可选图功能 |
| **Cognee** | https://github.com/topoteretes/cognee | 知识库、文档问答 | 深度知识图谱、批处理、复杂推理 |
| **Graphiti** | https://github.com/getzep/graphiti | 对话式 AI、个人助理 | 时序感知、情景记忆、时间感知的事实 |

---

## 快速开始

### 安装

```bash
# 基础安装
pip install kive

# 安装特定引擎
pip install kive[mem0]     # mem0
pip install kive[cognee]   # cognee
pip install kive[graphiti] # graphiti

# 安装所有引擎
pip install kive[all]
```

### 基本用法

在你的代码中直接使用记忆引擎：

```python
import asyncio
from kive import Memory, engines

# 1. 选择并配置一个引擎
engine = engines.Mem0(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
)

# 2. 创建记忆实例
memory = Memory(engine=engine)

# 3. 使用它！
await memory.add(text="Python 是一种编程语言")
results = await memory.search("Python 是什么？")
for memo in results:
    print(memo.text, memo.score)
```

**查看完整示例：**
- \- [Mem0 example](examples/memory_with_mem0.py) - Fast vector search

  \- [Cognee example](examples/memory_with_cognee.py) - Knowledge graph

  \- [Graphiti example](examples/memory_with_graphiti.py) - Temporal graph

---

## 切换记忆后端

三个支持的引擎各有优势：

- **Mem0**：快速向量搜索、实时查询、可选图功能
- **Cognee**：深度知识图谱、复杂关系、批处理
- **Graphiti**：时序知识图谱、时间感知的情景记忆

切换记忆引擎非常简单：

```python
from kive import Memory, engines

# 使用 Mem0 进行快速搜索
engine = engines.Mem0(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

# 切换到 Cognee 用于知识图谱
engine = engines.Cognee(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

# 或者使用 Graphiti 获得时序感知能力
engine = engines.Graphiti(
    llm_provider="openai",
    llm_model="gpt-4",
    llm_api_key="YOUR_KEY"
)

memory = Memory(engine=engine)
# 所有引擎使用相同的 API！
```

**查看完整示例：**
- \- [Mem0 example](examples/memory_with_mem0.py) - Fast vector search

  \- [Cognee example](examples/memory_with_cognee.py) - Knowledge graph

  \- [Graphiti example](examples/memory_with_graphiti.py) - Temporal graph

---

## 统一操作

所有引擎都支持相同的操作，并提供全面的多租户和上下文隔离功能：

### 核心 API 方法

```python
from kive import Memory, engines

# 创建记忆实例
engine = engines.Mem0(llm_provider="openai", llm_api_key="YOUR_KEY")
memory = Memory(engine=engine)

# 添加单条记忆
await memory.add(text="需要记住的知识")

# 语义搜索
results = await memory.search("查询", limit=10)

# 根据 ID 获取
memo = await memory.get(memo_id="uuid-here")

# 更新
await memory.update(memo, text="更新后的内容")

# 删除
await memory.delete(memo)

# 处理/认知化（如果支持）
await memory.process()
```

### 内容输入类型

Kive 支持多种输入格式来添加记忆，让你在内容处理方式上有更大的灵活性：

```python
# 文本内容（最常用）
await memory.add(
    text="Python 是一种强大的编程语言",
    user_id="用户_123"
)

# 文件内容（PDF、DOCX、TXT 等）
await memory.add(
    file="/path/to/document.pdf",
    user_id="用户_123"
)

# 网页内容（自动获取和提取）
await memory.add(
    url="https://example.com/article",
    user_id="用户_123"
)

# 对话消息（聊天历史）
await memory.add(
    messages=[
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "今天晴朗，气温25°C。"}
    ],
    user_id="用户_123"
)

# 带附加元数据
await memory.add(
    text="重要会议记录",
    metadata={
        "category": "工作",
        "priority": "高",
        "tags": ["会议", "项目alpha"],
        "created_by": "用户_123"
    },
    user_id="用户_123"
)
```

#### 输入格式详解

- **`text`**: 纯文本内容，直接存储到记忆中
- **`file`**: 本地文件路径 - 支持 PDF、DOCX、TXT、MD 等常见格式
- **`url`**: 网页链接 - 自动获取并提取网页内容
- **`messages`**: OpenAI 聊天格式的对话历史 - 保留对话上下文
- **`metadata`**: 附加结构化数据 - 标签、分类、时间戳等

### 上下文与多租户参数

Kive 通过分层 ID 参数提供全面的上下文隔离。这些参数帮助你在不同范围组织记忆，并确保多用户、多应用场景下的数据正确隔离。

#### 参数层次结构（从最广泛到最具体）

```python
# 所有 add/search 操作都支持这些上下文参数：
await memory.add(
    text="你的内容在这里",
    
    # 基础设施与组织级（可选）
    tenant_id="acme_corp",      #   组织/公司级 B2B SaaS 隔离
                                #   • 代表整个客户/组织
                                #   • 确保企业间的数据完全分离
                                #   • 可选：单租户应用使用 "default"
    
    # 应用级（可选）
    app_id="健康助手_v2",        #   具体应用或产品标识符
                                #   • 区分不同的 AI 产品
                                #   • 防止多产品平台中的跨应用数据泄露
                                #   • 例如："健康助手" vs "财务助手" vs "聊天机器人"
                                #   • 建议：生产应用总是设置
    
    # AI 代理级（可选）
    ai_id="健康教练",            #   AI 代理或角色标识符
                                #   • 区分不同的 AI 个性/角色
                                #   • 对用户+AI 协作记忆很重要
                                #   • 例如："客服" vs "健康教练" vs "导师"
                                #   • 单 AI 系统使用 "default"
    
    # 群组/项目级（可选）
    namespace="家庭_2024",      #   共享记忆空间标识符
                                #   • 最灵活的隔离级别
                                #   • 可代表：项目ID、工作空间、团队、家庭、班级
                                #   • 个人记忆：namespace = user_id
                                #   • 共享记忆：namespace = "team_123"（多用户访问）
                                #   • 推荐作为群组上下文的统一抽象
    
    # 用户级（必需）
    user_id="用户_10086",       #   最终用户标识符（关键）
                                #   • 个人记忆的最终所有者
                                #   • 几乎所有系统都必需
                                #   • 在共享上下文中：作为贡献者出现
    
    # 会话级（可选）
    session_id="聊天_abc123",   #   对话/会话标识符
                                #   • 代表当前交互会话
                                #   • 将短期记忆绑定到特定对话
                                #   • 用于审计、调试和临时上下文
                                #   • 长期操作可为 None，但建议保留
)
```

#### 实际使用模式

```python
# 个人助手（单用户，单应用）
await memory.add(
    text="用户喜欢早晨开会",
    user_id="用户_123",
    namespace="用户_123",  # 个人命名空间 = 用户ID
    app_id="个人助手"
)

# 团队项目记忆（共享工作空间）
await memory.add(
    text="项目截止日期是3月15日",
    user_id="用户_123",        # 贡献者
    namespace="项目_alpha",    # 共享团队命名空间
    app_id="项目管理器",
    tenant_id="acme_corp"
)

# 多产品平台（不同 AI 服务）
# 健康机器人记忆
await memory.add(
    text="用户有糖尿病，监测血糖",
    user_id="用户_123",
    namespace="用户_123",
    app_id="健康机器人",
    ai_id="健康教练"
)

# 财务机器人记忆（同一用户，不同应用 - 隔离！）
await memory.add(
    text="用户每月投资预算5000元", 
    user_id="用户_123",
    namespace="用户_123",
    app_id="财务机器人",  # 不同应用 = 独立记忆空间
    ai_id="财务顾问"
)
```

#### 带上下文的搜索

所有上下文参数都可在搜索时使用，以查询特定的记忆范围：

```python
# 仅搜索用户的个人记忆
个人记忆 = await memory.search(
    query="健康偏好",
    user_id="用户_123",
    namespace="用户_123"
)

# 搜索团队项目记忆
团队记忆 = await memory.search(
    query="项目截止日期",
    namespace="项目_alpha"
)

# 跨整个组织搜索（管理用途）
组织记忆 = await memory.search(
    query="公司政策",
    tenant_id="acme_corp"
)
```

#### 数据隔离保证

- **tenant_id**：完整的企业级数据分离
- **app_id**：防止跨应用数据泄露  
- **namespace**：控制记忆共享范围（个人 vs 团队）
- **user_id**：个人记忆所有权和访问控制
- **ai_id**：基于角色的记忆差异化
- **session_id**：临时对话绑定

#### 最佳实践

1. **始终设置 `user_id`** - 个人记忆所有权所必需
2. **对共享上下文使用 `namespace`** - 比 project_id/space_id 更直观
3. **为多产品平台设置 `app_id`** - 防止意外数据共享
4. **为 B2B SaaS 考虑 `tenant_id`** - 企业客户必需
5. **为多代理系统使用 `ai_id`** - 区分 AI 角色和视角

---

## 可选：HTTP 网关

需要从不同语言调用吗？启动一个本地网关：

```python
from kive.server import Server
from kive import engines

# 启动服务器
engine = engines.Mem0(llm_provider="openai", llm_api_key="YOUR_KEY")
server = Server(engine=engine, port=12306)
server.run()
```

然后使用 HTTP 客户端：

```python
from kive.client import Client

client = Client("http://localhost:12306")
await client.add(text="需要记住的知识")
results = await client.search("查询")
```

**查看服务器示例：**

- [Server quickstart](examples/server_quickstart.py)
- [Client usage](examples/client_crud.py)

---

## Kive 适合你吗？

如果你：
- 需要与多个记忆引擎协同工作
- 希望跨后端使用统一、简单的 API
- 希望无需更改代码就能切换记忆策略
- 希望专注于构建 AI 应用，而不是费心处理记忆复杂性

那么 Kive 就是为你打造的。

<div align="right"><a href="#"><img src="https://raw.githubusercontent.com/zhixiangxue/kive-ai/main/docs/assets/logo.png" alt="Kive Logo" width="120"></a></div>