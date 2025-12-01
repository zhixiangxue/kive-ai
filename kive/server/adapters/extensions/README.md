# Kive Adapter Extensions

本目录包含对第三方记忆库的扩展实现，用于支持国产 LLM 生态和补充缺失功能。

## 设计理念

这些扩展不仅仅是"补丁"（patches），更是：
- **生态适配** - 支持国产 LLM/Embedding 服务（百炼、千帆等）
- **功能补充** - 补充第三方库缺失或未完整实现的功能
- **Bug 修复** - 临时修复上游库的已知问题

## 目录结构

```
extensions/
├── graphiti/          # Graphiti 扩展
│   ├── llm_clients/   # LLM 客户端实现
│   ├── embedders/     # Embedding 实现（未来）
│   └── drivers/       # 驱动补充实现
└── cognee/            # Cognee 扩展（未来）
```

---

## Graphiti Extensions

### LLM Clients (`llm_clients/`)

#### `bailian.py` - 百炼 LLM 客户端

**用途**: 适配阿里云百炼（DashScope）API

**问题**:
- Graphiti 使用 OpenAI Structured Output (json_schema)
- 百炼 OpenAI 兼容 API 只支持 json_object 模式
- 百炼要求 prompt 中包含 "json" 关键词

**解决方案**:
- 重写 `_generate_response` 使用 json_object
- 将 JSON schema 注入到 prompt 中
- 添加显式的 JSON 格式说明

**状态**: 生产可用 ✅

**使用方式**:
```python
from kive.server.adapters.extensions.graphiti.llm_clients import BailianLLMClient

llm_client = BailianLLMClient(config=llm_config)
```

**上游问题**: N/A（API 限制，非 bug）

---

### Drivers (`drivers/`)

#### `kuzu_indices.py` - Kuzu 全文索引补充

**用途**: 创建 Kuzu driver 缺失的全文索引

**问题**:
- `KuzuDriver.build_indices_and_constraints()` 是空操作（no-op）
- 导致搜索时报错：`Table Entity doesn't have an index with name node_name_and_summary`

**根本原因**:
```python
# graphiti_core/driver/kuzu_driver.py line 143-147
async def build_indices_and_constraints(self, delete_existing: bool = False):
    # Kuzu doesn't support dynamic index creation like Neo4j or FalkorDB
    # Schema and indices are created during setup_schema()
    # This method is required by the abstract base class but is a no-op for Kuzu
    pass
```

但 `setup_schema()` 只创建表，不创建全文索引。

**解决方案**:
手动执行 FTS 索引创建（基于 `graphiti_core/graph_queries.py`）:
- `CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary'])`
- `CREATE_FTS_INDEX('Episodic', 'episode_content', ...)`
- `CREATE_FTS_INDEX('Community', 'community_name', ...)`
- `CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ...)`

**状态**: 临时方案 ⚠️

**使用方式**:
```python
from graphiti_core.driver.kuzu_driver import KuzuDriver
from kive.server.adapters.extensions.graphiti.drivers import patch_kuzu_fulltext_indices

# 创建 driver 后立即调用
driver = KuzuDriver(db=db_path)
await patch_kuzu_fulltext_indices(driver)
```

**上游问题**: https://github.com/getzep/graphiti (待报告)

**TODO**: Graphiti 修复后移除此扩展

---

## 贡献指南

### 添加新扩展

1. **选择合适的目录**:
   - LLM 客户端 → `llm_clients/`
   - Embedding 实现 → `embedders/`
   - 驱动补充 → `drivers/`

2. **文件命名**:
   - 使用描述性名称（如 `bailian.py`, `kuzu_indices.py`）
   - 避免使用 `xxx_patch.py`（我们是扩展，不只是补丁）

3. **代码规范**:
   ```python
   """模块标题
   
   Extension: 简短描述
   
   Issue:
       详细描述问题
   
   Root Cause: (可选)
       根本原因分析
   
   Solution:
       解决方案说明
   
   Status: Production-ready ✅ | Temporary workaround ⚠️
   Upstream Issue: 链接或 N/A
   
   Usage:
       使用示例代码
   """
   ```

4. **更新文档**:
   - 在本 README.md 中添加说明
   - 标注状态（生产可用/临时方案）
   - 如果是临时方案，添加移除条件

### 扩展生命周期

**生产扩展** ✅:
- 用于生态适配（如百炼 LLM 客户端）
- 长期维护，定期同步上游 API 变化

**临时扩展** ⚠️:
- 用于修复上游 bug 或补充缺失功能
- 定期检查上游是否已修复
- 修复后立即移除，更新引用

---

## 维护清单

### 定期检查（每次升级第三方库时）

- [ ] Graphiti Kuzu 索引 - 检查 `KuzuDriver.build_indices_and_constraints()` 是否已实现
- [ ] 百炼 LLM 客户端 - 检查百炼 API 是否支持 json_schema

### 版本兼容性

| 扩展 | 第三方库 | 测试版本 | 状态 |
|------|---------|---------|------|
| bailian.py | graphiti-core | 0.24.1 | ✅ |
| kuzu_indices.py | graphiti-core | 0.24.1 | ⚠️ |
| kuzu_indices.py | kuzu | 0.11.3 | ⚠️ |

---

## 未来规划

### Graphiti

- [ ] `llm_clients/qianfan.py` - 百度千帆适配
- [ ] `llm_clients/tongyi.py` - 阿里通义千问直连（非百炼）
- [ ] `embedders/bailian.py` - 百炼 Embedding 优化

### Cognee

- [ ] 待评估

---

## 问题反馈

如发现扩展失效或需要新的扩展，请：
1. 检查上游库版本是否兼容
2. 查看本文档的维护清单
3. 提交 issue 或直接修复
