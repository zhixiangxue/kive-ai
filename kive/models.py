"""
数据模型定义

定义了Kive系统中的核心数据结构
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer
from pydantic import ConfigDict


class BackendType(str, Enum):
    """Backend type enumeration"""
    COGNEE = "cognee"
    GRAPHITI = "graphiti"
    MEM0 = "mem0"


class BackendData(BaseModel):
    """Base backend tracking data
    
    All backend-specific data should inherit from this class.
    """
    type: BackendType = Field(..., description="Backend type")
    version: str = Field(..., description="Backend version (e.g. 0.4.1)")
    
    class Config:
        use_enum_values = True  # Serialize enum as string value


class CogneeBackendData(BackendData):
    """Cognee backend specific data"""
    data_id: str = Field(..., description="Cognee data UUID")
    dataset_id: str = Field(..., description="Cognee dataset UUID")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "type": "cognee",
                "version": "0.4.1",
                "data_id": "58422078-c384-5642-9621-2ca0b90bb97b",
                "dataset_id": "a3c54e4c-bf9d-59c7-b4a2-ff66b6ca1122",
            }
        }


class GraphitiBackendData(BackendData):
    """Graphiti backend specific data"""
    episode_id: str = Field(..., description="Graphiti episode UUID")
    source: str = Field(default="text", description="Episode source type (text/json)")
    source_description: Optional[str] = Field(None, description="Episode source description")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "type": "graphiti",
                "version": "0.24.1",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "source": "text",
                "source_description": "user input",
            }
        }


class Mem0BackendData(BackendData):
    """Mem0 backend specific data"""
    memory_id: str = Field(..., description="Mem0 memory ID")
    user_id: str = Field(..., description="User ID in mem0 (required by mem0 API)")
    agent_id: Optional[str] = Field(None, description="Agent ID in mem0")
    run_id: Optional[str] = Field(None, description="Run ID in mem0")
    
    # Graph relations (if graph store enabled)
    # mem0 returns: {"deleted_entities": [...], "added_entities": [...]}
    relations: Optional[Dict[str, Any]] = Field(None, description="Graph relations from mem0")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "type": "mem0",
                "version": "0.1.0",
                "memory_id": "m-abc123xyz",
                "user_id": "kive_user",
                "agent_id": None,
                "run_id": None,
                "relations": None,
            }
        }


class Memo(BaseModel):
    """Memory Entry
    
    Core data structure for storing memory content with backend tracking info.
    """
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "id": "memo_abc12345",
                "text": "This is memory content",
                "metadata": {"source": "file", "path": "/path/to/doc.pdf"},
                "backend": {
                    "type": "cognee",
                    "version": "0.4.1",
                    "data_id": "58422078-c384-5642-9621-2ca0b90bb97b",
                    "dataset_id": "a3c54e4c-bf9d-59c7-b4a2-ff66b6ca1122",
                },
                "score": 0.95,
                "created_at": "2025-01-01T12:00:00",
                "updated_at": "2025-01-01T12:00:00",
            }
        }
    )
    
    id: str = Field(..., description="Unique memo ID")
    text: str = Field(..., description="Memory text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    backend: BackendData = Field(..., description="Backend tracking data")
    score: Optional[float] = Field(None, description="Search similarity score (only in search results)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update time")
    
    @field_serializer('backend')
    def serialize_backend(self, backend: BackendData, _info):
        """Serialize backend with type information"""
        return backend.model_dump(mode='json')


class SearchResult(BaseModel):
    """搜索结果"""
    
    memos: List[Memo] = Field(default_factory=list, description="Memory list")
    total: int = Field(..., description="Total count")
    query: str = Field(..., description="Query string")
    took_ms: float = Field(..., description="Query time (ms)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "memos": [],
                "total": 10,
                "query": "查找相关记忆",
                "took_ms": 123.45,
            }
        }


class AddMemoRequest(BaseModel):
    """添加记忆请求"""
    
    text: Optional[str] = Field(None, description="文本内容")
    file: Optional[str] = Field(None, description="文件路径")
    url: Optional[str] = Field(None, description="URL地址")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "这是一段记忆",
                "metadata": {"source": "user_input"},
            }
        }


class AddMemoBatchRequest(BaseModel):
    """批量添加记忆请求"""
    
    items: List[AddMemoRequest] = Field(..., description="记忆列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {"text": "记忆1"},
                    {"file": "/path/to/doc.pdf"},
                ]
            }
        }


class UpdateMemoRequest(BaseModel):
    """更新记忆请求"""
    
    text: Optional[str] = Field(None, description="新的文本内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="新的元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "更新后的内容",
                "metadata": {"updated": True},
            }
        }


class DeleteMemosRequest(BaseModel):
    """删除记忆请求"""
    
    memo_ids: List[str] = Field(..., description="要删除的记忆ID列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "memo_ids": ["memo_1", "memo_2"]
            }
        }


class AddMemoResponse(BaseModel):
    """添加记忆响应"""
    
    memos: List[Memo] = Field(..., description="添加的记忆列表")
    count: int = Field(..., description="添加的记忆数量")
    
    class Config:
        json_schema_extra = {
            "example": {
                "memos": [],
                "count": 1
            }
        }


class UpdateMemoResponse(BaseModel):
    """更新记忆响应"""
    
    memo: Memo = Field(..., description="更新后的记忆")
    
    class Config:
        json_schema_extra = {
            "example": {
                "memo": {}
            }
        }


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessTask(BaseModel):
    """Process task info"""
    
    task_id: str = Field(..., description="Task ID")
    status: TaskStatus = Field(..., description="Task status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "task_id": "task_abc123",
                "status": "completed",
                "created_at": "2025-01-01T12:00:00",
                "result": {"status": "success", "message": "Process completed"}
            }
        }
