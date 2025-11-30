"""
Kive Client implementation

Provides a simple client API to access Kive service
"""

from typing import Any, Dict, List, Optional

import httpx

from ..exceptions import KiveError
from ..models import AddMemoResponse, Memo, ProcessTask, SearchResult, UpdateMemoResponse


class Client:
    """Kive客户端
    
    使用示例:
        client = Client("http://localhost:8000")
        
        # 添加记忆
        ids = await client.add(text="这是一段记忆")
        
        # 搜索
        results = await client.search("查找记忆")
        for memo in results.memos:
            print(memo.text)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0
    ):
        """
        Args:
            base_url: Kive服务器地址
            timeout: 请求超时时间(秒)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self):
        """关闭客户端连接"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _handle_response(self, response: httpx.Response) -> Any:
        """处理HTTP响应"""
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            raise KiveError(
                f"Request failed with status {response.status_code}: {error_detail}"
            )
        return response.json()
    
    # ===== 添加记忆 =====
    
    async def add(
        self,
        text: Optional[str] = None,
        file: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AddMemoResponse:
        """Add single memo
        
        Args:
            text: Text content
            file: File path
            url: URL address
            metadata: Metadata
            
        Returns:
            AddMemoResponse containing memos list and count
            
        Raises:
            KiveError: Raised when add fails
        """
        data = {"metadata": metadata or {}}
        if text:
            data["text"] = text
        elif file:
            data["file"] = file
        elif url:
            data["url"] = url
        else:
            raise KiveError("At least one of text/file/url must be provided")
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/memos",
            json=data
        )
        result = self._handle_response(response)
        return AddMemoResponse(**result)
    
    async def add_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> AddMemoResponse:
        """Batch add memos
        
        Args:
            items: Memo list, each item contains text/file/url and metadata
                Example: [
                    {"text": "...", "metadata": {}},
                    {"file": "path/to/file"},
                ]
                
        Returns:
            AddMemoResponse containing memos list and count
            
        Raises:
            KiveError: Raised when add fails
        """
        response = await self.client.post(
            f"{self.base_url}/api/v1/memos/batch",
            json={"items": items}
        )
        result = self._handle_response(response)
        return AddMemoResponse(**result)
    
    # ===== Process data =====
    
    async def process(self, background: bool = False) -> ProcessTask:
        """Trigger backend data processing (cognee's cognify)
        
        Args:
            background: Whether to execute in background, default False (sync wait)
        
        Returns:
            ProcessTask object containing task status and result
            
        Raises:
            KiveError: Raised when process fails
        """
        response = await self.client.post(
            f"{self.base_url}/api/v1/process",
            params={"background": background}
        )
        result = self._handle_response(response)
        return ProcessTask(**result)
    
    async def get_process_task(self, task_id: str) -> ProcessTask:
        """Get process task status
        
        Args:
            task_id: Task ID
        
        Returns:
            ProcessTask object
            
        Raises:
            KiveError: Raised when query fails
        """
        response = await self.client.get(
            f"{self.base_url}/api/v1/process/{task_id}"
        )
        result = self._handle_response(response)
        return ProcessTask(**result)
    
    # ===== 查询记忆 =====
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """搜索记忆
        
        Args:
            query: 查询文本
            limit: 返回数量限制
            **kwargs: 其他查询参数
            
        Returns:
            搜索结果
            
        Raises:
            KiveError: 搜索失败时抛出
        """
        params = {"query": query, "limit": limit}
        params.update(kwargs)
        
        response = await self.client.get(
            f"{self.base_url}/api/v1/memos/search",
            params=params
        )
        result = self._handle_response(response)
        return SearchResult(**result)
    
    async def get(self, memo_id: str) -> Optional[Memo]:
        """Get single memo
        
        Args:
            memo_id: Memo ID
            
        Returns:
            Memo object, returns None if adapter doesn't support or not found
            
        Raises:
            KiveError: Raised when get fails
        """
        response = await self.client.get(
            f"{self.base_url}/api/v1/memos/{memo_id}"
        )
        result = self._handle_response(response)
        return Memo(**result) if result else None
    
    # ===== 更新/删除 =====
    
    async def update(
        self,
        memo_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UpdateMemoResponse:
        """Update memo
        
        Args:
            memo_id: Memo ID
            text: New text content
            metadata: New metadata
            
        Returns:
            UpdateMemoResponse containing updated memo
            
        Raises:
            KiveError: Raised when update fails
        """
        data = {}
        if text is not None:
            data["text"] = text
        if metadata is not None:
            data["metadata"] = metadata
        
        response = await self.client.put(
            f"{self.base_url}/api/v1/memos/{memo_id}",
            json=data
        )
        result = self._handle_response(response)
        return UpdateMemoResponse(**result)
    
    async def delete(self, memo_ids: List[str]) -> bool:
        """删除记忆
        
        Args:
            memo_ids: 记忆ID列表
            
        Returns:
            是否删除成功
            
        Raises:
            KiveError: 删除失败时抛出
        """
        response = await self.client.request(
            "DELETE",
            f"{self.base_url}/api/v1/memos",
            json={"memo_ids": memo_ids}
        )
        result = self._handle_response(response)
        return result["success"]
    
    # ===== 健康检查 =====
    
    async def health(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            服务状态信息
        """
        response = await self.client.get(
            f"{self.base_url}/api/v1/health"
        )
        return self._handle_response(response)
