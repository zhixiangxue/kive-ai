"""
FastAPI route definitions
"""

import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from ...exceptions import KiveError, NotFoundError, ValidationError
from ...models import (
    AddMemoBatchRequest,
    AddMemoRequest,
    AddMemoResponse,
    DeleteMemosRequest,
    Memo,
    ProcessTask,
    SearchResult,
    TaskStatus,
    UpdateMemoRequest,
    UpdateMemoResponse,
)
from ...utils.document import create_document_from_request, url_to_documents
from ...utils.logger import logger

router = APIRouter(prefix="/api/v1")

# Background task storage (in-memory)
_background_tasks: Dict[str, ProcessTask] = {}


def get_adapter(request: Request):
    """Get adapter from app state"""
    return request.app.state.adapter


def get_cache(request: Request):
    """Get cache from app state"""
    return request.app.state.cache


@router.get("/health")
async def health_check(request: Request):
    """健康检查"""
    adapter = get_adapter(request)
    status = adapter.get_status()
    
    return {
        "status": "healthy",
        "version": "0.1.0",
        **status
    }


@router.post("/memos", response_model=AddMemoResponse)
async def add_memo(req: AddMemoRequest, request: Request):
    """添加单条记忆"""
    try:
        adapter = get_adapter(request)
        cache = get_cache(request)
        
        # 转换为Documents
        if req.url:
            # URL需要异步处理
            documents = await url_to_documents(req.url)
            if req.metadata:
                for doc in documents:
                    doc.metadata.update(req.metadata)
        else:
            documents = create_document_from_request(
                text=req.text,
                file=req.file,
                metadata=req.metadata
            )
        
        # 添加到adapter, 返回Memo列表
        memos = await adapter.add(documents)
        
        # 保存到cache
        for memo in memos:
            cache.save(memo)
        
        # 检查是否需要立即触发处理
        if adapter.should_trigger_process():
            asyncio.create_task(adapter.trigger_process())
        
        return AddMemoResponse(
            memos=memos,
            count=len(memos)
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memos/batch", response_model=AddMemoResponse)
async def add_memos_batch(req: AddMemoBatchRequest, request: Request):
    """批量添加记忆"""
    try:
        adapter = get_adapter(request)
        cache = get_cache(request)
        all_memos = []
        
        for item in req.items:
            # 转换为Documents
            if item.url:
                documents = await url_to_documents(item.url)
                if item.metadata:
                    for doc in documents:
                        doc.metadata.update(item.metadata)
            else:
                documents = create_document_from_request(
                    text=item.text,
                    file=item.file,
                    metadata=item.metadata
                )
            
            # 添加到adapter, 返回Memo列表
            memos = await adapter.add(documents)
            all_memos.extend(memos)
        
        # 保存到cache
        for memo in all_memos:
            cache.save(memo)
        
        # 检查是否需要触发处理
        if adapter.should_trigger_process():
            asyncio.create_task(adapter.trigger_process())
        
        return AddMemoResponse(
            memos=all_memos,
            count=len(all_memos)
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=ProcessTask)
async def trigger_process(request: Request, background: bool = False):
    """手动触发处理
    
    Args:
        background: 是否在后台执行，默认False（同步等待）
    """
    try:
        adapter = get_adapter(request)
        
        if background:
            # 异步执行
            task_id = str(uuid.uuid4())
            task = ProcessTask(
                task_id=task_id,
                status=TaskStatus.PENDING,
            )
            _background_tasks[task_id] = task
            
            # 创建后台任务
            async def _run_process():
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                try:
                    result = await adapter.trigger_process()
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    logger.error(f"Background process task {task_id} failed: {e}")
                finally:
                    task.completed_at = datetime.now()
            
            asyncio.create_task(_run_process())
            logger.info(f"Created background process task: {task_id}")
            return task
        else:
            # 同步执行
            task_id = str(uuid.uuid4())
            task = ProcessTask(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(),
            )
            
            try:
                result = await adapter.trigger_process()
                task.status = TaskStatus.COMPLETED
                task.result = result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                raise
            finally:
                task.completed_at = datetime.now()
            
            return task
        
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process/{task_id}", response_model=ProcessTask)
async def get_process_task(task_id: str):
    """查询后台处理任务状态"""
    task = _background_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


@router.get("/memos/search", response_model=SearchResult)
async def search_memos(
    query: str,
    limit: int = 10,
    request: Request = None
):
    """搜索记忆"""
    try:
        adapter = get_adapter(request)
        
        start_time = time.time()
        # adapter.search() 现在直接返回Memo列表
        memos = await adapter.search(query, limit=limit)
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            memos=memos,
            total=len(memos),
            query=query,
            took_ms=took_ms
        )
        
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memos/{memo_id}", response_model=Optional[Memo])
async def get_memo(memo_id: str, request: Request):
    """获取单条记忆
    
    Note: 如果adapter不支持get操作（如cognee），则从cache返回
    """
    try:
        adapter = get_adapter(request)
        cache = get_cache(request)
        
        # 先尝试从 adapter 获取
        memo = await adapter.get(memo_id)
        
        # 如果 adapter 不支持，从cache获取
        if memo is None:
            memo = cache.get(memo_id)
        
        return memo
        
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/memos/{memo_id}", response_model=UpdateMemoResponse)
async def update_memo(
    memo_id: str,
    req: UpdateMemoRequest,
    request: Request
):
    """更新记忆"""
    try:
        adapter = get_adapter(request)
        cache = get_cache(request)
        
        # 从cache获取原始memo
        original_memo = cache.get(memo_id)
        if not original_memo:
            raise NotFoundError(f"Memo not found: {memo_id}")
        
        # 创建新的Document
        from llama_index.core.schema import Document
        document = Document(
            text=req.text or original_memo.text,
            metadata=req.metadata or original_memo.metadata
        )
        
        # 返回更新后的Memo
        updated_memo = await adapter.update(original_memo, document)
        
        # 更新cache
        cache.save(updated_memo)
        
        return UpdateMemoResponse(memo=updated_memo)
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memos", response_model=dict)
async def delete_memos(req: DeleteMemosRequest, request: Request):
    """删除记忆"""
    try:
        adapter = get_adapter(request)
        cache = get_cache(request)
        
        # 从cache获取所有memo对象
        memos = []
        for memo_id in req.memo_ids:
            memo = cache.get(memo_id)
            if memo:
                memos.append(memo)
            else:
                logger.warning(f"Memo not found in cache: {memo_id}")
        
        if not memos:
            raise NotFoundError("No memos found for deletion")
        
        # 执行删除
        success = await adapter.delete(memos)
        
        # 从cache中删除
        if success:
            for memo_id in req.memo_ids:
                cache.delete(memo_id)
        
        return {
            "success": success,
            "count": len(memos)
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KiveError as e:
        raise HTTPException(status_code=500, detail=str(e))
