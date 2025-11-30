"""
Document conversion utilities

Provides conversion functionality from various data sources to llama-index Documents
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from ..exceptions import ValidationError
from .logger import logger


def text_to_document(text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
    """文本转Document
    
    Args:
        text: 文本内容
        metadata: 元数据
        
    Returns:
        Document对象
    """
    if not text:
        raise ValidationError("Text cannot be empty")
    
    return Document(
        text=text,
        metadata=metadata or {}
    )


def file_to_documents(file_path: str) -> List[Document]:
    """文件转Documents
    
    使用llama-index的SimpleDirectoryReader自动解析文件
    支持的格式: txt, pdf, docx, csv等
    
    Args:
        file_path: 文件路径
        
    Returns:
        Document列表
        
    Raises:
        ValidationError: 文件不存在或格式不支持
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    try:
        reader = SimpleDirectoryReader(input_files=[str(path)])
        documents = reader.load_data()
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
        
    except Exception as e:
        raise ValidationError(f"Failed to load file {file_path}: {e}")


async def url_to_documents(url: str) -> List[Document]:
    """URL转Documents
    
    需要安装llama-index-readers-web
    
    Args:
        url: URL地址
        
    Returns:
        Document列表
        
    Raises:
        ValidationError: URL无效或无法访问
    """
    if not url.startswith(("http://", "https://")):
        raise ValidationError(f"Invalid URL: {url}")
    
    try:
        # 尝试使用llama-index的web reader
        from llama_index.readers.web import SimpleWebPageReader
        
        reader = SimpleWebPageReader()
        documents = await reader.aload_data([url])
        
        logger.info(f"Loaded {len(documents)} documents from {url}")
        return documents
        
    except ImportError:
        raise ValidationError(
            "Web reader not available. "
            "URL support requires llama-index, but we don't force this dependency. "
            "Please handle URL content yourself or install: pip install llama-index-readers-web"
        )
    except Exception as e:
        raise ValidationError(f"Failed to load URL {url}: {e}")


def create_document_from_request(
    text: Optional[str] = None,
    file: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """根据请求参数创建Documents
    
    Args:
        text: 文本内容
        file: 文件路径
        url: URL地址
        metadata: 元数据
        
    Returns:
        Document列表
        
    Raises:
        ValidationError: 参数无效
    """
    if not any([text, file, url]):
        raise ValidationError("At least one of text/file/url must be provided")
    
    if sum([bool(text), bool(file), bool(url)]) > 1:
        raise ValidationError("Only one of text/file/url can be provided")
    
    if text:
        return [text_to_document(text, metadata)]
    
    if file:
        docs = file_to_documents(file)
        # 合并用户提供的metadata
        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)
        return docs
    
    if url:
        # URL是async的,这里需要调用者使用await
        raise ValidationError(
            "URL conversion requires async handling. "
            "Use url_to_documents() directly with await."
        )
