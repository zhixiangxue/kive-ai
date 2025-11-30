"""
Kive - A lightweight memory system glue layer

支持模块化安装:
- pip install kive[client]  # 只安装客户端
- pip install kive[server]  # 只安装服务端
- pip install kive[cognee]  # 安装cognee后端支持
- pip install kive[all]     # 安装所有组件
"""

__version__ = "0.1.0"

# Public exports
from .models import Memo, SearchResult
from .exceptions import (
    KiveError,
    AdapterError,
    ConnectionError,
    SearchError,
    ProcessError,
)

__all__ = [
    "__version__",
    "Memo",
    "SearchResult",
    "KiveError",
    "AdapterError",
    "ConnectionError",
    "SearchError",
    "ProcessError",
]
