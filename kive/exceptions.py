"""
统一异常定义

定义了Kive系统中的所有异常类型,用于统一错误处理
"""


class KiveError(Exception):
    """Kive基础异常"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self):
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class AdapterError(KiveError):
    """Adapter层错误"""
    pass


class ConnectionError(AdapterError):
    """连接错误"""
    pass


class SearchError(AdapterError):
    """搜索错误"""
    pass


class ProcessError(AdapterError):
    """处理错误"""
    pass


class NotFoundError(KiveError):
    """资源不存在"""
    pass


class ValidationError(KiveError):
    """参数验证错误"""
    pass
