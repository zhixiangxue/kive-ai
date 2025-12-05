"""
Backend Adapters

Provides adapter implementations for different memory engines
"""

from .base import BaseMemoryAdapter
from .llm_bridge import LLMProvider, LLMProviderType

__all__ = ["BaseMemoryAdapter", "LLMProvider", "LLMProviderType"]

# Optional imports - can only import if corresponding dependencies are installed
try:
    from .cognee_adapter import CogneeAdapter
    __all__.append("CogneeAdapter")
except ImportError:
    pass

try:
    from .graphiti_adapter import GraphitiAdapter
    __all__.append("GraphitiAdapter")
except ImportError:
    pass

try:
    from .mem0_adapter import Mem0Adapter
    __all__.append("Mem0Adapter")
except ImportError:
    pass
