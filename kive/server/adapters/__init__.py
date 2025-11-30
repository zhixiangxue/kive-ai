"""
Backend Adapters

Provides adapter implementations for different memory engines
"""

from .base import BaseMemoryAdapter

__all__ = ["BaseMemoryAdapter"]

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
