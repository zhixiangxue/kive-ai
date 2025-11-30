"""
Memory engines (alias for adapters)

Provides a more user-friendly interface to memory adapters.
Users interact with "engines" while the implementation uses Adapter Pattern.
"""

from .adapters.base import BaseMemoryAdapter as BaseMemoryEngine
from .adapters.cognee_adapter import CogneeAdapter as Cognee

__all__ = [
    "BaseMemoryEngine",
    "Cognee",
]
