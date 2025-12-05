"""
Memory engines (alias for adapters)

Provides a more user-friendly interface to memory adapters.
Users interact with "engines" while the implementation uses Adapter Pattern.
"""

from .adapters.base import BaseMemoryAdapter as BaseMemoryEngine
from .adapters.cognee_adapter import CogneeAdapter as Cognee
from .adapters.graphiti_adapter import GraphitiAdapter as Graphiti
from .adapters.mem0_adapter import Mem0Adapter as Mem0

__all__ = [
    "BaseMemoryEngine",
    "Cognee",
    "Graphiti",
    "Mem0",
]
