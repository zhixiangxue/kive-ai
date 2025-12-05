"""Graphiti Driver Extensions

Driver supplements and bug fixes for Graphiti graph database drivers.
"""

from .kuzu_indices import patch_kuzu_fulltext_indices

__all__ = ["patch_kuzu_fulltext_indices"]
