"""Kuzu Full-Text Indices Supplement for Graphiti

Extension: Kuzu driver full-text index creation

Issue:
    KuzuDriver.build_indices_and_constraints() is a no-op in Graphiti.
    The method is required by the abstract base class but does nothing,
    leaving required full-text indices (node_name_and_summary, etc.) uncreated.
    
    This causes search operations to fail with:
    "Table Entity doesn't have an index with name node_name_and_summary"

Root Cause:
    Graphiti's kuzu_driver.py line 143-147:
    ```python
    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Kuzu doesn't support dynamic index creation like Neo4j or FalkorDB
        # Schema and indices are created during setup_schema()
        # This method is required by the abstract base class but is a no-op for Kuzu
        pass
    ```
    
    However, setup_schema() only creates tables, not full-text indices.

Solution:
    Manually execute FTS index creation queries defined in:
    graphiti_core/graph_queries.py - get_fulltext_indices(GraphProvider.KUZU)

Status: Temporary workaround ⚠️
TODO: Remove when Graphiti fixes this issue upstream
Upstream Issue: https://github.com/getzep/graphiti (pending report)

Usage:
    from kive.server.adapters.extensions.graphiti.drivers import patch_kuzu_fulltext_indices
    
    # After creating KuzuDriver, before initializing Graphiti
    driver = KuzuDriver(db=db_path)
    await patch_kuzu_fulltext_indices(driver)
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.driver.kuzu_driver import KuzuDriver

logger = logging.getLogger(__name__)


async def patch_kuzu_fulltext_indices(driver: "KuzuDriver") -> None:
    """Create missing full-text indices for Kuzu driver
    
    This function supplements KuzuDriver.build_indices_and_constraints() by
    manually creating the required full-text indices that should have been
    created but weren't due to the no-op implementation.
    
    The index creation queries are based on Graphiti's own definition in
    graphiti_core/graph_queries.py:get_fulltext_indices(GraphProvider.KUZU)
    
    Args:
        driver: KuzuDriver instance (must be initialized with database)
        
    Note:
        Safe to call multiple times - will skip if indices already exist.
        Does not raise exceptions on duplicate index errors.
    """
    logger.info("Applying Kuzu FTS indices patch...")
    try:
        # These queries are from graphiti_core/graph_queries.py
        # See get_fulltext_indices(GraphProvider.KUZU)
        fts_index_queries = [
            # Episode content full-text index
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            
            # Entity name and summary full-text index (CRITICAL - most common search target)
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            
            # Community name full-text index
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            
            # Edge (relationship) name and fact full-text index
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]
        
        for query in fts_index_queries:
            try:
                await driver.execute_query(query)
                logger.debug(f"Created FTS index: {query[:50]}...")
            except Exception as e:
                # Index might already exist, which is fine
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.debug(f"FTS index already exists (skipped): {query[:50]}...")
                else:
                    # Log but don't fail - some indices might not be critical
                    logger.warning(f"Failed to create FTS index: {e}")
        
        logger.info("Kuzu FTS indices patch applied successfully")
    except Exception as e:
        logger.error(f"Error applying Kuzu FTS indices patch: {e}")
        # Don't fail initialization, as schema might already be set up
