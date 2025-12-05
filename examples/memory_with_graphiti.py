"""Kive Memory with Graphiti Example

Shows how to use Memory class with Graphiti engine for temporal knowledge graph operations.
"""

import asyncio
import os
from dotenv import load_dotenv
from kive import Memory, engines

load_dotenv()


async def main():
    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-***")
    
    # Create graphiti engine with full configuration
    engine = engines.Graphiti(
        # Graph DB configuration
        graph_db_provider="kuzu",
        graph_db_uri=".kive/memory_graphiti_example.kuzu",
        
        # LLM configuration (for knowledge extraction)
        llm_provider="bailian",
        llm_model="qwen-plus",
        llm_api_key=api_key,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        
        # Embedding configuration (for vector search)
        embedding_provider="bailian",
        embedding_model="text-embedding-v3",
        embedding_api_key=api_key,
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_dimensions=1024,
    )
    
    # Create memory with engine
    memory = Memory(engine=engine)
    
    try:
        # Add memo (auto-initializes on first call)
        print("\n=== Adding memo ===")
        memos = await memory.add(
            text="Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.",
            namespace="politics",
            user_id="analyst"
        )
        print(f"Added {len(memos)} memo(s):")
        for memo in memos:
            print(f"  - ID: {memo.id}")
            print(f"    Text: {memo.text}")
            print(f"    Backend: {memo.backend.type}")
        
        # Search (Graphiti processes in real-time)
        print("\n=== Searching ===")
        results = await memory.search(
            query="Who was the California Attorney General?",
            namespace="politics",
            user_id="analyst",
            limit=5
        )
        print(f"Found {len(results)} result(s):")
        for result in results:
            print(f"  - {result.text} (score: {result.score})")
        
    finally:
        # Close connection
        await memory.close()
        print("\n=== Connection closed ===")


async def main_with_context():
    """Alternative: using async context manager"""
    
    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-***")
    
    # Create engine with full configuration
    engine = engines.Graphiti(
        graph_db_provider="kuzu",
        graph_db_uri=".kive/memory_graphiti_example.kuzu",
        llm_provider="bailian",
        llm_model="qwen-plus",
        llm_api_key=api_key,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_provider="bailian",
        embedding_model="text-embedding-v3",
        embedding_api_key=api_key,
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_dimensions=1024,
    )
    
    # Context manager also auto-initializes
    async with Memory(engine=engine) as memory:
        memos = await memory.add(
            text="Gavin Newsom is the Governor of California. He previously served as Lieutenant Governor.",
            namespace="politics",
            user_id="analyst"
        )
        print(f"Added: {memos[0].text}")
        
        # Search (no process needed, real-time)
        results = await memory.search(
            query="Who is the Governor of California?",
            namespace="politics",
            user_id="analyst"
        )
        print(f"Found: {results[0].text if results else 'No results'}")
        
        # Automatically closed when exiting context


if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())
    
    # Uncomment to try context manager version:
    # asyncio.run(main_with_context())
