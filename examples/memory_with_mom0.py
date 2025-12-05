"""Kive Memory Quick Start Example

Shows how to use Memory class for local memory operations (no server needed).
"""

import asyncio
import os
from dotenv import load_dotenv
from kive import Memory, engines

load_dotenv()


async def main():
    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-***")
    
    # Create mem0 engine with full configuration
    engine = engines.Mem0(
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

        # Vector DB configuration
        vector_db_provider="chroma",
        vector_db_uri=None,  # Embedded mode, will use .kive/chroma
        
        # Graph DB configuration (optional)
        graph_db_provider="kuzu",
        graph_db_uri=".kive/memory_example.kuzu",
        
        # Multi-tenancy defaults
        default_user_id="kive_user",
    )
    
    # Create memory with engine
    memory = Memory(engine=engine)
    
    try:
        # Add memo (auto-initializes on first call)
        print("\n=== Adding memo ===")
        memos = await memory.add(
            text="John likes to play football",
            namespace="personal",
            user_id="john"
        )
        print(f"Added {len(memos)} memo(s):")
        for memo in memos:
            print(f"  - ID: {memo.id}")
            print(f"    Text: {memo.text}")
            print(f"    Backend: {memo.backend.type}")
        
        # Search
        print("\n=== Searching ===")
        results = await memory.search(
            query="What does John like?",
            namespace="personal",
            user_id="john",
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
    engine = engines.Mem0(
        vector_db_provider="chroma",
        graph_db_provider="kuzu",
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
            text="Alice loves reading sci-fi novels",
            namespace="books",
            user_id="alice"
        )
        print(f"Added: {memos[0].text}")
        
        # Search
        results = await memory.search(
            query="What does Alice love?",
            namespace="books",
            user_id="alice"
        )
        print(f"Found: {results[0].text if results else 'No results'}")
        
        # Automatically closed when exiting context


if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())
    
    # Uncomment to try context manager version:
    # asyncio.run(main_with_context())
