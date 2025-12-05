"""Kive Memory with Cognee Example

Shows how to use Memory class with Cognee engine for knowledge graph operations.
"""

import asyncio
import os
from dotenv import load_dotenv
from kive import Memory, engines

load_dotenv()


async def main():
    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-***")
    
    # Create cognee engine with full configuration
    engine = engines.Cognee(
        # LLM configuration (for knowledge extraction)
        llm_provider="bailian",
        llm_model="qwen-plus",
        llm_api_key=api_key,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        
        # Embedding configuration (for vector search)
        embedding_provider="ollama",  # Use Ollama local embedding service
        embedding_model="nomic-embed-text:latest",
        embedding_base_url="http://localhost:11434/api/embed",
        embedding_dimensions=768,
        huggingface_tokenizer="nomic-ai/nomic-embed-text-v1.5",  # Required for Ollama
        
        # Vector DB configuration
        vector_db_provider="lancedb",  # File-based, no service needed
        
        # Graph DB configuration
        graph_db_provider="kuzu",
    )
    
    # Create memory with engine
    memory = Memory(engine=engine)
    
    try:
        # Add memo (auto-initializes on first call)
        print("\n=== Adding memo ===")
        memos = await memory.add(
            text="Alice is a software engineer who loves coffee and reading tech books.",
            namespace="personal",
            user_id="alice"
        )
        print(f"Added {len(memos)} memo(s):")
        for memo in memos:
            print(f"  - ID: {memo.id}")
            print(f"    Text: {memo.text}")
            print(f"    Backend: {memo.backend.type}")
        
        # Process (cognify) - required before search
        print("\n=== Processing (cognify) ===")
        result = await memory.process()
        print(f"Process status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        
        # Search
        print("\n=== Searching ===")
        results = await memory.search(
            query="Who loves coffee?",
            namespace="personal",
            user_id="alice",
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
    engine = engines.Cognee(
        llm_provider="bailian",
        llm_model="qwen-plus",
        llm_api_key=api_key,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text:latest",
        embedding_base_url="http://localhost:11434/api/embed",
        embedding_dimensions=768,
        huggingface_tokenizer="nomic-ai/nomic-embed-text-v1.5",
        vector_db_provider="lancedb",
        graph_db_provider="kuzu",
    )
    
    # Context manager also auto-initializes
    async with Memory(engine=engine) as memory:
        memos = await memory.add(
            text="Bob is a data scientist who specializes in machine learning.",
            namespace="work",
            user_id="bob"
        )
        print(f"Added: {memos[0].text}")
        
        # Process before search
        await memory.process()
        
        # Search
        results = await memory.search(
            query="Who is Bob?",
            namespace="work",
            user_id="bob"
        )
        print(f"Found: {results[0].text if results else 'No results'}")
        
        # Automatically closed when exiting context


if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())
    
    # Uncomment to try context manager version:
    # asyncio.run(main_with_context())
