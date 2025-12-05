"""
Kive Server Quickstart Examples

This example demonstrates how to start Kive server with different memory engines.
Choose the engine that best fits your needs:

- Cognee: Knowledge graph + vector search, best for complex knowledge relationships
- Graphiti: Temporal knowledge graph, best for time-aware episodic memory
- Mem0: Hybrid memory (vector + graph), best for simple setup and fast queries

Requirements:
- Install dependencies: pip install kive[cognee] or kive[graphiti] or kive[mem0]
- For local embedding: Start Ollama service (ollama serve)
"""

import os
from dotenv import load_dotenv
from kive.server import Server

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Common Configuration (shared by all engines)
# =============================================================================

# Get API key from environment
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-***")

# LLM Configuration (for knowledge extraction)
LLM_CONFIG = {
    "llm_provider": "bailian",  # openai, anthropic, bailian, moonshot, ollama, etc.
    "llm_model": "qwen-plus",
    "llm_api_key": API_KEY,
    "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# Embedding Configuration (for vector search)
EMBEDDING_CONFIG = {
    "embedding_provider": "ollama",  # ollama (local) or bailian/openai (cloud)
    "embedding_model": "nomic-embed-text:latest",
    "embedding_base_url": "http://localhost:11434/api/embed",
    "embedding_dimensions": 768,
}

# Database Configuration
DB_CONFIG = {
    "vector_db_provider": "lancedb",  # lancedb (file-based) or chroma (service)
    "graph_db_provider": "kuzu",      # kuzu (embedded) or neo4j (service)
}


# =============================================================================
# Example 1: Cognee Engine (Knowledge Graph)
# =============================================================================

def start_with_cognee():
    """
    Cognee Engine - Best for complex knowledge relationships
    
    Features:
    - Deep knowledge extraction with LLM
    - Graph-based knowledge storage
    - Semantic search with cognify processing
    
    Best for:
    - Building knowledge bases from documents
    - Understanding relationships between concepts
    - Answering complex questions requiring reasoning
    """
    from kive.server.engines import Cognee
    
    engine = Cognee(
        **LLM_CONFIG,
        **EMBEDDING_CONFIG,
        **DB_CONFIG,
        # Cognee-specific: HuggingFace tokenizer for Ollama
        huggingface_tokenizer="nomic-ai/nomic-embed-text-v1.5",
        # Auto-process: automatically run cognify after adding data
        auto_process=False,
    )
    
    print("✓ Cognee engine configured")
    print("  Features: Knowledge graph + semantic search")
    
    return engine


# =============================================================================
# Example 2: Graphiti Engine (Temporal Knowledge Graph)
# =============================================================================

def start_with_graphiti():
    """
    Graphiti Engine - Best for time-aware episodic memory
    
    Features:
    - Temporal knowledge graph (time-aware facts)
    - Real-time processing (no separate cognify step)
    - Hybrid search (semantic + graph traversal)
    
    Best for:
    - Tracking events and timelines
    - Understanding how knowledge evolves over time
    - Episodic memory for conversational AI
    """
    from kive.server.engines import Graphiti
    
    engine = Graphiti(
        **LLM_CONFIG,
        **EMBEDDING_CONFIG,
        # Graphiti uses graph DB only
        graph_db_provider=DB_CONFIG["graph_db_provider"],
    )
    
    print("✓ Graphiti engine configured")
    print("  Features: Temporal knowledge graph + hybrid search")
    
    return engine


# =============================================================================
# Example 3: Mem0 Engine (Hybrid Memory)
# =============================================================================

def start_with_mem0():
    """
    Mem0 Engine - Best for simple setup and fast queries
    
    Features:
    - Vector search for semantic similarity
    - Optional graph storage for relationships
    - Real-time processing (no cognify needed)
    
    Best for:
    - Quick semantic search
    - Simple knowledge storage
    - Fast prototyping
    """
    from kive.server.engines import Mem0
    
    engine = Mem0(
        **LLM_CONFIG,
        **EMBEDDING_CONFIG,
        # Mem0 specific: uses chroma by default
        vector_db_provider="chroma",
        graph_db_provider=DB_CONFIG["graph_db_provider"],  # Optional
    )
    
    print("✓ Mem0 engine configured")
    print("  Features: Vector search + optional graph storage")
    
    return engine


# =============================================================================
# Start Server
# =============================================================================

def main():
    """Start Kive server with selected engine"""
    print("\n" + "="*60)
    print("Kive Server - Choose Your Memory Engine")
    print("="*60)
    print("\nAvailable engines:")
    print("  1. Cognee  - Knowledge graph + semantic search")
    print("  2. Graphiti - Temporal knowledge graph")
    print("  3. Mem0    - Hybrid memory (vector + graph)")
    print("\n" + "="*60)
    
    # Choose engine here (change number to switch engines)
    engine_choice = 1  # 1=Cognee, 2=Graphiti, 3=Mem0
    
    if engine_choice == 1:
        print("\nStarting with Cognee...")
        engine = start_with_cognee()
    elif engine_choice == 2:
        print("\nStarting with Graphiti...")
        engine = start_with_graphiti()
    elif engine_choice == 3:
        print("\nStarting with Mem0...")
        engine = start_with_mem0()
    else:
        raise ValueError(f"Invalid engine choice: {engine_choice}")
    
    # Create server
    server = Server(
        engine=engine,
        host="0.0.0.0",
        port=12306,
        log_level="INFO"
    )
    
    print("\n" + "="*60)
    print("Server starting on http://0.0.0.0:12306")
    print("API Docs: http://localhost:12306/docs")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    server.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
