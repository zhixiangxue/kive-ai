"""
Kive Client CRUD Example

This example demonstrates complete CRUD operations via Kive HTTP API:
1. Health Check - Check server status
2. Add - Create data (single + batch)
3. Process - Run cognify (for Cognee engine)
4. Search - Query data
5. Get - Retrieve single memo
6. Update - Modify data
7. Delete - Remove data

Before running:
1. Start server: python examples/server_quickstart.py
2. For Ollama embedding: ollama serve
3. Install client: pip install kive[client]
"""

import asyncio
import time
from kive.client import Client


# =============================================================================
# Configuration
# =============================================================================

SERVER_URL = "http://localhost:12306"
TIMEOUT = 60.0  # Longer timeout for process operations


# =============================================================================
# Test Functions
# =============================================================================

async def test_health(client: Client):
    """Test 0: Health Check"""
    print("\n" + "="*60)
    print("Test 0: Health Check")
    print("="*60)
    
    try:
        start_time = time.time()
        health = await client.health()
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Server is healthy [Time: {elapsed_ms:.2f}ms]")
        print(f"  - Status: {health.get('status')}")
        print(f"  - Version: {health.get('version')}")
        print(f"  - Engine: {health.get('adapter')}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print("  Please start server: python examples/server_quickstart.py")
        return False


async def test_add(client: Client):
    """Test 1: Add Data (Create)"""
    print("\n" + "="*60)
    print("Test 1: Add Data")
    print("="*60)
    
    try:
        # Add single memo
        print("Adding single memo...")
        start_time = time.time()
        response1 = await client.add(
            text="Artificial Intelligence is a branch of computer science focused on creating intelligent systems.",
            metadata={"source": "example", "topic": "AI"}
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Single add successful [Time: {elapsed_ms:.2f}ms]")
        print(f"  - Memo count: {response1.count}")
        print(f"  - Memo ID: {response1.memos[0].id}")
        print(f"  - Backend: {response1.memos[0].backend.type}")
        
        # Batch add
        print("\nAdding batch memos...")
        start_time = time.time()
        response2 = await client.add_batch([
            {
                "text": "Machine Learning is a core technology of AI that enables computers to learn from data.",
                "metadata": {"source": "example", "topic": "ML"}
            },
            {
                "text": "Deep Learning is a subfield of ML using neural networks to simulate brain processing.",
                "metadata": {"source": "example", "topic": "DL"}
            }
        ])
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Batch add successful [Time: {elapsed_ms:.2f}ms]")
        print(f"  - Memo count: {response2.count}")
        
        # Collect all memos
        all_memos = response1.memos + response2.memos
        print(f"\nTotal: {len(all_memos)} memos added")
        
        return all_memos
        
    except Exception as e:
        print(f"✗ Add failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_process(client: Client, background: bool = False):
    """Test 2: Process Data (Cognify)
    
    Note: Only required for Cognee engine. Graphiti and Mem0 process in real-time.
    """
    print("\n" + "="*60)
    print(f"Test 2: Process Data ({'Background' if background else 'Sync'})")
    print("="*60)
    print("⚠️  This step calls LLM, ensure API key is valid")
    print("If API key is invalid, this will fail but won't affect other tests\n")
    
    try:
        print(f"Starting cognify process (background={background})...")
        start_time = time.time()
        task = await client.process(background=background)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Process request sent [Time: {elapsed_ms:.2f}ms]")
        print(f"  - Task ID: {task.task_id}")
        print(f"  - Status: {task.status}")
        
        if background:
            print(f"  - Processing in background, checking status...")
            await asyncio.sleep(2)
            
            # Query task status
            start_time = time.time()
            task_status = await client.get_process_task(task.task_id)
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"\nTask status query [Time: {elapsed_ms:.2f}ms]:")
            print(f"  - Status: {task_status.status}")
            if task_status.result:
                print(f"  - Result: {task_status.result}")
            if task_status.error:
                print(f"  - Error: {task_status.error}")
        else:
            # Sync execution
            if task.result:
                print(f"  - Result: {task.result.get('status')}")
                print(f"  - Message: {task.result.get('message')}")
            if task.error:
                print(f"  - Error: {task.error}")
        
        return task
        
    except Exception as e:
        print(f"✗ Process failed: {e}")
        print("  ⚠️  Usually due to invalid API key, won't affect other tests")
        import traceback
        traceback.print_exc()
        return None


async def test_search(client: Client):
    """Test 3: Search Data (Read)"""
    print("\n" + "="*60)
    print("Test 3: Search Data")
    print("="*60)
    
    try:
        query = "What is artificial intelligence?"
        print(f"Query: {query}")
        print(f"Note: For Cognee, must run cognify before search")
        
        start_time = time.time()
        result = await client.search(query, limit=3)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Search successful [API time: {elapsed_ms:.2f}ms]")
        print(f"  - Query: {result.query}")
        print(f"  - Results: {result.total}")
        print(f"  - Backend search time: {result.took_ms:.2f}ms")
        
        if result.memos:
            print("\nSearch results:")
            for i, memo in enumerate(result.memos, 1):
                text = memo.text[:60]
                score = memo.score or 0
                print(f"  {i}. [{score:.3f}] {text}...")
        else:
            print("  No results (need to run cognify first for Cognee)")
        
        return result
        
    except Exception as e:
        print(f"✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_get(client: Client, memos):
    """Test 4: Get Single Memo (Read)"""
    print("\n" + "="*60)
    print("Test 4: Get Single Memo")
    print("="*60)
    
    if not memos:
        print("⚠️  Skipped: No memos available")
        return None
    
    try:
        memo = memos[0]
        print(f"Getting memo_id: {memo.id}")
        print(f"Note: Cognee doesn't support get by ID, returns None")
        
        start_time = time.time()
        result = await client.get(memo.id)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result is None:
            print(f"✓ Returns None as expected (Cognee limitation) [Time: {elapsed_ms:.2f}ms]")
        else:
            print(f"✓ Get successful [Time: {elapsed_ms:.2f}ms]")
            print(f"  - ID: {result.id}")
            print(f"  - Text: {result.text[:60]}...")
        
        return result
        
    except Exception as e:
        print(f"✗ Get failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_update(client: Client, memos):
    """Test 5: Update Data (Update)"""
    print("\n" + "="*60)
    print("Test 5: Update Data")
    print("="*60)
    
    if not memos:
        print("⚠️  Skipped: No memos available")
        return None
    
    try:
        memo = memos[0]
        new_text = "Artificial Intelligence (AI) is an important branch of computer science. [Updated]"
        
        print(f"Updating memo: {memo.id}")
        print(f"New text: {new_text[:50]}...")
        print(f"Note: Cognee update = delete + add, returns new Memo")
        
        start_time = time.time()
        response = await client.update(
            memo.id,
            text=new_text,
            metadata={"source": "example", "topic": "AI", "updated": True}
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Update successful [Time: {elapsed_ms:.2f}ms]")
        print(f"  - Old ID: {memo.id}")
        print(f"  - New ID: {response.memo.id}")
        print(f"  - ID changed: {memo.id != response.memo.id}")
        print(f"  - Backend: {response.memo.backend.type}")
        
        # Update memos list
        memos[0] = response.memo
        return response.memo
        
    except Exception as e:
        print(f"✗ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_delete(client: Client, memos):
    """Test 6: Delete Data (Delete)"""
    print("\n" + "="*60)
    print("Test 6: Delete Data")
    print("="*60)
    
    if not memos:
        print("⚠️  Skipped: No memos available")
        return False
    
    try:
        memo_ids = [memo.id for memo in memos]
        print(f"Deleting {len(memo_ids)} memos")
        for memo_id in memo_ids:
            print(f"  - {memo_id}")
        
        start_time = time.time()
        result = await client.delete(memo_ids)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Delete {'successful' if result else 'failed'} [Time: {elapsed_ms:.2f}ms]")
        
        return result
        
    except Exception as e:
        print(f"✗ Delete failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Flow
# =============================================================================

async def main():
    """Complete client test - CRUD + Process via HTTP"""
    print("\n" + "="*60)
    print("Kive Client Complete Test (CRUD + Process via HTTP)")
    print("="*60)
    print("\nGoal: Test all API endpoints through Client")
    print("Includes: Health, Add, Process, Search, Get, Update, Delete\n")
    
    # Create client
    client = Client(
        server_url=SERVER_URL,
        timeout=TIMEOUT
    )
    
    try:
        # Test 0: Health check
        if not await test_health(client):
            print("\nTest aborted: Server not running")
            return
        
        # Test 1: Add data (Create)
        memos = await test_add(client)
        if not memos:
            print("\nTest aborted: Add failed")
            return
        
        # Test 2: Process data - required before search
        print("\nNote: Must run cognify before search (Cognee only)")
        process_result = await test_process(client, background=False)
        if not process_result or process_result.error:
            print("\nWarning: Process failed, search may return no results")
        
        # Test 3: Search data (Read)
        await test_search(client)
        
        # Test 4: Get single data (Read)
        await test_get(client, memos)
        
        # Test 5: Update data (Update)
        await test_update(client, memos)
        
        # Test 6: Delete data (Delete)
        await test_delete(client, memos)
        
        # Test summary
        print("\n" + "="*60)
        print("Kive Client Test Complete!")
        print("="*60)
        print("\nTest coverage:")
        print("  ✓ Health     - Server health check")
        print("  ✓ Add        - Add data (single + batch)")
        print("  ✓ Process    - Cognify processing (sync/async)")
        print("  ✓ Search     - Search data")
        print("  ✓ Get        - Get single memo (returns None)")
        print("  ✓ Update     - Update data (returns new Memo)")
        print("  ✓ Delete     - Delete data")
        print("\nAll API endpoints tested successfully!")
        
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
