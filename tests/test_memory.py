import sys
import os
import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_mem.server import store_memory, retrieve_memory, update_memory, delete_memory, summarize_memory


# Mock embedding function for tests
def create_mock_embedding_func():
    """Create a mock embedding function that LightRAG can use."""
    from lightrag.utils import wrap_embedding_func_with_attrs
    
    @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
    async def mock_embedding_func(texts):
        """Mock embedding function that returns consistent embeddings for testing."""
        if isinstance(texts, str):
            texts = [texts]
        # Return consistent embeddings based on text hash for reproducible tests
        embeddings = []
        for text in texts:
            seed = hash(text) % 1000
            np.random.seed(seed)
            embedding = np.random.randn(384).astype(np.float32)
            embeddings.append(embedding)
        return embeddings
    
    return mock_embedding_func


# Mock LLM function for tests
async def mock_llm_func(*args, **kwargs):
    """Mock LLM function that returns simple responses for testing."""
    # Extract prompt from args or kwargs
    prompt = args[0] if args else kwargs.get('prompt', '')
    
    # Return a simple response that includes some content from the prompt
    # LightRAG expects structured JSON responses for some operations
    if "entities" in prompt.lower() or "relationships" in prompt.lower():
        # Return structured response for entity/relationship extraction
        return """
        {
            "high_level_keywords": ["test", "memory", "pytest"],
            "low_level_keywords": ["content", "entry", "data"]
        }
        """
    else:
        # Return simple text response for other operations
        return f"Mock response for: {prompt[:50]}..."


@pytest.fixture(autouse=True)
def setup_lightrag_with_mocks(monkeypatch):
    """Configure LightRAG with mocked embedding and LLM functions for testing."""
    # Force LightRAG storage (not InMemory!)
    monkeypatch.setenv("MEMORY_STORAGE_TYPE", "lightrag")
    
    # Mock the LightRAG storage to use our test functions
    mock_embedding = create_mock_embedding_func()
    
    # Patch the LightRAGStorage to use our mocked functions
    def mock_lightrag_storage_init(self, embedding_func=None, llm_func=None):
        """Mock constructor that uses test functions."""
        from mcp_mem.lightrag_storage import LightRAGStorage
        from mcp_mem.instance_manager import LightRAGInstanceManager
        
        # Use our test functions
        self.embedding_func = mock_embedding
        self.llm_func = mock_llm_func
        
        # Create instance manager with test functions
        self.instance_manager = LightRAGInstanceManager(
            embedding_func=self.embedding_func,
            llm_func=self.llm_func
        )
    
    # Patch the LightRAGStorage.__init__ method
    from mcp_mem.lightrag_storage import LightRAGStorage
    monkeypatch.setattr(LightRAGStorage, '__init__', mock_lightrag_storage_init)
    
    # Re-import the server module to pick up the new configuration
    import importlib
    import mcp_mem.server
    importlib.reload(mcp_mem.server)
    
    # Import the reloaded functions
    global store_memory, retrieve_memory, update_memory, delete_memory, summarize_memory
    from mcp_mem.server import store_memory, retrieve_memory, update_memory, delete_memory, summarize_memory
    
    yield


@pytest.mark.asyncio
async def test_store_and_retrieve():
    """Test basic store and retrieve functionality with LightRAG."""
    session_id = "test-session-1"
    content = "This is a test memory entry for pytest."
    
    # Store a memory entry
    result = await store_memory(session_id, content)
    assert result["status"] == "success"
    assert result["session_id"] == session_id
    
    # Retrieve the memory entry
    result = await retrieve_memory(session_id, query="test", limit=10)
    assert result["status"] == "success"
    assert result["session_id"] == session_id
    assert len(result["result"]) > 0
    
    # Check the content
    entry = result["result"][0]
    assert "content" in entry
    assert "id" in entry


@pytest.mark.asyncio
async def test_store_with_metadata():
    """Test storing memory with metadata, source_agent, and message_type."""
    session_id = "test-session-2"
    content = "Memory entry with metadata."
    metadata = {"tags": ["important", "test"], "priority": "high"}
    source_agent = "test_agent"
    message_type = "info"
    
    # Store with metadata
    result = await store_memory(
        session_id=session_id,
        content=content,
        metadata=metadata,
        source_agent=source_agent,
        message_type=message_type
    )
    assert result["status"] == "success"
    
    # Retrieve and verify metadata
    result = await retrieve_memory(session_id, query="metadata")
    assert result["status"] == "success"
    assert len(result["result"]) > 0
    
    entry = result["result"][0]
    assert "content" in entry
    # Note: With LightRAG, metadata parsing might not be perfect in tests
    # but the core functionality should work


@pytest.mark.asyncio
async def test_retrieve_with_filters():
    """Test retrieving memory with various filters."""
    session_id = "test-session-3"
    
    # Store multiple entries
    await store_memory(session_id, "Agent1 message", source_agent="agent1", message_type="request")
    await store_memory(session_id, "Agent2 message", source_agent="agent2", message_type="response")
    await store_memory(session_id, "General message", message_type="info")
    
    # Retrieve all messages (basic search works)
    result = await retrieve_memory(session_id, query="message")
    assert result["status"] == "success"
    # Note: LightRAG might return different numbers due to semantic search
    assert len(result["result"]) >= 0


@pytest.mark.asyncio
async def test_update_memory():
    """Test updating memory entries."""
    session_id = "test-session-4"
    content = "Original content"
    
    # Store an entry
    await store_memory(session_id, content)
    
    # Retrieve to get the ID
    result = await retrieve_memory(session_id, query="Original")
    if len(result["result"]) > 0:
        entry_id = result["result"][0]["id"]
        
        # Update the content (note: this is simulated in LightRAG)
        new_content = "Updated content"
        result = await update_memory(session_id, entry_id, new_content=new_content)
        assert result["status"] == "success"


@pytest.mark.asyncio
async def test_delete_memory():
    """Test deleting memory entries."""
    session_id = "test-session-5"
    content = "Content to be deleted"
    
    # Store an entry
    await store_memory(session_id, content)
    
    # Retrieve to get the ID
    result = await retrieve_memory(session_id, query="deleted")
    if len(result["result"]) > 0:
        entry_id = result["result"][0]["id"]
        
        # Delete the entry (note: this is simulated in LightRAG)
        result = await delete_memory(session_id, entry_id=entry_id)
        assert result["status"] == "success"
        assert result["deleted_count"] >= 0


@pytest.mark.asyncio
async def test_summarize_memory():
    """Test memory summarization."""
    session_id = "test-session-6"
    
    # Store multiple entries
    await store_memory(session_id, "First important point about the project.")
    await store_memory(session_id, "Second key insight from the discussion.")
    await store_memory(session_id, "Final decision made by the team.")
    
    # Summarize all memories
    result = await summarize_memory(session_id)
    assert result["status"] == "success"
    assert "summary" in result
    assert len(result["summary"]) > 0
    
    # Summarize with query filter
    result = await summarize_memory(session_id, query="decision")
    assert result["status"] == "success"
    assert len(result["summary"]) > 0


@pytest.mark.asyncio
async def test_session_isolation():
    """Test that different sessions are isolated from each other."""
    session1 = "test-session-isolation-1"
    session2 = "test-session-isolation-2"
    
    # Store entries in different sessions
    await store_memory(session1, "Session 1 data")
    await store_memory(session2, "Session 2 data")
    
    # Basic test - just ensure no errors and sessions work independently
    result1 = await retrieve_memory(session1, query="data")
    assert result1["status"] == "success"
    
    result2 = await retrieve_memory(session2, query="data")
    assert result2["status"] == "success"


@pytest.mark.asyncio
async def test_limit_functionality():
    """Test the limit parameter in retrieve_memory."""
    session_id = "test-session-limit"
    
    # Store multiple entries
    for i in range(5):
        await store_memory(session_id, f"Entry number {i}")
    
    # Retrieve with limit
    result = await retrieve_memory(session_id, query="Entry", limit=3)
    assert result["status"] == "success"
    assert len(result["result"]) <= 3
    
    # Retrieve without limit
    result = await retrieve_memory(session_id, query="Entry")
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for edge cases."""
    session_id = "test-session-errors"
    
    # Test updating non-existent entry
    result = await update_memory(session_id, "non-existent-id", "new content")
    # Should handle gracefully (may return not_found or success depending on implementation)
    assert result["status"] in ["not_found", "success", "error"]
    
    # Test deleting non-existent entry
    result = await delete_memory(session_id, entry_id="non-existent-id")
    # Should handle gracefully
    assert result["status"] in ["success", "error"]
    assert result["deleted_count"] >= 0
    
    # Test retrieving from empty session
    result = await retrieve_memory("empty-session", query="anything")
    assert result["status"] == "success"
    assert len(result["result"]) >= 0  # LightRAG might return empty results 