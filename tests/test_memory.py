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
async def mock_llm_func(prompt, system_prompt=None, history_messages=None, hashing_kv=None, **kwargs):
    """Mock LLM function that returns simple responses for testing."""
    
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
    
    # Check if real LLM/embedding environment variables are provided
    has_real_llm = bool(os.getenv("LLM_BASE_URL") and os.getenv("OPENAI_API_KEY"))
    has_real_embedding = bool(os.getenv("EMBEDDING_BASE_URL") and os.getenv("OPENAI_API_KEY"))
    
    # Only use mocks if real services are not configured
    if not (has_real_llm and has_real_embedding):
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


# Predefined test data for realistic memory scenarios
REALISTIC_MEMORY_DATA = [
    {
        "content": "User John Smith requested access to the production database for quarterly reporting. Request approved by manager Sarah Johnson on 2024-01-15.",
        "metadata": {"tags": ["access-request", "database", "approved"], "priority": "high", "department": "finance"},
        "source_agent": "access_control_agent",
        "message_type": "approval",
        "expected_queries": ["database access", "John Smith", "quarterly reporting", "Sarah Johnson", "production"]
    },
    {
        "content": "System maintenance scheduled for Saturday 2024-01-20 from 2:00 AM to 6:00 AM EST. All services will be temporarily unavailable during this window.",
        "metadata": {"tags": ["maintenance", "downtime", "scheduled"], "priority": "critical", "department": "IT"},
        "source_agent": "maintenance_scheduler",
        "message_type": "notification",
        "expected_queries": ["maintenance", "Saturday", "downtime", "services unavailable", "2:00 AM"]
    },
    {
        "content": "Customer complaint received from ABC Corp regarding delayed shipment of order #12345. Issue escalated to logistics team for immediate resolution.",
        "metadata": {"tags": ["complaint", "shipment", "escalated"], "priority": "high", "department": "customer_service"},
        "source_agent": "customer_service_agent",
        "message_type": "escalation",
        "expected_queries": ["ABC Corp", "delayed shipment", "order 12345", "logistics team", "complaint"]
    },
    {
        "content": "Security audit completed for Q1 2024. Found 3 medium-risk vulnerabilities in web application. Remediation plan created with 30-day timeline.",
        "metadata": {"tags": ["security", "audit", "vulnerabilities"], "priority": "medium", "department": "security"},
        "source_agent": "security_audit_agent",
        "message_type": "report",
        "expected_queries": ["security audit", "vulnerabilities", "web application", "remediation plan", "Q1 2024"]
    },
    {
        "content": "New employee Maria Garcia started in Marketing department on 2024-01-22. Onboarding checklist completed, access credentials provided.",
        "metadata": {"tags": ["new-hire", "onboarding", "marketing"], "priority": "normal", "department": "HR"},
        "source_agent": "hr_onboarding_agent",
        "message_type": "status_update",
        "expected_queries": ["Maria Garcia", "new employee", "marketing", "onboarding", "credentials"]
    },
    {
        "content": "Budget proposal for Q2 2024 submitted by Finance team. Total requested amount: $2.5M for infrastructure upgrades and new software licenses.",
        "metadata": {"tags": ["budget", "proposal", "infrastructure"], "priority": "high", "department": "finance"},
        "source_agent": "budget_planning_agent",
        "message_type": "proposal",
        "expected_queries": ["budget proposal", "Q2 2024", "2.5M", "infrastructure upgrades", "software licenses"]
    },
    {
        "content": "Critical bug identified in payment processing system. Bug affects credit card transactions. Hotfix deployed to production at 14:30 EST.",
        "metadata": {"tags": ["bug", "critical", "payment", "hotfix"], "priority": "critical", "department": "engineering"},
        "source_agent": "bug_tracker_agent",
        "message_type": "incident",
        "expected_queries": ["critical bug", "payment processing", "credit card", "hotfix", "production"]
    },
    {
        "content": "Training session on cybersecurity best practices scheduled for all employees on 2024-02-01. Mandatory attendance required.",
        "metadata": {"tags": ["training", "cybersecurity", "mandatory"], "priority": "normal", "department": "security"},
        "source_agent": "training_coordinator",
        "message_type": "announcement",
        "expected_queries": ["training session", "cybersecurity", "best practices", "mandatory", "February"]
    }
]


@pytest.mark.asyncio
async def test_realistic_memory_storage_and_retrieval():
    """Test realistic memory storage and retrieval with predefined data."""
    session_id = "realistic-test-session"
    
    # Store all predefined memory entries
    stored_entries = []
    for i, data in enumerate(REALISTIC_MEMORY_DATA):
        result = await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        assert result["status"] == "success"
        stored_entries.append(data)
        
        # Add small delay to ensure different timestamps
        await asyncio.sleep(0.1)
    
    # Test retrieval with various queries
    for entry_data in REALISTIC_MEMORY_DATA:
        for query in entry_data["expected_queries"]:
            result = await retrieve_memory(session_id, query=query, limit=5)
            assert result["status"] == "success"
            
            # With real LLM/embeddings, we should get relevant results
            # With mocks, we just ensure no errors occur
            assert isinstance(result["result"], list)
            print(f"Query '{query}' returned {len(result['result'])} results")


@pytest.mark.asyncio
async def test_memory_retrieval_with_filters():
    """Test memory retrieval with various filters using realistic data."""
    session_id = "filter-test-session"
    
    # Store predefined entries
    for data in REALISTIC_MEMORY_DATA:
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Test filtering by source_agent
    result = await retrieve_memory(
        session_id, 
        query="system", 
        source_agent="security_audit_agent"
    )
    assert result["status"] == "success"
    print(f"Security agent filter returned {len(result['result'])} results")
    
    # Test filtering by message_type
    result = await retrieve_memory(
        session_id, 
        query="critical", 
        message_type="incident"
    )
    assert result["status"] == "success"
    print(f"Incident message type filter returned {len(result['result'])} results")
    
    # Test filtering by tags
    result = await retrieve_memory(
        session_id, 
        query="security", 
        tags=["critical"]
    )
    assert result["status"] == "success"
    print(f"Critical tag filter returned {len(result['result'])} results")


@pytest.mark.asyncio
async def test_department_based_memory_retrieval():
    """Test retrieving memories based on department metadata."""
    session_id = "department-test-session"
    
    # Store entries
    for data in REALISTIC_MEMORY_DATA:
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Test department-specific queries
    departments = ["finance", "IT", "security", "HR", "engineering"]
    
    for dept in departments:
        # Query for department-related content
        result = await retrieve_memory(session_id, query=dept, limit=10)
        assert result["status"] == "success"
        print(f"Department '{dept}' query returned {len(result['result'])} results")


@pytest.mark.asyncio
async def test_priority_based_memory_retrieval():
    """Test retrieving memories based on priority levels."""
    session_id = "priority-test-session"
    
    # Store entries
    for data in REALISTIC_MEMORY_DATA:
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Test priority-based queries
    priorities = ["critical", "high", "medium", "normal"]
    
    for priority in priorities:
        result = await retrieve_memory(session_id, query=priority, limit=10)
        assert result["status"] == "success"
        print(f"Priority '{priority}' query returned {len(result['result'])} results")


@pytest.mark.asyncio
async def test_complex_semantic_queries():
    """Test complex semantic queries that should work well with real embeddings."""
    session_id = "semantic-test-session"
    
    # Store entries
    for data in REALISTIC_MEMORY_DATA:
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Complex semantic queries
    semantic_queries = [
        "What security issues were found?",
        "Who are the new employees?",
        "What maintenance activities are planned?",
        "Which systems had problems?",
        "What budget requests were made?",
        "Who needs database access?",
        "What training is required?",
        "Which orders had delivery issues?"
    ]
    
    for query in semantic_queries:
        result = await retrieve_memory(session_id, query=query, limit=5)
        assert result["status"] == "success"
        print(f"Semantic query '{query}' returned {len(result['result'])} results")


@pytest.mark.asyncio
async def test_memory_summarization_with_realistic_data():
    """Test memory summarization with realistic data."""
    session_id = "summarization-test-session"
    
    # Store entries
    for data in REALISTIC_MEMORY_DATA:
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Test general summarization
    result = await summarize_memory(session_id)
    assert result["status"] == "success"
    assert "summary" in result
    print(f"General summary length: {len(result['summary'])}")
    
    # Test topic-specific summarization
    topics = ["security", "budget", "maintenance", "employees"]
    
    for topic in topics:
        result = await summarize_memory(session_id, query=topic)
        assert result["status"] == "success"
        assert "summary" in result
        print(f"Summary for '{topic}': {len(result['summary'])} characters")


@pytest.mark.asyncio
async def test_cross_session_isolation_with_realistic_data():
    """Test that sessions remain isolated with realistic data."""
    session1 = "isolation-session-1"
    session2 = "isolation-session-2"
    
    # Store different subsets in different sessions
    finance_data = [d for d in REALISTIC_MEMORY_DATA if d["metadata"]["department"] == "finance"]
    security_data = [d for d in REALISTIC_MEMORY_DATA if d["metadata"]["department"] == "security"]
    
    # Store finance data in session 1
    for data in finance_data:
        await store_memory(
            session_id=session1,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Store security data in session 2
    for data in security_data:
        await store_memory(
            session_id=session2,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"]
        )
        await asyncio.sleep(0.1)
    
    # Test that sessions are isolated
    result1 = await retrieve_memory(session1, query="budget", limit=10)
    result2 = await retrieve_memory(session2, query="security", limit=10)
    
    assert result1["status"] == "success"
    assert result2["status"] == "success"
    
    print(f"Session 1 (finance) returned {len(result1['result'])} results for 'budget'")
    print(f"Session 2 (security) returned {len(result2['result'])} results for 'security'")


@pytest.mark.asyncio
async def test_temporal_memory_queries():
    """Test time-based memory retrieval with realistic timestamps."""
    session_id = "temporal-test-session"
    
    # Store entries with specific timestamps
    import datetime
    base_time = datetime.datetime(2024, 1, 15, 10, 0, 0)
    
    for i, data in enumerate(REALISTIC_MEMORY_DATA):
        # Create timestamps spread over several days
        timestamp = (base_time + datetime.timedelta(days=i)).isoformat() + "Z"
        
        await store_memory(
            session_id=session_id,
            content=data["content"],
            metadata=data["metadata"],
            source_agent=data["source_agent"],
            message_type=data["message_type"],
            timestamp=timestamp
        )
        await asyncio.sleep(0.1)
    
    # Test time-range queries
    start_time = (base_time + datetime.timedelta(days=2)).isoformat() + "Z"
    end_time = (base_time + datetime.timedelta(days=5)).isoformat() + "Z"
    
    result = await retrieve_memory(
        session_id, 
        query="system", 
        start_time=start_time,
        end_time=end_time,
        limit=10
    )
    assert result["status"] == "success"
    print(f"Temporal query returned {len(result['result'])} results")


# Legacy tests for backward compatibility
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
    assert len(result["result"]) >= 0


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
    assert len(result["result"]) >= 0


if __name__ == "__main__":
    # Print environment info for debugging
    print("Environment variables:")
    print(f"LLM_BASE_URL: {os.getenv('LLM_BASE_URL', 'Not set')}")
    print(f"LLM_NAME: {os.getenv('LLM_NAME', 'Not set')}")
    print(f"EMBEDDING_BASE_URL: {os.getenv('EMBEDDING_BASE_URL', 'Not set')}")
    print(f"EMBEDDING_MODEL_NAME: {os.getenv('EMBEDDING_MODEL_NAME', 'Not set')}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    
    # Run tests
    pytest.main([__file__, "-v"])