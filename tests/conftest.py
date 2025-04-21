"""Pytest configuration and fixtures for mcp-mem tests."""

import os
import shutil
import tempfile
import pytest
import sys
from pathlib import Path

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_mem.config import MemoryConfig, update_config


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage during tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_memory_dir):
    """Create a test configuration with a temporary memory directory."""
    original_config = MemoryConfig()
    
    # Update config with test settings
    test_config = update_config({
        "memory_dir": temp_memory_dir,
        "default_retrieve_limit": 5,
        "default_metadata": {"test": True},
        "session_ttl_days": None
    })
    
    yield test_config
    
    # Restore original config after test
    update_config({
        "memory_dir": original_config.memory_dir,
        "default_retrieve_limit": original_config.default_retrieve_limit,
        "default_metadata": original_config.default_metadata,
        "session_ttl_days": original_config.session_ttl_days
    })


@pytest.fixture
def mock_session():
    """Create a mock session for testing."""
    return {
        "session_id": "test-session-123",
        "memories": [
            {
                "id": "mem1",
                "content": "This is test memory 1",
                "timestamp": "2025-04-18T10:00:00",
                "metadata": {"type": "test", "priority": "high"}
            },
            {
                "id": "mem2",
                "content": "This is test memory 2",
                "timestamp": "2025-04-18T11:00:00",
                "metadata": {"type": "test", "priority": "medium"}
            },
            {
                "id": "mem3",
                "content": "This is another test memory",
                "timestamp": "2025-04-18T12:00:00",
                "metadata": {"type": "test", "priority": "low"}
            }
        ]
    }