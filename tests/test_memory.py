"""Tests for the mcp-mem package."""

import unittest
from unittest.mock import patch, MagicMock
import datetime
import json
import os
import sys
import pytest

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_mem.server import create_memory, store_memory, retrieve_memory


class TestMemoryTools(unittest.TestCase):
    """Test the memory tools functionality."""

    @patch('mcp_mem.server.datetime')
    @patch('mcp_mem.server.save_session_state')
    @patch('mcp_mem.server.ensure_session_exists')
    @patch('mcp_mem.server.initialize_hipporag')
    async def test_create_memory(self, mock_initialize_hipporag, mock_ensure_session_exists, 
                                mock_save_session_state, mock_datetime):
        """Test that the create_memory tool correctly creates a new memory session."""
        # Setup mock datetime
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2025-04-18T22:00:00"
        mock_datetime.datetime.now.return_value = mock_now
        
        # Mock HippoRAG initialization
        mock_hipporag = MagicMock()
        mock_initialize_hipporag.return_value = mock_hipporag
        
        # Call the create_memory function
        session_id = "test-session-123"
        result = await create_memory(session_id)
        
        # Check the result
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["status"], "created")
        
        # Check that the session was created
        from mcp_mem.server import session_memories
        self.assertIn(session_id, session_memories)
        self.assertEqual(session_memories[session_id]["created_at"], "2025-04-18T22:00:00")
        self.assertEqual(session_memories[session_id]["hipporag"], mock_hipporag)
        self.assertEqual(session_memories[session_id]["memories"], [])
        
        # Check that the necessary functions were called
        mock_ensure_session_exists.assert_called_once_with(session_id)
        mock_initialize_hipporag.assert_called_once_with(session_id)
        mock_save_session_state.assert_called_once_with(session_id)

    @patch('mcp_mem.server.datetime')
    @patch('mcp_mem.server.save_session_state')
    @patch('uuid.uuid4')
    async def test_store_memory(self, mock_uuid4, mock_save_session_state, mock_datetime):
        """Test that the store_memory tool correctly stores memory in a session."""
        # Setup mock datetime and UUID
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2025-04-18T22:01:00"
        mock_datetime.datetime.now.return_value = mock_now
        
        mock_uuid = MagicMock()
        mock_uuid.return_value = "test-memory-456"
        mock_uuid4.return_value = mock_uuid
        mock_uuid.__str__.return_value = "test-memory-456"
        
        # Setup test session
        from mcp_mem.server import session_memories
        session_id = "test-session-123"
        mock_hipporag = MagicMock()
        session_memories[session_id] = {
            "hipporag": mock_hipporag,
            "memories": [],
            "created_at": "2025-04-18T22:00:00"
        }
        
        # Call the store_memory function
        content = "This is a test memory"
        metadata = {"source": "test", "importance": "high"}
        result = await store_memory(session_id, content, metadata)
        
        # Check the result
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["memory_id"], "test-memory-456")
        self.assertEqual(result["status"], "stored")
        
        # Check that the memory was stored
        self.assertEqual(len(session_memories[session_id]["memories"]), 1)
        stored_memory = session_memories[session_id]["memories"][0]
        self.assertEqual(stored_memory["id"], "test-memory-456")
        self.assertEqual(stored_memory["content"], content)
        self.assertEqual(stored_memory["timestamp"], "2025-04-18T22:01:00")
        self.assertEqual(stored_memory["metadata"], metadata)
        
        # Check that HippoRAG index was called
        mock_hipporag.index.assert_called_once_with([content])
        
        # Check that save_session_state was called
        mock_save_session_state.assert_called_once_with(session_id)

    @patch('mcp_mem.server.datetime')
    async def test_retrieve_memory_basic(self, mock_datetime):
        """Test that the retrieve_memory tool correctly retrieves memories from a session."""
        # Setup test session with memories
        from mcp_mem.server import session_memories
        session_id = "test-session-123"
        mock_hipporag = MagicMock()
        
        # Create test memories with different timestamps
        memories = [
            {
                "id": f"memory-{i}",
                "content": f"Test memory {i}",
                "timestamp": f"2025-04-18T22:{i:02d}:00",
                "metadata": {"index": i}
            }
            for i in range(5)
        ]
        
        session_memories[session_id] = {
            "hipporag": mock_hipporag,
            "memories": memories,
            "created_at": "2025-04-18T22:00:00"
        }
        
        # Call the retrieve_memory function without a query
        result = await retrieve_memory(session_id, limit=3)
        
        # Check the result
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["memories"]), 3)
        
        # Memories should be sorted by timestamp (most recent first)
        self.assertEqual(result["memories"][0]["id"], "memory-4")
        self.assertEqual(result["memories"][1]["id"], "memory-3")
        self.assertEqual(result["memories"][2]["id"], "memory-2")
        
        # Test with a query
        result = await retrieve_memory(session_id, query="memory 1")
        
        # Check that the query filtered correctly
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["id"], "memory-1")


if __name__ == "__main__":
    unittest.main()