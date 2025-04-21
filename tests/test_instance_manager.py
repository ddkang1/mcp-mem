"""Tests for the HippoRAG instance manager."""

import unittest
import time
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from mcp_mem.instance_manager import HippoRAGInstanceManager
from mcp_mem.config import MemoryConfig, update_config


class TestHippoRAGInstanceManager(unittest.TestCase):
    """Test cases for the HippoRAGInstanceManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a short TTL for testing
        update_config({"instance_ttl_minutes": 1})
        
        # Create a mock HippoRAG class
        self.mock_hipporag = MagicMock()
        
        # Patch the cleanup loop to prevent background thread from running
        with patch('mcp_mem.instance_manager.threading.Thread'):
            self.manager = HippoRAGInstanceManager()
            # Manually set running to False to prevent cleanup thread
            self.manager._running = False

    def test_add_instance(self):
        """Test adding instances."""
        # Add an instance
        self.manager.add("session1", self.mock_hipporag)
        
        # Verify it was added
        self.assertTrue(self.manager.contains("session1"))
        self.assertEqual(self.manager._instances["session1"], self.mock_hipporag)
    
    @patch('mcp_mem.instance_manager.HippoRAGInstanceManager.create_memory')
    def test_get_existing_instance(self, mock_create_memory):
        """Test retrieving an existing instance."""
        # Add an instance
        self.manager.add("session1", self.mock_hipporag)
        
        # Verify it can be retrieved without calling create_memory
        import asyncio
        instance = asyncio.run(self.manager.get("session1"))
        self.assertEqual(instance, self.mock_hipporag)
        mock_create_memory.assert_not_called()
    
    @patch('mcp_mem.instance_manager.HippoRAGInstanceManager.create_memory')
    def test_get_nonexistent_instance(self, mock_create_memory):
        """Test retrieving a non-existent instance creates it."""
        # Setup the mock to do nothing (we'll test create_memory separately)
        mock_create_memory.return_value = None
        
        # Call get on a non-existent session
        import asyncio
        asyncio.run(self.manager.get("nonexistent"))
        
        # Verify create_memory was called
        mock_create_memory.assert_called_once_with("nonexistent")

    def test_contains_and_remove(self):
        """Test contains and remove functionality."""
        # Add an instance
        self.manager.add("session1", self.mock_hipporag)
        
        # Verify contains works
        self.assertTrue(self.manager.contains("session1"))
        self.assertFalse(self.manager.contains("nonexistent"))
        
        # Remove the instance
        self.manager.remove("session1")
        
        # Verify it's gone
        self.assertFalse(self.manager.contains("session1"))

    def test_get_all_session_ids(self):
        """Test getting all session IDs."""
        # Add some instances
        self.manager.add("session1", self.mock_hipporag)
        self.manager.add("session2", self.mock_hipporag)
        
        # Verify all sessions are returned
        session_ids = self.manager.get_all_session_ids()
        self.assertEqual(session_ids, {"session1", "session2"})

    def test_cleanup_expired_instances(self):
        """Test cleanup of expired instances."""
        # Add some instances
        self.manager.add("session1", self.mock_hipporag)
        self.manager.add("session2", self.mock_hipporag)
        
        # Manually set last_accessed time for session1 to be expired
        self.manager._last_accessed["session1"] = datetime.now() - timedelta(minutes=2)
        
        # Run cleanup
        self.manager._cleanup_expired_instances()
        
        # Verify session1 was removed but session2 remains
        self.assertFalse(self.manager.contains("session1"))
        self.assertTrue(self.manager.contains("session2"))

    def test_ttl_disabled(self):
        """Test that cleanup doesn't happen when TTL is disabled."""
        # Set TTL to 0 (disabled)
        update_config({"instance_ttl_minutes": 0})
        
        # Add an instance
        self.manager.add("session1", self.mock_hipporag)
        
        # Manually set last_accessed time to be expired
        self.manager._last_accessed["session1"] = datetime.now() - timedelta(minutes=10)
        
        # Run cleanup
        self.manager._cleanup_expired_instances()
        
        # Verify session1 still exists
        self.assertTrue(self.manager.contains("session1"))

    @pytest.mark.asyncio
    async def test_access_updates_timestamp(self):
        """Test that accessing an instance updates its timestamp."""
        # Add an instance
        self.manager.add("session1", self.mock_hipporag)
        
        # Get the initial timestamp
        initial_timestamp = self.manager._last_accessed["session1"]
        
        # Wait a moment
        time.sleep(0.1)
        
        # Access the instance
        await self.manager.get("session1")
        
        # Verify timestamp was updated
        self.assertGreater(self.manager._last_accessed["session1"], initial_timestamp)

    def test_shutdown(self):
        """Test manager shutdown."""
        # Create a manager with a mocked thread
        with patch('mcp_mem.instance_manager.threading.Thread') as mock_thread:
            manager = HippoRAGInstanceManager()
            mock_thread_instance = mock_thread.return_value
            
            # Call shutdown
            manager.shutdown()
            
            # Verify running flag is set to False
            self.assertFalse(manager._running)
            
            # Verify join was called on the thread
            mock_thread_instance.join.assert_called_once()


    @patch('mcp_mem.instance_manager.os')
    @patch('mcp_mem.instance_manager.HippoRAG')
    def test_create_memory(self, mock_hipporag_class, mock_os):
        """Test the create_memory method."""
        # Setup
        mock_hipporag_instance = MagicMock()
        mock_hipporag_class.return_value = mock_hipporag_instance
        
        # Create a manager with mocked methods
        manager = HippoRAGInstanceManager()
        manager._ensure_session_exists = MagicMock()
        manager._initialize_hipporag = MagicMock(return_value=mock_hipporag_instance)
        
        # Call create_memory synchronously for testing
        import asyncio
        asyncio.run(manager.create_memory("test-session"))
        
        # Verify the session was created
        self.assertTrue(manager.contains("test-session"))
        manager._ensure_session_exists.assert_called_once_with("test-session")
        manager._initialize_hipporag.assert_called_once_with("test-session")
    
    def test_get_session_path(self):
        """Test the _get_session_path method."""
        # Setup
        manager = HippoRAGInstanceManager()
        manager._config.memory_dir = "/tmp/memory"
        
        # Test
        path = manager._get_session_path("test-session")
        
        # Verify
        self.assertEqual(path, "/tmp/memory/session_test-session")
    
    @patch('mcp_mem.instance_manager.os')
    def test_ensure_session_exists(self, mock_os):
        """Test the _ensure_session_exists method."""
        # Setup
        manager = HippoRAGInstanceManager()
        manager._get_session_path = MagicMock(return_value="/tmp/memory/session_test")
        
        # Reset the mock to clear the call from __init__
        mock_os.makedirs.reset_mock()
        
        # Test
        manager._ensure_session_exists("test-session")
        
        # Verify
        mock_os.makedirs.assert_called_once_with("/tmp/memory/session_test", exist_ok=True)
    
    @patch('mcp_mem.instance_manager.BaseConfig')
    @patch('mcp_mem.instance_manager.HippoRAG')
    def test_initialize_hipporag(self, mock_hipporag_class, mock_base_config):
        """Test the _initialize_hipporag method."""
        # Setup
        mock_config = MagicMock()
        mock_base_config.return_value = mock_config
        mock_hipporag_instance = MagicMock()
        mock_hipporag_class.return_value = mock_hipporag_instance
        
        manager = HippoRAGInstanceManager()
        manager._get_session_path = MagicMock(return_value="/tmp/memory/session_test")
        manager._config.hipporag_config = {"model_name": "gpt-4", "embedding_model": "text-embedding-3-large"}
        
        # Test
        result = manager._initialize_hipporag("test-session")
        
        # Verify
        self.assertEqual(result, mock_hipporag_instance)
        self.assertEqual(mock_config.save_dir, "/tmp/memory/session_test")
        mock_hipporag_class.assert_called_once_with(
            global_config=mock_config,
            save_dir="/tmp/memory/session_test"
        )


if __name__ == '__main__':
    unittest.main()