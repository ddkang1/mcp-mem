"""Tests for the mcp-mem package."""

import unittest
from unittest.mock import patch, MagicMock
import datetime
import json
import os
import sys
import pytest
import pytest_asyncio

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_mem.server import store_memory, retrieve_memory
from mcp_mem.instance_manager import HippoRAGInstanceManager

# Helper class for async mocks
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@pytest.mark.asyncio
class TestMemoryTools:
    """Test the memory tools functionality."""

    @patch('mcp_mem.instance_manager.datetime')
    @patch('mcp_mem.instance_manager.HippoRAG')
    async def test_create_memory(self, mock_hipporag_class, mock_datetime):
        """Test that the create_memory method correctly creates a new memory session."""
        # Setup mock datetime
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2025-04-18T22:00:00"
        mock_datetime.datetime.now.return_value = mock_now
        
        # Mock HippoRAG initialization
        mock_hipporag = MagicMock()
        mock_hipporag_class.return_value = mock_hipporag
        
        # Create instance manager with mocked methods
        manager = HippoRAGInstanceManager()
        manager._ensure_session_exists = MagicMock()
        
        # Call the create_memory method
        session_id = "test-session-123"
        await manager.create_memory(session_id)
        
        # Check that the session was created
        assert manager.contains(session_id)
        
        # Check that the necessary methods were called
        manager._ensure_session_exists.assert_called_once_with(session_id)

    @patch('mcp_mem.server.datetime')
    @patch('mcp_mem.server.hipporag_manager')
    async def test_store_memory(self, mock_manager, mock_datetime):
        """Test that the store_memory tool correctly stores memory in a session."""
        # Setup mock datetime
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2025-04-18T22:01:00"
        mock_datetime.datetime.now.return_value = mock_now
        
        # Setup test session
        session_id = "test-session-123"
        mock_hipporag = MagicMock()
        mock_manager.get = AsyncMock(return_value=mock_hipporag)
        
        # Call the store_memory function with explicit memory_id
        content = "This is a test memory"
        result = await store_memory(session_id, content)
        
        # Check the result
        assert result["session_id"] == session_id
        assert result["status"] == "success"
        
        # Check that the memory was stored in HippoRAG
        mock_hipporag.index.assert_called_once_with([content])
        
        # Check that HippoRAG index was called
        mock_hipporag.index.assert_called_once_with([content])

    @patch('mcp_mem.server.datetime')
    @patch('mcp_mem.server.hipporag_manager')
    async def test_retrieve_memory_basic(self, mock_manager, mock_datetime):
        """Test that the retrieve_memory tool correctly retrieves memories from a session."""
        # Setup test session with memories
        session_id = "test-session-123"
        mock_hipporag = MagicMock()
        mock_manager.get = AsyncMock(return_value=mock_hipporag)
        
        # Test retrieving memory without a query
        no_query_result = MagicMock()
        no_query_result.docs = ["Test memory 4", "Test memory 3", "Test memory 2"]
        no_query_result.doc_scores = [0.98, 0.96, 0.94]
        mock_hipporag.retrieve.return_value = [no_query_result]
        
        # Call the retrieve_memory function with empty query
        result = await retrieve_memory(session_id, query="", limit=3)
        
        # Check that HippoRAG was used for retrieval with empty string
        mock_hipporag.retrieve.assert_called_once_with([""], num_to_retrieve=3)
        
        # Check the result
        assert result["session_id"] == session_id
        assert result["status"] == "success"
        assert len(result["memories"]) == 3
        
        # Check the retrieved memories
        assert result["memories"][0]["content"] == "Test memory 4"
        assert result["memories"][0]["score"] == 0.98
        assert result["memories"][0]["rank"] == 1
        
        # Reset the mock
        mock_hipporag.retrieve.reset_mock()
        
        # Test with a query - mock HippoRAG retrieve method
        with_query_result = MagicMock()
        with_query_result.docs = ["Test memory 1"]
        with_query_result.doc_scores = [0.95]
        mock_hipporag.retrieve.return_value = [with_query_result]
        
        result = await retrieve_memory(session_id, query="memory 1", limit=10)
        
        # Check that HippoRAG was used for retrieval
        mock_hipporag.retrieve.assert_called_once_with(["memory 1"], num_to_retrieve=10)
        
        # Check the result format
        assert result["session_id"] == session_id
        assert result["status"] == "success"
        assert result["query"] == "memory 1"
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "Test memory 1"
        assert result["memories"][0]["score"] == 0.95
        assert result["memories"][0]["rank"] == 1

