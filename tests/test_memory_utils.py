"""Tests for the memory_utils module."""

import os
import json
import datetime
from unittest.mock import patch, MagicMock
import pytest

from mcp_mem.memory_utils import (
    get_session_memories,
    search_memories,
    cleanup_old_sessions,
    get_memory_stats
)


class TestMemoryUtils:
    """Test the memory utilities."""

    def test_get_session_memories_nonexistent(self, test_config):
        """Test getting memories from a nonexistent session."""
        memories = get_session_memories("nonexistent-session")
        assert memories == []

    def test_get_session_memories_existing(self, test_config, tmp_path):
        """Test getting memories from an existing session."""
        # Create a test session directory and state file
        session_id = "test-session"
        session_dir = os.path.join(test_config.memory_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        test_memories = [
            {
                "id": "mem1",
                "content": "Test memory 1",
                "timestamp": "2025-04-18T10:00:00",
                "metadata": {"test": True}
            },
            {
                "id": "mem2",
                "content": "Test memory 2",
                "timestamp": "2025-04-18T11:00:00",
                "metadata": {"test": True}
            }
        ]
        
        state_data = {
            "memories": test_memories,
            "created_at": "2025-04-18T09:00:00"
        }
        
        with open(os.path.join(session_dir, "session_state.json"), "w") as f:
            json.dump(state_data, f)
        
        # Test retrieving the memories
        memories = get_session_memories(session_id)
        assert len(memories) == 2
        assert memories[0]["id"] == "mem1"
        assert memories[1]["id"] == "mem2"

    def test_search_memories_by_content(self, test_config, tmp_path):
        """Test searching memories by content."""
        # Create a test session directory and state file
        session_id = "test-session"
        session_dir = os.path.join(test_config.memory_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        test_memories = [
            {
                "id": "mem1",
                "content": "Apple is a fruit",
                "timestamp": "2025-04-18T10:00:00",
                "metadata": {"category": "fruits"}
            },
            {
                "id": "mem2",
                "content": "Banana is yellow",
                "timestamp": "2025-04-18T11:00:00",
                "metadata": {"category": "fruits"}
            },
            {
                "id": "mem3",
                "content": "Carrot is orange",
                "timestamp": "2025-04-18T12:00:00",
                "metadata": {"category": "vegetables"}
            }
        ]
        
        state_data = {
            "memories": test_memories,
            "created_at": "2025-04-18T09:00:00"
        }
        
        with open(os.path.join(session_dir, "session_state.json"), "w") as f:
            json.dump(state_data, f)
        
        # Test searching by content
        results = search_memories(session_id, query="apple")
        assert len(results) == 1
        assert results[0]["id"] == "mem1"
        
        results = search_memories(session_id, query="is")
        assert len(results) == 3
        # Should be sorted by timestamp (most recent first)
        assert results[0]["id"] == "mem3"
        assert results[1]["id"] == "mem2"
        assert results[2]["id"] == "mem1"

    def test_search_memories_by_metadata(self, test_config, tmp_path):
        """Test searching memories by metadata."""
        # Create a test session directory and state file
        session_id = "test-session"
        session_dir = os.path.join(test_config.memory_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        test_memories = [
            {
                "id": "mem1",
                "content": "Apple is a fruit",
                "timestamp": "2025-04-18T10:00:00",
                "metadata": {"category": "fruits", "color": "red"}
            },
            {
                "id": "mem2",
                "content": "Banana is yellow",
                "timestamp": "2025-04-18T11:00:00",
                "metadata": {"category": "fruits", "color": "yellow"}
            },
            {
                "id": "mem3",
                "content": "Carrot is orange",
                "timestamp": "2025-04-18T12:00:00",
                "metadata": {"category": "vegetables", "color": "orange"}
            }
        ]
        
        state_data = {
            "memories": test_memories,
            "created_at": "2025-04-18T09:00:00"
        }
        
        with open(os.path.join(session_dir, "session_state.json"), "w") as f:
            json.dump(state_data, f)
        
        # Test searching by metadata
        results = search_memories(session_id, metadata_filters={"category": "fruits"})
        assert len(results) == 2
        assert results[0]["id"] == "mem2"  # Most recent first
        assert results[1]["id"] == "mem1"
        
        results = search_memories(session_id, metadata_filters={"color": "red"})
        assert len(results) == 1
        assert results[0]["id"] == "mem1"
        
        # Test combined content and metadata search
        results = search_memories(
            session_id, 
            query="is", 
            metadata_filters={"category": "vegetables"}
        )
        assert len(results) == 1
        assert results[0]["id"] == "mem3"

    @patch("mcp_mem.memory_utils.datetime")
    def test_cleanup_old_sessions(self, mock_datetime, test_config, tmp_path):
        """Test cleaning up old sessions."""
        # Mock the current date
        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now
        mock_now.isoformat.return_value = "2025-04-20T00:00:00"
        mock_datetime.fromisoformat.side_effect = lambda x: datetime.datetime.fromisoformat(x)
        
        # Create test session directories with different ages
        sessions = [
            # Recent session (2 days old)
            {
                "id": "recent-session",
                "created_at": "2025-04-18T00:00:00",
            },
            # Old session (10 days old)
            {
                "id": "old-session",
                "created_at": "2025-04-10T00:00:00",
            },
            # Very old session (30 days old)
            {
                "id": "very-old-session",
                "created_at": "2025-03-21T00:00:00",
            }
        ]
        
        for session in sessions:
            session_dir = os.path.join(test_config.memory_dir, f"session_{session['id']}")
            os.makedirs(session_dir, exist_ok=True)
            
            state_data = {
                "memories": [],
                "created_at": session["created_at"]
            }
            
            with open(os.path.join(session_dir, "session_state.json"), "w") as f:
                json.dump(state_data, f)
        
        # Test cleanup with 7-day TTL
        removed = cleanup_old_sessions(max_age_days=7)
        assert removed == 2  # Should remove old-session and very-old-session
        
        # Check that only the recent session remains
        assert os.path.exists(os.path.join(test_config.memory_dir, "session_recent-session"))
        assert not os.path.exists(os.path.join(test_config.memory_dir, "session_old-session"))
        assert not os.path.exists(os.path.join(test_config.memory_dir, "session_very-old-session"))

    def test_get_memory_stats(self, test_config, tmp_path):
        """Test getting memory statistics."""
        # Create test session directories with memories
        sessions = [
            {
                "id": "session1",
                "created_at": "2025-04-10T00:00:00",
                "memories": [{"id": "mem1"}, {"id": "mem2"}]
            },
            {
                "id": "session2",
                "created_at": "2025-04-15T00:00:00",
                "memories": [{"id": "mem3"}]
            }
        ]
        
        for session in sessions:
            session_dir = os.path.join(test_config.memory_dir, f"session_{session['id']}")
            os.makedirs(session_dir, exist_ok=True)
            
            state_data = {
                "memories": session["memories"],
                "created_at": session["created_at"]
            }
            
            with open(os.path.join(session_dir, "session_state.json"), "w") as f:
                json.dump(state_data, f)
        
        # Test getting stats
        stats = get_memory_stats()
        assert stats["session_count"] == 2
        assert stats["total_memories"] == 3
        assert stats["oldest_session"] == "2025-04-10T00:00:00"
        assert stats["newest_session"] == "2025-04-15T00:00:00"
        assert stats["avg_memories_per_session"] == 1.5