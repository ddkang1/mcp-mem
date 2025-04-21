"""Tests for the memory_utils module."""

import os
import json
import datetime
from unittest.mock import patch, MagicMock
import pytest

from mcp_mem.memory_utils import (
    cleanup_old_sessions
)


class TestMemoryUtils:
    """Test the memory utilities."""



    def test_cleanup_old_sessions(self, test_config, tmp_path):
        """Test cleaning up old sessions."""
        # Use real datetime instead of mocks to avoid comparison issues
        current_time = datetime.datetime.now()
        old_time = current_time - datetime.timedelta(days=10)
        very_old_time = current_time - datetime.timedelta(days=30)
        recent_time = current_time - datetime.timedelta(days=2)
        
        # Create test session directories with different ages
        sessions = [
            # Recent session (2 days old)
            {
                "id": "recent-session",
                "created_at": recent_time.isoformat(),
            },
            # Old session (10 days old)
            {
                "id": "old-session",
                "created_at": old_time.isoformat(),
            },
            # Very old session (30 days old)
            {
                "id": "very-old-session",
                "created_at": very_old_time.isoformat(),
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
