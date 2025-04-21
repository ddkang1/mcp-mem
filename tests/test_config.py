"""Tests for the config module."""

import os
import pytest
from mcp_mem.config import MemoryConfig, get_config, update_config


class TestConfig:
    """Test the configuration functionality."""

    def test_default_config(self):
        """Test that the default configuration has expected values."""
        config = get_config()
        
        assert config.memory_dir == os.path.expanduser("~/.mcp-mem")
        assert config.default_retrieve_limit == 10
        assert isinstance(config.hipporag_config, dict)
        assert isinstance(config.default_metadata, dict)
        assert config.session_ttl_days is None

    def test_update_config(self):
        """Test updating the configuration."""
        original_config = get_config()
        
        # Save original values to restore later
        original_memory_dir = original_config.memory_dir
        original_retrieve_limit = original_config.default_retrieve_limit
        
        # Update config
        updated_config = update_config({
            "memory_dir": "/tmp/test-mcp-mem",
            "default_retrieve_limit": 20
        })
        
        # Check that the config was updated
        assert updated_config.memory_dir == "/tmp/test-mcp-mem"
        assert updated_config.default_retrieve_limit == 20
        
        # Check that get_config returns the updated config
        current_config = get_config()
        assert current_config.memory_dir == "/tmp/test-mcp-mem"
        assert current_config.default_retrieve_limit == 20
        
        # Restore original config
        update_config({
            "memory_dir": original_memory_dir,
            "default_retrieve_limit": original_retrieve_limit
        })

    def test_memory_config_post_init(self, tmp_path):
        """Test that the post_init method creates the memory directory."""
        test_dir = os.path.join(tmp_path, "test-mcp-mem")
        
        # Directory should not exist yet
        assert not os.path.exists(test_dir)
        
        # Create config with the test directory
        config = MemoryConfig(memory_dir=test_dir)
        
        # Directory should now exist
        assert os.path.exists(test_dir)
        
        # Default HippoRAG config should be set
        assert "embedding_model_name" in config.hipporag_config
        assert "llm_name" in config.hipporag_config

    def test_invalid_config_update(self):
        """Test that updating with invalid keys doesn't change the config."""
        original_config = get_config()
        
        # Try to update with an invalid key
        updated_config = update_config({
            "invalid_key": "some value"
        })
        
        # Config should be unchanged
        assert not hasattr(updated_config, "invalid_key")
        assert updated_config == original_config