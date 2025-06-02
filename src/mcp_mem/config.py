"""Configuration module for mcp-mem (LightRAG only)."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import lightrag.utils as lightrag_utils


@dataclass
class MemoryConfig:
    """Configuration settings for the MCP Memory server (LightRAG only)."""
    
    # Base directory for storing memory data
    memory_dir: str = os.path.expanduser("~/.mcp-mem")
    
    # Maximum number of memories to return in retrieve_memory (not enforced by LightRAG)
    default_retrieve_limit: Optional[int] = None
    
    # Session cleanup settings
    session_ttl_days: Optional[int] = None  # None means no automatic cleanup
    
    # Instance TTL settings (in minutes)
    # After this period of inactivity, hipporag instances will be offloaded from memory
    instance_ttl_minutes: int = 30  # Default 30 minutes
    
    def __post_init__(self):
        """Ensure memory directory exists and set up default HippoRAG config."""
        os.makedirs(self.memory_dir, exist_ok=True)


# Default configuration instance
default_config = MemoryConfig()


def get_config() -> MemoryConfig:
    """Get the current configuration."""
    return default_config


def update_config(config_updates: Dict[str, Any]) -> MemoryConfig:
    """Update the configuration with new values."""
    global default_config
    
    for key, value in config_updates.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    return default_config