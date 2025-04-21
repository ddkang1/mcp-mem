"""Configuration module for mcp-mem."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MemoryConfig:
    """Configuration settings for the MCP Memory server."""
    
    # Base directory for storing memory data
    memory_dir: str = os.path.expanduser("~/.mcp-mem")
    
    # Maximum number of memories to return in retrieve_memory
    # default_retrieve_limit: int = 10
    
    # HippoRAG configuration
    hipporag_config: Dict[str, Any] = field(default_factory=dict)
    
    # Default metadata to include with all memories
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Session cleanup settings
    session_ttl_days: Optional[int] = None  # None means no automatic cleanup
    
    # Instance TTL settings (in minutes)
    # After this period of inactivity, hipporag instances will be offloaded from memory
    instance_ttl_minutes: int = 30  # Default 30 minutes
    
    def __post_init__(self):
        """Ensure memory directory exists and set up default HippoRAG config."""
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Set default HippoRAG configuration if not provided
        if not self.hipporag_config:
            self.hipporag_config = {
                "llm_base_url": os.environ.get("LLM_BASE_URL", None),
                "llm_name": os.environ.get("LLM_NAME", 'gpt-4o-mini'),
                "embedding_base_url": os.environ.get("EMBEDDING_BASE_URL", None),
                "embedding_model_name": os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
            }


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