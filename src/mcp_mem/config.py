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
    default_retrieve_limit: int = 10
    
    # Whether to use HippoRAG for advanced memory features
    use_hipporag: bool = True
    
    # HippoRAG configuration
    hipporag_config: Dict[str, Any] = field(default_factory=dict)
    
    # Default metadata to include with all memories
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Session cleanup settings
    session_ttl_days: Optional[int] = None  # None means no automatic cleanup
    
    def __post_init__(self):
        """Ensure memory directory exists and set up default HippoRAG config."""
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Set default HippoRAG configuration if not provided
        if not self.hipporag_config:
            self.hipporag_config = {
                "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
                "synonymy_edge_topk": 5,
                "synonymy_edge_sim_threshold": 0.7,
                "retrieval_top_k": 10,
                "qa_top_k": 5,
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