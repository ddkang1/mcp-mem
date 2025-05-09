"""Configuration module for mcp-mem."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal


@dataclass
class MemoryConfig:
    """Configuration settings for the MCP Memory server."""
    
    # Base directory for storing memory data
    memory_dir: str = os.path.expanduser("~/.mcp-mem")
    
    # Maximum number of memories to return in retrieve_memory
    default_retrieve_limit: Optional[int] = 5
    
    # Integration type: "direct" or "api"
    integration_type: Literal["direct", "api"] = "direct"
    
    # LightRAG storage configuration (for direct integration)
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    
    # LightRAG API configuration (for API integration)
    lightrag_api_base_url: str = "http://localhost:8000"
    lightrag_api_key: Optional[str] = None
    
    # Embedding configuration
    embedding_provider: str = "openai"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_base_url: Optional[str] = None
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_base_url: Optional[str] = None
    
    # LightRAG specific settings
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_max_async: int = 4
    max_parallel_insert: int = 2
    force_llm_summary_on_merge: int = 3
    
    # Default metadata to include with all memories
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Session cleanup settings
    session_ttl_days: Optional[int] = None  # None means no automatic cleanup
    
    # Instance TTL settings (in minutes)
    # After this period of inactivity, lightrag instances will be offloaded from memory
    instance_ttl_minutes: int = 30  # Default 30 minutes
    
    def __post_init__(self):
        """Ensure memory directory exists and set up default configuration."""
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Apply environment variables if available
        if os.environ.get("OPENAI_API_KEY"):
            # If OPENAI_API_KEY is set, use OpenAI as the default provider
            self.embedding_provider = "openai"
            self.llm_provider = "openai"
            
            # Apply OpenAI API base URL if set
            if os.environ.get("OPENAI_API_BASE"):
                self.embedding_base_url = os.environ.get("OPENAI_API_BASE")
                self.llm_base_url = os.environ.get("OPENAI_API_BASE")
        
        # Apply LightRAG API configuration from environment variables
        if os.environ.get("LIGHTRAG_API_BASE_URL"):
            self.lightrag_api_base_url = os.environ.get("LIGHTRAG_API_BASE_URL")
            
        if os.environ.get("LIGHTRAG_API_KEY"):
            self.lightrag_api_key = os.environ.get("LIGHTRAG_API_KEY")
            
        # If LightRAG API key is set, use API integration by default
        if self.lightrag_api_key:
            self.integration_type = "api"


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