"""MCP Memory module for permanent external memory using LightRAG."""

from .config import MemoryConfig, get_config, update_config
from .instance_manager import LightRAGInstanceManager
from .lightrag_interface import LightRAGInterface, DirectLightRAG, ApiLightRAG
from .lightrag_client import LightRAGClient
from .memory_utils import (
    cleanup_old_sessions, 
    create_session_state, 
    update_session_access,
    get_session_info,
    list_sessions
)

__version__ = "0.2.0"  # Updated version to reflect LightRAG integration