"""MCP Memory - A Model Context Protocol server for memory management (LightRAG backend)."""

from .__about__ import __version__
from .server import main, store_memory, retrieve_memory, mcp
from .instance_manager import LightRAGInstanceManager
from .config import get_config, update_config, MemoryConfig
from .memory_utils import cleanup_old_sessions

__all__ = [
    "__version__",
    "main",
    "store_memory",
    "retrieve_memory",
    "mcp",
    "get_config",
    "update_config",
    "MemoryConfig",
    "cleanup_old_sessions",
    "LightRAGInstanceManager"
]