"""MCP Memory - A Model Context Protocol server for memory management."""

from .__about__ import __version__
from .server import main, create_memory, store_memory, retrieve_memory, mcp
from .config import get_config, update_config, MemoryConfig
from .memory_utils import cleanup_old_sessions

__all__ = [
    "__version__",
    "main",
    "create_memory",
    "store_memory",
    "retrieve_memory",
    "mcp",
    "get_config",
    "update_config",
    "MemoryConfig",
    "cleanup_old_sessions"
]