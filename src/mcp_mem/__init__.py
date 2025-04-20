"""MCP Memory - A Model Context Protocol server for memory management."""

from .__about__ import __version__
from .server import main, create_memory, store_memory, retrieve_memory, mcp
from .config import get_config, update_config, MemoryConfig
from .memory_utils import (
    get_session_memories,
    search_memories,
    cleanup_old_sessions,
    get_memory_stats
)

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
    "get_session_memories",
    "search_memories",
    "cleanup_old_sessions",
    "get_memory_stats"
]