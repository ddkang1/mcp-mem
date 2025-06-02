"""Memory utility functions for mcp-mem (LightRAG only)."""

import os
import shutil
from datetime import datetime, timedelta
from .config import get_config


def cleanup_old_sessions() -> int:
    """Remove session directories older than the configured TTL (in days)."""
    config = get_config()
    if not config.session_ttl_days:
        return 0
    base_dir = config.memory_dir
    now = datetime.now()
    removed = 0
    for session_dir in os.listdir(base_dir):
        session_path = os.path.join(base_dir, session_dir)
        if not os.path.isdir(session_path):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
        if now - mtime > timedelta(days=config.session_ttl_days):
            shutil.rmtree(session_path)
            removed += 1
    return removed
