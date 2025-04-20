"""Utility functions for memory operations."""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

from .config import get_config

logger = logging.getLogger(__name__)

def get_session_memories(session_id: str) -> List[Dict[str, Any]]:
    """Get all memories for a specific session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        List of memory entries
    """
    config = get_config()
    session_path = os.path.join(config.memory_dir, f"session_{session_id}")
    state_file = os.path.join(session_path, "session_state.json")
    
    if not os.path.exists(state_file):
        return []
    
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            return data.get("memories", [])
    except Exception as e:
        logger.error(f"Error loading session memories: {str(e)}")
        return []

def search_memories(
    session_id: str, 
    query: str = None, 
    metadata_filters: Dict[str, Any] = None,
    limit: int = None
) -> List[Dict[str, Any]]:
    """Search memories based on content and metadata filters.
    
    Args:
        session_id: The session identifier
        query: Optional text query to search in content
        metadata_filters: Optional metadata filters to apply
        limit: Maximum number of results to return
        
    Returns:
        List of matching memory entries
    """
    config = get_config()
    if limit is None:
        limit = config.default_retrieve_limit
        
    memories = get_session_memories(session_id)
    
    # Apply text search if query is provided
    if query:
        memories = [
            mem for mem in memories 
            if query.lower() in mem["content"].lower()
        ]
    
    # Apply metadata filters if provided
    if metadata_filters:
        filtered_memories = []
        for mem in memories:
            match = True
            for key, value in metadata_filters.items():
                if key not in mem["metadata"] or mem["metadata"][key] != value:
                    match = False
                    break
            if match:
                filtered_memories.append(mem)
        memories = filtered_memories
    
    # Sort by timestamp (most recent first) and limit
    sorted_memories = sorted(
        memories,
        key=lambda x: x["timestamp"],
        reverse=True
    )[:limit]
    
    return sorted_memories

def cleanup_old_sessions(max_age_days: int = None) -> int:
    """Clean up old session data.
    
    Args:
        max_age_days: Maximum age of sessions to keep (None means use config value)
        
    Returns:
        Number of sessions removed
    """
    config = get_config()
    
    # Use config value if not specified
    if max_age_days is None:
        max_age_days = config.session_ttl_days
    
    # If still None, no cleanup
    if max_age_days is None:
        return 0
    
    if not os.path.exists(config.memory_dir):
        return 0
    
    session_dirs = [
        d for d in os.listdir(config.memory_dir) 
        if os.path.isdir(os.path.join(config.memory_dir, d)) and d.startswith("session_")
    ]
    
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    removed_count = 0
    
    for session_dir in session_dirs:
        dir_path = os.path.join(config.memory_dir, session_dir)
        state_file = os.path.join(dir_path, "session_state.json")
        
        try:
            # Check if session is older than cutoff
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    created_at = datetime.fromisoformat(data.get("created_at", ""))
                    
                    if created_at < cutoff_date:
                        # Remove session directory
                        import shutil
                        shutil.rmtree(dir_path)
                        removed_count += 1
                        logger.info(f"Removed old session: {session_dir}")
            else:
                # If no state file, check directory modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
                if mod_time < cutoff_date:
                    import shutil
                    shutil.rmtree(dir_path)
                    removed_count += 1
                    logger.info(f"Removed old session with no state file: {session_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_dir}: {str(e)}")
    
    return removed_count

def get_memory_stats() -> Dict[str, Any]:
    """Get statistics about stored memories.
    
    Returns:
        Dictionary with memory statistics
    """
    config = get_config()
    
    if not os.path.exists(config.memory_dir):
        return {
            "session_count": 0,
            "total_memories": 0,
            "oldest_session": None,
            "newest_session": None,
            "avg_memories_per_session": 0
        }
    
    session_dirs = [
        d for d in os.listdir(config.memory_dir) 
        if os.path.isdir(os.path.join(config.memory_dir, d)) and d.startswith("session_")
    ]
    
    session_count = len(session_dirs)
    total_memories = 0
    oldest_timestamp = None
    newest_timestamp = None
    
    for session_dir in session_dirs:
        dir_path = os.path.join(config.memory_dir, session_dir)
        state_file = os.path.join(dir_path, "session_state.json")
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    memories = data.get("memories", [])
                    total_memories += len(memories)
                    
                    created_at = datetime.fromisoformat(data.get("created_at", ""))
                    
                    if oldest_timestamp is None or created_at < oldest_timestamp:
                        oldest_timestamp = created_at
                    
                    if newest_timestamp is None or created_at > newest_timestamp:
                        newest_timestamp = created_at
        except Exception as e:
            logger.error(f"Error getting stats for session {session_dir}: {str(e)}")
    
    avg_memories_per_session = total_memories / session_count if session_count > 0 else 0
    
    return {
        "session_count": session_count,
        "total_memories": total_memories,
        "oldest_session": oldest_timestamp.isoformat() if oldest_timestamp else None,
        "newest_session": newest_timestamp.isoformat() if newest_timestamp else None,
        "avg_memories_per_session": avg_memories_per_session
    }