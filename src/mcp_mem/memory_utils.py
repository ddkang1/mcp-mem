"""Utility functions for memory operations."""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

from .config import get_config

logger = logging.getLogger(__name__)

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
