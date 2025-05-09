"""Utility functions for memory operations."""

import json
import os
import logging
import shutil
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
                        shutil.rmtree(dir_path)
                        removed_count += 1
                        logger.info(f"Removed old session: {session_dir}")
            else:
                # If no state file, check directory modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
                if mod_time < cutoff_date:
                    shutil.rmtree(dir_path)
                    removed_count += 1
                    logger.info(f"Removed old session with no state file: {session_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_dir}: {str(e)}")
    
    return removed_count

def create_session_state(session_id: str) -> None:
    """Create a session state file to track session metadata.
    
    Args:
        session_id: The session ID
    """
    config = get_config()
    session_dir = os.path.join(config.memory_dir, f"session_{session_id}")
    state_file = os.path.join(session_dir, "session_state.json")
    
    # Create session directory if it doesn't exist
    os.makedirs(session_dir, exist_ok=True)
    
    # Create state file if it doesn't exist
    if not os.path.exists(state_file):
        state_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "integration_type": config.integration_type,
            "metadata": config.default_metadata
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.debug(f"Created session state file for session {session_id}")

def update_session_access(session_id: str) -> None:
    """Update the last accessed time for a session.
    
    Args:
        session_id: The session ID
    """
    config = get_config()
    session_dir = os.path.join(config.memory_dir, f"session_{session_id}")
    state_file = os.path.join(session_dir, "session_state.json")
    
    # Update state file if it exists
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            state_data["last_accessed"] = datetime.now().isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"Updated last accessed time for session {session_id}")
        except Exception as e:
            logger.error(f"Error updating session access time for {session_id}: {str(e)}")

def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a session.
    
    Args:
        session_id: The session ID
        
    Returns:
        Session information or None if the session doesn't exist
    """
    config = get_config()
    session_dir = os.path.join(config.memory_dir, f"session_{session_id}")
    state_file = os.path.join(session_dir, "session_state.json")
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            return state_data
        except Exception as e:
            logger.error(f"Error reading session state for {session_id}: {str(e)}")
    
    return None

def list_sessions() -> List[Dict[str, Any]]:
    """List all sessions.
    
    Returns:
        List of session information dictionaries
    """
    config = get_config()
    
    if not os.path.exists(config.memory_dir):
        return []
    
    session_dirs = [
        d for d in os.listdir(config.memory_dir) 
        if os.path.isdir(os.path.join(config.memory_dir, d)) and d.startswith("session_")
    ]
    
    sessions = []
    for session_dir in session_dirs:
        session_id = session_dir[len("session_"):]
        session_info = get_session_info(session_id)
        if session_info:
            sessions.append(session_info)
    
    return sessions
