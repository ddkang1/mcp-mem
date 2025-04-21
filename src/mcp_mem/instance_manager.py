"""HippoRAG instance manager with TTL functionality."""

import threading
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from .config import get_config

logger = logging.getLogger(__name__)

class HippoRAGInstanceManager:
    """
    Manager for HippoRAG instances with TTL-based cleanup.
    
    This class manages HippoRAG instances and automatically offloads
    instances that haven't been accessed for a specified period of time.
    """
    
    def __init__(self):
        """Initialize the instance manager."""
        self._instances: Dict[str, HippoRAG] = {}
        self._last_accessed: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="hipporag-instance-cleanup"
        )
        self._running = True
        self._cleanup_thread.start()
        logger.info("HippoRAG instance manager initialized with TTL cleanup")
        
        # Get configuration
        self._config = get_config()
        
        # Ensure memory directory exists
        os.makedirs(self._config.memory_dir, exist_ok=True)
    
    async def get(self, session_id: str) -> HippoRAG:
        """
        Get a HippoRAG instance for the given session ID.
        If the instance doesn't exist, it will be created.
        
        Args:
            session_id: The session ID
            
        Returns:
            The HippoRAG instance
        """
        with self._lock:
            instance = self._instances.get(session_id)
            if instance:
                # Update last access time
                self._last_accessed[session_id] = datetime.now()
                logger.debug(f"Accessed HippoRAG instance for session {session_id}")
            else:
                # Create a new instance if it doesn't exist
                await self.create_memory(session_id)
                instance = self._instances.get(session_id)
                
            return instance
    
    def add(self, session_id: str, instance: HippoRAG) -> None:
        """
        Add a HippoRAG instance for the given session ID.
        
        Args:
            session_id: The session ID
            instance: The HippoRAG instance
        """
        with self._lock:
            self._instances[session_id] = instance
            self._last_accessed[session_id] = datetime.now()
            logger.info(f"Added HippoRAG instance for session {session_id}")
    
    def remove(self, session_id: str) -> None:
        """
        Remove a HippoRAG instance for the given session ID.
        
        Args:
            session_id: The session ID
        """
        with self._lock:
            if session_id in self._instances:
                del self._instances[session_id]
                del self._last_accessed[session_id]
                logger.info(f"Removed HippoRAG instance for session {session_id}")
    
    def contains(self, session_id: str) -> bool:
        """
        Check if a HippoRAG instance exists for the given session ID.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if the instance exists, False otherwise
        """
        with self._lock:
            return session_id in self._instances
    
    def get_all_session_ids(self) -> Set[str]:
        """
        Get all session IDs with active instances.
        
        Returns:
            A set of session IDs
        """
        with self._lock:
            return set(self._instances.keys())
    
    def _cleanup_loop(self) -> None:
        """Background thread that periodically cleans up expired instances."""
        while self._running:
            try:
                self._cleanup_expired_instances()
            except Exception as e:
                logger.error(f"Error in instance cleanup: {str(e)}")
            
            # Sleep for a minute before checking again
            time.sleep(60)
    
    def _cleanup_expired_instances(self) -> None:
        """Clean up instances that have expired based on TTL."""
        config = get_config()
        ttl_minutes = config.instance_ttl_minutes
        
        if ttl_minutes <= 0:
            # TTL is disabled
            return
        
        cutoff_time = datetime.now() - timedelta(minutes=ttl_minutes)
        expired_sessions = []
        
        with self._lock:
            # Find expired sessions
            for session_id, last_accessed in self._last_accessed.items():
                if last_accessed < cutoff_time:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self._instances[session_id]
                del self._last_accessed[session_id]
                logger.info(f"Offloaded inactive HippoRAG instance for session {session_id} (TTL expired)")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} inactive HippoRAG instances")
    
    def shutdown(self) -> None:
        """Shutdown the instance manager and stop the cleanup thread."""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        logger.info("HippoRAG instance manager shutdown")
        
    def _get_session_path(self, session_id: str) -> str:
        """Get the path for storing session data."""
        return os.path.join(self._config.memory_dir, f"session_{session_id}")
    
    def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure that a session directory exists."""
        session_path = self._get_session_path(session_id)
        os.makedirs(session_path, exist_ok=True)
    
    def _initialize_hipporag(self, session_id: str) -> HippoRAG:
        """Initialize HippoRAG for a session."""
        session_path = self._get_session_path(session_id)
        hippo_config = BaseConfig()
        hippo_config.save_dir = session_path
        
        # Apply HippoRAG configuration from our config
        for key, value in self._config.hipporag_config.items():
            if hasattr(hippo_config, key) and value is not None:
                setattr(hippo_config, key, value)
        
        return HippoRAG(
            global_config=hippo_config,
            save_dir=session_path
        )
    
    async def create_memory(self, session_id: str) -> None:
        """Create a new memory for a given chat session.
        
        Args:
            session_id: Unique identifier for the chat session.
        """
        # Check if session already exists
        if not self.contains(session_id):
            # Create session directory
            self._ensure_session_exists(session_id)
            
            # Initialize HippoRAG for knowledge graph
            hipporag_instance = self._initialize_hipporag(session_id)
            
            # Store HippoRAG instance in the manager
            self.add(session_id, hipporag_instance)