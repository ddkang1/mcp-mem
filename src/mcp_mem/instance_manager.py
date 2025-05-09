"""LightRAG instance manager with TTL functionality."""

import threading
import time
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any

from .config import get_config
from .lightrag_interface import LightRAGInterface, DirectLightRAG, ApiLightRAG

logger = logging.getLogger(__name__)

class LightRAGInstanceManager:
    """
    Manager for LightRAG instances with TTL-based cleanup.
    
    This class manages LightRAG instances and automatically offloads
    instances that haven't been accessed for a specified period of time.
    """
    
    def __init__(self):
        """Initialize the instance manager."""
        self._instances: Dict[str, LightRAGInterface] = {}
        self._last_accessed: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="lightrag-instance-cleanup"
        )
        self._running = True
        self._cleanup_thread.start()
        logger.info("LightRAG instance manager initialized with TTL cleanup")
        
        # Get configuration
        self._config = get_config()
        
        # Ensure memory directory exists
        os.makedirs(self._config.memory_dir, exist_ok=True)
    
    async def get(self, session_id: str) -> LightRAGInterface:
        """
        Get a LightRAG instance for the given session ID.
        If the instance doesn't exist, it will be created.
        
        Args:
            session_id: The session ID
            
        Returns:
            The LightRAG instance
        """
        with self._lock:
            instance = self._instances.get(session_id)
            if instance:
                # Update last access time
                self._last_accessed[session_id] = datetime.now()
                logger.debug(f"Accessed LightRAG instance for session {session_id}")
            else:
                # Create a new instance if it doesn't exist
                await self.create_memory(session_id)
                instance = self._instances.get(session_id)
                
            return instance
    
    def add(self, session_id: str, instance: LightRAGInterface) -> None:
        """
        Add a LightRAG instance for the given session ID.
        
        Args:
            session_id: The session ID
            instance: The LightRAG instance
        """
        with self._lock:
            self._instances[session_id] = instance
            self._last_accessed[session_id] = datetime.now()
            logger.info(f"Added LightRAG instance for session {session_id}")
    
    def remove(self, session_id: str) -> None:
        """
        Remove a LightRAG instance for the given session ID.
        
        Args:
            session_id: The session ID
        """
        with self._lock:
            if session_id in self._instances:
                del self._instances[session_id]
                del self._last_accessed[session_id]
                logger.info(f"Removed LightRAG instance for session {session_id}")
    
    def contains(self, session_id: str) -> bool:
        """
        Check if a LightRAG instance exists for the given session ID.
        
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
                instance = self._instances.get(session_id)
                if instance:
                    # Finalize the instance
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(instance.finalize())
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error finalizing LightRAG instance for session {session_id}: {str(e)}")
                
                del self._instances[session_id]
                del self._last_accessed[session_id]
                logger.info(f"Offloaded inactive LightRAG instance for session {session_id} (TTL expired)")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} inactive LightRAG instances")
    
    def shutdown(self) -> None:
        """Shutdown the instance manager and stop the cleanup thread."""
        self._running = False
        
        # Finalize all instances
        for session_id, instance in list(self._instances.items()):
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(instance.finalize())
                loop.close()
            except Exception as e:
                logger.error(f"Error finalizing LightRAG instance for session {session_id}: {str(e)}")
        
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        
        logger.info("LightRAG instance manager shutdown")
        
    def _get_session_path(self, session_id: str) -> str:
        """Get the path for storing session data."""
        return os.path.join(self._config.memory_dir, f"session_{session_id}")
    
    def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure that a session directory exists."""
        session_path = self._get_session_path(session_id)
        os.makedirs(session_path, exist_ok=True)
    
    def _create_lightrag_instance(self, session_id: str) -> LightRAGInterface:
        """
        Create a LightRAG instance based on configuration.
        
        Args:
            session_id: Session ID
            
        Returns:
            LightRAG instance
        """
        # Get configuration as dictionary
        config_dict = {
            # Basic configuration
            "kv_storage": self._config.kv_storage,
            "vector_storage": self._config.vector_storage,
            "graph_storage": self._config.graph_storage,
            "doc_status_storage": self._config.doc_status_storage,
            
            # Embedding configuration
            "embedding_provider": self._config.embedding_provider,
            "embedding_model_name": self._config.embedding_model_name,
            "embedding_base_url": self._config.embedding_base_url,
            
            # LLM configuration
            "llm_provider": self._config.llm_provider,
            "llm_model_name": self._config.llm_model_name,
            "llm_base_url": self._config.llm_base_url,
            
            # LightRAG specific settings
            "chunk_token_size": self._config.chunk_token_size,
            "chunk_overlap_token_size": self._config.chunk_overlap_token_size,
            "embedding_batch_num": self._config.embedding_batch_num,
            "embedding_func_max_async": self._config.embedding_func_max_async,
            "llm_model_max_async": self._config.llm_model_max_async,
            "max_parallel_insert": self._config.max_parallel_insert,
            "force_llm_summary_on_merge": self._config.force_llm_summary_on_merge,
            
            # API configuration
            "lightrag_api_base_url": self._config.lightrag_api_base_url,
            "lightrag_api_key": self._config.lightrag_api_key,
        }
        
        # Create instance based on integration type
        if self._config.integration_type == "direct":
            session_path = self._get_session_path(session_id)
            return DirectLightRAG(config=config_dict, session_path=session_path)
        else:  # api
            return ApiLightRAG(config=config_dict, session_id=session_id)
    
    async def create_memory(self, session_id: str) -> None:
        """Create a new memory for a given chat session.
        
        Args:
            session_id: Unique identifier for the chat session.
        """
        # Check if session already exists
        if not self.contains(session_id):
            # Create session directory if using direct integration
            if self._config.integration_type == "direct":
                self._ensure_session_exists(session_id)
            
            # Initialize LightRAG instance
            lightrag_instance = self._create_lightrag_instance(session_id)
            
            # Initialize the instance
            await lightrag_instance.initialize()
            
            # Store LightRAG instance in the manager
            self.add(session_id, lightrag_instance)