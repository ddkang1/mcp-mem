"""LightRAG instance manager with TTL functionality."""

import threading
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
import inspect

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from .config import get_config

logger = logging.getLogger(__name__)

class LightRAGInstanceManager:
    """
    Manager for LightRAG instances with TTL-based cleanup.
    Each session gets its own LightRAG instance.
    """
    def __init__(self, embedding_func: Optional[Callable] = None, llm_func: Optional[Callable] = None):
        self._instances: Dict[str, LightRAG] = {}
        self._last_accessed: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._ttl_minutes = get_config().instance_ttl_minutes
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self._embedding_func = embedding_func
        self._llm_func = llm_func

    def _get_session_dir(self, session_id: str) -> str:
        base_dir = get_config().memory_dir
        session_dir = os.path.join(base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _cleanup_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                now = datetime.now()
                expired = [sid for sid, last in self._last_accessed.items()
                           if (now - last).total_seconds() > self._ttl_minutes * 60]
                for sid in expired:
                    logger.info(f"Unloading LightRAG instance for session {sid} (TTL expired)")
                    self._instances.pop(sid, None)
                    self._last_accessed.pop(sid, None)
            time.sleep(60)

    async def get(self, session_id: str) -> LightRAG:
        with self._lock:
            if session_id not in self._instances:
                session_dir = self._get_session_dir(session_id)
                kwargs = {"working_dir": session_dir}
                
                # Check if we should use LightRAG's built-in OpenAI integration
                llm_base_url = os.getenv("LLM_BASE_URL")
                embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
                api_key = os.getenv("OPENAI_API_KEY")
                
                # Always provide functions - either custom or built-in
                if self._embedding_func is not None:
                    kwargs["embedding_func"] = self._embedding_func
                
                if self._llm_func is not None:
                    kwargs["llm_model_func"] = self._llm_func
                
                # If we have custom endpoints, set environment variables for LightRAG's built-in client
                if api_key and (llm_base_url or embedding_base_url):
                    if llm_base_url:
                        kwargs["llm_model_name"] = os.getenv("LLM_NAME", "gpt-4o-mini")
                        # Set environment variables for LightRAG's built-in OpenAI client
                        os.environ["OPENAI_BASE_URL"] = llm_base_url
                    
                    if embedding_base_url and not llm_base_url:
                        # LightRAG uses the same base URL for embeddings
                        os.environ["OPENAI_BASE_URL"] = embedding_base_url
                
                logger.debug(f"Creating LightRAG with kwargs: {list(kwargs.keys())}")
                
                # Create LightRAG instance
                lightrag_instance = LightRAG(**kwargs)
                
                # Initialize storages (required for LightRAG to work)
                await lightrag_instance.initialize_storages()
                
                # Initialize pipeline status - this is crucial for proper LightRAG operation
                await initialize_pipeline_status()
                
                self._instances[session_id] = lightrag_instance
            self._last_accessed[session_id] = datetime.now()
            return self._instances[session_id]

    def shutdown(self):
        self._stop_event.set()
        self._cleanup_thread.join()
        logger.info("LightRAGInstanceManager shutdown complete.")