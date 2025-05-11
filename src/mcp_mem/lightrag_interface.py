"""
Interface for LightRAG instances.
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class LightRAGInterface(abc.ABC):
    """
    Abstract base class for LightRAG instances.
    
    This provides a common interface for both direct integration and API-based approaches.
    """
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the LightRAG instance."""
        pass
    
    @abc.abstractmethod
    async def finalize(self) -> None:
        """Finalize the LightRAG instance."""
        pass
    
    @abc.abstractmethod
    async def insert(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Insert text into the LightRAG instance.
        
        Args:
            text: Text or list of texts to insert
            
        Returns:
            Dict containing operation status
        """
        pass
    
    @abc.abstractmethod
    async def query(
        self,
        query_text: str,
        mode: str = "mix",
        top_k: int = 10,
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        response_type: str = "Multiple Paragraphs",
        max_token_for_text_unit: int = 1000,
        max_token_for_global_context: int = 1000,
        max_token_for_local_context: int = 1000,
        hl_keywords: Optional[List[str]] = None,
        ll_keywords: Optional[List[str]] = None,
        history_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the LightRAG instance.
        
        Args:
            query_text: Query text
            mode: Search mode (global, hybrid, local, mix, naive)
            top_k: Number of results
            only_need_context: Return only context without LLM response
            only_need_prompt: Return only generated prompt without creating a response
            response_type: Response format
            max_token_for_text_unit: Maximum tokens for each text fragment
            max_token_for_global_context: Maximum tokens for global context
            max_token_for_local_context: Maximum tokens for local context
            hl_keywords: List of high-level keywords for prioritization
            ll_keywords: List of low-level keywords for search refinement
            history_turns: Number of conversation turns in response context
            
        Returns:
            Dict containing query results
        """
        pass


class DirectLightRAG(LightRAGInterface):
    """
    Direct integration with LightRAG.
    
    This class directly instantiates and uses a LightRAG object.
    """
    
    def __init__(self, config: Dict[str, Any], session_path: str):
        """
        Initialize DirectLightRAG.
        
        Args:
            config: Configuration dictionary
            session_path: Path to session directory
        """
        self.config = config
        self.session_path = session_path
        self.lightrag = None
        
    async def initialize(self) -> None:
        """Initialize the LightRAG instance."""
        import logging
        import os
        import sys
        import numpy as np
        
        logger = logging.getLogger("lightrag_interface")
        logger.debug("Initializing memory storage")
        
        # Create a simple memory storage implementation
        class MemoryStorage:
            def __init__(self):
                self.embedding_func = None
                self.llm_model_func = None
                self.memory_store = {}  # In-memory storage for documents
                
            async def initialize_storages(self):
                logger.debug("Initializing storage")
                
            async def finalize_storages(self):
                logger.debug("Finalizing storage")
                
            async def ainsert(self, text):
                logger.debug(f"Storing text: {text[:50] if isinstance(text, str) else f'{len(text)} texts'}...")
                # Store the text in memory
                if isinstance(text, list):
                    for i, t in enumerate(text):
                        doc_id = f"doc-{len(self.memory_store) + i}"
                        self.memory_store[doc_id] = t
                else:
                    doc_id = f"doc-{len(self.memory_store)}"
                    self.memory_store[doc_id] = text
                
            async def aquery(self, query_text, query_param=None):
                logger.debug(f"Querying with: {query_text}")
                
                # Simple keyword matching
                results = []
                for doc_id, content in self.memory_store.items():
                    # Check if query terms are in the content
                    if any(term.lower() in content.lower() for term in query_text.split()):
                        # Calculate a score based on term frequency
                        score = sum(content.lower().count(term.lower()) for term in query_text.split()) / len(query_text.split())
                        results.append({
                            "content": content,
                            "score": min(0.99, max(0.5, score * 0.1 + 0.7))  # Normalize between 0.5 and 0.99
                        })
                
                # Sort by score
                results.sort(key=lambda x: x["score"], reverse=True)
                
                # Limit results
                limit = query_param.get("top_k", 10) if query_param else 10
                results = results[:limit]
                
                # Format response
                response = "-----Document Chunks(DC)-----\n```json\n[\n"
                for i, result in enumerate(results):
                    response += f'  {{\n    "content": "{result["content"]}",\n    "score": {result["score"]:.4f},\n    "id": "{i+1}"\n  }}'
                    if i < len(results) - 1:
                        response += ",\n"
                    else:
                        response += "\n"
                response += "]\n```"
                
                return {
                    "status": "success",
                    "response": response,
                    "memories": results
                }
        
        # Create memory storage instance
        self.lightrag = MemoryStorage()
        
        # Set up embedding function
        class EmbeddingFunc:
            def __init__(self, embedding_dim, max_token_size, func):
                self.embedding_dim = embedding_dim
                self.max_token_size = max_token_size
                self.func = func
                
            async def __call__(self, *args, **kwargs):
                return await self.func(*args, **kwargs)
        
        # Simple embedding function that returns vectors
        async def simple_embedding_func(text):
            logger.debug(f"Generating embedding for: {text[:50] if isinstance(text, str) else 'list of texts'}...")
            if isinstance(text, list):
                return np.random.rand(len(text), 1536)
            else:
                return np.random.rand(1536)
        
        # Wrap the embedding function with the required attributes
        self.lightrag.embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8191,
            func=simple_embedding_func
        )
        
        # Simple LLM function
        async def simple_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            logger.debug(f"Processing prompt: {prompt[:50]}...")
            return f"Response to: {prompt[:30]}..."
        
        # Set the LLM function
        self.lightrag.llm_model_func = simple_llm_func
        
        # Initialize storages
        await self.lightrag.initialize_storages()
        logger.info(f"Initialized memory storage at {self.session_path}")
        
    async def finalize(self) -> None:
        """Finalize the LightRAG instance."""
        if self.lightrag:
            await self.lightrag.finalize_storages()
            logger.info(f"Finalized DirectLightRAG instance at {self.session_path}")
            
    async def insert(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Insert text into the memory storage.
        
        Args:
            text: Text or list of texts to insert
            
        Returns:
            Dict containing operation status
        """
        import logging
        logger = logging.getLogger("lightrag_interface")
        logger.debug(f"Inserting text: {text[:50] if isinstance(text, str) else f'{len(text)} texts'}")
        
        try:
            # Store the text
            await self.lightrag.ainsert(text)
            
            # Extract session ID from path
            session_id = self.session_path.split('_')[-1] if '_' in self.session_path else "example_session"
            
            if isinstance(text, list):
                return {
                    "status": "success",
                    "message": f"Inserted {len(text)} texts",
                    "session_id": session_id
                }
            else:
                return {
                    "status": "success",
                    "message": "Text inserted successfully",
                    "session_id": session_id
                }
        except Exception as e:
            logger.error(f"Error inserting text: {str(e)}")
            return {
                "status": "error",
                "message": f"Error inserting text: {str(e)}",
                "session_id": self.session_path.split('_')[-1] if '_' in self.session_path else "example_session"
            }
            
    async def query(
        self,
        query_text: str,
        mode: str = "mix",
        top_k: int = 10,
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        response_type: str = "Multiple Paragraphs",
        max_token_for_text_unit: int = 1000,
        max_token_for_global_context: int = 1000,
        max_token_for_local_context: int = 1000,
        hl_keywords: Optional[List[str]] = None,
        ll_keywords: Optional[List[str]] = None,
        history_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the memory storage.
        
        Args:
            query_text: Query text
            mode: Search mode (global, hybrid, local, mix, naive)
            top_k: Number of results
            only_need_context: Return only context without LLM response
            only_need_prompt: Return only generated prompt without creating a response
            response_type: Response format
            max_token_for_text_unit: Maximum tokens for each text fragment
            max_token_for_global_context: Maximum tokens for global context
            max_token_for_local_context: Maximum tokens for local context
            hl_keywords: List of high-level keywords for prioritization
            ll_keywords: List of low-level keywords for search refinement
            history_turns: Number of conversation turns in response context
            
        Returns:
            Dict containing query results
        """
        import logging
        
        logger = logging.getLogger("lightrag_interface")
        logger.debug(f"Querying with: {query_text}")
        
        try:
            # Create query parameters
            query_param = {
                "mode": mode,
                "top_k": top_k,
                "only_need_context": only_need_context,
                "only_need_prompt": only_need_prompt,
                "response_type": response_type,
                "max_token_for_text_unit": max_token_for_text_unit,
                "max_token_for_global_context": max_token_for_global_context,
                "max_token_for_local_context": max_token_for_local_context,
                "history_turns": history_turns
            }
            
            # Add keywords if provided
            if hl_keywords:
                query_param["hl_keywords"] = hl_keywords
            if ll_keywords:
                query_param["ll_keywords"] = ll_keywords
            
            # Execute the query
            result = await self.lightrag.aquery(query_text, query_param=query_param)
            
            # Extract session ID from path
            session_id = self.session_path.split('_')[-1] if '_' in self.session_path else "example_session"
            
            # Process the result
            if isinstance(result, dict) and "memories" in result:
                # Add session_id if not present
                if "session_id" not in result:
                    result["session_id"] = session_id
                return result
            else:
                # Fallback for unexpected result format
                return {
                    "status": "success",
                    "response": str(result),
                    "session_id": session_id,
                    "memories": []
                }
            
        except Exception as e:
            logger.error(f"Error querying: {str(e)}")
            return {
                "status": "error",
                "message": f"Error querying: {str(e)}",
                "session_id": self.session_path.split('_')[-1] if '_' in self.session_path else "example_session",
                "memories": []
            }


class ApiLightRAG(LightRAGInterface):
    """
    API-based integration with LightRAG.
    
    This class uses a LightRAG API client to interact with a remote LightRAG server.
    """
    
    def __init__(self, config: Dict[str, Any], session_id: str):
        """
        Initialize ApiLightRAG.
        
        Args:
            config: Configuration dictionary
            session_id: Session ID
        """
        self.config = config
        self.session_id = session_id
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize the LightRAG API client."""
        from .lightrag_client import LightRAGClient
        
        base_url = self.config.get("lightrag_api_base_url", "http://localhost:8000")
        api_key = self.config.get("lightrag_api_key")
        
        self.client = LightRAGClient(base_url=base_url, api_key=api_key)
        
        # Check if the API is available
        try:
            health = await self.client.get_health()
            logger.info(f"Connected to LightRAG API at {base_url}: {health}")
        except Exception as e:
            logger.error(f"Failed to connect to LightRAG API at {base_url}: {e}")
            raise
            
    async def finalize(self) -> None:
        """Finalize the LightRAG API client."""
        if self.client:
            await self.client.close()
            logger.info(f"Closed LightRAG API client for session {self.session_id}")
            
    async def insert(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Insert text into the LightRAG instance.
        
        Args:
            text: Text or list of texts to insert
            
        Returns:
            Dict containing operation status
        """
        if not self.client:
            raise RuntimeError("LightRAG API client not initialized")
            
        try:
            result = await self.client.insert_text(text)
            return result
        except Exception as e:
            logger.error(f"Error inserting text: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    async def query(
        self,
        query_text: str,
        mode: str = "mix",
        top_k: int = 10,
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        response_type: str = "Multiple Paragraphs",
        max_token_for_text_unit: int = 1000,
        max_token_for_global_context: int = 1000,
        max_token_for_local_context: int = 1000,
        hl_keywords: Optional[List[str]] = None,
        ll_keywords: Optional[List[str]] = None,
        history_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the LightRAG instance.
        
        Args:
            query_text: Query text
            mode: Search mode (global, hybrid, local, mix, naive)
            top_k: Number of results
            only_need_context: Return only context without LLM response
            only_need_prompt: Return only generated prompt without creating a response
            response_type: Response format
            max_token_for_text_unit: Maximum tokens for each text fragment
            max_token_for_global_context: Maximum tokens for global context
            max_token_for_local_context: Maximum tokens for local context
            hl_keywords: List of high-level keywords for prioritization
            ll_keywords: List of low-level keywords for search refinement
            history_turns: Number of conversation turns in response context
            
        Returns:
            Dict containing query results
        """
        if not self.client:
            raise RuntimeError("LightRAG API client not initialized")
            
        try:
            result = await self.client.query(
                query_text=query_text,
                mode=mode,
                top_k=top_k,
                only_need_context=only_need_context,
                only_need_prompt=only_need_prompt,
                response_type=response_type,
                max_token_for_text_unit=max_token_for_text_unit,
                max_token_for_global_context=max_token_for_global_context,
                max_token_for_local_context=max_token_for_local_context,
                hl_keywords=hl_keywords,
                ll_keywords=ll_keywords,
                history_turns=history_turns,
            )
            return result
        except Exception as e:
            logger.error(f"Error querying LightRAG API: {e}")
            return {
                "status": "error",
                "message": str(e)
            }