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
        from lightrag import LightRAG
        
        # Create LightRAG configuration
        lightrag_config = {
            "working_dir": self.session_path,
            "kv_storage": self.config.get("kv_storage", "JsonKVStorage"),
            "vector_storage": self.config.get("vector_storage", "NanoVectorDBStorage"),
            "graph_storage": self.config.get("graph_storage", "NetworkXStorage"),
            "doc_status_storage": self.config.get("doc_status_storage", "JsonDocStatusStorage"),
            "embedding_func": None,  # Will be set below
            "llm_model_func": None,  # Will be set below
            "chunk_token_size": self.config.get("chunk_token_size", 1200),
            "chunk_overlap_token_size": self.config.get("chunk_overlap_token_size", 100),
            "embedding_batch_num": self.config.get("embedding_batch_num", 32),
            "embedding_func_max_async": self.config.get("embedding_func_max_async", 16),
            "llm_model_max_async": self.config.get("llm_model_max_async", 4),
            "max_parallel_insert": self.config.get("max_parallel_insert", 2),
            "force_llm_summary_on_merge": self.config.get("force_llm_summary_on_merge", 3),
        }
        
        # Create LightRAG instance
        self.lightrag = LightRAG(**lightrag_config)
        
        # Set up embedding function
        if self.config.get("embedding_provider") == "openai":
            from lightrag.embeddings.openai import OpenAIEmbedding
            import os
            
            self.lightrag.embedding_func = OpenAIEmbedding(
                model=self.config.get("embedding_model_name", "text-embedding-3-large"),
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_base=self.config.get("embedding_base_url"),
            ).get_embeddings
        
        # Set up LLM function
        if self.config.get("llm_provider") == "openai":
            from lightrag.llm.openai import OpenAIChat
            import os
            
            self.lightrag.llm_model_func = OpenAIChat(
                model=self.config.get("llm_model_name", "gpt-4o-mini"),
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_base=self.config.get("llm_base_url"),
            ).chat
        
        # Initialize storages
        await self.lightrag.initialize_storages()
        logger.info(f"Initialized DirectLightRAG instance at {self.session_path}")
        
    async def finalize(self) -> None:
        """Finalize the LightRAG instance."""
        if self.lightrag:
            await self.lightrag.finalize_storages()
            logger.info(f"Finalized DirectLightRAG instance at {self.session_path}")
            
    async def insert(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Insert text into the LightRAG instance.
        
        Args:
            text: Text or list of texts to insert
            
        Returns:
            Dict containing operation status
        """
        if not self.lightrag:
            raise RuntimeError("LightRAG instance not initialized")
            
        if isinstance(text, list):
            for t in text:
                await self.lightrag.ainsert(t)
            return {
                "status": "success",
                "message": f"Inserted {len(text)} texts"
            }
        else:
            await self.lightrag.ainsert(text)
            return {
                "status": "success",
                "message": "Text inserted successfully"
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
        if not self.lightrag:
            raise RuntimeError("LightRAG instance not initialized")
            
        from lightrag.base import QueryParam
        
        query_param = QueryParam(
            mode=mode,
            top_k=top_k,
            only_need_context=only_need_context,
            only_need_prompt=only_need_prompt,
            response_type=response_type,
            max_token_for_text_unit=max_token_for_text_unit,
            max_token_for_global_context=max_token_for_global_context,
            max_token_for_local_context=max_token_for_local_context,
            hl_keywords=hl_keywords or [],
            ll_keywords=ll_keywords or [],
            history_turns=history_turns,
        )
        
        result = await self.lightrag.aquery(query_text, query_param=query_param)
        
        return {
            "status": "success",
            "response": result
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