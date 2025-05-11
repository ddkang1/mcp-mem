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
    
    @abc.abstractmethod
    async def get_internal_state(self) -> Dict[str, Any]:
        """
        Get the internal state of the LightRAG instance.
        
        Returns:
            Dict containing information about nodes, edges, and other internal state
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
        logger.debug("Initializing LightRAG instance")
        
        # Import LightRAG
        sys.path.append(os.path.abspath("./LightRAG"))
        try:
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag.utils import wrap_embedding_func_with_attrs
        except ImportError as e:
            logger.error(f"Failed to import LightRAG: {e}")
            raise ImportError(f"LightRAG library is required but could not be imported: {e}")
        
        # Create LightRAG working directory if it doesn't exist
        os.makedirs(self.session_path, exist_ok=True)
        
        # Define a custom embedding function with proper attributes
        @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
        async def custom_embedding_func(texts):
            """Custom embedding function that returns random vectors."""
            logger.debug(f"Generating embedding for: {texts[:50] if isinstance(texts, str) else f'{len(texts)} texts'}...")
            if isinstance(texts, list):
                return np.random.rand(len(texts), 1536).astype(np.float32)
            else:
                return np.random.rand(1536).astype(np.float32)
        
        # Simple LLM function
        async def simple_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            logger.debug(f"Processing prompt: {prompt[:50]}...")
            return f"Response to: {prompt[:30]}..."
        
        # Create a real LightRAG instance
        self.lightrag = LightRAG(
            working_dir=self.session_path,
            # Use default storage types
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            # Configure chunking
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            tiktoken_model_name="gpt-4o-mini",
            # Use our custom embedding function
            embedding_func=custom_embedding_func,
            # Use our simple LLM function
            llm_model_func=simple_llm_func
        )
        
        # Initialize LightRAG storages
        await self.lightrag.initialize_storages()
        
        # Initialize pipeline status
        await initialize_pipeline_status()
        
        logger.info(f"Initialized LightRAG instance at {self.session_path}")
    
    async def finalize(self) -> None:
        """Finalize the LightRAG instance."""
        if self.lightrag:
            try:
                # Check if this is a real LightRAG instance with finalize_storages method
                if hasattr(self.lightrag, 'finalize_storages') and callable(getattr(self.lightrag, 'finalize_storages', None)):
                    await self.lightrag.finalize_storages()
                    logger.info(f"Finalized LightRAG instance at {self.session_path}")
                else:
                    # For our simple memory storage implementation
                    await self.lightrag.finalize_storages()
                    logger.info(f"Finalized memory storage at {self.session_path}")
            except Exception as e:
                logger.error(f"Error finalizing LightRAG instance: {e}")
    
    async def get_internal_state(self) -> Dict[str, Any]:
        """
        Get the internal state of the LightRAG instance.
        
        Returns:
            Dict containing information about nodes, edges, and other internal state
        """
        if not self.lightrag:
            return {"status": "error", "message": "LightRAG instance not initialized"}
        
        try:
            # Get basic stats about the memory store
            doc_count = len(getattr(self.lightrag, 'memory_store', {}))
            
            # Create a simplified representation of the memory store
            memory_store_preview = {}
            for doc_id, content in list(getattr(self.lightrag, 'memory_store', {}).items())[:10]:
                preview = content[:100] + "..." if len(content) > 100 else content
                memory_store_preview[doc_id] = preview
            
            # Get knowledge graph directly from LightRAG
            logger.debug("Getting knowledge graph from LightRAG instance")
            kg = await self.lightrag.get_knowledge_graph("*", max_depth=2, max_nodes=20)
            
            # Extract entities from nodes
            entities = [
                {
                    "entity_name": node.id,
                    "entity_type": node.labels[0] if node.labels else "unknown",
                    "description": node.properties.get("description", ""),
                    "source": node.properties.get("source", "")
                }
                for node in kg.nodes
            ]
            
            # Extract relationships from edges
            relationships = [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type or "unknown",
                    "description": edge.properties.get("description", "")
                }
                for edge in kg.edges
            ]
            
            logger.info(f"Retrieved knowledge graph with {len(entities)} entities and {len(relationships)} relationships")
            
            # Return the state information
            return {
                "status": "success",
                "document_count": doc_count,
                "memory_store_preview": memory_store_preview,
                "entities": entities,
                "relationships": relationships,
                "session_path": self.session_path
            }
        except Exception as e:
            logger.error(f"Error getting internal state: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting internal state: {str(e)}"
            }
    
    async def insert(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Insert text into the LightRAG instance.
        
        Args:
            text: Text or list of texts to insert
            
        Returns:
            Dict containing operation status
        """
        import logging
        logger = logging.getLogger("lightrag_interface")
        logger.debug(f"Inserting text: {text[:50] if isinstance(text, str) else f'{len(text)} texts'}")
        
        try:
            # Use the pipeline approach for LightRAG if available
            if hasattr(self.lightrag, 'apipeline_enqueue_documents'):
                await self.lightrag.apipeline_enqueue_documents(text)
                await self.lightrag.apipeline_process_enqueue_documents()
                logger.info("Used LightRAG pipeline for text insertion")
            else:
                # Use direct ainsert
                await self.lightrag.ainsert(text)
                logger.info("Used LightRAG direct ainsert for text insertion")
            
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
        import logging
        
        logger = logging.getLogger("lightrag_interface")
        logger.debug(f"Querying with: {query_text}")
        
        try:
            # Use the query method
            logger.info("Using LightRAG query method")
            result = await self.lightrag.query(
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
                history_turns=history_turns
            )
            
            # Extract session ID from path
            session_id = self.session_path.split('_')[-1] if '_' in self.session_path else "example_session"
            
            # Process the result
            if isinstance(result, dict):
                # Add session_id if not present
                if "session_id" not in result:
                    result["session_id"] = session_id
                # Ensure memories field exists
                if "memories" not in result:
                    result["memories"] = []
                return result
            else:
                # Convert non-dict result to standard format
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
    
    async def get_internal_state(self) -> Dict[str, Any]:
        """
        Get the internal state of the LightRAG instance.
        
        Returns:
            Dict containing information about nodes, edges, and other internal state
        """
        if not self.client:
            return {"status": "error", "message": "LightRAG API client not initialized"}
            
        try:
            # Get knowledge graph from API
            graph_info = await self.client.get_knowledge_graph()
            
            # Extract entities and relationships from graph_info
            entities = []
            relationships = []
            
            if isinstance(graph_info, dict) and "nodes" in graph_info and "edges" in graph_info:
                # Extract entities from nodes
                for node in graph_info.get("nodes", []):
                    entity = {
                        "entity_name": node.get("id", ""),
                        "entity_type": node.get("labels", ["unknown"])[0] if node.get("labels") else "unknown",
                        "description": node.get("properties", {}).get("description", ""),
                        "source": node.get("properties", {}).get("source", "")
                    }
                    entities.append(entity)
                
                # Extract relationships from edges
                for edge in graph_info.get("edges", []):
                    relationship = {
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "type": edge.get("type", "unknown"),
                        "description": edge.get("properties", {}).get("description", "")
                    }
                    relationships.append(relationship)
            else:
                logger.error("Invalid graph information format returned from API")
                raise ValueError("Invalid graph information format returned from API")
            
            return {
                "status": "success",
                "session_id": self.session_id,
                "entities": entities,
                "relationships": relationships,
                "graph_info": graph_info  # Include original graph_info for backward compatibility
            }
        except Exception as e:
            logger.error(f"Error getting internal state: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting internal state: {str(e)}"
            }
            
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