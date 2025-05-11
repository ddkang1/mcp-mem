"""
Interface for LightRAG instances.
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Union
from .custom_llm import gpt_4o_mini_complete, openai_embed

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
        
        # Extract addon_params from config if available
        addon_params = self.config.get("addon_params", {
            "language": "English",
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "CONCEPT"],
            "example_number": 3  # Number of examples to use for entity extraction
        })
        
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
            embedding_func=openai_embed,
            # Use our simple LLM function
            llm_model_func=gpt_4o_mini_complete,
            # Add entity extraction configuration
            addon_params=addon_params
        )
        
        # Initialize LightRAG storages
        await self.lightrag.initialize_storages()
        
        # Initialize pipeline status
        await initialize_pipeline_status()
        
        logger.info(f"Initialized LightRAG instance at {self.session_path}")
    
    async def clear_storage(self) -> Dict[str, Any]:
        """
        Clear all data from the LightRAG instance.
        
        This method drops all documents, entities, and relationships from storage.
        
        Returns:
            Dict containing operation status
        """
        try:
            # Drop document status data
            if hasattr(self.lightrag, 'doc_status') and hasattr(self.lightrag.doc_status, 'drop'):
                await self.lightrag.doc_status.drop()
                logger.info(f"Cleared document status data for {self.session_path}")
            
            # Drop entity data if available
            if hasattr(self.lightrag, 'entities_vdb') and hasattr(self.lightrag.entities_vdb, 'drop'):
                await self.lightrag.entities_vdb.drop()
                logger.info(f"Cleared entity data for {self.session_path}")
            
            # Drop relationship data if available
            if hasattr(self.lightrag, 'relationships_vdb') and hasattr(self.lightrag.relationships_vdb, 'drop'):
                await self.lightrag.relationships_vdb.drop()
                logger.info(f"Cleared relationship data for {self.session_path}")
            
            # Drop chunk data if available
            if hasattr(self.lightrag, 'chunks_vdb') and hasattr(self.lightrag.chunks_vdb, 'drop'):
                await self.lightrag.chunks_vdb.drop()
                logger.info(f"Cleared chunk data for {self.session_path}")
            
            # Drop knowledge graph if available
            if hasattr(self.lightrag, 'chunk_entity_relation_graph') and hasattr(self.lightrag.chunk_entity_relation_graph, 'drop'):
                await self.lightrag.chunk_entity_relation_graph.drop()
                logger.info(f"Cleared knowledge graph for {self.session_path}")
            
            return {
                "status": "success",
                "message": f"All storage cleared for {self.session_path}"
            }
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")
            return {
                "status": "error",
                "message": f"Error clearing storage: {e}"
            }
    
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
            # Get document count using status counts
            status_counts = await self.lightrag.doc_status.get_status_counts()
            doc_count = sum(status_counts.values())
            
            # Get processed documents
            from lightrag.base import DocStatus
            processed_docs = await self.lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            
            # Create a simplified representation of the documents
            memory_store_preview = {}
            for doc_id, doc_status in list(processed_docs.items())[:10]:
                content = getattr(doc_status, 'content', '')
                preview = content[:100] + "..." if content and len(content) > 100 else content
                memory_store_preview[doc_id] = preview
                
            # Wait for indexing and entity extraction to complete
            import asyncio
            logger.info("Waiting for indexing and entity extraction to complete...")
            await asyncio.sleep(2)  # Increased from 2 to 2 seconds to allow entity extraction to finish
            
            # Get knowledge graph directly from LightRAG
            logger.debug("Getting knowledge graph from LightRAG instance")
            
            # Try different methods to extract entities and relationships
            entities = []
            relationships = []
            
            # Method 1: Use get_knowledge_graph
            try:
                logger.debug("Attempting to get knowledge graph with get_knowledge_graph")
                kg = await self.lightrag.get_knowledge_graph("*", max_depth=5, max_nodes=100)
                logger.debug(f"Knowledge graph retrieved: {kg}")
                logger.debug(f"Nodes: {len(kg.nodes)}, Edges: {len(kg.edges)}")
                
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
                
                logger.info(f"Method 1: Retrieved {len(entities)} entities and {len(relationships)} relationships")
            except Exception as e:
                logger.warning(f"Error using get_knowledge_graph: {e}")
            
            # Method 2: Try to access entity and relationship storage directly
            if hasattr(self.lightrag, 'entities_vdb'):
                try:
                    logger.debug(f"Attempting to access entities_vdb directly: {self.lightrag.entities_vdb}")
                    # Try to get all entities from vector database
                    entity_data = {}
                    if hasattr(self.lightrag.entities_vdb, 'get_all'):
                        logger.debug("entities_vdb has get_all method, calling it")
                        entity_data = await self.lightrag.entities_vdb.get_all()
                        logger.debug(f"Entity data retrieved: {len(entity_data)} items")
                    
                    # Process entity data
                    for entity_id, entity_info in entity_data.items():
                        entity = {
                            "entity_name": entity_info.get("entity_name", "Unknown"),
                            "entity_type": entity_info.get("entity_type", "Unknown"),
                            "description": entity_info.get("content", "").replace(entity_info.get("entity_name", "") + "\n", ""),
                            "source": entity_info.get("source_id", "")
                        }
                        entities.append(entity)
                    
                    logger.info(f"Method 2: Retrieved {len(entities)} entities from vector database")
                except Exception as e:
                    logger.warning(f"Error accessing entities_vdb: {e}")
            
            # Try to get relationships from vector database
            if hasattr(self.lightrag, 'relationships_vdb'):
                try:
                    logger.debug(f"Attempting to access relationships_vdb directly: {self.lightrag.relationships_vdb}")
                    # Try to get all relationships from vector database
                    rel_data = {}
                    if hasattr(self.lightrag.relationships_vdb, 'get_all'):
                        logger.debug("relationships_vdb has get_all method, calling it")
                        rel_data = await self.lightrag.relationships_vdb.get_all()
                        logger.debug(f"Relationship data retrieved: {len(rel_data)} items")
                    
                    # Process relationship data
                    for rel_id, rel_info in rel_data.items():
                        relationship = {
                            "source": rel_info.get("src_id", "Unknown"),
                            "target": rel_info.get("tgt_id", "Unknown"),
                            "type": rel_info.get("keywords", "Unknown"),
                            "description": rel_info.get("content", "").split("\n")[-1] if rel_info.get("content") else ""
                        }
                        relationships.append(relationship)
                    
                    logger.info(f"Method 2: Retrieved {len(relationships)} relationships from vector database")
                except Exception as e:
                    logger.warning(f"Error accessing relationships_vdb: {e}")
            
            # Method 3: Try to access chunk_entity_relation_graph directly
            if not entities and hasattr(self.lightrag, 'chunk_entity_relation_graph'):
                try:
                    logger.debug(f"Attempting to access chunk_entity_relation_graph directly")
                    
                    # Try to get all nodes
                    if hasattr(self.lightrag.chunk_entity_relation_graph, 'get_all_nodes'):
                        logger.debug("chunk_entity_relation_graph has get_all_nodes method, calling it")
                        nodes = await self.lightrag.chunk_entity_relation_graph.get_all_nodes()
                        logger.debug(f"Nodes retrieved: {len(nodes) if nodes else 0}")
                        
                        # Process nodes
                        for node_id, node_data in nodes.items():
                            entity = {
                                "entity_name": node_id,
                                "entity_type": node_data.get("label", "Unknown"),
                                "description": node_data.get("description", ""),
                                "source": node_data.get("source", "")
                            }
                            entities.append(entity)
                        
                        logger.info(f"Method 3: Retrieved {len(entities)} entities from graph storage")
                    
                    # Try to get all edges
                    if hasattr(self.lightrag.chunk_entity_relation_graph, 'get_all_edges'):
                        logger.debug("chunk_entity_relation_graph has get_all_edges method, calling it")
                        edges = await self.lightrag.chunk_entity_relation_graph.get_all_edges()
                        logger.debug(f"Edges retrieved: {len(edges) if edges else 0}")
                        
                        # Process edges
                        for edge_id, edge_data in edges.items():
                            relationship = {
                                "source": edge_data.get("source", "Unknown"),
                                "target": edge_data.get("target", "Unknown"),
                                "type": edge_data.get("type", "Unknown"),
                                "description": edge_data.get("description", "")
                            }
                            relationships.append(relationship)
                        
                        logger.info(f"Method 3: Retrieved {len(relationships)} relationships from graph storage")
                except Exception as e:
                    logger.warning(f"Error accessing chunk_entity_relation_graph: {e}")
            
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
    
    async def aquery(self, query_text: str, param=None) -> Dict[str, Any]:
        """
        Query the LightRAG instance using the aquery method.
        
        Args:
            query_text: Query text as the first positional argument
            param: QueryParam object with additional parameters
            
        Returns:
            Dict containing query results
        """
        import logging
        
        logger = logging.getLogger("lightrag_interface")
        logger.debug(f"Querying with aquery: {query_text}")
        
        try:
            # Call the LightRAG aquery method
            logger.info("Using LightRAG aquery method")
            result = await self.lightrag.aquery(query_text, param=param)
            
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
            # Use the aquery method
            logger.info("Using LightRAG query method")
            from lightrag import QueryParam
            
            # Create QueryParam object with all the parameters
            param = QueryParam(
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
            
            # Call aquery with query_text as first argument and param as named parameter
            result = await self.lightrag.aquery(query_text, param=param)
            
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
        
        # Extract addon_params from config if available
        addon_params = self.config.get("addon_params", {
            "language": "English",
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "CONCEPT"],
            "example_number": 3  # Number of examples to use for entity extraction
        })
        
        self.client = LightRAGClient(
            base_url=base_url,
            api_key=api_key,
            addon_params=addon_params
        )
        
        # Check if the API is available
        try:
            health = await self.client.get_health()
            logger.info(f"Connected to LightRAG API at {base_url}: {health}")
        except Exception as e:
            logger.error(f"Failed to connect to LightRAG API at {base_url}: {e}")
            raise
            
    async def clear_storage(self) -> Dict[str, Any]:
        """
        Clear all data from the LightRAG instance.
        
        This method drops all documents, entities, and relationships from storage.
        
        Returns:
            Dict containing operation status
        """
        if not self.client:
            return {"status": "error", "message": "LightRAG API client not initialized"}
            
        try:
            # For API-based implementation, we need to call the appropriate API endpoints
            # This is a simplified implementation that assumes the API has a clear_documents endpoint
            try:
                # Try to call clear_documents endpoint if available
                await self.client.clear_documents()
                logger.info(f"Cleared all documents for session {self.session_id}")
                return {
                    "status": "success",
                    "message": f"All storage cleared for session {self.session_id}"
                }
            except Exception as e:
                logger.error(f"Error clearing documents via API: {e}")
                return {
                    "status": "error",
                    "message": f"Error clearing documents via API: {e}"
                }
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")
            return {
                "status": "error",
                "message": f"Error clearing storage: {e}"
            }
    
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
            
    async def aquery(self, query_text: str, param=None) -> Dict[str, Any]:
        """
        Query the LightRAG instance using the aquery method.
        
        Args:
            query_text: Query text as the first positional argument
            param: QueryParam object with additional parameters
            
        Returns:
            Dict containing query results
        """
        if not self.client:
            raise RuntimeError("LightRAG API client not initialized")
            
        try:
            # Extract parameters from QueryParam object if provided
            params = {}
            if param:
                # Convert QueryParam object to dictionary
                params = {
                    "mode": getattr(param, "mode", "mix"),
                    "top_k": getattr(param, "top_k", 10),
                    "only_need_context": getattr(param, "only_need_context", False),
                    "only_need_prompt": getattr(param, "only_need_prompt", False),
                    "response_type": getattr(param, "response_type", "Multiple Paragraphs"),
                    "max_token_for_text_unit": getattr(param, "max_token_for_text_unit", 1000),
                    "max_token_for_global_context": getattr(param, "max_token_for_global_context", 1000),
                    "max_token_for_local_context": getattr(param, "max_token_for_local_context", 1000),
                    "hl_keywords": getattr(param, "hl_keywords", None),
                    "ll_keywords": getattr(param, "ll_keywords", None),
                    "history_turns": getattr(param, "history_turns", 10),
                }
            
            # Call the query method with the appropriate parameters
            result = await self.client.query(query_text, **params)
            return result
        except Exception as e:
            logger.error(f"Error querying LightRAG API: {e}")
            return {
                "status": "error",
                "message": str(e),
                "session_id": self.session_id,
                "memories": []
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
            # Create parameters dictionary for API client
            params = {
                "mode": mode,
                "top_k": top_k,
                "only_need_context": only_need_context,
                "only_need_prompt": only_need_prompt,
                "response_type": response_type,
                "max_token_for_text_unit": max_token_for_text_unit,
                "max_token_for_global_context": max_token_for_global_context,
                "max_token_for_local_context": max_token_for_local_context,
                "hl_keywords": hl_keywords,
                "ll_keywords": ll_keywords,
                "history_turns": history_turns,
            }
            
            # Call the query method with the appropriate parameters
            # Note: The API client might have a different interface than the direct LightRAG instance
            result = await self.client.query(query_text, **params)
            return result
        except Exception as e:
            logger.error(f"Error querying LightRAG API: {e}")
            return {
                "status": "error",
                "message": str(e)
            }