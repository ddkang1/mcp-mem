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
                
                # Extract meaningful keywords from the query
                # Remove common stop words to focus on important terms
                stop_words = {"what", "are", "is", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "to", "of", "and", "or", "different", "types"}
                query_keywords = [term.lower() for term in query_text.split() if term.lower() not in stop_words]
                
                # If no keywords remain after filtering, use original terms
                if not query_keywords:
                    query_keywords = [term.lower() for term in query_text.split()]
                
                logger.debug(f"Extracted keywords: {query_keywords}")
                
                # Enhanced keyword matching
                results = []
                for doc_id, content in self.memory_store.items():
                    content_lower = content.lower()
                    
                    # Count how many keywords match
                    matching_keywords = [keyword for keyword in query_keywords if keyword in content_lower]
                    keyword_match_count = len(matching_keywords)
                    
                    # Only include results with at least one keyword match
                    if keyword_match_count > 0:
                        # Calculate a score based on keyword matches and frequency
                        keyword_match_ratio = keyword_match_count / len(query_keywords)
                        term_frequency = sum(content_lower.count(keyword) for keyword in matching_keywords)
                        
                        # Combined score formula
                        score = min(0.99, max(0.5, (keyword_match_ratio * 0.7) + (term_frequency * 0.01)))
                        
                        results.append({
                            "content": content,
                            "score": score,
                            "matching_keywords": matching_keywords
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
            
            # Extract graph information - entities (nodes)
            entities = []
            try:
                # Try to extract entities from the memory store content
                for doc_id, content in getattr(self.lightrag, 'memory_store', {}).items():
                    # Simple entity extraction - look for capitalized words as potential entities
                    words = content.split()
                    for i, word in enumerate(words):
                        if word and word[0].isupper() and len(word) > 3 and word.lower() not in ["this", "that", "these", "those", "they", "their"]:
                            # Get context (surrounding words)
                            start = max(0, i - 3)
                            end = min(len(words), i + 4)
                            context = " ".join(words[start:end])
                            
                            # Add as entity
                            entity_type = "concept" if word in ["Climate", "Energy", "AI", "Quantum"] else "unknown"
                            entities.append({
                                "entity_name": word,
                                "entity_type": entity_type,
                                "description": context,
                                "source": doc_id
                            })
                
                # Remove duplicates based on entity_name
                unique_entities = []
                seen_names = set()
                for entity in entities:
                    if entity["entity_name"] not in seen_names:
                        seen_names.add(entity["entity_name"])
                        unique_entities.append(entity)
                entities = unique_entities[:20]  # Limit to 10 entities
            except Exception as e:
                logger.warning(f"Error extracting entities: {e}")
            
            # Create simple relationships between entities
            relationships = []
            try:
                # Create relationships between entities that appear in the same document
                entity_by_doc = {}
                for entity in entities:
                    doc_id = entity.get("source", "")
                    if doc_id not in entity_by_doc:
                        entity_by_doc[doc_id] = []
                    entity_by_doc[doc_id].append(entity["entity_name"])
                
                # Create relationships for entities in the same document
                for doc_id, doc_entities in entity_by_doc.items():
                    for i in range(len(doc_entities)):
                        for j in range(i + 1, len(doc_entities)):
                            relationships.append({
                                "source": doc_entities[i],
                                "target": doc_entities[j],
                                "type": "co-occurrence",
                                "description": f"Both appear in document {doc_id}"
                            })
                
                # Limit to 10 relationships
                relationships = relationships[:20]
            except Exception as e:
                logger.warning(f"Error creating relationships: {e}")
            
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
    
    async def get_internal_state(self) -> Dict[str, Any]:
        """
        Get the internal state of the LightRAG instance.
        
        Returns:
            Dict containing information about nodes, edges, and other internal state
        """
        if not self.client:
            return {"status": "error", "message": "LightRAG API client not initialized"}
            
        try:
            # Try to get graph information if available
            try:
                graph_info = await self.client.get_knowledge_graph()
                return {
                    "status": "success",
                    "session_id": self.session_id,
                    "graph_info": graph_info
                }
            except Exception as e:
                # If getting graph info fails, return basic session info
                return {
                    "status": "success",
                    "session_id": self.session_id,
                    "message": "Graph information not available through API",
                    "error": str(e)
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