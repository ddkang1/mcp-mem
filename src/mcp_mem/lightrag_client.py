"""
Client for interacting with LightRAG API.
"""

import logging
import aiohttp
import json
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class LightRAGClient:
    """
    Client for interacting with LightRAG API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        addon_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LightRAG API client.

        Args:
            base_url (str): Base API URL.
            api_key (str, optional): API key (token).
            addon_params (Dict[str, Any], optional): Additional parameters for entity extraction.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.addon_params = addon_params or {
            "language": "English",
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "CONCEPT"],
            "example_number": 3
        }
        self.session = None
        logger.info(f"Initialized LightRAG API client: {base_url}")

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )

    async def _call_api(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call LightRAG API endpoint.

        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            json_data (Dict[str, Any], optional): JSON data for request body
            params (Dict[str, Any], optional): Query parameters

        Returns:
            Dict[str, Any]: API response
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Calling API: {method} {url}")
            
            async with self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                logger.debug(f"API call successful: {method} {url}")
                return result
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error during {method} {url}: {e.status} - {e.message}")
            raise
        except Exception as e:
            logger.error(f"Error during {method} {url}: {str(e)}")
            raise

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
        hl_keywords: List[str] = None,
        ll_keywords: List[str] = None,
        history_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute a query to LightRAG API.

        Args:
            query_text (str): Query text
            mode (str, optional): Search mode (global, hybrid, local, mix, naive). Default is "mix".
            response_type (str, optional): Response format. Default is "Multiple Paragraphs".
            top_k (int, optional): Number of results. Default is 10.
            only_need_context (bool, optional): Return only context without LLM response. Default is False.
            only_need_prompt (bool, optional): Return only generated prompt without creating a response. Default is False.
            max_token_for_text_unit (int, optional): Maximum tokens for each text fragment. Default is 1000.
            max_token_for_global_context (int, optional): Maximum tokens for global context. Default is 1000.
            max_token_for_local_context (int, optional): Maximum tokens for local context. Default is 1000.
            hl_keywords (list[str], optional): List of high-level keywords for prioritization. Default is [].
            ll_keywords (list[str], optional): List of low-level keywords for search refinement. Default is [].
            history_turns (int, optional): Number of conversation turns in response context. Default is 10.

        Returns:
            Dict[str, Any]: Query result
        """
        logger.debug(f"Executing query: {query_text[:100]}...")

        request = {
            "query": query_text,
            "mode": mode,
            "response_type": response_type,
            "top_k": top_k,
            "only_need_context": only_need_context,
            "only_need_prompt": only_need_prompt,
            "max_token_for_text_unit": max_token_for_text_unit,
            "max_token_for_global_context": max_token_for_global_context,
            "max_token_for_local_context": max_token_for_local_context,
            "hl_keywords": hl_keywords or [],
            "ll_keywords": ll_keywords or [],
            "history_turns": history_turns,
            "addon_params": self.addon_params,
        }

        return await self._call_api(
            method="POST",
            endpoint="/query",
            json_data=request,
        )

    async def insert_text(
        self,
        text: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Add text to LightRAG.

        Args:
            text (Union[str, List[str]]): Text or list of texts to add

        Returns:
            Dict[str, Any]: Operation result
        """
        logger.debug(f"Adding text: {str(text)[:100]}...")

        if isinstance(text, str):
            request = {
                "text": text,
                "addon_params": self.addon_params
            }
            return await self._call_api(
                method="POST",
                endpoint="/documents/text",
                json_data=request,
            )
        else:
            request = {
                "texts": text,
                "addon_params": self.addon_params
            }
            return await self._call_api(
                method="POST",
                endpoint="/documents/texts",
                json_data=request,
            )

    async def get_health(self) -> Dict[str, Any]:
        """
        Check health status of LightRAG service.

        Returns:
            Dict[str, Any]: Health status.
        """
        logger.debug("Checking service health status...")
        return await self._call_api(
            method="GET",
            endpoint="/health",
        )

    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """
        Get the knowledge graph from LightRAG.
        
        Returns:
            Dict[str, Any]: Knowledge graph containing nodes and edges
        """
        logger.debug("Getting knowledge graph...")
        return await self._call_api(
            method="GET",
            endpoint="/graphs",
            params={"node_label": "*", "max_depth": 5, "max_nodes": 100}
        )
    
    async def close(self):
        """Close HTTP client."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("LightRAG API client closed.")