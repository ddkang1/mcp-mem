"""
Custom LLM and embedding functions for LightRAG integration.
This module provides wrapper functions for OpenAI-compatible APIs.
"""

import os
import numpy as np
from typing import Any, List, Optional, Union, Dict

# Import the base functions from LightRAG
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed as base_openai_embed,
)
from lightrag.utils import wrap_embedding_func_with_attrs

# Environment variables
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://grai.gilead.com:4001")
LLM_NAME = os.environ.get("LLM_NAME", "gpt-4o-mini")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://grai.gilead.com:4001")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "key")


async def gpt_4o_mini_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """
    Custom wrapper for gpt-4o-mini completion using the specified environment variables.
    
    Args:
        prompt: The prompt to complete
        system_prompt: Optional system prompt to include
        history_messages: Optional list of previous messages in the conversation
        keyword_extraction: Whether to extract keywords from the response
        **kwargs: Additional keyword arguments to pass to the OpenAI API
        
    Returns:
        The completed text
    """
    if history_messages is None:
        history_messages = []
    
    # Remove keyword_extraction from kwargs to avoid duplicate
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    
    # Set response_format for keyword extraction if needed
    if keyword_extraction:
        from lightrag.types import GPTKeywordExtractionFormat
        kwargs["response_format"] = GPTKeywordExtractionFormat
    
    # Use the base OpenAI completion function with our custom parameters
    return await openai_complete_if_cache(
        model=LLM_NAME,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=OPENAI_API_KEY,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embed(
    texts: List[str],
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    client_configs: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Custom wrapper for OpenAI embeddings using the specified environment variables.
    
    Args:
        texts: List of texts to embed
        model: Optional model override (defaults to EMBEDDING_MODEL_NAME)
        base_url: Optional base URL override (defaults to EMBEDDING_BASE_URL)
        api_key: Optional API key override (defaults to OPENAI_API_KEY)
        client_configs: Additional configuration options for the AsyncOpenAI client
        
    Returns:
        A numpy array of embeddings, one per input text
    """
    # Use environment variables as defaults
    model = model or EMBEDDING_MODEL_NAME
    base_url = base_url or EMBEDDING_BASE_URL
    api_key = api_key or OPENAI_API_KEY
    
    # Use the base OpenAI embedding function with our custom parameters
    return await base_openai_embed(
        texts=texts,
        model=model,
        base_url=base_url,
        api_key=api_key,
        client_configs=client_configs,
    )