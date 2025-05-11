#!/usr/bin/env python3
"""MCP server implementation for memory management."""

import sys
import datetime
import argparse
import json
import os
import logging
import atexit
import re
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from fastmcp.server.server import FastMCP
from .config import get_config, MemoryConfig, update_config
from .memory_utils import cleanup_old_sessions, create_session_state, update_session_access
from .instance_manager import LightRAGInstanceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create LightRAG instance manager with TTL-based cleanup
lightrag_manager = LightRAGInstanceManager()

# Register shutdown handler to ensure clean shutdown of the manager
atexit.register(lightrag_manager.shutdown)

# Initialize FastMCP server
mcp = FastMCP(
    name="memory",
    host="127.0.0.1",
    port=8000,
    debug=False,
    log_level="INFO",
)

@mcp.tool()
async def store_memory(session_id: str, content: str) -> Dict[str, Any]:
    """Store content permanently in the unlimited external storage.
        
    IMPORTANT: Do NOT condense, summarize, or reduce information before storing.
    This tool is designed to store unlimited amounts of information in its complete form.
    Unlike typical memory features that require summarization due to space constraints,
    this tool preserves the complete content without any loss of information or detail.
    All stored data remains permanently accessible and retrievable in its original form
    to overcome context window limitations in future interactions.
        
    Args:
        session_id: Unique identifier for the permanent storage session.
        content: The complete content to store permanently. Send the full, unmodified content
                 as there are no storage limitations or need for summarization.
        
    Returns:
        Dict containing operation status.
    """
    # Get or create LightRAG instance
    lightrag_instance = await lightrag_manager.get(session_id)
    
    # Update session state
    create_session_state(session_id)
    update_session_access(session_id)
    
    # Store content using LightRAG
    result = await lightrag_instance.insert(content)
    logger.info(f"Indexed content in LightRAG for session {session_id}")
    
    return {
        "session_id": session_id,
        "status": "success",
        "message": f"Memory stored successfully in session {session_id}"
    }

@mcp.tool()
async def retrieve_memory(session_id: str, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve previously stored content from the AI's permanent memory system.
    
    This tool enables access to the unlimited storage of previously encountered
    information, allowing the AI to overcome context window limitations by
    retrieving complete, unaltered content that was stored in previous interactions.
    All content is preserved and retrieved in its original form without any
    summarization or information loss - exactly as it was stored, regardless of size.
    
    Args:
        session_id: Unique identifier for the permanent storage session.
        query: Search query to filter and retrieve relevant memories.
        limit: Maximum number of memories to return. When not specified, returns
               all relevant memories based on system configuration.
    
    Returns:
        Dict containing retrieved memories with their original, complete content
        exactly as they were stored, with no loss of information or detail.
    """
    # Get or create LightRAG instance
    lightrag_instance = await lightrag_manager.get(session_id)
    
    # Update session access time
    update_session_access(session_id)
    
    # Use default limit if not specified
    if limit is None:
        limit = config.default_retrieve_limit
    
    # Use LightRAG for retrieval
    result = await lightrag_instance.query(
        query_text=query,
        mode="hybrid",  # Use hybrid mode to leverage both vector and graph search
        top_k=limit,
        only_need_context=True,  # We only need the context, not the LLM response
    )
    
    # Check if the result already contains memories
    logger.debug(f"Query result from memory storage: {result}")
    logger.debug(f"Query result type: {type(result)}")
    logger.debug(f"Query result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    memories_from_result = result.get("memories", []) if isinstance(result, dict) else []
    logger.debug(f"Memories from result: {memories_from_result}")
    logger.debug(f"Memories type: {type(memories_from_result)}")
    logger.debug(f"Memories length: {len(memories_from_result)}")
    
    if isinstance(memories_from_result, list) and len(memories_from_result) > 0:
        retrieved_memories = memories_from_result
        logger.debug(f"Using memories from result: {retrieved_memories}")
        
        # Add rank to each memory if not present
        for idx, memory in enumerate(retrieved_memories):
            logger.debug(f"Processing memory {idx+1}: {memory}")
            logger.debug(f"Memory type: {type(memory)}")
            if isinstance(memory, dict):
                if "rank" not in memory:
                    memory["rank"] = idx + 1
                logger.debug(f"Memory {idx+1} after processing: {memory}")
            else:
                logger.debug(f"Memory {idx+1} is not a dict, skipping")
    else:
        # Parse the results from LightRAG response
        retrieved_memories = []
        
        try:
            # Extract document chunks section from the response
            if isinstance(result.get("response"), str):
                response = result["response"]
                # Find the document chunks section
                chunks_section = re.search(r'-----Document Chunks\(DC\)-----\s*```json\s*(.*?)\s*```',
                                          response, re.DOTALL)
                
                if chunks_section:
                    chunks_json = chunks_section.group(1).strip()
                    chunks = json.loads(chunks_json)
                    
                    # Format the chunks as memories
                    for idx, chunk in enumerate(chunks):
                        retrieved_memories.append({
                            "content": chunk.get("content", ""),
                            "score": float(chunk.get("score", idx + 1)),
                            "rank": idx + 1
                        })
        except Exception as e:
            logger.error(f"Error parsing LightRAG results: {str(e)}")
    
    # Force memories to be a list of dictionaries with content and score
    if not retrieved_memories:
        logger.debug("No memories found, returning empty list")
    
    # Ensure memories are properly serializable by converting to simple dictionaries
    serializable_memories = []
    for memory in retrieved_memories:
        # Convert each memory to a simple dictionary with only the essential fields
        serializable_memory = {
            "content": str(memory.get("content", "")),
            "score": float(memory.get("score", 0.0)),
            "rank": int(memory.get("rank", 1))
        }
        serializable_memories.append(serializable_memory)
    
    # Create the result with serializable memories
    result_to_return = {
        "session_id": session_id,
        "status": "success",
        "query": query,
        "memories": serializable_memories
    }
    
    logger.debug(f"Final result to return: {result_to_return}")
    logger.debug(f"Final memories count: {len(serializable_memories)}")
    for idx, memory in enumerate(serializable_memories):
        logger.debug(f"Final memory {idx+1}: {memory}")
    
    # Return the result as a simple dictionary that can be easily serialized
    return result_to_return

async def configure_memory(
    integration_type: Optional[str] = None,
    lightrag_api_base_url: Optional[str] = None,
    lightrag_api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Configure the memory system.
    
    This tool allows you to configure various aspects of the memory system,
    including the integration type (direct or API), API endpoints, and model settings.
    
    Args:
        integration_type: Integration type, either "direct" or "api".
        lightrag_api_base_url: Base URL for LightRAG API (for API integration).
        lightrag_api_key: API key for LightRAG API (for API integration).
        embedding_provider: Embedding provider, e.g., "openai".
        embedding_model_name: Embedding model name, e.g., "text-embedding-3-large".
        llm_provider: LLM provider, e.g., "openai".
        llm_model_name: LLM model name, e.g., "gpt-4o-mini".
        
    Returns:
        Dict containing operation status and current configuration.
    """
    # Create updates dictionary with only provided values
    updates = {}
    if integration_type is not None:
        if integration_type not in ["direct", "api"]:
            return {
                "status": "error",
                "message": "Invalid integration_type. Must be 'direct' or 'api'."
            }
        updates["integration_type"] = integration_type
    
    if lightrag_api_base_url is not None:
        updates["lightrag_api_base_url"] = lightrag_api_base_url
    
    if lightrag_api_key is not None:
        updates["lightrag_api_key"] = lightrag_api_key
    
    if embedding_provider is not None:
        updates["embedding_provider"] = embedding_provider
    
    if embedding_model_name is not None:
        updates["embedding_model_name"] = embedding_model_name
    
    if llm_provider is not None:
        updates["llm_provider"] = llm_provider
    
    if llm_model_name is not None:
        updates["llm_model_name"] = llm_model_name
    
    # Apply updates if any
    if updates:
        update_config(updates)
        logger.info(f"Updated configuration: {updates}")
    
    # Get current configuration
    current_config = get_config()
    
    # Return safe version of configuration (without sensitive data)
    safe_config = {
        "integration_type": current_config.integration_type,
        "lightrag_api_base_url": current_config.lightrag_api_base_url,
        "lightrag_api_key": "***" if current_config.lightrag_api_key else None,
        "embedding_provider": current_config.embedding_provider,
        "embedding_model_name": current_config.embedding_model_name,
        "llm_provider": current_config.llm_provider,
        "llm_model_name": current_config.llm_model_name,
        "default_retrieve_limit": current_config.default_retrieve_limit,
    }
    
    return {
        "status": "success",
        "message": "Configuration updated successfully" if updates else "Current configuration retrieved",
        "configuration": safe_config
    }

def main():
    """Main entry point for the MCP Memory server."""
    parser = argparse.ArgumentParser(description="Run MCP Memory server")

    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="Transport protocol to use (stdio or sse)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (for SSE transport)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (for SSE transport)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--integration-type",
        type=str,
        choices=["direct", "api"],
        help="Integration type (direct or api)"
    )
    parser.add_argument(
        "--lightrag-api-url",
        type=str,
        help="LightRAG API base URL (for API integration)"
    )
    parser.add_argument(
        "--lightrag-api-key",
        type=str,
        help="LightRAG API key (for API integration)"
    )
    args = parser.parse_args()

    # Configure logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Apply command line configuration
    updates = {}
    if args.integration_type:
        updates["integration_type"] = args.integration_type
    if args.lightrag_api_url:
        updates["lightrag_api_base_url"] = args.lightrag_api_url
    if args.lightrag_api_key:
        updates["lightrag_api_key"] = args.lightrag_api_key
    
    if updates:
        update_config(updates)
        logger.info(f"Applied command line configuration: {updates}")

    # Clean up old sessions if TTL is configured
    if config.session_ttl_days:
        removed = cleanup_old_sessions()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")

    # Log startup
    logger.info("Starting Memory MCP Server...")
    logger.info(f"Integration type: {config.integration_type}")
    if config.integration_type == "api":
        logger.info(f"LightRAG API URL: {config.lightrag_api_base_url}")
        logger.info(f"LightRAG API key: {'configured' if config.lightrag_api_key else 'not configured'}")
    
    # Run the server with the specified transport
    try:
        # Set debug mode in the server settings
        mcp.settings.debug = args.debug
        
        if args.transport == "stdio":
            # For stdio transport, don't pass host and port
            mcp.run(transport="stdio")
        else:
            # For SSE transport, pass only host and port
            mcp.run(
                transport="sse",
                host=args.host,
                port=args.port,
                log_level="debug" if args.debug else "info"
            )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down server...")
    finally:
        # Clean up resources
        lightrag_manager.shutdown()
        logger.info("Server shutdown complete.")

if __name__ == "__main__":
    main()