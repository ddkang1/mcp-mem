#!/usr/bin/env python3
"""MCP server implementation for memory management."""

import sys
import datetime
import argparse
import json
import os
import logging
import atexit
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from fastmcp.server.server import FastMCP
from .config import get_config, MemoryConfig
from .memory_utils import cleanup_old_sessions
from .instance_manager import HippoRAGInstanceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create HippoRAG instance manager with TTL-based cleanup
hipporag_manager = HippoRAGInstanceManager()

# Register shutdown handler to ensure clean shutdown of the manager
atexit.register(hipporag_manager.shutdown)

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
    # Get or create HippoRAG instance
    hipporag_instance = await hipporag_manager.get(session_id)
    hipporag_instance.index([content])
    logger.info(f"Indexed content in HippoRAG for session {session_id}")
    
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
    # Get or create HippoRAG instance
    hipporag_instance = await hipporag_manager.get(session_id)
    
    # Use default limit if not specified
    # if limit is None:
    #     limit = config.default_retrieve_limit
    
    # Use HippoRAG for retrieval
    retrieval_results = hipporag_instance.retrieve([query], num_to_retrieve=limit)
    
    # Format results
    retrieved_memories = []
    for idx, result in enumerate(retrieval_results):
        for doc_idx, doc in enumerate(result.docs):
            retrieved_memories.append({
                "content": doc,
                "score": float(result.doc_scores[doc_idx]) if doc_idx < len(result.doc_scores) else 0.0,
                "rank": doc_idx + 1
            })
    
    return {
        "session_id": session_id,
        "status": "success",
        "query": query,
        "memories": retrieved_memories
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
    args = parser.parse_args()

    # Configure logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Clean up old sessions if TTL is configured
    if config.session_ttl_days:
        removed = cleanup_old_sessions()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")

    # Log startup
    logger.info("Starting Memory MCP Server...")
    
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
        hipporag_manager.shutdown()
        logger.info("Server shutdown complete.")

if __name__ == "__main__":
    main()