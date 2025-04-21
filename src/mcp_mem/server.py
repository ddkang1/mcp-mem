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
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from .config import get_config, MemoryConfig
from .memory_utils import cleanup_old_sessions
from .instance_manager import HippoRAGInstanceManager
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

# Import HippoRAG for knowledge graph management
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("memory")

# Get configuration
config = get_config()

# Create HippoRAG instance manager with TTL-based cleanup
hipporag_manager = HippoRAGInstanceManager()

# Register shutdown handler to ensure clean shutdown of the manager
atexit.register(hipporag_manager.shutdown)

# Ensure memory directory exists is now handled by the instance manager
    
@mcp.tool()
async def store_memory(session_id: str, content: str) -> Dict[str, Any]:
    """Add memory to a specific session.
    
    Args:
        session_id: Unique identifier for the chat session.
        content: The content to store in memory.
    
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
async def retrieve_memory(session_id: str, query: str, limit: int = None) -> Dict[str, Any]:
    """Retrieve memory from a specific session.
    
    Args:
        session_id: Unique identifier for the chat session.
        query: search query to filter memories.
        limit: Maximum number of memories to return.
    
    Returns:
        Dict containing retrieved memories.
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

# The get_hipporag_instance function is no longer needed as hipporag_manager.get now handles creation

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette app for SSE transport."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

def main():
    """Main entry point for the MCP Memory server."""
    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description="Run MCP Memory server")

    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run the server with SSE transport rather than STDIO (default: False)",
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 8000)"
    )
    args = parser.parse_args()

    if not args.sse and (args.host or args.port):
        parser.error("Host and port arguments are only valid when using SSE transport.")
        sys.exit(1)

    # Clean up old sessions if TTL is configured
    if config.session_ttl_days:
        removed = cleanup_old_sessions()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")

    # Log startup
    logger.info("Starting Memory MCP Server...")

    print("Starting Memory MCP Server...")
    
    if args.sse:
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(
            starlette_app,
            host=args.host if args.host else "127.0.0.1",
            port=args.port if args.port else 8000,
        )
    else:
        mcp.run()

if __name__ == "__main__":
    main()