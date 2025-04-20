#!/usr/bin/env python3
"""MCP server implementation for memory management."""

import sys
import datetime
import argparse
import json
import os
import logging
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from .config import get_config, MemoryConfig
from .memory_utils import get_session_memories, search_memories, cleanup_old_sessions, get_memory_stats
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

# Import HippoRAG for knowledge graph management
try:
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
except ImportError:
    print("Warning: HippoRAG not found. Please install it to use advanced memory features.")
    HippoRAG = None
    BaseConfig = None

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

# Store for session memories
# Structure: {session_id: {"hipporag": HippoRAG instance, "memories": [memory entries]}}
session_memories = {}

# Ensure memory directory exists
os.makedirs(config.memory_dir, exist_ok=True)

def get_session_path(session_id: str) -> str:
    """Get the path for storing session data."""
    return os.path.join(config.memory_dir, f"session_{session_id}")

def ensure_session_exists(session_id: str) -> None:
    """Ensure that a session directory exists."""
    session_path = get_session_path(session_id)
    os.makedirs(session_path, exist_ok=True)

def initialize_hipporag(session_id: str) -> Optional[HippoRAG]:
    """Initialize HippoRAG for a session."""
    if HippoRAG is None or not config.use_hipporag:
        logger.warning("HippoRAG not available or disabled. Using basic memory storage.")
        return None
    
    session_path = get_session_path(session_id)
    hippo_config = BaseConfig()
    hippo_config.save_dir = session_path
    
    # Apply HippoRAG configuration from our config
    for key, value in config.hipporag_config.items():
        if hasattr(hippo_config, key) and value is not None:
            setattr(hippo_config, key, value)
    
    return HippoRAG(
        global_config=hippo_config,
        save_dir=session_path
    )

@mcp.tool()
async def create_memory(session_id: str) -> Dict[str, Any]:
    """Create a new memory for a given chat session.
    
    Args:
        session_id: Unique identifier for the chat session.
    
    Returns:
        Dict containing session information.
    """
    # Check if session already exists
    if session_id in session_memories:
        return {
            "session_id": session_id,
            "status": "exists",
            "message": f"Memory for session {session_id} already exists"
        }
    
    # Create session directory
    ensure_session_exists(session_id)
    
    # Initialize HippoRAG for knowledge graph
    hipporag_instance = initialize_hipporag(session_id)
    
    # Create session memory
    session_memories[session_id] = {
        "hipporag": hipporag_instance,
        "memories": [],
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Save initial session state
    save_session_state(session_id)
    
    return {
        "session_id": session_id,
        "status": "created",
        "message": f"Memory for session {session_id} created successfully"
    }

@mcp.tool()
async def store_memory(session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add memory to a specific session.
    
    Args:
        session_id: Unique identifier for the chat session.
        content: The content to store in memory.
        metadata: Optional metadata associated with the memory.
    
    Returns:
        Dict containing operation status.
    """
    # Check if session exists
    if session_id not in session_memories:
        # Create session if it doesn't exist
        await create_memory(session_id)
    
    # Create memory entry
    timestamp = datetime.datetime.now().isoformat()
    memory_id = str(uuid4())
    
    # Combine provided metadata with default metadata
    combined_metadata = {**config.default_metadata}
    if metadata:
        combined_metadata.update(metadata)
    
    memory_entry = {
        "id": memory_id,
        "content": content,
        "timestamp": timestamp,
        "metadata": combined_metadata
    }
    
    # Add to session memories
    session_memories[session_id]["memories"].append(memory_entry)
    
    # If HippoRAG is available, index the content
    hipporag_instance = session_memories[session_id]["hipporag"]
    if hipporag_instance:
        try:
            hipporag_instance.index([content])
            logger.info(f"Indexed content in HippoRAG for session {session_id}")
        except Exception as e:
            logger.error(f"Error indexing content in HippoRAG: {str(e)}")
    
    # Save updated session state
    save_session_state(session_id)
    
    return {
        "session_id": session_id,
        "memory_id": memory_id,
        "status": "stored",
        "message": f"Memory stored successfully in session {session_id}"
    }

@mcp.tool()
async def retrieve_memory(session_id: str, query: str = None, limit: int = None) -> Dict[str, Any]:
    """Retrieve memory from a specific session.
    
    Args:
        session_id: Unique identifier for the chat session.
        query: Optional search query to filter memories.
        limit: Maximum number of memories to return.
    
    Returns:
        Dict containing retrieved memories.
    """
    # Check if session exists
    if session_id not in session_memories:
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Session {session_id} does not exist",
            "memories": []
        }
    
    session_data = session_memories[session_id]
    hipporag_instance = session_data["hipporag"]
    
    # Use default limit if not specified
    if limit is None:
        limit = config.default_retrieve_limit
    
    # If query is provided and HippoRAG is available, use it for retrieval
    if query and hipporag_instance:
        try:
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
        except Exception as e:
            logger.error(f"Error retrieving from HippoRAG: {str(e)}")
            # Fall back to basic retrieval
    
    # Use memory_utils for basic retrieval
    sorted_memories = search_memories(session_id, query, limit=limit)
    
    return {
        "session_id": session_id,
        "status": "success",
        "query": query,
        "memories": sorted_memories
    }

def save_session_state(session_id: str) -> None:
    """Save the current state of a session to disk."""
    session_path = get_session_path(session_id)
    state_file = os.path.join(session_path, "session_state.json")
    
    # Extract serializable data
    session_data = session_memories[session_id]
    serializable_data = {
        "memories": session_data["memories"],
        "created_at": session_data["created_at"]
    }
    
    with open(state_file, 'w') as f:
        json.dump(serializable_data, f)

def load_session_state(session_id: str) -> None:
    """Load a session state from disk."""
    session_path = get_session_path(session_id)
    state_file = os.path.join(session_path, "session_state.json")
    
    if not os.path.exists(state_file):
        return
    
    with open(state_file, 'r') as f:
        serializable_data = json.load(f)
    
    # Initialize HippoRAG
    hipporag_instance = initialize_hipporag(session_id)
    
    # Restore session data
    session_memories[session_id] = {
        "hipporag": hipporag_instance,
        "memories": serializable_data["memories"],
        "created_at": serializable_data["created_at"]
    }

def load_all_sessions() -> None:
    """Load all saved sessions from disk."""
    if not os.path.exists(config.memory_dir):
        return
    
    session_dirs = [
        d for d in os.listdir(config.memory_dir)
        if os.path.isdir(os.path.join(config.memory_dir, d)) and d.startswith("session_")
    ]
    
    for session_dir in session_dirs:
        session_id = session_dir.replace("session_", "")
        load_session_state(session_id)
        logger.info(f"Loaded session {session_id}")
    
    # Clean up old sessions if TTL is configured
    if config.session_ttl_days:
        removed = cleanup_old_sessions()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")

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

@mcp.tool()
async def get_stats() -> Dict[str, Any]:
    """Get statistics about stored memories.
    
    Returns:
        Dict containing memory statistics.
    """
    return get_memory_stats()

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

    # Load existing sessions
    load_all_sessions()

    # Log memory stats
    stats = get_memory_stats()
    logger.info(f"Memory stats: {stats}")

    print(f"Starting Memory MCP Server...")
    
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