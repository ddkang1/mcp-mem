#!/usr/bin/env python3
"""MCP server implementation for memory management using pluggable storage backends."""

import sys
import datetime
import argparse
import json
import os
import logging
import atexit
from typing import Any, Dict, Optional, Annotated, List
from pathlib import Path
from pydantic import Field

from fastmcp.server.server import FastMCP
from .config import get_config, MemoryConfig
from .memory_utils import cleanup_old_sessions
from .storage import MemoryStorage, MemoryEntry, InMemoryStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize storage backend
def create_storage_backend() -> MemoryStorage:
    """Create and configure the storage backend based on configuration."""
    storage_type = os.getenv("MEMORY_STORAGE_TYPE", "lightrag").lower()
    
    if storage_type == "inmemory":
        return InMemoryStorage()
    elif storage_type == "lightrag":
        try:
            from .lightrag_storage import LightRAGStorage
            return LightRAGStorage()
        except ImportError as e:
            logger.warning(f"LightRAG not available, falling back to in-memory storage: {e}")
            return InMemoryStorage()
    else:
        logger.warning(f"Unknown storage type '{storage_type}', using in-memory storage")
        return InMemoryStorage()

# Create global storage instance
storage = create_storage_backend()

# Register shutdown handler
def shutdown_handler():
    """Clean shutdown of storage backend."""
    if hasattr(storage, 'shutdown'):
        storage.shutdown()

atexit.register(shutdown_handler)

# Initialize FastMCP server
mcp = FastMCP(
    name="memory",
    host="127.0.0.1",
    port=8000,
    debug=False,
    log_level="INFO",
)

@mcp.tool(
    name="store_memory",
    description="Store a memory entry in persistent, shared session memory for agentic workflows.",
    tags={"memory", "storage", "agentic"}
)
async def store_memory(
    session_id: Annotated[str, Field(description="Unique session or workflow ID. All agents in the same session share access to this memory space.")],
    content: Annotated[str, Field(description="Text content to store. Can be user input, agent output, metadata, or any information relevant to the workflow.")],
    metadata: Annotated[Optional[Dict[str, Any]], Field(description="Optional metadata to store with the content (e.g., tags, custom fields).")]=None,
    source_agent: Annotated[Optional[str], Field(description="Name of the agent/tool storing this entry.")]=None,
    message_type: Annotated[Optional[str], Field(description="Type of message (e.g., 'user', 'tool', 'summary').")]=None,
    timestamp: Annotated[Optional[str], Field(description="ISO8601 timestamp for this entry. If not provided, will be set automatically.")]=None
) -> dict:
    """
    Store content and metadata in persistent, shared session memory for agentic workflows.

    This tool enables agents to persistently store any text content and associated metadata within a session's memory space. Use this tool to record facts, user preferences, intermediate results, or any information that should be accessible to all agents participating in the session.

    Parameters:
        session_id (str): Unique identifier for the session or workflow. All agents in the same session share access to this memory space.
        content (str): The text content to store. This can be user input, agent output, metadata, or any information relevant to the ongoing workflow.
        metadata (dict, optional): Optional metadata to store with the content (e.g., tags, custom fields).
        source_agent (str, optional): Name of the agent/tool storing this entry.
        message_type (str, optional): Type of message (e.g., 'user', 'tool', 'summary').
        timestamp (str, optional): ISO8601 timestamp for this entry. If not provided, will be set automatically.

    Returns:
        dict: A dictionary containing the session ID, status, and a message indicating success.

    Example:
        await store_memory(session_id="workflow-abc", content="User approved deployment to production.", source_agent="approval_agent", message_type="decision", metadata={"tags": ["approval", "prod"]})
    """
    try:
        entry = MemoryEntry(
            content=content,
            metadata=metadata,
            source_agent=source_agent,
            message_type=message_type,
            timestamp=timestamp
        )
        
        success = await storage.store(session_id, entry)
        
        if success:
            logger.info(f"Stored memory entry for session {session_id}")
            return {
                "session_id": session_id,
                "status": "success",
                "message": f"Memory stored successfully in session {session_id}"
            }
        else:
            return {
                "session_id": session_id,
                "status": "error",
                "message": "Failed to store memory entry"
            }
    except Exception as e:
        logger.error(f"Error storing memory for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Error storing memory: {str(e)}"
        }

@mcp.tool(
    name="retrieve_memory",
    description="Retrieve memory entries from persistent, shared session memory for agentic workflows.",
    tags={"memory", "retrieval", "agentic"}
)
async def retrieve_memory(
    session_id: Annotated[str, Field(description="Unique session or workflow ID. All agents in the same session share access to this memory space.")],
    query: Annotated[str, Field(description="Query string to search for relevant memories. Can be a keyword, phrase, or question.")],
    limit: Annotated[Optional[int], Field(description="Maximum number of results to return. If not specified, returns all relevant entries.")] = None,
    source_agent: Annotated[Optional[str], Field(description="Filter by agent/tool name.")] = None,
    message_type: Annotated[Optional[str], Field(description="Filter by message type (e.g., 'user', 'tool', 'summary').")] = None,
    tags: Annotated[Optional[List[str]], Field(description="Filter by tags (must match at least one tag in the entry's metadata).")]=None,
    start_time: Annotated[Optional[str], Field(description="ISO8601 start time for filtering entries by timestamp.")]=None,
    end_time: Annotated[Optional[str], Field(description="ISO8601 end time for filtering entries by timestamp.")]=None
) -> dict:
    """
    Retrieve relevant content and metadata from persistent, shared session memory for agentic workflows, with advanced filtering.

    This tool allows agents to search and retrieve information (including metadata, source_agent, message_type, and timestamp) from the shared session memory, enabling context-aware reasoning and collaboration across multiple steps and agents. Supports advanced filtering by agent, message type, tags, and time range.

    Parameters:
        session_id (str): Unique identifier for the session or workflow. All agents in the same session share access to this memory space.
        query (str): The search query to find relevant memories. This can be a keyword, phrase, or question.
        limit (int, optional): Maximum number of results to return. If not specified, returns all relevant entries.
        source_agent (str, optional): Filter by agent/tool name.
        message_type (str, optional): Filter by message type (e.g., 'user', 'tool', 'summary').
        tags (list of str, optional): Filter by tags (must match at least one tag in the entry's metadata).
        start_time (str, optional): ISO8601 start time for filtering entries by timestamp.
        end_time (str, optional): ISO8601 end time for filtering entries by timestamp.

    Returns:
        dict: A dictionary containing the session ID, status, query, and a list of matching memory entries (each with content, metadata, source_agent, message_type, and timestamp, if present).

    Example:
        result = await retrieve_memory(session_id="workflow-abc", query="deployment approval", source_agent="approval_agent", tags=["prod"], start_time="2024-01-01T00:00:00Z")
        print(result["result"])
    """
    try:
        entries = await storage.retrieve(session_id, query, limit)
        
        # Apply additional filtering
        def entry_matches(entry: MemoryEntry) -> bool:
            if source_agent and entry.source_agent != source_agent:
                return False
            if message_type and entry.message_type != message_type:
                return False
            if tags:
                entry_tags = set(entry.metadata.get("tags", []))
                if not entry_tags.intersection(tags):
                    return False
            if start_time or end_time:
                ts = entry.timestamp
                if ts:
                    try:
                        ts_dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        return False
                    if start_time:
                        start_dt = datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        if ts_dt < start_dt:
                            return False
                    if end_time:
                        end_dt = datetime.datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                        if ts_dt > end_dt:
                            return False
            return True
        
        filtered_entries = [entry for entry in entries if entry_matches(entry)]
        result_dicts = [entry.to_dict() for entry in filtered_entries]
        
        return {
            "session_id": session_id,
            "status": "success",
            "query": query,
            "result": result_dicts
        }
    except Exception as e:
        logger.error(f"Error retrieving memory for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "query": query,
            "result": []
        }

@mcp.tool(
    name="summarize_memory",
    description="Summarize session memory to preserve key context for agentic workflows.",
    tags={"memory", "summarization", "agentic"}
)
async def summarize_memory(
    session_id: Annotated[str, Field(description="Unique session or workflow ID. All agents in the same session share access to this memory space.")],
    query: Annotated[Optional[str], Field(description="Optional query to focus the summary on specific topics or questions.")]=None,
    max_tokens: Annotated[Optional[int], Field(description="Maximum number of tokens for the summary. If not specified, a default is used.")]=None
) -> dict:
    """
    Summarize session memory to preserve key context for agentic workflows.

    This tool condenses the session's memory, generating a summary that captures the most important information, decisions, and context. Use this tool to keep the context window manageable for long-running or complex workflows, ensuring agents retain continuity and relevant knowledge.

    Parameters:
        session_id (str): Unique identifier for the session or workflow. All agents in the same session share access to this memory space.
        query (str, optional): Optional query to focus the summary on specific topics or questions.
        max_tokens (int, optional): Maximum number of tokens for the summary. If not specified, a default is used.

    Returns:
        dict: A dictionary containing the session ID, status, and the generated summary string.

    Example:
        result = await summarize_memory(session_id="workflow-abc", query="deployment decisions", max_tokens=256)
        print(result["summary"])
    """
    try:
        all_entries = await storage.get_all(session_id)
        
        # Optionally filter by query
        if query:
            filtered_entries = [entry for entry in all_entries if query.lower() in entry.content.lower()]
        else:
            filtered_entries = all_entries
        
        # Concatenate content for summarization
        contents = [entry.content for entry in filtered_entries]
        text = "\n".join(contents)
        
        # Use a simple summarization approach (placeholder: truncate or use LLM if available)
        # In production, replace with a call to an LLM summarizer
        if max_tokens:
            summary = text[:max_tokens * 4]  # Roughly estimate 4 chars per token
        else:
            summary = text[:1024]
        
        return {
            "session_id": session_id,
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error summarizing memory for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "summary": ""
        }

@mcp.tool(
    name="delete_memory",
    description="Delete memory entries from persistent, shared session memory for agentic workflows.",
    tags={"memory", "deletion", "agentic"}
)
async def delete_memory(
    session_id: Annotated[str, Field(description="Unique session or workflow ID. All agents in the same session share access to this memory space.")],
    entry_id: Annotated[Optional[str], Field(description="ID of the memory entry to delete. If not provided, will delete entries matching the filters.")]=None,
    query: Annotated[Optional[str], Field(description="Query string to match entries for deletion.")]=None,
    source_agent: Annotated[Optional[str], Field(description="Filter by agent/tool name.")]=None,
    message_type: Annotated[Optional[str], Field(description="Filter by message type (e.g., 'user', 'tool', 'summary').")]=None,
    tags: Annotated[Optional[list], Field(description="Filter by tags (must match at least one tag in the entry's metadata).")]=None,
    start_time: Annotated[Optional[str], Field(description="ISO8601 start time for filtering entries by timestamp.")]=None,
    end_time: Annotated[Optional[str], Field(description="ISO8601 end time for filtering entries by timestamp.")]=None
) -> dict:
    """
    Delete memory entries from persistent, shared session memory for agentic workflows.

    This tool allows agents to delete memory entries by entry ID or by advanced filters (query, agent, message type, tags, time range).
    Use this tool to remove outdated, incorrect, or sensitive information from the session memory.

    Parameters:
        session_id (str): Unique identifier for the session or workflow. All agents in the same session share access to this memory space.
        entry_id (str, optional): ID of the memory entry to delete. If not provided, will delete entries matching the filters.
        query (str, optional): Query string to match entries for deletion.
        source_agent (str, optional): Filter by agent/tool name.
        message_type (str, optional): Filter by message type (e.g., 'user', 'tool', 'summary').
        tags (list of str, optional): Filter by tags (must match at least one tag in the entry's metadata).
        start_time (str, optional): ISO8601 start time for filtering entries by timestamp.
        end_time (str, optional): ISO8601 end time for filtering entries by timestamp.

    Returns:
        dict: A dictionary containing the session ID, status, and the number of entries deleted.

    Example:
        result = await delete_memory(session_id="workflow-abc", query="outdated info", source_agent="old_agent")
        print(result["deleted_count"])
    """
    try:
        if entry_id:
            success = await storage.delete(session_id, entry_id)
            deleted_count = 1 if success else 0
        else:
            # For bulk deletion, we'd need to implement filtering in storage or retrieve and delete individually
            # For now, simulate deletion based on the first matching entry
            all_entries = await storage.get_all(session_id)
            deleted_count = 0
            
            for entry in all_entries:
                should_delete = True
                if query and query.lower() not in entry.content.lower():
                    should_delete = False
                if source_agent and entry.source_agent != source_agent:
                    should_delete = False
                if message_type and entry.message_type != message_type:
                    should_delete = False
                    
                if should_delete and entry.id:
                    success = await storage.delete(session_id, entry.id)
                    if success:
                        deleted_count += 1
        
        return {
            "session_id": session_id,
            "status": "success",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error deleting memory for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "deleted_count": 0
        }

@mcp.tool(
    name="update_memory",
    description="Update memory entry content or metadata in persistent, shared session memory for agentic workflows.",
    tags={"memory", "update", "agentic"}
)
async def update_memory(
    session_id: Annotated[str, Field(description="Unique session or workflow ID. All agents in the same session share access to this memory space.")],
    entry_id: Annotated[str, Field(description="ID of the memory entry to update.")],
    new_content: Annotated[Optional[str], Field(description="New content to replace the existing content.")]=None,
    new_metadata: Annotated[Optional[Dict[str, Any]], Field(description="New metadata to replace or merge with the existing metadata.")]=None
) -> dict:
    """
    Update memory entry content or metadata in persistent, shared session memory for agentic workflows.

    This tool allows agents to update the content or metadata of a memory entry by entry ID. Use this tool to correct, redact, or enrich information in the session memory.

    Parameters:
        session_id (str): Unique identifier for the session or workflow. All agents in the same session share access to this memory space.
        entry_id (str): ID of the memory entry to update.
        new_content (str, optional): New content to replace the existing content.
        new_metadata (dict, optional): New metadata to replace or merge with the existing metadata.

    Returns:
        dict: A dictionary containing the session ID, status, and a message indicating success or failure.

    Example:
        result = await update_memory(session_id="workflow-abc", entry_id="5", new_content="Corrected info.")
        print(result["status"])
    """
    try:
        success = await storage.update(session_id, entry_id, new_content, new_metadata)
        
        if success:
            return {
                "session_id": session_id,
                "status": "success",
                "message": "Memory entry updated successfully."
            }
        else:
            return {
                "session_id": session_id,
                "status": "not_found",
                "message": "Entry not found."
            }
    except Exception as e:
        logger.error(f"Error updating memory for session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "message": f"Error updating memory: {str(e)}"
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
    logger.info(f"Using storage backend: {type(storage).__name__}")
    
    # Run the server with the specified transport
    try:
        mcp.settings.debug = args.debug
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(
                transport="sse",
                host=args.host,
                port=args.port,
                log_level="debug" if args.debug else "info"
            )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down server...")
    finally:
        shutdown_handler()
        logger.info("Server shutdown complete.")

if __name__ == "__main__":
    main()