#!/usr/bin/env python3
"""
Run script for the MCP Memory server.

This script provides a convenient way to start the MCP Memory server
with various configuration options.
"""

import argparse
import sys
from mcp_mem import main, update_config, get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP Memory server")
    
    # Transport options
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="Transport protocol to use (stdio or sse)"
    )
    parser.add_argument(
        "--host",
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
    
    # Memory configuration options
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Directory to store memory data (default: ~/.mcp-mem)",
    )
    parser.add_argument(
        "--disable-hipporag",
        action="store_true",
        help="Disable HippoRAG and use basic memory storage",
    )
    parser.add_argument(
        "--retrieve-limit",
        type=int,
        default=None,
        help="Default number of memories to retrieve",
    )
    
    args = parser.parse_args()
    
    # Update configuration based on command-line arguments
    config_updates = {}
    
    if args.memory_dir:
        config_updates["memory_dir"] = args.memory_dir
    
    if args.disable_hipporag:
        config_updates["use_hipporag"] = False
    
    if args.retrieve_limit:
        config_updates["default_retrieve_limit"] = args.retrieve_limit
    
    if config_updates:
        update_config(config_updates)
    
    # Pass transport arguments to main function
    sys.argv = [sys.argv[0]]
    
    # Add transport argument
    sys.argv.append("--transport")
    sys.argv.append(args.transport)
    
    # Add host and port arguments only for SSE transport
    if args.transport == "sse":
        sys.argv.append("--host")
        sys.argv.append(args.host)
        
        sys.argv.append("--port")
        sys.argv.append(str(args.port))
    
    # Always add debug flag (it will be handled correctly in server.py)
    if args.debug:
        sys.argv.append("--debug")
    
    # Start the server
    main()