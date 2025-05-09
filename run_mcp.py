#!/usr/bin/env python3
"""
Simple script to run the MCP Memory server.
This is a convenience wrapper around the mcp_mem.server module.
"""

import sys
import os
import logging
import argparse
from mcp_mem.server import main
from mcp_mem.config import update_config

if __name__ == "__main__":
    # Parse command line arguments
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
    
    # Parse arguments and update sys.argv for the main function
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    
    if args.transport:
        sys.argv.extend(["--transport", args.transport])
    if args.host:
        sys.argv.extend(["--host", args.host])
    if args.port:
        sys.argv.extend(["--port", str(args.port)])
    if args.debug:
        sys.argv.append("--debug")
    if args.integration_type:
        sys.argv.extend(["--integration-type", args.integration_type])
    if args.lightrag_api_url:
        sys.argv.extend(["--lightrag-api-url", args.lightrag_api_url])
    if args.lightrag_api_key:
        sys.argv.extend(["--lightrag-api-key", args.lightrag_api_key])
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Add current directory to path to ensure imports work
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run the server
    main()