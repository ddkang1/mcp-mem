 #!/usr/bin/env python3
"""
Example script demonstrating how to use the MCP Memory server with LightRAG.
This script shows how to store and retrieve memories using the MCP client.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import Dict, Any, List
from fastmcp.client.client import Client

import httpx
# Add parent directory to path to ensure imports work

async def main():
    """Main function demonstrating MCP Memory usage."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Memory Client Example")
    parser.add_argument(
        "--integration-type",
        type=str,
        choices=["direct", "api"],
        default="direct",
        help="Integration type (direct or api)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    parser.add_argument(
        "--lightrag-api-url",
        type=str,
        default="http://localhost:8000",
        help="LightRAG API base URL (for API integration)"
    )
    parser.add_argument(
        "--lightrag-api-key",
        type=str,
        help="LightRAG API key (for API integration)"
    )
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("mcp_memory_client")
    
    if args.debug:
        logger.debug("Debug mode enabled")
        logger.debug(f"Arguments: {args}")
    
    # Pre-flight check: only for API mode
    if args.integration_type == "api":
        try:
            async with httpx.AsyncClient() as ac:
                resp = await ac.get(args.lightrag_api_url)
                resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: Could not connect to LightRAG API at {args.lightrag_api_url}.")
            print("Please ensure the LightRAG server is running and accessible at this address.")
            print(f"Details: {e}")
            return

    # Create MCP client
    if args.integration_type == "api":
        client_transport = args.lightrag_api_url
        if args.debug:
            logger.debug(f"Using API integration with URL: {args.lightrag_api_url}")
    else:
        client_transport = {
            "mcpServers": {
                "memory": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "mcp_mem.server"],
                    "env": {
                        "PYTHONPATH": "."
                    }
                }
            }
        }
        if args.debug:
            logger.debug("Using direct integration with stdio transport")
            logger.debug(f"Client transport configuration: {json.dumps(client_transport, indent=2)}")

    async with Client(client_transport) as client:
        print("Connected to MCP Memory server")
        if args.debug:
            logger.debug("Client connection established")

        # List available tools for debugging
        tools = await client.list_tools()
        print("Available tools:", tools)
        if args.debug:
            logger.debug(f"Available tools: {json.dumps(tools, indent=2)}")
        # Configure the memory system (API mode only)
        if args.integration_type == "api":
            print("\nConfiguring memory system...")
            config_result = await client.call_tool(
                "configure_memory",
                {
                    "integration_type": "api",
                    "lightrag_api_base_url": args.lightrag_api_url,
                    "lightrag_api_key": args.lightrag_api_key,
                }
            )
            print(f"Configuration result: {json.dumps(config_result, indent=2)}")
        
        # Define a session ID
        session_id = "example_session"
        
        # Example content to store
        content1 = """
        Climate change is the long-term alteration of temperature and typical weather patterns in a place.
        Climate change could refer to a particular location or the planet as a whole.
        Climate change may cause weather patterns to be less predictable.
        These unexpected weather patterns can make it difficult to maintain and grow crops in regions that rely on farming.
        """
        
        content2 = """
        The primary cause of climate change is human activities, particularly the burning of fossil fuels,
        like coal, oil, and natural gas. Burning these materials releases what are called greenhouse gases into Earth's atmosphere.
        These gases trap heat from the sun's rays inside the atmosphere causing Earth's average temperature to rise.
        """
        
        # Store memories
        print("\nStoring first memory...")
        if args.debug:
            logger.debug(f"Storing memory with session_id: {session_id}")
            logger.debug(f"Content length: {len(content1)} characters")
        
        store_result1 = await client.call_tool(
            "store_memory",
            {
                "session_id": session_id,
                "content": content1
            }
        )
        print(f"Result: {json.dumps(store_result1, indent=2)}")
        if args.debug:
            logger.debug(f"Store result: {json.dumps(store_result1)}")
        
        print("\nStoring second memory...")
        if args.debug:
            logger.debug(f"Storing second memory with session_id: {session_id}")
            logger.debug(f"Content length: {len(content2)} characters")
            
        store_result2 = await client.call_tool(
            "store_memory",
            {
                "session_id": session_id,
                "content": content2
            }
        )
        print(f"Result: {json.dumps(store_result2, indent=2)}")
        if args.debug:
            logger.debug(f"Store result: {json.dumps(store_result2)}")
        
        # Wait a moment for indexing to complete
        print("\nWaiting for indexing to complete...")
        if args.debug:
            logger.debug("Waiting for indexing to complete (2 seconds)")
        await asyncio.sleep(2)
        
        # Retrieve memories
        print("\nRetrieving memories about 'greenhouse gases'...")
        query = "What are greenhouse gases?"
        limit = 3
        
        if args.debug:
            logger.debug(f"Retrieving memories with query: '{query}'")
            logger.debug(f"Session ID: {session_id}, Limit: {limit}")
            
        retrieve_result = await client.call_tool(
            "retrieve_memory",
            {
                "session_id": session_id,
                "query": query,
                "limit": limit
            }
        )
        
        if args.debug:
            logger.debug(f"Retrieved {len(retrieve_result.get('memories', []))} memories")
            logger.debug(f"Full retrieve result: {json.dumps(retrieve_result, indent=2)}")
        
        print("\nRetrieved memories:")
        for i, memory in enumerate(retrieve_result.get("memories", [])):
            print(f"\nMemory {i+1} (Score: {memory.get('score')}):")
            print(memory.get("content"))
        
        print("\nDisconnected from MCP Memory server")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)