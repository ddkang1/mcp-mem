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
from typing import Dict, Any, List

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MCP client
try:
    from fastmcp.client.client import FastMCPClient
except ImportError:
    print("Error: fastmcp package not found. Please install it with 'pip install fastmcp'")
    sys.exit(1)

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
    
    # Create MCP client
    client = FastMCPClient(
        server_name="memory",
        transport="stdio",  # Use "sse" for HTTP-based access
        server_command="python -m mcp_mem.server",  # Command to start the server
    )
    
    # Connect to the server
    await client.connect()
    print("Connected to MCP Memory server")
    
    # Configure the memory system
    print("\nConfiguring memory system...")
    config_result = await client.use_tool("configure_memory", {
        "integration_type": args.integration_type,
        "lightrag_api_base_url": args.lightrag_api_url if args.integration_type == "api" else None,
        "lightrag_api_key": args.lightrag_api_key if args.integration_type == "api" else None,
    })
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
    store_result1 = await client.use_tool("store_memory", {
        "session_id": session_id,
        "content": content1
    })
    print(f"Result: {json.dumps(store_result1, indent=2)}")
    
    print("\nStoring second memory...")
    store_result2 = await client.use_tool("store_memory", {
        "session_id": session_id,
        "content": content2
    })
    print(f"Result: {json.dumps(store_result2, indent=2)}")
    
    # Wait a moment for indexing to complete
    print("\nWaiting for indexing to complete...")
    await asyncio.sleep(2)
    
    # Retrieve memories
    print("\nRetrieving memories about 'greenhouse gases'...")
    retrieve_result = await client.use_tool("retrieve_memory", {
        "session_id": session_id,
        "query": "What are greenhouse gases?",
        "limit": 3
    })
    
    print("\nRetrieved memories:")
    for i, memory in enumerate(retrieve_result.get("memories", [])):
        print(f"\nMemory {i+1} (Score: {memory.get('score')}):")
        print(memory.get("content"))
    
    # Disconnect from the server
    await client.disconnect()
    print("\nDisconnected from MCP Memory server")

if __name__ == "__main__":
    asyncio.run(main())