 #!/usr/bin/env python3
"""
Example script demonstrating how to use the MCP Memory server with LightRAG.
This script shows how to store and retrieve memories using the MCP client.
"""

import os
import sys
import json
import re
import asyncio
import argparse
import logging
from typing import Dict, Any, List
from fastmcp.client.client import Client

import httpx
# Add parent directory to path to ensure imports work

# Custom JSON encoder for Tool objects
class ToolJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if object has the attributes we need without relying on class type
        if hasattr(obj, 'name') and hasattr(obj, 'description') and hasattr(obj, 'inputSchema'):
            return {
                "name": obj.name,
                "description": obj.description,
                "inputSchema": obj.inputSchema
            }
        return super().default(obj)

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
                    "args": ["-m", "mcp_mem.server", "--debug"],
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
            logger.debug(f"Available tools: {json.dumps(tools, indent=2, cls=ToolJSONEncoder)}")
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
        
        try:
            store_result1 = await client.call_tool(
                "store_memory",
                {
                    "session_id": session_id,
                    "content": content1
                }
            )
            
            # Convert complex objects to simple dictionaries for JSON serialization
            if isinstance(store_result1, dict):
                store_result1_json = {
                    "status": store_result1.get("status", "unknown"),
                    "message": store_result1.get("message", "No message"),
                    "session_id": store_result1.get("session_id", session_id)
                }
            else:
                store_result1_json = {"status": "success", "message": "Memory stored successfully"}
                
            print(f"Result: {json.dumps(store_result1_json, indent=2)}")
            if args.debug:
                logger.debug(f"Store result: {json.dumps(store_result1_json)}")
        except Exception as e:
            print(f"Error storing memory: {e}")
            if args.debug:
                logger.debug(f"Error details: {str(e)}")
        
        print("\nStoring second memory...")
        if args.debug:
            logger.debug(f"Storing second memory with session_id: {session_id}")
            logger.debug(f"Content length: {len(content2)} characters")
            
        try:
            store_result2 = await client.call_tool(
                "store_memory",
                {
                    "session_id": session_id,
                    "content": content2
                }
            )
            
            # Convert complex objects to simple dictionaries for JSON serialization
            if isinstance(store_result2, dict):
                store_result2_json = {
                    "status": store_result2.get("status", "unknown"),
                    "message": store_result2.get("message", "No message"),
                    "session_id": store_result2.get("session_id", session_id)
                }
            else:
                store_result2_json = {"status": "success", "message": "Memory stored successfully"}
                
            print(f"Result: {json.dumps(store_result2_json, indent=2)}")
            if args.debug:
                logger.debug(f"Store result: {json.dumps(store_result2_json)}")
        except Exception as e:
            print(f"Error storing memory: {e}")
            if args.debug:
                logger.debug(f"Error details: {str(e)}")
        
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
            
        try:
            # Use the low-level MCP call to get the raw result
            raw_result = await client.call_tool_mcp(
                "retrieve_memory",
                {
                    "session_id": session_id,
                    "query": query,
                    "limit": limit
                }
            )
            
            logger.debug(f"Raw call_tool_mcp result type: {type(raw_result)}")
            logger.debug(f"Raw call_tool_mcp result dir: {dir(raw_result)}")
            
            # Try to extract memories directly from the raw result
            memories = []
            
            # Check if the result has content
            if hasattr(raw_result, 'content') and isinstance(raw_result.content, list):
                logger.debug(f"Raw content length: {len(raw_result.content)}")
                
                # Process each content item
                for item in raw_result.content:
                    logger.debug(f"Content item type: {type(item)}")
                    
                    # If it's a TextContent object
                    if hasattr(item, 'text'):
                        logger.debug(f"Text content: {item.text[:100]}...")
                        try:
                            # Try to parse the text as JSON
                            content_json = json.loads(item.text)
                            logger.debug(f"Parsed JSON: {content_json}")
                            
                            # Check if it contains memories
                            if isinstance(content_json, dict):
                                if 'memories' in content_json:
                                    memories = content_json['memories']
                                    logger.debug(f"Found memories in content: {memories}")
                                elif 'content' in content_json:
                                    # It might be a single memory
                                    memories = [content_json]
                                    logger.debug(f"Found single memory in content: {memories}")
                        except Exception as e:
                            logger.debug(f"Error parsing content as JSON: {e}")
                            # It might be the raw memory content
                            memories = [{
                                "content": item.text,
                                "score": 1.0
                            }]
                            logger.debug(f"Using text as memory content: {memories}")
            
            # If we didn't find any memories in the raw result, try the regular call
            if not memories:
                # Make the regular call to get the processed result
                retrieve_result = await client.call_tool(
                    "retrieve_memory",
                    {
                        "session_id": session_id,
                        "query": query,
                        "limit": limit
                    }
                )
                
                # Convert complex objects to simple dictionaries for JSON serialization
                if isinstance(retrieve_result, dict):
                    # Extract memories safely - first try direct access, then try parsing from response
                    memories = retrieve_result.get('memories', [])
                    logger.debug(f"Raw memories from retrieve_result: {memories}")
                    
                    # If memories is empty but we have a response field, try to extract from there
                    if not memories and 'response' in retrieve_result:
                        try:
                            response = retrieve_result['response']
                            # Find the document chunks section
                            chunks_section = re.search(r'-----Document Chunks\(DC\)-----\s*```json\s*(.*?)\s*```',
                                                     response, re.DOTALL)
                            
                            if chunks_section:
                                chunks_json = chunks_section.group(1).strip()
                                chunks = json.loads(chunks_json)
                                
                                # Format the chunks as memories
                                for idx, chunk in enumerate(chunks):
                                    memories.append({
                                        "content": chunk.get("content", ""),
                                        "score": float(chunk.get("score", 0.0))
                                    })
                                logger.debug(f"Extracted {len(memories)} memories from response field")
                        except Exception as e:
                            logger.debug(f"Error extracting memories from response: {e}")
            
            # Process the memories we found
            simple_memories = []
            for i, memory in enumerate(memories):
                logger.debug(f"Processing memory {i+1}: {memory}")
                if isinstance(memory, dict):
                    simple_memory = {
                        "content": str(memory.get("content", "")),
                        "score": float(memory.get("score", 0.0))
                    }
                    simple_memories.append(simple_memory)
                    logger.debug(f"Added simple memory: {simple_memory}")
            
            # Create the result JSON
            retrieve_result_json = {
                "status": "success",
                "query": query,
                "memories": simple_memories
            }
            
            # If no memories were found, ensure we have an empty list
            if not simple_memories:
                retrieve_result_json = {
                    "status": "success",
                    "query": query,
                    "memories": []
                }
            
            if args.debug:
                memories = retrieve_result_json.get('memories', [])
                logger.debug(f"Retrieved {len(memories)} memories")
                for i, memory in enumerate(memories):
                    logger.debug(f"Final memory {i+1}: {memory}")
                logger.debug(f"Full retrieve result: {json.dumps(retrieve_result_json, indent=2)}")
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            if args.debug:
                logger.debug(f"Error details: {str(e)}")
            retrieve_result_json = {
                "status": "error",
                "query": query,
                "memories": []
            }
        
        print("\nRetrieved memories:")
        for i, memory in enumerate(retrieve_result_json.get("memories", [])):
            print(f"\nMemory {i+1} (Score: {memory.get('score')}):")
            print(memory.get("content"))
        
        print("\nDisconnected from MCP Memory server")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)