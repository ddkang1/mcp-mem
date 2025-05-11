#!/usr/bin/env python3
"""
Example script demonstrating how to use the MCP Memory server with LightRAG.
This script shows how to store and retrieve memories using the MCP client,
testing various features of both LightRAG and FastMCP.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import Dict, Any, List, Optional
from fastmcp.client.client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_memory_client")

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

async def test_store_memory(client: Client, session_id: str, content: str) -> Dict[str, Any]:
    """Store a memory and return the result."""
    logger.info(f"Storing memory in session '{session_id}'")
    logger.debug(f"Content length: {len(content)} characters")
    
    try:
        result = await client.call_tool(
            "store_memory",
            {
                "session_id": session_id,
                "content": content
            }
        )
        
        # Format result for display
        if isinstance(result, dict):
            formatted_result = {
                "status": result.get("status", "unknown"),
                "message": result.get("message", "No message"),
                "session_id": result.get("session_id", session_id)
            }
        else:
            formatted_result = {"status": "success", "message": "Memory stored successfully"}
            
        logger.info(f"Memory stored: {json.dumps(formatted_result, indent=2)}")
        return formatted_result
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return {"status": "error", "message": str(e)}

async def display_lightrag_state(client: Client, session_id: str) -> None:
    """Display the internal state of LightRAG."""
    try:
        # Call the tool to get the state
        raw_result = await client.call_tool_mcp(
            "get_lightrag_state",
            {
                "session_id": session_id
            }
        )
        
        print("\n----- LightRAG Internal State -----")
        
        # Process the raw result
        if hasattr(raw_result, 'content'):
            # Try to extract content from TextContent object
            content_list = raw_result.content
            
            # Process each content item
            for item in content_list:
                if hasattr(item, 'text'):
                    # Try to parse as JSON
                    try:
                        state = json.loads(item.text)
                        
                        if isinstance(state, dict) and state.get("status") == "success":
                            # Display document count if available
                            if "document_count" in state:
                                print(f"Document count: {state['document_count']}")
                            
                            # Display memory store preview if available
                            if "memory_store_preview" in state and state["memory_store_preview"]:
                                print("\nMemory Store Preview:")
                                for doc_id, content in state["memory_store_preview"].items():
                                    print(f"  {doc_id}: {content}")
                            
                            # Display entities (nodes) if available
                            if "entities" in state and state["entities"]:
                                print("\nEntities (Nodes):")
                                for i, entity in enumerate(state["entities"]):
                                    name = entity.get("entity_name", "Unknown")
                                    entity_type = entity.get("entity_type", "Unknown")
                                    description = entity.get("description", "")
                                    print(f"  Entity {i+1}: {name} (Type: {entity_type})")
                                    if description:
                                        print(f"    Description: {description}")
                            
                            # Display relationships (edges) if available
                            if "relationships" in state and state["relationships"]:
                                print("\nRelationships (Edges):")
                                for i, rel in enumerate(state["relationships"]):
                                    source = rel.get("source", "Unknown")
                                    target = rel.get("target", "Unknown")
                                    rel_type = rel.get("type", "Unknown")
                                    description = rel.get("description", "")
                                    print(f"  Relationship {i+1}: {source} → {target} (Type: {rel_type})")
                                    if description:
                                        print(f"    Description: {description}")
                            
                            # Display session information
                            if "session_path" in state:
                                print(f"\nSession path: {state['session_path']}")
                            elif "session_id" in state:
                                print(f"\nSession ID: {state['session_id']}")
                    except json.JSONDecodeError:
                        # If not JSON, display the raw text
                        print(f"Raw content: {item.text[:200]}...")
                else:
                    # If no text attribute, display what we can
                    print(f"Content item type: {type(item)}")
        else:
            # If no content attribute, try to process as a regular dict
            try:
                if isinstance(raw_result, dict) and raw_result.get("status") == "success":
                    # Display document count if available
                    if "document_count" in raw_result:
                        print(f"Document count: {raw_result['document_count']}")
                    
                    # Display memory store preview if available
                    if "memory_store_preview" in raw_result and raw_result["memory_store_preview"]:
                        print("\nMemory Store Preview:")
                        for doc_id, content in raw_result["memory_store_preview"].items():
                            print(f"  {doc_id}: {content}")
                
                # If all else fails, print what we can about the object
                print(f"\nResult type: {type(raw_result)}")
                if hasattr(raw_result, "__dict__"):
                    print(f"Attributes: {dir(raw_result)}")
            except Exception as inner_e:
                print(f"Error processing result: {inner_e}")
                print(f"Result type: {type(raw_result)}")
    except Exception as e:
        logger.error(f"Error displaying LightRAG state: {e}")

async def test_retrieve_memory(
    client: Client,
    session_id: str,
    query: str,
    limit: int = 3,
    expected_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Retrieve memories using the configured search mode and return the result.
    
    Args:
        client: MCP client
        session_id: Session ID
        query: Query text
        limit: Maximum number of memories to return
        expected_keywords: List of keywords expected to be found in the retrieved content
        
    Returns:
        Dict containing retrieval results and accuracy metrics
    """
    logger.info(f"Retrieving memories with query: '{query}'")
    logger.debug(f"Session ID: {session_id}, Limit: {limit}")
    if expected_keywords:
        logger.debug(f"Expected keywords: {expected_keywords}")
    
    try:
        # Call the retrieve_memory tool
        result = await client.call_tool(
            "retrieve_memory",
            {
                "session_id": session_id,
                "query": query,
                "limit": limit
            }
        )
        
        # Process the result - handle various possible formats
        memories = []
        
        if isinstance(result, dict):
            memories = result.get("memories", [])
        elif isinstance(result, list):
            # If result is a list, assume it's a list of memories
            memories = result
            # Convert to standard format
            result = {
                "status": "success",
                "query": query,
                "memories": memories
            }
        else:
            # Handle TextContent or other object types
            try:
                # Try to access text attribute if it exists
                if hasattr(result, 'text'):
                    content = result.text
                    # Try to parse as JSON if possible
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "memories" in parsed:
                            memories = parsed["memories"]
                        else:
                            memories = [{"content": content, "score": 1.0}]
                    except:
                        memories = [{"content": content, "score": 1.0}]
                elif hasattr(result, 'content'):
                    # If it has content attribute
                    memories = [{"content": str(result.content), "score": 1.0}]
                else:
                    # Last resort: convert to string
                    memories = [{"content": str(result), "score": 1.0}]
                
                # Create a standardized result
                result = {
                    "status": "success",
                    "query": query,
                    "memories": memories
                }
            except Exception as e:
                logger.warning(f"Failed to process result of type {type(result)}: {e}")
                result = {
                    "status": "error",
                    "message": f"Unexpected result format: {type(result)}",
                    "memories": []
                }
        
        logger.info(f"Retrieved {len(memories)} memories")
        
        # Print the memories - safely access attributes
        for i, memory in enumerate(memories):
            try:
                # Handle different memory formats safely
                if isinstance(memory, dict):
                    score = memory.get('score', 'N/A')
                    content = memory.get("content", "")
                elif hasattr(memory, 'score') and hasattr(memory, 'content'):
                    score = getattr(memory, 'score', 'N/A')
                    content = getattr(memory, 'content', "")
                else:
                    score = 'N/A'
                    content = str(memory)
                
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    logger.info(f"\nMemory {i+1} (Score: {score}):")
                    logger.info(preview)
            except Exception as e:
                logger.warning(f"Error processing memory {i+1}: {e}")
        
        # Evaluate accuracy if expected keywords are provided
        if expected_keywords and memories:
            accuracy_metrics = {}
            
            # Calculate keyword match rate
            total_keywords = len(expected_keywords)
            matched_keywords = 0
            matched_keyword_list = []
            
            # Combine all memory content for keyword search
            all_content = ""
            for memory in memories:
                if isinstance(memory, dict) and "content" in memory:
                    all_content += memory["content"].lower() + " "
                elif hasattr(memory, "content"):
                    all_content += str(memory.content).lower() + " "
                else:
                    all_content += str(memory).lower() + " "
            
            # Check for each expected keyword
            for keyword in expected_keywords:
                if keyword.lower() in all_content:
                    matched_keywords += 1
                    matched_keyword_list.append(keyword)
            
            # Calculate accuracy metrics
            keyword_match_rate = matched_keywords / total_keywords if total_keywords > 0 else 0
            
            # Add metrics to result
            accuracy_metrics = {
                "total_keywords": total_keywords,
                "matched_keywords": matched_keywords,
                "matched_keyword_list": matched_keyword_list,
                "keyword_match_rate": keyword_match_rate
            }
            
            # Log accuracy metrics
            logger.info(f"Accuracy metrics: {matched_keywords}/{total_keywords} keywords matched ({keyword_match_rate:.2%})")
            logger.info(f"Matched keywords: {', '.join(matched_keyword_list)}")
            
            # Add accuracy metrics to result
            if isinstance(result, dict):
                result["accuracy_metrics"] = accuracy_metrics
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return {"status": "error", "message": str(e), "memories": []}

async def test_different_search_modes(client: Client, session_id: str, query: str) -> None:
    """Test different search modes available in LightRAG."""
    search_modes = ["hybrid", "local", "global", "mix", "naive"]
    
    logger.info("\n===== Testing Different Search Modes =====")
    for mode in search_modes:
        logger.info(f"\n----- Testing '{mode}' search mode -----")
        try:
            # Configure the memory system to use the specified mode
            config_result = await client.call_tool(
                "configure_memory",
                {
                    "search_mode": mode
                }
            )
            
            # Safely log configuration result
            try:
                if isinstance(config_result, dict) and "configuration" in config_result:
                    logger.info(f"Configuration set to {mode} mode")
                else:
                    logger.info(f"Configuration updated for {mode} mode")
            except Exception as e:
                logger.debug(f"Error processing config result: {e}")
            
            # Retrieve memories using the configured mode with expected keywords
            result = await test_retrieve_memory(
                client=client,
                session_id=session_id,
                query=query,
                expected_keywords=["climate", "change", "impacts", "weather", "temperature"]
            )
            
            # Log the number of memories retrieved - safely access memories
            try:
                if isinstance(result, dict) and "memories" in result:
                    memories = result["memories"]
                    logger.info(f"Mode '{mode}' returned {len(memories)} memories")
                else:
                    logger.info(f"Mode '{mode}' query completed")
            except Exception as e:
                logger.debug(f"Error processing result memories: {e}")
                
        except Exception as e:
            logger.error(f"Error testing search mode '{mode}': {e}")

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
    
    if args.debug:
        logger.debug("Debug mode enabled")
        logger.debug(f"Arguments: {args}")
    
    # Pre-flight check: only for API mode
    if args.integration_type == "api":
        try:
            import httpx
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
                    "args": ["-m", "mcp_mem.server", "--debug"] if args.debug else ["-m", "mcp_mem.server"],
                }
            }
        }
        if args.debug:
            logger.debug("Using direct integration with stdio transport")
            logger.debug(f"Client transport configuration: {json.dumps(client_transport, indent=2)}")

    async with Client(client_transport) as client:
        print("\n===== MCP Memory Client Example =====")
        print("Connected to MCP Memory server")
        
        # List available tools for debugging
        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
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
        
        content3 = """
        Solutions to climate change include transitioning to renewable energy sources like solar and wind power,
        improving energy efficiency in buildings and transportation, protecting and restoring forests and other ecosystems,
        and adopting more sustainable agricultural practices. International cooperation through agreements like the Paris Climate Accord
        aims to limit global temperature increases and address climate change impacts.
        """
        
        # Additional content on different topics
        content4 = """
        Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed
        to think and learn like humans. The term may also be applied to any machine that exhibits traits associated
        with a human mind such as learning and problem-solving. AI can be categorized as either weak AI or strong AI.
        Weak AI, also known as narrow AI, is designed to perform a specific task, like voice recognition.
        Strong AI, also known as artificial general intelligence, is AI that more fully replicates the autonomy of
        the human brain—a machine with consciousness, sentience, and mind.
        """
        
        content5 = """
        Quantum computing is an area of computing focused on developing computer technology based on the principles
        of quantum theory. Quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously,
        allowing them to perform complex calculations at speeds unattainable by traditional computers.
        Potential applications include cryptography, optimization problems, drug discovery, and materials science.
        However, quantum computers are still in early development stages and face significant technical challenges.
        """
        
        content6 = """
        Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass power.
        Unlike fossil fuels, these energy sources replenish naturally and produce minimal greenhouse gas emissions.
        Solar power harnesses energy from the sun using photovoltaic cells or solar thermal systems.
        Wind power captures kinetic energy from air movement using turbines. Hydroelectric power generates
        electricity from flowing water, typically using dams. Geothermal energy taps into the Earth's internal heat.
        Biomass energy comes from organic materials like plants and waste. The transition to renewable energy
        is crucial for addressing climate change and creating sustainable energy systems.
        """
        
        # Store memories and display internal state after each insertion
        print("\n===== Storing Memories =====")
        print("\n----- Storing Climate Change Content -----")
        await test_store_memory(client, session_id, content1)
        await display_lightrag_state(client, session_id)
        
        await test_store_memory(client, session_id, content2)
        await display_lightrag_state(client, session_id)
        
        await test_store_memory(client, session_id, content3)
        await display_lightrag_state(client, session_id)
        
        print("\n----- Storing Additional Topics Content -----")
        await test_store_memory(client, session_id, content4)
        await display_lightrag_state(client, session_id)
        
        await test_store_memory(client, session_id, content5)
        await display_lightrag_state(client, session_id)
        
        await test_store_memory(client, session_id, content6)
        await display_lightrag_state(client, session_id)
        
        # Wait a moment for indexing to complete
        print("\nWaiting for indexing to complete...")
        await asyncio.sleep(5)  # Increased from 2 to 5 seconds for better indexing
        
        # Test basic memory retrieval with expected keywords
        print("\n===== Basic Memory Retrieval =====")
        query1 = "What are greenhouse gases?"
        await test_retrieve_memory(
            client,
            session_id,
            query1,
            expected_keywords=["greenhouse gases", "fossil fuels", "atmosphere", "temperature"]
        )
        
        # Test memory retrieval with different query
        print("\n===== Memory Retrieval with Different Query =====")
        query2 = "What are solutions to climate change?"
        await test_retrieve_memory(
            client,
            session_id,
            query2,
            expected_keywords=["renewable energy", "solar", "wind", "sustainable", "Paris Climate Accord"]
        )
        
        # Test memory retrieval with different limit
        print("\n===== Memory Retrieval with Different Limit =====")
        query3 = "What is climate change?"
        await test_retrieve_memory(
            client,
            session_id,
            query3,
            limit=5,
            expected_keywords=["long-term", "temperature", "weather patterns", "predictable"]
        )
        
        # Test retrieval of new content
        print("\n===== Testing Retrieval of AI Content =====")
        query4 = "What is artificial intelligence?"
        await test_retrieve_memory(
            client,
            session_id,
            query4,
            expected_keywords=["simulation", "human intelligence", "machines", "weak AI", "strong AI"]
        )
        
        print("\n===== Testing Retrieval of Quantum Computing Content =====")
        query5 = "Explain quantum computing"
        await test_retrieve_memory(
            client,
            session_id,
            query5,
            expected_keywords=["quantum", "qubits", "multiple states", "calculations", "cryptography"]
        )
        
        print("\n===== Testing Retrieval of Renewable Energy Content =====")
        query6 = "What are different types of renewable energy?"
        await test_retrieve_memory(
            client,
            session_id,
            query6,
            expected_keywords=["solar", "wind", "hydroelectric", "geothermal", "biomass"]
        )
        
        print("\n===== Testing Cross-Topic Query =====")
        query7 = "How can AI help with renewable energy and climate change?"
        await test_retrieve_memory(
            client,
            session_id,
            query7,
            limit=5,
            expected_keywords=["AI", "renewable", "climate change", "energy"]
        )
        
        # Test different search modes if supported
        try:
            await test_different_search_modes(client, session_id, "climate change impacts")
        except Exception as e:
            logger.warning(f"Could not test different search modes: {e}")
        
        print("\nDisconnected from MCP Memory server")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)