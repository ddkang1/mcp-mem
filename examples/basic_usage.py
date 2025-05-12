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
import traceback
from typing import Dict, Any, List, Optional
from fastmcp.client.client import Client
from mcp_mem.retrieval_evaluation import (
    QueryType, DifficultyLevel, RetrievalQuery,
    generate_retrieval_queries, evaluate_retrieval_system,
    extract_memory_content, print_evaluation_results,
    test_single_query, get_all_content,
    # Ground truth evaluation
    GroundTruthItem, generate_ground_truth_items,
    test_with_ground_truth, evaluate_with_ground_truth_set,
    print_ground_truth_evaluation_results
)

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

# Using the test_single_query function from the retrieval_evaluation module

async def test_different_search_modes(client: Client, session_id: str, query: str) -> None:
    """Test different search modes available in LightRAG."""
    search_modes = ["hybrid", "local", "global", "mix", "naive"]
    
    print("\n" + "=" * 80)
    print(f"COMPARING SEARCH MODES FOR QUERY: '{query}'")
    print("=" * 80)
    
    results = {}
    
    for mode in search_modes:
        print(f"\n{'-' * 30} TESTING '{mode.upper()}' MODE {'-' * 30}")
        try:
            # Use the test_single_query function with the specified search mode
            result = await test_single_query(
                client=client,
                session_id=session_id,
                query=query,
                required_concepts=["climate", "change", "impacts", "weather", "temperature"],
                search_mode=mode
            )
            
            # Store results for comparison
            results[mode] = {
                "count": result["retrieved_count"],
                "score": result["metrics"]["overall_score"],
                "matched_concepts": len(result["matched_concepts"]),
                "total_concepts": len(result["query"]["required_concepts"])
            }
                
        except Exception as e:
            logger.error(f"Error testing search mode '{mode}': {e}")
            results[mode] = {"error": str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("SEARCH MODE COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort modes by score
    sorted_modes = sorted(
        [mode for mode in results if "score" in results[mode]],
        key=lambda x: results[x]["score"],
        reverse=True
    )
    
    if sorted_modes:
        print(f"\nBest performing mode: {sorted_modes[0].upper()} (Score: {results[sorted_modes[0]]['score']:.2f})")
        
        print("\nAll modes ranked by performance:")
        for i, mode in enumerate(sorted_modes):
            r = results[mode]
            print(f"{i+1}. {mode.upper()}: Score {r['score']:.2f}, " +
                  f"Concepts {r['matched_concepts']}/{r['total_concepts']}, " +
                  f"Retrieved {r['count']} memories")
    
    # Print modes with errors
    error_modes = [mode for mode in results if "error" in results[mode]]
    if error_modes:
        print("\nModes with errors:")
        for mode in error_modes:
            print(f"- {mode}: {results[mode]['error']}")

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
        
        # Get sample content from the retrieval evaluation module
        content1, content2, content3, content4, content5, content6 = get_all_content()
        
        # Clear any existing data to start fresh
        print("\n===== Clearing Existing Memory Storage =====")
        try:
            clear_result = await client.call_tool(
                "clear_memory_storage",
                {
                    "session_id": session_id
                }
            )
            
            # Handle different result formats
            if isinstance(clear_result, dict):
                print(f"Storage cleared: {json.dumps(clear_result, indent=2)}")
            elif hasattr(clear_result, 'text'):
                print(f"Storage cleared: {clear_result.text}")
            elif hasattr(clear_result, 'content'):
                print(f"Storage cleared: {clear_result.content}")
            else:
                print(f"Storage cleared with result type: {type(clear_result)}")
                
            logger.info("Storage cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear storage: {e}")
            logger.warning("Continuing with tests anyway...")
            
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
        
        # Wait for indexing and entity extraction to complete
        print("\nWaiting for indexing and entity extraction to complete...")
        await asyncio.sleep(2)  # Increased from 5 to 2 seconds to allow entity extraction to finish
        
        # Test basic memory retrieval with expected keywords
        print("\n===== Basic Memory Retrieval =====")
        query1 = "What are greenhouse gases?"
        await test_single_query(
            client,
            session_id,
            query1,
            expected_answers=["greenhouse gases are released by burning fossil fuels"],
            required_concepts=["greenhouse gases", "fossil fuels", "atmosphere", "temperature"]
        )
        
        # Test memory retrieval with different query
        print("\n===== Memory Retrieval with Different Query =====")
        query2 = "What are solutions to climate change?"
        await test_single_query(
            client,
            session_id,
            query2,
            expected_answers=["renewable energy sources", "improving energy efficiency"],
            required_concepts=["renewable energy", "solar", "wind", "sustainable", "Paris Climate Accord"]
        )
        
        # Test memory retrieval with different limit
        print("\n===== Memory Retrieval with Different Limit =====")
        query3 = "What is climate change?"
        await test_single_query(
            client,
            session_id,
            query3,
            limit=5,
            expected_answers=["long-term alteration of temperature and typical weather patterns"],
            required_concepts=["long-term", "temperature", "weather patterns", "predictable"]
        )
        
        # Test retrieval of new content
        print("\n===== Testing Retrieval of AI Content =====")
        query4 = "What is artificial intelligence?"
        await test_single_query(
            client,
            session_id,
            query4,
            expected_answers=["simulation of human intelligence in machines"],
            required_concepts=["simulation", "human intelligence", "machines", "weak AI", "strong AI"]
        )
        
        print("\n===== Testing Retrieval of Quantum Computing Content =====")
        query5 = "Explain quantum computing"
        await test_single_query(
            client,
            session_id,
            query5,
            expected_answers=["computer technology based on the principles of quantum theory"],
            required_concepts=["quantum", "qubits", "multiple states", "calculations", "cryptography"]
        )
        
        print("\n===== Testing Retrieval of Renewable Energy Content =====")
        query6 = "What are different types of renewable energy?"
        await test_single_query(
            client,
            session_id,
            query6,
            expected_answers=["solar, wind, hydroelectric, geothermal, and biomass"],
            required_concepts=["solar", "wind", "hydroelectric", "geothermal", "biomass"]
        )
        
        print("\n===== Testing Cross-Topic Query =====")
        query7 = "How can AI help with renewable energy and climate change?"
        await test_single_query(
            client,
            session_id,
            query7,
            limit=5,
            required_concepts=["AI", "renewable", "climate change", "energy"]
        )
        
        # Test different search modes if supported
        try:
            await test_different_search_modes(client, session_id, "climate change impacts")
        except Exception as e:
            logger.warning(f"Could not test different search modes: {e}")
            
        # Test the advanced structured evaluation approach
            print("\n===== Testing Advanced Structured Evaluation =====")
            
            # Generate structured retrieval queries
            retrieval_queries = generate_retrieval_queries(
                content1, content2, content3, content4, content5, content6
            )
            print(f"Generated {len(retrieval_queries)} structured retrieval queries for evaluation")
            
            # Run the advanced evaluation
            advanced_results = await evaluate_retrieval_system(
                client=client,
                session_id=session_id,
                queries=retrieval_queries,
                retrieval_methods=["hybrid", "local", "global"]  # Using a subset for brevity
            )
            
            # Print comprehensive summary of results using the utility function
            print_evaluation_results(advanced_results)
            
            # Test the ground truth evaluation approach
            print("\n===== Testing Ground Truth Evaluation =====")
            
            # Generate ground truth items
            ground_truth_items = generate_ground_truth_items()
            print(f"Generated {len(ground_truth_items)} ground truth items for evaluation")
            
            # Test a single ground truth item
            print("\n----- Testing Single Ground Truth Item -----")
            item = ground_truth_items[0]  # Use the first item
            print(f"Testing query: '{item.question}'")
            print(f"Expected context IDs: {item.context_ids}")
            print(f"Expected context snippets: {item.context_snippets}")
            
            single_result = await test_with_ground_truth(
                client=client,
                session_id=session_id,
                ground_truth_item=item,
                search_mode="hybrid"
            )
            
            # Run the comprehensive ground truth evaluation
            print("\n----- Running Comprehensive Ground Truth Evaluation -----")
            ground_truth_results = await evaluate_with_ground_truth_set(
                client=client,
                session_id=session_id,
                ground_truth_items=ground_truth_items[:3],  # Use first 3 items for brevity
                search_modes=["hybrid", "local"]  # Using a subset for brevity
            )
            
            # Print ground truth evaluation results
            print_ground_truth_evaluation_results(ground_truth_results)
            
        except Exception as e:
            logger.error(f"Error in retrieval evaluation: {e}")
            logger.error(traceback.format_exc())
        
        print("\nDisconnected from MCP Memory server")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)