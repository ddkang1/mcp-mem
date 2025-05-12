"""
Functions for evaluating retrieval systems.
"""

import json
import logging
import traceback
import math
from typing import Dict, Any, List, Tuple

from fastmcp.client.client import Client
from .types import QueryType, DifficultyLevel, RetrievalQuery
from .utils import extract_memory_content

logger = logging.getLogger("mcp_memory_client")

async def evaluate_retrieval_system(
    client: Client,
    session_id: str,
    queries: List[RetrievalQuery],
    retrieval_methods: List[str] = ["hybrid", "local", "global"],
    limit: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of retrieval system using structured queries.
    
    Args:
        client: MCP client
        session_id: Session ID
        queries: List of RetrievalQuery objects
        retrieval_methods: List of retrieval methods to test
        limit: Maximum number of memories to return per query
        
    Returns:
        Dict containing detailed evaluation results
    """
    logger.info(f"Evaluating retrieval system with {len(queries)} queries across {len(retrieval_methods)} methods")
    results = {}
    
    # Create document ID mapping for reference
    doc_id_map = {}
    
    # Try to get document IDs from LightRAG if possible
    try:
        state_result = await client.call_tool_mcp(
            "get_lightrag_state",
            {"session_id": session_id}
        )
        
        # Extract document IDs if available
        if hasattr(state_result, 'content'):
            for item in state_result.content:
                if hasattr(item, 'text'):
                    try:
                        state = json.loads(item.text)
                        if isinstance(state, dict) and state.get("status") == "success":
                            if "memory_store_preview" in state:
                                for i, (doc_id, _) in enumerate(state["memory_store_preview"].items()):
                                    doc_id_map[doc_id] = f"doc{i+1}"
                    except:
                        pass
    except Exception as e:
        logger.warning(f"Could not get document IDs: {e}")
    
    for method in retrieval_methods:
        logger.info(f"\n===== Evaluating '{method}' retrieval method =====")
        
        # Configure retrieval method
        try:
            await client.call_tool(
                "configure_memory",
                {"search_mode": method}
            )
            logger.info(f"Configured search mode to '{method}'")
        except Exception as e:
            logger.warning(f"Failed to configure search mode '{method}': {e}")
            continue
        
        method_results = []
        
        # Group queries by type for more organized evaluation
        query_groups = {}
        for query in queries:
            if query.query_type.value not in query_groups:
                query_groups[query.query_type.value] = []
            query_groups[query.query_type.value].append(query)
        
        # Process each query group
        for query_type, group_queries in query_groups.items():
            logger.info(f"\n----- Evaluating {query_type} queries -----")
            
            for i, query in enumerate(group_queries):
                logger.info(f"\nQuery {i+1} ({query.difficulty.value}): {query.question}")
                logger.debug(f"Expected answers: {query.answers}")
                logger.debug(f"Source docs: {query.source_docs}")
                
                # Retrieve memories
                try:
                    result = await client.call_tool(
                        "retrieve_memory",
                        {
                            "session_id": session_id,
                            "query": query.question,
                            "limit": limit
                        }
                    )
                    
                    # Process result to get retrieved content
                    memories = []
                    if isinstance(result, dict) and "memories" in result:
                        memories = result["memories"]
                    elif isinstance(result, list):
                        memories = result
                    
                    # Extract content and metadata from memories
                    memory_contents = []
                    memory_ids = []
                    memory_scores = []
                    
                    for memory in memories:
                        content = extract_memory_content(memory)
                        memory_contents.append(content)
                        
                        # Try to extract memory ID if available
                        memory_id = "unknown"
                        if isinstance(memory, dict) and "id" in memory:
                            memory_id = memory["id"]
                        elif hasattr(memory, "id"):
                            memory_id = memory.id
                        memory_ids.append(memory_id)
                        
                        # Try to extract score if available
                        score = 0.0
                        if isinstance(memory, dict) and "score" in memory:
                            score = float(memory["score"])
                        elif hasattr(memory, "score"):
                            score = float(memory.score)
                        memory_scores.append(score)
                    
                    # Combine all memory content for answer checking
                    all_content = " ".join(memory_contents).lower()
                    
                    # Check if retrieved content contains any of the acceptable answers
                    contains_answer = any(answer.lower() in all_content for answer in query.answers)
                    
                    # Find positions of documents containing answers (1-indexed)
                    answer_positions = []
                    for i, content in enumerate(memory_contents):
                        if any(answer.lower() in content.lower() for answer in query.answers):
                            answer_positions.append(i + 1)
                    
                    # Check for required concepts
                    matched_concepts = []
                    for concept in query.required_concepts:
                        if concept.lower() in all_content:
                            matched_concepts.append(concept)
                    
                    # Calculate metrics
                    metrics = {}
                    
                    # Reciprocal Rank (1/rank of first relevant document)
                    if answer_positions:
                        metrics["mrr"] = 1.0 / min(answer_positions)
                    else:
                        metrics["mrr"] = 0.0
                    
                    # Precision (fraction of retrieved documents that are relevant)
                    metrics["precision"] = len(answer_positions) / len(memories) if memories else 0.0
                    
                    # Recall for source documents (fraction of source docs that were retrieved)
                    retrieved_source_docs = set(query.source_docs).intersection(set(memory_ids))
                    metrics["source_recall"] = len(retrieved_source_docs) / len(query.source_docs) if query.source_docs else 0.0
                    
                    # Concept coverage (fraction of required concepts found)
                    metrics["concept_coverage"] = len(matched_concepts) / len(query.required_concepts) if query.required_concepts else 0.0
                    
                    # Normalized DCG (Discounted Cumulative Gain)
                    # This rewards relevant documents appearing earlier in results
                    dcg = 0.0
                    idcg = 0.0
                    for i in range(len(memories)):
                        relevance = 1.0 if (i + 1) in answer_positions else 0.0
                        dcg += relevance / (math.log2(i + 2))  # i+2 because log2(1) = 0
                    
                    # Ideal DCG (best possible ranking)
                    for i in range(min(len(answer_positions), len(memories))):
                        idcg += 1.0 / (math.log2(i + 2))
                    
                    metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0
                    
                    # Overall score (weighted average of metrics)
                    metrics["overall_score"] = (
                        metrics["mrr"] * 0.3 +
                        metrics["precision"] * 0.2 +
                        metrics["source_recall"] * 0.2 +
                        metrics["concept_coverage"] * 0.2 +
                        metrics["ndcg"] * 0.1
                    )
                    
                    # Log results
                    logger.info(f"Retrieved {len(memories)} memories")
                    logger.info(f"Contains answer: {contains_answer}")
                    logger.info(f"Answer positions: {answer_positions}")
                    logger.info(f"Matched concepts: {len(matched_concepts)}/{len(query.required_concepts)}")
                    logger.info(f"Overall score: {metrics['overall_score']:.2f}")
                    
                    # Create result object
                    retrieval_result = {
                        "query": query.to_dict(),
                        "retrieved_count": len(memories),
                        "retrieved_docs": memory_ids,
                        "contains_answer": contains_answer,
                        "answer_positions": answer_positions,
                        "matched_concepts": matched_concepts,
                        "metrics": metrics
                    }
                    
                    method_results.append(retrieval_result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating query '{query.question}': {e}")
                    logger.error(traceback.format_exc())
                    method_results.append({
                        "query": query.to_dict(),
                        "error": str(e),
                        "contains_answer": False,
                        "metrics": {"overall_score": 0.0}
                    })
        
        # Calculate aggregate metrics by query type and difficulty
        aggregates = {
            "overall": {
                "count": len(method_results),
                "success_rate": sum(1 for r in method_results if r.get("contains_answer", False)) / len(method_results) if method_results else 0,
                "avg_score": sum(r.get("metrics", {}).get("overall_score", 0) for r in method_results) / len(method_results) if method_results else 0,
                "mrr": sum(r.get("metrics", {}).get("mrr", 0) for r in method_results) / len(method_results) if method_results else 0,
            },
            "by_type": {},
            "by_difficulty": {}
        }
        
        # Calculate metrics by query type
        for query_type in QueryType:
            type_results = [r for r in method_results if r.get("query", {}).get("query_type") == query_type.value]
            if type_results:
                aggregates["by_type"][query_type.value] = {
                    "count": len(type_results),
                    "success_rate": sum(1 for r in type_results if r.get("contains_answer", False)) / len(type_results),
                    "avg_score": sum(r.get("metrics", {}).get("overall_score", 0) for r in type_results) / len(type_results)
                }
        
        # Calculate metrics by difficulty
        for difficulty in DifficultyLevel:
            diff_results = [r for r in method_results if r.get("query", {}).get("difficulty") == difficulty.value]
            if diff_results:
                aggregates["by_difficulty"][difficulty.value] = {
                    "count": len(diff_results),
                    "success_rate": sum(1 for r in diff_results if r.get("contains_answer", False)) / len(diff_results),
                    "avg_score": sum(r.get("metrics", {}).get("overall_score", 0) for r in diff_results) / len(diff_results)
                }
        
        results[method] = {
            "aggregates": aggregates,
            "detailed_results": method_results
        }
    
    return results

def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: The results dictionary returned by evaluate_retrieval_system
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)
    
    # Compare methods
    methods = list(results.keys())
    if len(methods) > 1:
        # Calculate average scores for each method
        method_scores = {
            method: results[method]["aggregates"]["overall"]["avg_score"]
            for method in methods
        }
        
        # Find best method
        best_method = max(method_scores.items(), key=lambda x: x[1])
        
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method[0].upper()} (Score: {best_method[1]:.2f})")
        
        # Print all methods ranked
        print("\nAll methods ranked by performance:")
        for i, (method, score) in enumerate(sorted(method_scores.items(), key=lambda x: x[1], reverse=True)):
            success_rate = results[method]["aggregates"]["overall"]["success_rate"]
            print(f"{i+1}. {method.upper()}: Score {score:.2f}, Success Rate {success_rate:.1%}")
    
    # Print detailed results for each method
    for method, method_results in results.items():
        print(f"\n{'=' * 40} {method.upper()} METHOD {'=' * 40}")
        
        # Overall metrics
        overall = method_results["aggregates"]["overall"]
        success_rate = overall['success_rate']
        avg_score = overall['avg_score']
        mrr = overall['mrr']
        
        # Determine overall quality rating
        if avg_score > 0.8:
            quality = "Excellent"
        elif avg_score > 0.6:
            quality = "Good"
        elif avg_score > 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
            
        print(f"\n📊 OVERALL PERFORMANCE: {quality}")
        print(f"  Success rate: {success_rate:.1%} of queries returned correct answers")
        print(f"  Mean Reciprocal Rank: {mrr:.2f} (higher is better, max 1.0)")
        print(f"  Overall score: {avg_score:.2f}/1.0")
        
        # Metrics by query type
        print("\n📋 PERFORMANCE BY QUERY TYPE:")
        query_types = sorted(
            method_results["aggregates"]["by_type"].items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        )
        
        for query_type, metrics in query_types:
            success_emoji = "✅" if metrics['success_rate'] >= 0.7 else "⚠️" if metrics['success_rate'] >= 0.4 else "❌"
            print(f"  {success_emoji} {query_type.ljust(15)}: {metrics['success_rate']:.1%} success, score {metrics['avg_score']:.2f} ({metrics['count']} queries)")
        
        # Metrics by difficulty
        print("\n📈 PERFORMANCE BY DIFFICULTY:")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in method_results["aggregates"]["by_difficulty"]:
                metrics = method_results["aggregates"]["by_difficulty"][difficulty]
                success_emoji = "✅" if metrics['success_rate'] >= 0.7 else "⚠️" if metrics['success_rate'] >= 0.4 else "❌"
                print(f"  {success_emoji} {difficulty.ljust(15)}: {metrics['success_rate']:.1%} success, score {metrics['avg_score']:.2f} ({metrics['count']} queries)")
        
        # Identify best and worst performing query types
        by_type = method_results["aggregates"]["by_type"]
        if by_type:
            best_type = max(by_type.items(), key=lambda x: x[1]['avg_score'])
            worst_type = min(by_type.items(), key=lambda x: x[1]['avg_score'])
            print(f"\n🔍 QUERY TYPE ANALYSIS:")
            print(f"  Strongest: {best_type[0]} ({best_type[1]['avg_score']:.2f} score)")
            print(f"  Weakest: {worst_type[0]} ({worst_type[1]['avg_score']:.2f} score)")
        
        # Identify cross-document retrieval performance specifically
        if "cross_doc" in by_type:
            cross_doc = by_type["cross_doc"]
            print(f"\n🔄 CROSS-DOCUMENT RETRIEVAL PERFORMANCE:")
            success_emoji = "✅" if cross_doc['success_rate'] >= 0.7 else "⚠️" if cross_doc['success_rate'] >= 0.4 else "❌"
            print(f"  {success_emoji} Success rate: {cross_doc['success_rate']:.1%}")
            print(f"  Score: {cross_doc['avg_score']:.2f}/1.0")
            
            if cross_doc['success_rate'] < 0.5:
                print("  ⚠️ Cross-document retrieval needs improvement")
            elif cross_doc['success_rate'] >= 0.8:
                print("  ✅ Cross-document retrieval performing well")

async def test_single_query(
    client: Any,
    session_id: str,
    query: str,
    expected_answers: List[str] = None,
    required_concepts: List[str] = None,
    search_mode: str = "hybrid",
    limit: int = 3
) -> Dict[str, Any]:
    """
    Test retrieval for a single query with comprehensive metrics.
    This provides a simpler interface for testing individual queries.
    
    Args:
        client: MCP client
        session_id: Session ID
        query: The query text
        expected_answers: List of acceptable answers (optional)
        required_concepts: Key concepts that should be present in retrieved content (optional)
        search_mode: Retrieval method to use (default: "hybrid")
        limit: Maximum number of memories to return (default: 3)
        
    Returns:
        Dictionary with retrieval results and metrics
    """
    import logging
    logger = logging.getLogger("mcp_memory_client")
    
    # Configure search mode if provided
    try:
        await client.call_tool(
            "configure_memory",
            {"search_mode": search_mode}
        )
        logger.info(f"Configured search mode to '{search_mode}'")
    except Exception as e:
        logger.warning(f"Failed to configure search mode '{search_mode}': {e}")
    
    # Create a simple query object
    test_query = RetrievalQuery(
        question=query,
        answers=expected_answers or [""],
        query_type=QueryType.FACTOID,
        difficulty=DifficultyLevel.MEDIUM,
        source_docs=[],
        required_concepts=required_concepts or []
    )
    
    # Retrieve memories
    try:
        result = await client.call_tool(
            "retrieve_memory",
            {
                "session_id": session_id,
                "query": query,
                "limit": limit
            }
        )
        
        # Process result to get retrieved content
        memories = []
        if isinstance(result, dict) and "memories" in result:
            memories = result["memories"]
        elif isinstance(result, list):
            memories = result
        
        # Extract content and metadata from memories
        memory_contents = []
        memory_ids = []
        memory_scores = []
        
        for memory in memories:
            content = extract_memory_content(memory)
            memory_contents.append(content)
            
            # Try to extract memory ID if available
            memory_id = "unknown"
            if isinstance(memory, dict) and "id" in memory:
                memory_id = memory["id"]
            elif hasattr(memory, "id"):
                memory_id = memory.id
            memory_ids.append(memory_id)
            
            # Try to extract score if available
            score = 0.0
            if isinstance(memory, dict) and "score" in memory:
                score = float(memory["score"])
            elif hasattr(memory, "score"):
                score = float(memory.score)
            memory_scores.append(score)
        
        # Combine all memory content for answer checking
        all_content = " ".join(memory_contents).lower()
        
        # Check if retrieved content contains any of the acceptable answers
        contains_answer = False
        answer_positions = []
        
        if expected_answers:
            contains_answer = any(answer.lower() in all_content for answer in expected_answers)
            
            # Find positions of documents containing answers (1-indexed)
            for i, content in enumerate(memory_contents):
                if any(answer.lower() in content.lower() for answer in expected_answers):
                    answer_positions.append(i + 1)
        
        # Check for required concepts
        matched_concepts = []
        if required_concepts:
            for concept in required_concepts:
                if concept.lower() in all_content:
                    matched_concepts.append(concept)
        
        # Calculate metrics
        metrics = {}
        
        # Reciprocal Rank (1/rank of first relevant document)
        if answer_positions:
            metrics["mrr"] = 1.0 / min(answer_positions)
        else:
            metrics["mrr"] = 0.0
        
        # Precision (fraction of retrieved documents that are relevant)
        metrics["precision"] = len(answer_positions) / len(memories) if memories else 0.0
        
        # Concept coverage (fraction of required concepts found)
        metrics["concept_coverage"] = len(matched_concepts) / len(required_concepts) if required_concepts else 0.0
        
        # Overall score (weighted average of metrics)
        metrics["overall_score"] = (
            metrics["mrr"] * 0.4 +
            metrics["precision"] * 0.3 +
            metrics["concept_coverage"] * 0.3
        )
        
        # Determine retrieval success
        success = False
        if expected_answers and contains_answer:
            success = True
        elif not expected_answers and required_concepts and len(matched_concepts) >= len(required_concepts) * 0.7:
            success = True  # At least 70% of required concepts found
        
        # Print a clear summary of the results
        logger.info(f"\n{'=' * 50}")
        if success:
            logger.info(f"✅ RETRIEVAL SUCCESS: Query '{query}'")
        else:
            logger.info(f"❌ RETRIEVAL FAILURE: Query '{query}'")
        logger.info(f"{'=' * 50}")
        
        # Print retrieval statistics
        logger.info(f"Retrieved {len(memories)} memories")
        
        # Analyze failure reasons when retrieval fails
        if not success:
            logger.info("\n🔍 FAILURE ANALYSIS:")
            failure_reasons = []
            improvement_suggestions = []
            
            # Check if any memories were retrieved
            if not memories:
                failure_reasons.append("No memories were retrieved for this query")
                improvement_suggestions.append("Check if content was properly stored in the memory system")
                improvement_suggestions.append("Try a different search mode or query formulation")
            
            # Check if expected answers were provided but not found
            if expected_answers and not contains_answer:
                failure_reasons.append("Expected answer not found in retrieved content")
                
                # Check if the answer might be present but in a different form
                answer_fragments_found = []
                for answer in expected_answers:
                    words = answer.lower().split()
                    fragments_found = [word for word in words if word in all_content and len(word) > 3]
                    if fragments_found:
                        answer_fragments_found.extend(fragments_found)
                
                if answer_fragments_found:
                    logger.info(f"  Found fragments of expected answer: {', '.join(set(answer_fragments_found))}")
                    failure_reasons.append("Answer fragments found but complete answer missing")
                    improvement_suggestions.append("The retriever found related content but not the exact answer")
                else:
                    logger.info("  No fragments of expected answer found in retrieved content")
                    failure_reasons.append("No parts of the answer were found")
                    improvement_suggestions.append("The content containing the answer may not be in the memory system")
            
            # Check concept coverage
            if required_concepts:
                missing_concepts = [c for c in required_concepts if c not in matched_concepts]
                if missing_concepts:
                    if len(missing_concepts) == len(required_concepts):
                        failure_reasons.append("None of the required concepts were found")
                    else:
                        failure_reasons.append(f"Missing key concepts: {', '.join(missing_concepts)}")
                    
                    # Check if concepts might be present in different forms
                    for concept in missing_concepts:
                        concept_words = concept.lower().split()
                        for word in concept_words:
                            if len(word) > 3 and word in all_content:
                                logger.info(f"  Found partial match for concept '{concept}': '{word}'")
            
            # Check if retrieved content is relevant but doesn't contain exact matches
            if memories and (expected_answers or required_concepts):
                # Calculate text similarity between query and retrieved content
                query_words = set(query.lower().split())
                content_words = set(all_content.lower().split())
                common_words = query_words.intersection(content_words)
                
                if len(common_words) / len(query_words) > 0.5:
                    logger.info("  Retrieved content is topically related but doesn't contain expected information")
                    failure_reasons.append("Content is related but missing specific information")
                    improvement_suggestions.append("The retriever found related content but not the specific information needed")
            
            # Print failure reasons
            logger.info("\n❌ FAILURE REASONS:")
            for i, reason in enumerate(failure_reasons):
                logger.info(f"  {i+1}. {reason}")
            
            # Print improvement suggestions
            if improvement_suggestions:
                logger.info("\n💡 SUGGESTIONS FOR IMPROVEMENT:")
                for i, suggestion in enumerate(improvement_suggestions):
                    logger.info(f"  {i+1}. {suggestion}")
                
                # Add general suggestions
                if "hybrid" not in search_mode.lower():
                    logger.info("  • Try using the 'hybrid' search mode which often performs better")
                logger.info("  • Consider reformulating the query to be more specific")
                logger.info("  • Check if the expected content is actually stored in the system")
        
        # Print answer information if expected answers were provided
        if expected_answers:
            if contains_answer:
                logger.info(f"\n✓ Found expected answer in retrieved content")
                if answer_positions:
                    logger.info(f"  Answer found in position(s): {', '.join(map(str, answer_positions))}")
                    
                    # Show the matching content
                    for pos in answer_positions[:2]:  # Show at most 2 matches
                        content = memory_contents[pos-1]
                        # Find the sentence containing the answer
                        sentences = content.split('.')
                        matching_sentences = []
                        for sentence in sentences:
                            if any(answer.lower() in sentence.lower() for answer in expected_answers):
                                matching_sentences.append(sentence.strip())
                        
                        if matching_sentences:
                            logger.info(f"  Matching content from position {pos}:")
                            for sentence in matching_sentences[:2]:  # Show at most 2 sentences
                                logger.info(f"    \"{sentence}\"")
            else:
                logger.info(f"\n✗ Expected answer NOT found in retrieved content")
                logger.info(f"  Expected to find: \"{expected_answers[0]}\"")
        
        # Print concept coverage information
        if required_concepts:
            concept_percentage = len(matched_concepts) / len(required_concepts) * 100
            logger.info(f"\nConcept coverage: {len(matched_concepts)}/{len(required_concepts)} concepts ({concept_percentage:.1f}%)")
            if matched_concepts:
                logger.info(f"  Found concepts: {', '.join(matched_concepts)}")
            if len(matched_concepts) < len(required_concepts):
                missing_concepts = [c for c in required_concepts if c not in matched_concepts]
                logger.info(f"  Missing concepts: {', '.join(missing_concepts)}")
        
        # Print overall quality score
        score = metrics["overall_score"]
        quality = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Fair" if score > 0.4 else "Poor"
        logger.info(f"\nOverall quality: {quality} ({score:.2f}/1.0)")
        
        # Print memory previews (limited to first 3 for clarity)
        if memories:
            logger.info(f"\nTop retrieved memories:")
            for i, (content, score) in enumerate(zip(memory_contents[:3], memory_scores[:3])):
                preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"  {i+1}. (Score: {score:.2f}) {preview}")
            if len(memories) > 3:
                logger.info(f"  ... and {len(memories) - 3} more memories")
        else:
            logger.info("\n⚠️ No memories were retrieved for this query")
        
        # Create result object
        retrieval_result = {
            "query": test_query.to_dict(),
            "retrieved_count": len(memories),
            "retrieved_docs": memory_ids,
            "contains_answer": contains_answer,
            "answer_positions": answer_positions,
            "matched_concepts": matched_concepts,
            "metrics": metrics,
            "memories": memories
        }
        
        return retrieval_result
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"Error evaluating query '{query}': {error_message}")
        logger.error(error_traceback)
        
        # Print a clear error message
        logger.info(f"\n{'=' * 50}")
        logger.info(f"❌ RETRIEVAL ERROR: Query '{query}'")
        logger.info(f"{'=' * 50}")
        
        # Analyze the error
        logger.info("\n🔍 ERROR ANALYSIS:")
        
        # Check for common error types and provide helpful information
        if "connection" in error_message.lower() or "timeout" in error_message.lower():
            logger.info("  Connection issue detected. The retrieval system may be unavailable.")
            logger.info("  💡 SUGGESTIONS:")
            logger.info("    • Check if the memory server is running")
            logger.info("    • Verify network connectivity")
            logger.info("    • Try again after a few moments")
        elif "not found" in error_message.lower() or "404" in error_message:
            logger.info("  Resource not found error detected.")
            logger.info("  💡 SUGGESTIONS:")
            logger.info("    • Check if the session ID is correct")
            logger.info("    • Verify that the memory system has been properly initialized")
        elif "permission" in error_message.lower() or "unauthorized" in error_message.lower():
            logger.info("  Authorization issue detected.")
            logger.info("  💡 SUGGESTIONS:")
            logger.info("    • Check authentication credentials")
            logger.info("    • Verify that you have permission to access this resource")
        elif "format" in error_message.lower() or "parse" in error_message.lower() or "json" in error_message.lower():
            logger.info("  Data format issue detected.")
            logger.info("  💡 SUGGESTIONS:")
            logger.info("    • Check the format of the query")
            logger.info("    • Verify that the memory system is returning properly formatted data")
        else:
            logger.info(f"  Unexpected error: {error_message}")
            logger.info("  💡 SUGGESTIONS:")
            logger.info("    • Check the error message for clues")
            logger.info("    • Review the system logs for more details")
            logger.info("    • Try a simpler query to isolate the issue")
        
        # Return a structured error result
        return {
            "query": test_query.to_dict(),
            "error": error_message,
            "error_traceback": error_traceback,
            "contains_answer": False,
            "metrics": {"overall_score": 0.0},
            "memories": []
        }