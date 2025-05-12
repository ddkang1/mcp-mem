"""
Ground truth evaluation for retrieval systems.

This module provides tools for evaluating retrieval systems using ground truth
context-question pairs. This approach is more methodological and allows for
more precise evaluation of retrieval quality.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional
import re
import logging
from .types import QueryType, DifficultyLevel

logger = logging.getLogger("mcp_memory_client")

@dataclass
class GroundTruthItem:
    """
    A ground truth item for retrieval evaluation.
    
    Attributes:
        question: The query text
        context_ids: List of document IDs that should be retrieved for this question
        context_snippets: List of text snippets that should be present in retrieved content
        query_type: Type of query (factoid, multi-hop, etc.)
        difficulty: Difficulty level
    """
    question: str
    context_ids: List[str]
    context_snippets: List[str]
    query_type: QueryType
    difficulty: DifficultyLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "context_ids": self.context_ids,
            "context_snippets": self.context_snippets,
            "query_type": self.query_type.value,
            "difficulty": self.difficulty.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthItem':
        """Create from dictionary."""
        return cls(
            question=data["question"],
            context_ids=data["context_ids"],
            context_snippets=data["context_snippets"],
            query_type=QueryType(data["query_type"]),
            difficulty=DifficultyLevel(data["difficulty"])
        )


def generate_ground_truth_items() -> List[GroundTruthItem]:
    """
    Generate a list of ground truth items for evaluation.
    
    Returns:
        List of GroundTruthItem objects
    """
    items = []
    
    # Document IDs for reference - using simple IDs for clarity
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    
    # FACTOID QUERIES - Single document, direct answers
    items.append(
        GroundTruthItem(
            question="What is climate change?",
            context_ids=[doc_ids[0]],
            context_snippets=[
                "Climate change is the long-term alteration of temperature and typical weather patterns",
                "Climate change could refer to a particular location or the planet as a whole"
            ],
            query_type=QueryType.FACTOID,
            difficulty=DifficultyLevel.EASY
        )
    )
    
    items.append(
        GroundTruthItem(
            question="What is the primary cause of climate change?",
            context_ids=[doc_ids[1]],
            context_snippets=[
                "The primary cause of climate change is human activities",
                "burning of fossil fuels",
                "greenhouse gases into Earth's atmosphere"
            ],
            query_type=QueryType.FACTOID,
            difficulty=DifficultyLevel.EASY
        )
    )
    
    # DESCRIPTIVE QUERIES - Require more detailed information
    items.append(
        GroundTruthItem(
            question="How do greenhouse gases affect Earth's temperature?",
            context_ids=[doc_ids[1]],
            context_snippets=[
                "gases trap heat from the sun's rays inside the atmosphere",
                "causing Earth's average temperature to rise"
            ],
            query_type=QueryType.DESCRIPTIVE,
            difficulty=DifficultyLevel.MEDIUM
        )
    )
    
    # MULTI-HOP QUERIES - Require connecting information within a document
    items.append(
        GroundTruthItem(
            question="Why is it difficult to grow crops in regions affected by climate change?",
            context_ids=[doc_ids[0]],
            context_snippets=[
                "Climate change may cause weather patterns to be less predictable",
                "These unexpected weather patterns can make it difficult to maintain and grow crops"
            ],
            query_type=QueryType.MULTI_HOP,
            difficulty=DifficultyLevel.MEDIUM
        )
    )
    
    # CROSS-DOCUMENT QUERIES - Require information from multiple documents
    items.append(
        GroundTruthItem(
            question="How can renewable energy help address the primary cause of climate change?",
            context_ids=[doc_ids[1], doc_ids[5]],
            context_snippets=[
                "primary cause of climate change is human activities, particularly the burning of fossil fuels",
                "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass power",
                "Unlike fossil fuels, these energy sources replenish naturally and produce minimal greenhouse gas emissions"
            ],
            query_type=QueryType.CROSS_DOC,
            difficulty=DifficultyLevel.HARD
        )
    )
    
    return items


def evaluate_retrieval_with_ground_truth(
    retrieved_content: List[str],
    retrieved_ids: List[str],
    ground_truth: GroundTruthItem
) -> Dict[str, Any]:
    """
    Evaluate retrieval results against ground truth.
    
    Args:
        retrieved_content: List of retrieved content strings
        retrieved_ids: List of retrieved document IDs
        ground_truth: Ground truth item to evaluate against
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Combine all retrieved content
    all_content = " ".join(retrieved_content).lower()
    
    # Check for context ID matches
    retrieved_ids_set = set(retrieved_ids)
    expected_ids_set = set(ground_truth.context_ids)
    
    id_matches = retrieved_ids_set.intersection(expected_ids_set)
    id_precision = len(id_matches) / len(retrieved_ids) if retrieved_ids else 0.0
    id_recall = len(id_matches) / len(expected_ids_set) if expected_ids_set else 1.0
    id_f1 = 2 * (id_precision * id_recall) / (id_precision + id_recall) if (id_precision + id_recall) > 0 else 0.0
    
    # Check for context snippet matches
    snippet_matches = []
    for snippet in ground_truth.context_snippets:
        snippet = snippet.lower()
        if snippet in all_content:
            snippet_matches.append(snippet)
        else:
            # Try to find partial matches
            words = snippet.split()
            if len(words) > 3:  # Only check substantial snippets
                # Check if at least 70% of words are present
                words_found = [word for word in words if word in all_content and len(word) > 3]
                if len(words_found) / len(words) >= 0.7:
                    snippet_matches.append(snippet + " (partial match)")
    
    snippet_precision = len(snippet_matches) / len(ground_truth.context_snippets) if ground_truth.context_snippets else 1.0
    
    # Calculate overall score
    overall_score = (id_f1 * 0.5) + (snippet_precision * 0.5)
    
    # Determine success
    success = overall_score >= 0.7
    
    return {
        "success": success,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1,
        "snippet_matches": snippet_matches,
        "snippet_precision": snippet_precision,
        "overall_score": overall_score
    }


async def test_with_ground_truth(
    client: Any,
    session_id: str,
    ground_truth_item: GroundTruthItem,
    search_mode: str = "hybrid",
    limit: int = 3
) -> Dict[str, Any]:
    """
    Test retrieval using a ground truth item.
    
    Args:
        client: MCP client
        session_id: Session ID
        ground_truth_item: Ground truth item to test with
        search_mode: Retrieval method to use (default: "hybrid")
        limit: Maximum number of memories to return (default: 3)
        
    Returns:
        Dictionary with retrieval results and evaluation metrics
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
    
    # Retrieve memories
    try:
        result = await client.call_tool(
            "retrieve_memory",
            {
                "session_id": session_id,
                "query": ground_truth_item.question,
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
        from .utils import extract_memory_content
        
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
        
        # Evaluate against ground truth
        evaluation = evaluate_retrieval_with_ground_truth(
            retrieved_content=memory_contents,
            retrieved_ids=memory_ids,
            ground_truth=ground_truth_item
        )
        
        # Print a clear summary of the results
        logger.info(f"\n{'=' * 50}")
        if evaluation["success"]:
            logger.info(f"✅ RETRIEVAL SUCCESS: Query '{ground_truth_item.question}'")
        else:
            logger.info(f"❌ RETRIEVAL FAILURE: Query '{ground_truth_item.question}'")
        logger.info(f"{'=' * 50}")
        
        # Print retrieval statistics
        logger.info(f"Retrieved {len(memories)} memories")
        
        # Print ground truth evaluation results
        logger.info("\n📊 GROUND TRUTH EVALUATION:")
        
        # Document ID matching
        id_match_percentage = evaluation["id_recall"] * 100
        logger.info(f"Document ID matching: {id_match_percentage:.1f}% of expected documents retrieved")
        logger.info(f"  Expected IDs: {', '.join(ground_truth_item.context_ids)}")
        logger.info(f"  Retrieved IDs: {', '.join(memory_ids)}")
        
        # Context snippet matching
        snippet_match_percentage = evaluation["snippet_precision"] * 100
        logger.info(f"\nContext snippet matching: {snippet_match_percentage:.1f}% of expected snippets found")
        
        if evaluation["snippet_matches"]:
            logger.info("  Matched snippets:")
            for i, snippet in enumerate(evaluation["snippet_matches"]):
                logger.info(f"    {i+1}. {snippet}")
        
        missing_snippets = [s for s in ground_truth_item.context_snippets 
                           if s not in [m.split(" (partial match)")[0] for m in evaluation["snippet_matches"]]]
        if missing_snippets:
            logger.info("  Missing snippets:")
            for i, snippet in enumerate(missing_snippets):
                logger.info(f"    {i+1}. {snippet}")
        
        # Print overall quality score
        score = evaluation["overall_score"]
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
            "query": ground_truth_item.to_dict(),
            "retrieved_count": len(memories),
            "retrieved_ids": memory_ids,
            "retrieved_contents": memory_contents,
            "evaluation": evaluation,
            "memories": memories
        }
        
        return retrieval_result
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"Error evaluating query '{ground_truth_item.question}': {error_message}")
        logger.error(error_traceback)
        
        # Print a clear error message
        logger.info(f"\n{'=' * 50}")
        logger.info(f"❌ RETRIEVAL ERROR: Query '{ground_truth_item.question}'")
        logger.info(f"{'=' * 50}")
        
        # Return a structured error result
        return {
            "query": ground_truth_item.to_dict(),
            "error": error_message,
            "error_traceback": error_traceback,
            "evaluation": {
                "success": False,
                "overall_score": 0.0
            },
            "memories": []
        }


async def evaluate_with_ground_truth_set(
    client: Any,
    session_id: str,
    ground_truth_items: List[GroundTruthItem] = None,
    search_modes: List[str] = ["hybrid"],
    limit: int = 3
) -> Dict[str, Any]:
    """
    Evaluate retrieval using a set of ground truth items.
    
    Args:
        client: MCP client
        session_id: Session ID
        ground_truth_items: List of ground truth items to test with (default: generated items)
        search_modes: List of search modes to test (default: ["hybrid"])
        limit: Maximum number of memories to return (default: 3)
        
    Returns:
        Dictionary with evaluation results for each search mode
    """
    if ground_truth_items is None:
        ground_truth_items = generate_ground_truth_items()
    
    results = {}
    
    for mode in search_modes:
        logger.info(f"\n\n{'=' * 70}")
        logger.info(f"EVALUATING '{mode.upper()}' SEARCH MODE")
        logger.info(f"{'=' * 70}")
        
        mode_results = []
        
        for item in ground_truth_items:
            result = await test_with_ground_truth(
                client=client,
                session_id=session_id,
                ground_truth_item=item,
                search_mode=mode,
                limit=limit
            )
            mode_results.append(result)
        
        # Calculate aggregate metrics
        success_count = sum(1 for r in mode_results if r.get("evaluation", {}).get("success", False))
        success_rate = success_count / len(mode_results) if mode_results else 0
        
        avg_id_precision = sum(r.get("evaluation", {}).get("id_precision", 0) for r in mode_results) / len(mode_results) if mode_results else 0
        avg_id_recall = sum(r.get("evaluation", {}).get("id_recall", 0) for r in mode_results) / len(mode_results) if mode_results else 0
        avg_id_f1 = sum(r.get("evaluation", {}).get("id_f1", 0) for r in mode_results) / len(mode_results) if mode_results else 0
        
        avg_snippet_precision = sum(r.get("evaluation", {}).get("snippet_precision", 0) for r in mode_results) / len(mode_results) if mode_results else 0
        
        avg_overall_score = sum(r.get("evaluation", {}).get("overall_score", 0) for r in mode_results) / len(mode_results) if mode_results else 0
        
        # Group results by query type
        by_query_type = {}
        for result in mode_results:
            query_type = result.get("query", {}).get("query_type")
            if query_type:
                if query_type not in by_query_type:
                    by_query_type[query_type] = []
                by_query_type[query_type].append(result)
        
        # Calculate metrics by query type
        query_type_metrics = {}
        for query_type, type_results in by_query_type.items():
            success_count = sum(1 for r in type_results if r.get("evaluation", {}).get("success", False))
            success_rate = success_count / len(type_results) if type_results else 0
            
            avg_score = sum(r.get("evaluation", {}).get("overall_score", 0) for r in type_results) / len(type_results) if type_results else 0
            
            query_type_metrics[query_type] = {
                "count": len(type_results),
                "success_rate": success_rate,
                "avg_score": avg_score
            }
        
        # Group results by difficulty
        by_difficulty = {}
        for result in mode_results:
            difficulty = result.get("query", {}).get("difficulty")
            if difficulty:
                if difficulty not in by_difficulty:
                    by_difficulty[difficulty] = []
                by_difficulty[difficulty].append(result)
        
        # Calculate metrics by difficulty
        difficulty_metrics = {}
        for difficulty, diff_results in by_difficulty.items():
            success_count = sum(1 for r in diff_results if r.get("evaluation", {}).get("success", False))
            success_rate = success_count / len(diff_results) if diff_results else 0
            
            avg_score = sum(r.get("evaluation", {}).get("overall_score", 0) for r in diff_results) / len(diff_results) if diff_results else 0
            
            difficulty_metrics[difficulty] = {
                "count": len(diff_results),
                "success_rate": success_rate,
                "avg_score": avg_score
            }
        
        # Store results for this mode
        results[mode] = {
            "aggregates": {
                "overall": {
                    "count": len(mode_results),
                    "success_rate": success_rate,
                    "avg_id_precision": avg_id_precision,
                    "avg_id_recall": avg_id_recall,
                    "avg_id_f1": avg_id_f1,
                    "avg_snippet_precision": avg_snippet_precision,
                    "avg_overall_score": avg_overall_score
                },
                "by_query_type": query_type_metrics,
                "by_difficulty": difficulty_metrics
            },
            "detailed_results": mode_results
        }
    
    return results


def print_ground_truth_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of ground truth evaluation results.
    
    Args:
        results: The results dictionary returned by evaluate_with_ground_truth_set
    """
    print("\n" + "=" * 80)
    print("GROUND TRUTH RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)
    
    # Compare methods
    methods = list(results.keys())
    if len(methods) > 1:
        # Calculate average scores for each method
        method_scores = {
            method: results[method]["aggregates"]["overall"]["avg_overall_score"] 
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
        avg_score = overall['avg_overall_score']
        avg_id_f1 = overall['avg_id_f1']
        avg_snippet_precision = overall['avg_snippet_precision']
        
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
        print(f"  Success rate: {success_rate:.1%} of queries returned correct context")
        print(f"  Document ID F1 score: {avg_id_f1:.2f} (higher is better, max 1.0)")
        print(f"  Context snippet precision: {avg_snippet_precision:.2f} (higher is better, max 1.0)")
        print(f"  Overall score: {avg_score:.2f}/1.0")
        
        # Metrics by query type
        print("\n📋 PERFORMANCE BY QUERY TYPE:")
        query_types = sorted(
            method_results["aggregates"]["by_query_type"].items(),
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
        by_type = method_results["aggregates"]["by_query_type"]
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