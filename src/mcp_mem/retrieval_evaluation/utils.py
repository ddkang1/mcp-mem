"""
Utility functions for retrieval evaluation.
"""

from typing import Any, Dict, List, Union

def extract_memory_content(memory: Any) -> str:
    """
    Extract content from a memory object regardless of its format.
    
    Args:
        memory: A memory object which could be a dict with a 'content' key,
               an object with a 'content' attribute, or any other object
               
    Returns:
        The extracted content as a string
    """
    if isinstance(memory, dict) and "content" in memory:
        return memory["content"]
    elif hasattr(memory, "content"):
        return str(memory.content)
    else:
        return str(memory)

def calculate_metrics(
    memory_contents: List[str],
    memory_ids: List[str],
    query_answers: List[str],
    query_source_docs: List[str],
    query_required_concepts: List[str]
) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics.
    
    Args:
        memory_contents: List of retrieved memory contents
        memory_ids: List of retrieved memory IDs
        query_answers: List of acceptable answers for the query
        query_source_docs: List of document IDs containing information needed for the answer
        query_required_concepts: Key concepts that should be present in retrieved content
        
    Returns:
        Dictionary of metrics including MRR, precision, recall, etc.
    """
    import math
    
    # Combine all memory content for answer checking
    all_content = " ".join(memory_contents).lower()
    
    # Check if retrieved content contains any of the acceptable answers
    contains_answer = any(answer.lower() in all_content for answer in query_answers)
    
    # Find positions of documents containing answers (1-indexed)
    answer_positions = []
    for i, content in enumerate(memory_contents):
        if any(answer.lower() in content.lower() for answer in query_answers):
            answer_positions.append(i + 1)
    
    # Check for required concepts
    matched_concepts = []
    for concept in query_required_concepts:
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
    metrics["precision"] = len(answer_positions) / len(memory_contents) if memory_contents else 0.0
    
    # Recall for source documents (fraction of source docs that were retrieved)
    retrieved_source_docs = set(query_source_docs).intersection(set(memory_ids))
    metrics["source_recall"] = len(retrieved_source_docs) / len(query_source_docs) if query_source_docs else 0.0
    
    # Concept coverage (fraction of required concepts found)
    metrics["concept_coverage"] = len(matched_concepts) / len(query_required_concepts) if query_required_concepts else 0.0
    
    # Normalized DCG (Discounted Cumulative Gain)
    # This rewards relevant documents appearing earlier in results
    dcg = 0.0
    idcg = 0.0
    for i in range(len(memory_contents)):
        relevance = 1.0 if (i + 1) in answer_positions else 0.0
        dcg += relevance / (math.log2(i + 2))  # i+2 because log2(1) = 0
    
    # Ideal DCG (best possible ranking)
    for i in range(min(len(answer_positions), len(memory_contents))):
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
    
    return metrics, contains_answer, answer_positions, matched_concepts