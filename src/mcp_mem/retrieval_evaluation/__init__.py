"""
Retrieval Evaluation Module

This module provides tools for evaluating retrieval systems using structured queries
and comprehensive metrics.
"""

from .types import QueryType, DifficultyLevel, RetrievalQuery
from .query_generation import generate_retrieval_queries
from .evaluation import evaluate_retrieval_system, print_evaluation_results, test_single_query
from .utils import extract_memory_content
from .sample_content import (
    CLIMATE_CHANGE_DEFINITION, CLIMATE_CHANGE_CAUSES, CLIMATE_CHANGE_SOLUTIONS,
    ARTIFICIAL_INTELLIGENCE, QUANTUM_COMPUTING, RENEWABLE_ENERGY,
    SAMPLE_CONTENT, get_all_content
)
from .ground_truth import (
    GroundTruthItem, generate_ground_truth_items, evaluate_retrieval_with_ground_truth,
    test_with_ground_truth, evaluate_with_ground_truth_set, print_ground_truth_evaluation_results
)

__all__ = [
    # Types
    'QueryType',
    'DifficultyLevel',
    'RetrievalQuery',
    'GroundTruthItem',
    
    # Query generation
    'generate_retrieval_queries',
    'generate_ground_truth_items',
    
    # Evaluation functions
    'evaluate_retrieval_system',
    'print_evaluation_results',
    'test_single_query',
    'evaluate_retrieval_with_ground_truth',
    'test_with_ground_truth',
    'evaluate_with_ground_truth_set',
    'print_ground_truth_evaluation_results',
    
    # Utilities
    'extract_memory_content',
    
    # Sample content
    'CLIMATE_CHANGE_DEFINITION',
    'CLIMATE_CHANGE_CAUSES',
    'CLIMATE_CHANGE_SOLUTIONS',
    'ARTIFICIAL_INTELLIGENCE',
    'QUANTUM_COMPUTING',
    'RENEWABLE_ENERGY',
    'SAMPLE_CONTENT',
    'get_all_content',
]