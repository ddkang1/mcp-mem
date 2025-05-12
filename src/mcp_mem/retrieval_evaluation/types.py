"""
Data structures and type definitions for retrieval evaluation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class QueryType(Enum):
    """Types of queries for evaluating different retrieval capabilities."""
    FACTOID = "factoid"           # Simple fact-based queries
    DESCRIPTIVE = "descriptive"   # Queries requiring longer descriptive answers
    MULTI_HOP = "multi_hop"       # Queries requiring multiple reasoning steps
    CROSS_DOC = "cross_doc"       # Queries requiring information from multiple documents
    COMPARATIVE = "comparative"   # Queries requiring comparison between concepts
    TEMPORAL = "temporal"         # Queries involving time relationships
    CAUSAL = "causal"             # Queries about cause-effect relationships

class DifficultyLevel(Enum):
    """Difficulty levels for retrieval queries."""
    EASY = "easy"       # Direct answer present in text
    MEDIUM = "medium"   # Some inference required
    HARD = "hard"       # Significant synthesis or reasoning required

@dataclass
class RetrievalQuery:
    """
    A structured query for evaluating retrieval systems.
    
    Attributes:
        question: The query text
        answers: List of acceptable answers (multiple formulations)
        query_type: Type of query (factoid, multi-hop, etc.)
        difficulty: Difficulty level
        source_docs: List of document IDs containing information needed for the answer
        required_concepts: Key concepts that should be present in retrieved content
    """
    question: str
    answers: List[str]
    query_type: QueryType
    difficulty: DifficultyLevel
    source_docs: List[str]
    required_concepts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answers": self.answers,
            "query_type": self.query_type.value,
            "difficulty": self.difficulty.value,
            "source_docs": self.source_docs,
            "required_concepts": self.required_concepts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalQuery':
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answers=data["answers"],
            query_type=QueryType(data["query_type"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            source_docs=data["source_docs"],
            required_concepts=data["required_concepts"]
        )