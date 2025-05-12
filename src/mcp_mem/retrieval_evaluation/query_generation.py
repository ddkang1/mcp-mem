"""
Functions for generating structured retrieval queries.
"""

from typing import List, Dict, Any
from .types import QueryType, DifficultyLevel, RetrievalQuery

def generate_retrieval_queries(
    content1: str, content2: str, content3: str, 
    content4: str, content5: str, content6: str
) -> List[RetrievalQuery]:
    """
    Generate structured retrieval queries from content.
    
    Args:
        content1-content6: The content strings to generate questions from
        
    Returns:
        List of RetrievalQuery objects
    """
    queries = []
    
    # Document IDs for reference - using simple IDs for clarity
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    
    # FACTOID QUERIES - Single document, direct answers
    queries.extend([
        RetrievalQuery(
            question="What is climate change?",
            answers=["long-term alteration of temperature and typical weather patterns"],
            query_type=QueryType.FACTOID,
            difficulty=DifficultyLevel.EASY,
            source_docs=[doc_ids[0]],
            required_concepts=["climate change", "temperature", "weather patterns"]
        ),
        RetrievalQuery(
            question="What is the primary cause of climate change?",
            answers=["human activities", "burning of fossil fuels"],
            query_type=QueryType.FACTOID,
            difficulty=DifficultyLevel.EASY,
            source_docs=[doc_ids[1]],
            required_concepts=["human activities", "fossil fuels"]
        ),
        RetrievalQuery(
            question="What are qubits?",
            answers=["quantum bits", "bits that can exist in multiple states simultaneously"],
            query_type=QueryType.FACTOID,
            difficulty=DifficultyLevel.EASY,
            source_docs=[doc_ids[4]],
            required_concepts=["quantum bits", "qubits", "multiple states"]
        )
    ])
    
    # DESCRIPTIVE QUERIES - Require more detailed information
    queries.extend([
        RetrievalQuery(
            question="How do greenhouse gases affect Earth's temperature?",
            answers=[
                "trap heat from the sun's rays inside the atmosphere causing Earth's average temperature to rise"
            ],
            query_type=QueryType.DESCRIPTIVE,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[1]],
            required_concepts=["greenhouse gases", "trap heat", "atmosphere", "temperature rise"]
        ),
        RetrievalQuery(
            question="What is the difference between weak AI and strong AI?",
            answers=[
                "Weak AI is designed for specific tasks while strong AI replicates human brain autonomy"
            ],
            query_type=QueryType.DESCRIPTIVE,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[3]],
            required_concepts=["weak AI", "strong AI", "specific task", "human brain"]
        )
    ])
    
    # MULTI-HOP QUERIES - Require connecting information within a document
    queries.extend([
        RetrievalQuery(
            question="Why is it difficult to grow crops in regions affected by climate change?",
            answers=[
                "climate change makes weather patterns less predictable, making it difficult to maintain and grow crops"
            ],
            query_type=QueryType.MULTI_HOP,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[0]],
            required_concepts=["climate change", "weather patterns", "predictable", "crops", "farming"]
        ),
        RetrievalQuery(
            question="What advantages do quantum computers have over traditional computers?",
            answers=[
                "perform complex calculations at speeds unattainable by traditional computers"
            ],
            query_type=QueryType.MULTI_HOP,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[4]],
            required_concepts=["quantum computers", "complex calculations", "speeds", "traditional computers"]
        )
    ])
    
    # CROSS-DOCUMENT QUERIES - Require information from multiple documents
    queries.extend([
        RetrievalQuery(
            question="How can renewable energy help address the primary cause of climate change?",
            answers=[
                "renewable energy produces minimal greenhouse gas emissions unlike fossil fuels which are the primary cause of climate change"
            ],
            query_type=QueryType.CROSS_DOC,
            difficulty=DifficultyLevel.HARD,
            source_docs=[doc_ids[1], doc_ids[5]],  # Climate causes + Renewable energy
            required_concepts=["renewable energy", "fossil fuels", "greenhouse gas emissions", "climate change"]
        ),
        RetrievalQuery(
            question="What technologies mentioned could help solve climate change?",
            answers=[
                "renewable energy sources", "solar power", "wind power", "quantum computing applications"
            ],
            query_type=QueryType.CROSS_DOC,
            difficulty=DifficultyLevel.HARD,
            source_docs=[doc_ids[2], doc_ids[4], doc_ids[5]],  # Climate solutions + Quantum + Renewable
            required_concepts=["renewable energy", "technology", "climate change", "solutions"]
        ),
        RetrievalQuery(
            question="How might AI and quantum computing together address climate challenges?",
            answers=[
                "AI and quantum computing could optimize renewable energy systems and solve complex climate models"
            ],
            query_type=QueryType.CROSS_DOC,
            difficulty=DifficultyLevel.HARD,
            source_docs=[doc_ids[2], doc_ids[3], doc_ids[4]],  # Climate solutions + AI + Quantum
            required_concepts=["AI", "quantum computing", "climate", "optimization"]
        )
    ])
    
    # COMPARATIVE QUERIES - Require comparing different concepts
    queries.extend([
        RetrievalQuery(
            question="Compare solar power and wind power as renewable energy sources.",
            answers=[
                "solar power harnesses energy from the sun using photovoltaic cells while wind power captures kinetic energy from air movement using turbines"
            ],
            query_type=QueryType.COMPARATIVE,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[5]],  # Renewable energy
            required_concepts=["solar power", "wind power", "photovoltaic", "turbines"]
        ),
        RetrievalQuery(
            question="How do the challenges of climate change differ from the challenges of quantum computing development?",
            answers=[
                "climate change is a global environmental issue requiring policy solutions while quantum computing faces technical development challenges"
            ],
            query_type=QueryType.COMPARATIVE,
            difficulty=DifficultyLevel.HARD,
            source_docs=[doc_ids[0], doc_ids[1], doc_ids[4]],  # Climate def + causes + Quantum
            required_concepts=["climate change", "quantum computing", "challenges", "development"]
        )
    ])
    
    # CAUSAL QUERIES - About cause-effect relationships
    queries.extend([
        RetrievalQuery(
            question="Why does burning fossil fuels lead to climate change?",
            answers=[
                "burning fossil fuels releases greenhouse gases which trap heat in the atmosphere causing temperature rise"
            ],
            query_type=QueryType.CAUSAL,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[1]],  # Climate causes
            required_concepts=["fossil fuels", "greenhouse gases", "temperature", "atmosphere"]
        ),
        RetrievalQuery(
            question="How does transitioning to renewable energy impact greenhouse gas emissions?",
            answers=[
                "renewable energy produces minimal greenhouse gas emissions unlike fossil fuels"
            ],
            query_type=QueryType.CAUSAL,
            difficulty=DifficultyLevel.MEDIUM,
            source_docs=[doc_ids[2], doc_ids[5]],  # Climate solutions + Renewable
            required_concepts=["renewable energy", "greenhouse gas emissions", "fossil fuels"]
        )
    ])
    
    return queries