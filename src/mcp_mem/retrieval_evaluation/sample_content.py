"""
Sample content for retrieval evaluation.

This module provides sample content on various topics that can be used
for testing and evaluating retrieval systems.
"""

# Climate change definition
CLIMATE_CHANGE_DEFINITION = """
Climate change is the long-term alteration of temperature and typical weather patterns in a place.
Climate change could refer to a particular location or the planet as a whole.
Climate change may cause weather patterns to be less predictable.
These unexpected weather patterns can make it difficult to maintain and grow crops in regions that rely on farming.
"""

# Climate change causes
CLIMATE_CHANGE_CAUSES = """
The primary cause of climate change is human activities, particularly the burning of fossil fuels,
like coal, oil, and natural gas. Burning these materials releases what are called greenhouse gases into Earth's atmosphere.
These gases trap heat from the sun's rays inside the atmosphere causing Earth's average temperature to rise.
"""

# Climate change solutions
CLIMATE_CHANGE_SOLUTIONS = """
Solutions to climate change include transitioning to renewable energy sources like solar and wind power,
improving energy efficiency in buildings and transportation, protecting and restoring forests and other ecosystems,
and adopting more sustainable agricultural practices. International cooperation through agreements like the Paris Climate Accord
aims to limit global temperature increases and address climate change impacts.
"""

# Artificial intelligence
ARTIFICIAL_INTELLIGENCE = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed
to think and learn like humans. The term may also be applied to any machine that exhibits traits associated
with a human mind such as learning and problem-solving. AI can be categorized as either weak AI or strong AI.
Weak AI, also known as narrow AI, is designed to perform a specific task, like voice recognition.
Strong AI, also known as artificial general intelligence, is AI that more fully replicates the autonomy of
the human brain—a machine with consciousness, sentience, and mind.
"""

# Quantum computing
QUANTUM_COMPUTING = """
Quantum computing is an area of computing focused on developing computer technology based on the principles
of quantum theory. Quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously,
allowing them to perform complex calculations at speeds unattainable by traditional computers.
Potential applications include cryptography, optimization problems, drug discovery, and materials science.
However, quantum computers are still in early development stages and face significant technical challenges.
"""

# Renewable energy
RENEWABLE_ENERGY = """
Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass power.
Unlike fossil fuels, these energy sources replenish naturally and produce minimal greenhouse gas emissions.
Solar power harnesses energy from the sun using photovoltaic cells or solar thermal systems.
Wind power captures kinetic energy from air movement using turbines. Hydroelectric power generates
electricity from flowing water, typically using dams. Geothermal energy taps into the Earth's internal heat.
Biomass energy comes from organic materials like plants and waste. The transition to renewable energy
is crucial for addressing climate change and creating sustainable energy systems.
"""

# Dictionary of all sample content
SAMPLE_CONTENT = {
    "climate_change_definition": CLIMATE_CHANGE_DEFINITION,
    "climate_change_causes": CLIMATE_CHANGE_CAUSES,
    "climate_change_solutions": CLIMATE_CHANGE_SOLUTIONS,
    "artificial_intelligence": ARTIFICIAL_INTELLIGENCE,
    "quantum_computing": QUANTUM_COMPUTING,
    "renewable_energy": RENEWABLE_ENERGY
}

def get_all_content() -> tuple:
    """
    Get all sample content as a tuple.
    
    Returns:
        Tuple of (climate_change_definition, climate_change_causes, climate_change_solutions,
                 artificial_intelligence, quantum_computing, renewable_energy)
    """
    return (
        CLIMATE_CHANGE_DEFINITION,
        CLIMATE_CHANGE_CAUSES,
        CLIMATE_CHANGE_SOLUTIONS,
        ARTIFICIAL_INTELLIGENCE,
        QUANTUM_COMPUTING,
        RENEWABLE_ENERGY
    )