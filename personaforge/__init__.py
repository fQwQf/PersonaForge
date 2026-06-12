"""PersonaForge — Psychology-Grounded Dual-Process Architecture for
Personality-Consistent Role-Playing Agents.

This package is the paper's core contribution, decoupled from the BookWorld
simulation engine it was originally built on:

  - personality_model    Three-layer persona (Big Five + Defense Mechanism,
                          Speaking Style, Dynamic State) and its enums.
  - dual_process_agent    Selective "Think-then-Speak" Inner Monologue.
  - dynamic_state_manager Mood / energy / relationship state updates.
  - style_vector_db       Speaking-style retrieval for few-shot styling
                          (imports `embedding`; pulls in torch/chromadb).
  - embedding             Embedding-model wrapper.
  - llm/ utils/ db/       Provider adapters and shared helpers.

The lightweight data model and agents below import only the standard library,
so `import personaforge` stays cheap; import `style_vector_db` / `embedding`
explicitly when you need the heavier ML dependencies.
"""

from personaforge.personality_model import (
    PersonalityProfile,
    CoreTraits,
    DynamicState,
    RelationshipInfo,
    DefenseMechanism,
)
from personaforge.dual_process_agent import DualProcessAgent
from personaforge.dynamic_state_manager import DynamicStateManager

__all__ = [
    "PersonalityProfile",
    "CoreTraits",
    "DynamicState",
    "RelationshipInfo",
    "DefenseMechanism",
    "DualProcessAgent",
    "DynamicStateManager",
]
