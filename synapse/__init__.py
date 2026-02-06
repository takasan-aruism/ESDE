"""
ESDE Synapse Package
====================
Shared Synapse data layer for all ESDE subsystems.

SynapseStore is the SINGLE SOURCE OF TRUTH for Synapse data.
All consumers (Phase 8 Sensor, Phase 7 Engine, Observation C)
MUST use SynapseStore rather than loading Synapse JSON directly.

GO Condition (GPT Audit):
  SynapseStore is used by Phase 8 Sensor (sensor/loader_synapse.py)
  AND Observation C (integration/relations/relation_logger.py).
  Patch effects are reflected in BOTH systems simultaneously.
  If either system bypasses SynapseStore, tests MUST fail.

Design Spec: v2.1 (Gemini design, GPT audit, Claude implementation)
"""

from .store import SynapseStore
from .schema import SynapsePatchEntry

__all__ = [
    "SynapseStore",
    "SynapsePatchEntry",
]

__version__ = "1.0.0"
