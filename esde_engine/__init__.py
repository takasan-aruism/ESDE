"""
ESDE Engine v5.3.2
==================

Emotional Semantic Detection Engine with Multi-Hypothesis Routing.

Modules:
- config: Configuration constants and settings
- utils: Utility functions (tokenize, typo detection, etc.)
- loaders: Data loaders (SynapseLoader, GlossaryLoader)
- extractors: Synset extraction (SynsetExtractor)
- collectors: Activation collection (ProximityExplorer, ActivationCollector, OutputGate)
- routing: Unknown token routing (UnknownTokenRouter)
- queue: Unknown queue management (UnknownQueueWriter)
- engine: Main engine (ESDEEngine)
"""

from .config import VERSION
from .engine import ESDEEngine
from .routing import UnknownTokenRouter
from .queue import UnknownQueueWriter
from .loaders import SynapseLoader, GlossaryLoader
from .extractors import SynsetExtractor
from .collectors import ActivationCollector, OutputGate, ProximityExplorer

__version__ = VERSION
__all__ = [
    "ESDEEngine",
    "UnknownTokenRouter",
    "UnknownQueueWriter",
    "SynapseLoader",
    "GlossaryLoader",
    "SynsetExtractor",
    "ActivationCollector",
    "OutputGate",
    "ProximityExplorer",
]
