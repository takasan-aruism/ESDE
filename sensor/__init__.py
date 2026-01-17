"""
ESDE Sensor Package
Phase 8: Synapse-Integrated Concept Extraction

Structure:
  - loader_synapse.py:   SynapseLoader (memory-resident)
  - extract_synset.py:   SynsetExtractor (WordNet → synsets)
  - rank_candidates.py:  CandidateRanker (aggregation, sorting, hash)
  - legacy_trigger.py:   LegacyTriggerMatcher (v1 fallback)
  - audit_trace.py:      AuditTracer (counters, hash, evidence)
  
Phase 8-2:
  - molecule_generator.py: MoleculeGenerator (LLM-based, mock)
  - molecule_validator.py: MoleculeValidator (integrity checks)

Phase 8-3:
  - molecule_generator_live.py: MoleculeGeneratorLive (Real LLM with guardrails)

Usage:
  # Phase 8 (core)
  from sensor import SynapseLoader, SynsetExtractor, CandidateRanker
  
  # Phase 8-2/8-3 (direct import to avoid circular issues)
  from sensor.molecule_validator import MoleculeValidator
  from sensor.molecule_generator import MoleculeGenerator
  from sensor.molecule_generator_live import MoleculeGeneratorLive
"""

# Phase 8 core imports only (no side effects)
from .loader_synapse import SynapseLoader
from .extract_synset import SynsetExtractor
from .rank_candidates import CandidateRanker
from .legacy_trigger import LegacyTriggerMatcher
from .audit_trace import AuditTracer

# Phase 8-2/8-3: Lazy import to avoid RuntimeWarning
# Users should import directly: from sensor.molecule_validator import ...
_LAZY_IMPORTS = {
    # Phase 8-2
    "MoleculeValidator": "molecule_validator",
    "ValidationResult": "molecule_validator", 
    "GlossaryValidator": "molecule_validator",
    "MoleculeGenerator": "molecule_generator",
    "MockMoleculeGenerator": "molecule_generator",
    "GenerationResult": "molecule_generator",
    # Phase 8-3
    "MoleculeGeneratorLive": "molecule_generator_live",
    "MockMoleculeGeneratorLive": "molecule_generator_live",
    "LiveGenerationResult": "molecule_generator_live",
    "SpanCalculator": "molecule_generator_live",
    "CoordinateCoercer": "molecule_generator_live",
    "FormulaValidator": "molecule_generator_live",
}

def __getattr__(name):
    """Lazy import for Phase 8-2/8-3 modules."""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Phase 8
    "SynapseLoader",
    "SynsetExtractor", 
    "CandidateRanker",
    "LegacyTriggerMatcher",
    "AuditTracer",
    # Phase 8-2 (lazy)
    "MoleculeValidator",
    "ValidationResult",
    "GlossaryValidator",
    "MoleculeGenerator",
    "MockMoleculeGenerator",
    "GenerationResult",
    # Phase 8-3 (lazy)
    "MoleculeGeneratorLive",
    "MockMoleculeGeneratorLive",
    "LiveGenerationResult",
    "SpanCalculator",
    "CoordinateCoercer",
    "FormulaValidator",
]

__version__ = "2.0.0"
