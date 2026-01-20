# ==========================================
# sensor/__init__.py への追記差分
# ==========================================
# 
# 既存の __init__.py に以下を追加してください：
#

# --- ADD THESE IMPORTS ---
from .constants import VALID_OPERATORS, BRACKET_PAIRS
from .glossary_validator import GlossaryValidator
from .validator_v83 import MoleculeValidatorV83, ValidationResultV83

# --- ADD TO __all__ ---
# "VALID_OPERATORS",
# "BRACKET_PAIRS",
# "GlossaryValidator",
# "MoleculeValidatorV83",
# "ValidationResultV83",

# ==========================================
# 完全な __init__.py の例（参考）
# ==========================================
"""
ESDE Sensor Package
Phase 8 Exports

Canonical (v8.3):
  - VALID_OPERATORS, BRACKET_PAIRS
  - GlossaryValidator
  - MoleculeValidatorV83, ValidationResultV83

Existing:
  - SynapseLoader
  - SynsetExtractor
  - CandidateRanker
  - LegacyTriggerMatcher
  - AuditTracer
  - MoleculeGeneratorLive
"""

# Existing exports
from .loader_synapse import SynapseLoader
from .extract_synset import SynsetExtractor
from .rank_candidates import CandidateRanker
from .legacy_trigger import LegacyTriggerMatcher
from .audit_trace import AuditTracer
from .molecule_generator_live import MoleculeGeneratorLive

# NEW: Canonical v8.3 exports
from .constants import VALID_OPERATORS, BRACKET_PAIRS
from .glossary_validator import GlossaryValidator
from .validator_v83 import MoleculeValidatorV83, ValidationResultV83

__all__ = [
    # Existing
    "SynapseLoader",
    "SynsetExtractor",
    "CandidateRanker",
    "LegacyTriggerMatcher",
    "AuditTracer",
    "MoleculeGeneratorLive",
    # NEW: Canonical v8.3
    "VALID_OPERATORS",
    "BRACKET_PAIRS",
    "GlossaryValidator",
    "MoleculeValidatorV83",
    "ValidationResultV83",
]
