"""
ESDE Phase 9-6: Discovery Package
=================================
W6 (Weak Structural Observation) components.

Theme: The Observatory - From Structure to Evidence (構造から証拠へ)

Components:
  - schema_w6.py: Data structures (W6Observatory, W6IslandDetail, etc.)
  - w6_analyzer.py: Evidence extraction and topology calculation
  - w6_exporter.py: Export to MD/CSV/JSON formats

Design Philosophy:
  - W6 = Observation Window, NOT computation layer
  - No New Math: Transforms/extracts only
  - Strict Determinism: Bit-level reproducibility
  - Evidence-based: All outputs traceable to W3/W4/W5

Invariants (W6 Constitution):
  INV-W6-001: No Synthetic Labels
  INV-W6-002: Deterministic Export
  INV-W6-003: Read Only
  INV-W6-004: No New Math
  INV-W6-005: Evidence Provenance
  INV-W6-006: Stable Ordering
  INV-W6-007: No Hypothesis
  INV-W6-008: Strict Versioning
  INV-W6-009: Scope Closure

Spec: v5.4.6-P9.6-Final
"""

from .schema_w6 import (
    # Constants
    W6_VERSION,
    W6_CODE_VERSION,
    W6_EVIDENCE_POLICY,
    W6_SNIPPET_POLICY,
    W6_METRIC_POLICY,
    W6_DIGEST_POLICY,
    W6_SNIPPET_LENGTH,
    W6_DIGEST_K,
    W6_EVIDENCE_K,
    W6_REP_ARTICLE_K,
    W6_TOPOLOGY_K,
    W6_DISTANCE_ROUNDING,
    W6_EVIDENCE_ROUNDING,
    DEFAULT_W6_OUTPUT_DIR,
    
    # Utilities
    get_canonical_json,
    compute_canonical_hash,
    
    # Data Structures
    W6EvidenceToken,
    W6RepresentativeArticle,
    W6IslandDetail,
    W6TopologyPair,
    W6Observatory,
    
    # Comparison
    compare_w6_observations,
)

from .w6_analyzer import (
    W6Analyzer,
)

from .w6_exporter import (
    W6Exporter,
    export_observation,
    check_no_labels,
    FORBIDDEN_LABELS,
)

__all__ = [
    # Constants
    "W6_VERSION",
    "W6_CODE_VERSION",
    "W6_EVIDENCE_POLICY",
    "W6_SNIPPET_POLICY",
    "W6_METRIC_POLICY",
    "W6_DIGEST_POLICY",
    "W6_SNIPPET_LENGTH",
    "W6_DIGEST_K",
    "W6_EVIDENCE_K",
    "W6_REP_ARTICLE_K",
    "W6_TOPOLOGY_K",
    "W6_DISTANCE_ROUNDING",
    "W6_EVIDENCE_ROUNDING",
    "DEFAULT_W6_OUTPUT_DIR",
    
    # Utilities
    "get_canonical_json",
    "compute_canonical_hash",
    
    # Data Structures
    "W6EvidenceToken",
    "W6RepresentativeArticle",
    "W6IslandDetail",
    "W6TopologyPair",
    "W6Observatory",
    
    # Comparison
    "compare_w6_observations",
    
    # Analyzer
    "W6Analyzer",
    
    # Exporter
    "W6Exporter",
    "export_observation",
    "check_no_labels",
    "FORBIDDEN_LABELS",
]

__version__ = "5.4.6-P9.6"
