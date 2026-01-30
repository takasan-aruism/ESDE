"""
ESDE Phase 9: Pipeline Package
==============================

Internal condition-based statistics pipeline.

This package provides:
  - ConditionProvider: Pluggable condition extraction from internal structure
  - W2Aggregator: Condition-based token statistics
  - W3Calculator: S-Score calculation
  - W4Projector: Article vector projection
  - W5/W6Adapter: Clustering and export
  - run_full_pipeline: Complete W2-W6 pipeline

Key Innovation:
  - Conditions extracted from INTERNAL structure (section, passive, quote, etc.)
  - NOT from external metadata (source_type, language)
  - This solves the "1-condition death" problem

Spec: Phase 9 Pipeline v1.0
"""

from .condition_provider import (
    BaseConditionProvider,
    AggregationContext,
    SectionConditionProvider,
    PassiveConditionProvider,
    ParenthesesConditionProvider,
    QuoteConditionProvider,
    ProperNounConditionProvider,
    CONDITION_PROVIDERS,
    get_condition_provider,
)

from .w2_aggregator import (
    W2Aggregator,
    W2Stats,
    ConditionStats,
    aggregate_by_axis,
)

from .w3_calculator import (
    W3Calculator,
    W3Result,
    ConditionCandidates,
    CandidateToken,
)

from .w4_projector import (
    W4Projector,
    W4Result,
    ArticleVector,
    cosine_similarity,
    compute_pairwise_similarities,
)

from .w2_adapter import (
    convert_to_legacy_format,
    get_conversion_summary,
    LegacyW1GlobalStats,
    LegacyW2GlobalStats,
    LegacyW1Record,
    LegacyW2Record,
    LegacyConditionEntry,
)

from .w5_w6_adapter import (
    convert_to_w4_records,
    SimpleCondensator,
    SimpleIsland,
    SimpleStructure,
    export_structure_markdown,
    export_structure_json,
)

__all__ = [
    # Condition Provider
    "BaseConditionProvider",
    "AggregationContext",
    "SectionConditionProvider",
    "PassiveConditionProvider",
    "ParenthesesConditionProvider",
    "QuoteConditionProvider",
    "ProperNounConditionProvider",
    "CONDITION_PROVIDERS",
    "get_condition_provider",
    
    # W2 Aggregator
    "W2Aggregator",
    "W2Stats",
    "ConditionStats",
    "aggregate_by_axis",
    
    # W3 Calculator
    "W3Calculator",
    "W3Result",
    "ConditionCandidates",
    "CandidateToken",
    
    # W4 Projector
    "W4Projector",
    "W4Result",
    "ArticleVector",
    "cosine_similarity",
    "compute_pairwise_similarities",
    
    # W2 Adapter (Legacy)
    "convert_to_legacy_format",
    "get_conversion_summary",
    "LegacyW1GlobalStats",
    "LegacyW2GlobalStats",
    "LegacyW1Record",
    "LegacyW2Record",
    "LegacyConditionEntry",
    
    # W5/W6 Adapter
    "convert_to_w4_records",
    "SimpleCondensator",
    "SimpleIsland",
    "SimpleStructure",
    "export_structure_markdown",
    "export_structure_json",
]

__version__ = "1.0.0"
