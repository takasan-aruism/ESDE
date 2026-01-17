"""
ESDE Engine v5.3.4 - Phase 7B+: Resolver Package (Cleaned)

Unknown Queue Resolution with Evidence Ledger.
Target: No Winner, Volatility First, Append-Only.

Components:
- aggregate_state: New 7B+ State Lifecycle (NEW/OBSERVING/STABLE/VOLATILE)
- state: Legacy Queue state management (keep for compatibility)
- ledger: Evidence ledger (audit trail)
- cache: Search result caching
- online: Online search and evidence extraction
- hypothesis: Multi-hypothesis scoring with volatility (7B+)
"""

# 1. State Management
from .state import QueueStateManager
from .aggregate_state import AggregateStateManager  # <--- 重要: 7B+の新顔を追加

# 2. Audit & Ledger
from .ledger import EvidenceLedger, compute_reliability, compute_volatility

# 3. External Resources
from .cache import SearchCache
from .online import (
    SearchProvider,
    MockSearchProvider,
    SearXNGProvider,  # 2026/01/09追加
    EvidenceCollection,
    EvidenceItem,
    collect_evidence,
    generate_queries,
    get_source_tier,
    TIER1_DOMAINS,
    TIER2_DOMAINS
)

# 4. Configuration (SINGLE SOURCE OF TRUTH)
from ..config import (
    COMPETE_TH,
    ROUTE_A_MIN_SCORE,
    VOL_LOW_TH,
    VOL_HIGH_TH,
    DEFAULT_LOW_VOLATILITY_THRESHOLD,
    DEFAULT_HIGH_VOLATILITY_THRESHOLD,
    DEFAULT_CONFIDENCE_FLOOR
)

# 5. Logic Core (Hypothesis Evaluation)
from .hypothesis import (
    # 7B+ primary exports
    HypothesisResult,
    EvaluationReport,
    evaluate_all_hypotheses,
    determine_status,
    compute_global_volatility,
    compute_competing_count,
    compute_competing_info,
    compute_sense_conflict,
    compute_source_conflict,
    compute_evidence_dispersion,
    # Weights
    VOLATILITY_WEIGHTS,
    # Legacy exports (keep for safety)
    ScoringDecision,
    HypothesisScore,
    score_all_hypotheses,
    determine_action,
    compute_inter_hypothesis_conflict,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_VOLATILITY_THRESHOLD
)

# Export definition
__all__ = [
    # State
    "QueueStateManager",
    "AggregateStateManager", # <--- Added
    # Ledger
    "EvidenceLedger",
    "compute_reliability",
    "compute_volatility",
    # Cache
    "SearchCache",
    # Online    仕様変更につきコメントアウト　2026/01/10
    "SearchProvider",
    "MockSearchProvider",
    "SearXNGProvider",  # 2026/01/09 added
    "EvidenceCollection",
    "EvidenceItem",
    "collect_evidence",
    "generate_queries",
    "get_source_tier",
    "TIER1_DOMAINS",
    "TIER2_DOMAINS",
    # Hypothesis (7B+ primary)
    "HypothesisResult",
    "EvaluationReport",
    "evaluate_all_hypotheses",
    "determine_status",
    "compute_global_volatility",
    "compute_competing_count",
    "compute_competing_info",
    "compute_sense_conflict",
    "compute_source_conflict",
    "compute_evidence_dispersion",
    "VOLATILITY_WEIGHTS",
    # Constants
    "COMPETE_TH",
    "ROUTE_A_MIN_SCORE",
    "VOL_LOW_TH",
    "VOL_HIGH_TH",
    "DEFAULT_LOW_VOLATILITY_THRESHOLD",
    "DEFAULT_HIGH_VOLATILITY_THRESHOLD",
    "DEFAULT_CONFIDENCE_FLOOR",
    # Hypothesis (legacy)
    "ScoringDecision",
    "HypothesisScore",
    "score_all_hypotheses",
    "determine_action",
    "compute_inter_hypothesis_conflict",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_VOLATILITY_THRESHOLD",
]