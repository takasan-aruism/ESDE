"""
ESDE Phase 9-1/9-2/9-3: Statistics Package
==========================================
W1 (Weak Cross-Sectional Statistics) - The Pulse of Weakness
W2 (Weak Conditional Statistics) - The Context of Weakness
W3 (Weak Axis Candidates) - The Shadow of Structure

This package provides cross-sectional, conditional statistics,
and axis candidate calculation from W0 observations.

Invariants:
  INV-W1-001 (Recalculability):
      W1 can always be regenerated from W0.
  
  INV-W1-002 (Input Authority):
      W1 input is ArticleRecord.raw_text sliced by segment_span.
  
  INV-W2-001 (Read-Only Condition):
      W2Aggregator reads condition factors, never writes/infers.
  
  INV-W2-002 (Time Bucket Fixed):
      In v9.2, time_bucket is fixed to YYYY-MM (monthly).
  
  INV-W2-003 (Denominator Ownership):
      ConditionEntry.total_token_count is written by W2Aggregator only.
  
  INV-W3-001 (No Labeling):
      W3 output is factual only. No axis labels.
  
  INV-W3-002 (Immutable Input):
      W3 does not modify W1/W2 data.
  
  INV-W3-003 (Deterministic):
      Same W1/W2 input produces identical W3 output.

Spec: v5.4.3-P9.3
"""

from .schema import (
    # Constants
    AGGREGATION_VERSION,
    SURFACE_FORMS_LIMIT,
    SURFACE_FORMS_OTHER_KEY,
    SURFACE_FORMS_LONG_KEY,
    
    # Data Structures
    W1Record,
    W1GlobalStats,
)

from .schema_w2 import (
    # Constants
    W2_VERSION,
    SOURCE_META_VERSION,
    SOURCE_TYPES,
    LANGUAGE_PROFILES,
    DEFAULT_SOURCE_TYPE,
    DEFAULT_LANGUAGE_PROFILE,
    
    # Data Structures
    W2Record,
    W2GlobalStats,
    ConditionEntry,
    
    # Functions
    compute_condition_signature,
    compute_record_id,
    compute_time_bucket,
)

from .schema_w3 import (
    # Constants
    W3_VERSION,
    W3_ALGORITHM,
    EPSILON,
    DEFAULT_MIN_COUNT_FOR_W3,
    DEFAULT_TOP_K,
    
    # Data Structures
    W3Record,
    CandidateToken,
    
    # Functions
    compute_analysis_id,
    compute_stats_hash,
)

from .tokenizer import (
    # Base Class
    W1Tokenizer,
    TokenResult,
    
    # Implementations
    EnglishWordTokenizer,
    CJKBigramTokenizer,
    HybridTokenizer,
    
    # Factory
    get_w1_tokenizer,
)

from .normalizer import (
    normalize_token,
    is_valid_token,
    NORMALIZER_VERSION,
)

from .w1_aggregator import (
    W1Aggregator,
    compare_w1_stats,
    format_source_id,
    DEFAULT_STORAGE_PATH,
)

from .w2_aggregator import (
    W2Aggregator,
    compare_w2_stats,
    DEFAULT_W2_RECORDS_PATH,
    DEFAULT_W2_CONDITIONS_PATH,
)

from .w3_calculator import (
    W3Calculator,
    compare_w3_records,
    DEFAULT_W3_OUTPUT_DIR,
)

__all__ = [
    # W1 Schema
    "AGGREGATION_VERSION",
    "SURFACE_FORMS_LIMIT",
    "SURFACE_FORMS_OTHER_KEY",
    "SURFACE_FORMS_LONG_KEY",
    "W1Record",
    "W1GlobalStats",
    
    # W2 Schema
    "W2_VERSION",
    "SOURCE_META_VERSION",
    "SOURCE_TYPES",
    "LANGUAGE_PROFILES",
    "DEFAULT_SOURCE_TYPE",
    "DEFAULT_LANGUAGE_PROFILE",
    "W2Record",
    "W2GlobalStats",
    "ConditionEntry",
    "compute_condition_signature",
    "compute_record_id",
    "compute_time_bucket",
    
    # W3 Schema
    "W3_VERSION",
    "W3_ALGORITHM",
    "EPSILON",
    "DEFAULT_MIN_COUNT_FOR_W3",
    "DEFAULT_TOP_K",
    "W3Record",
    "CandidateToken",
    "compute_analysis_id",
    "compute_stats_hash",
    
    # Tokenizer
    "W1Tokenizer",
    "TokenResult",
    "EnglishWordTokenizer",
    "CJKBigramTokenizer",
    "HybridTokenizer",
    "get_w1_tokenizer",
    
    # Normalizer
    "normalize_token",
    "is_valid_token",
    "NORMALIZER_VERSION",
    
    # W1 Aggregator
    "W1Aggregator",
    "compare_w1_stats",
    "format_source_id",
    "DEFAULT_STORAGE_PATH",
    
    # W2 Aggregator
    "W2Aggregator",
    "compare_w2_stats",
    "DEFAULT_W2_RECORDS_PATH",
    "DEFAULT_W2_CONDITIONS_PATH",
    
    # W3 Calculator
    "W3Calculator",
    "compare_w3_records",
    "DEFAULT_W3_OUTPUT_DIR",
]

__version__ = "5.4.3-P9.3"
