"""
ESDE Phase 9-1: Statistics Package
==================================
W1 (Weak Cross-Sectional Statistics) - The Pulse of Weakness

This package provides cross-sectional statistics aggregation
from W0 observations.

Invariants:
  INV-W1-001 (Recalculability):
      W1 can always be regenerated from W0. Loss of W1 is not system corruption.
  
  INV-W1-002 (Input Authority):
      W1 input is ArticleRecord.raw_text sliced by ObservationEvent.segment_span.
      ObservationEvent.segment_text is cache only, not authoritative.

Components:
  - schema.py: W1Record, W1GlobalStats
  - tokenizer.py: EnglishWordTokenizer, CJKBigramTokenizer, HybridTokenizer
  - normalizer.py: Token normalization (NFKC + punctuation handling)
  - w1_aggregator.py: W1Aggregator (main entry point)

Spec: v5.4.1-P9.1
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

__all__ = [
    # Schema
    "AGGREGATION_VERSION",
    "SURFACE_FORMS_LIMIT",
    "SURFACE_FORMS_OTHER_KEY",
    "SURFACE_FORMS_LONG_KEY",
    "W1Record",
    "W1GlobalStats",
    
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
    
    # Aggregator
    "W1Aggregator",
    "compare_w1_stats",
    "format_source_id",
    "DEFAULT_STORAGE_PATH",
]

__version__ = "5.4.1-P9.1"
