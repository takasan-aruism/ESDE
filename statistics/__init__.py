"""
ESDE Phase 9: Statistics Package
================================
W1/W2/W3/W4/W5 Statistics Components

Spec: v5.4.5-P9.5
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

# ==========================================
# Phase 9-2: W2 (Conditional Statistics)
# ==========================================

from .schema_w2 import (
    W2Record,
    W2GlobalStats,
    ConditionEntry,
    compute_condition_signature,
    compute_record_id,
    compute_time_bucket,
    W2_VERSION,
    SOURCE_TYPES,
    LANGUAGE_PROFILES,
)

from .w2_aggregator import (
    W2Aggregator,
)

# ==========================================
# Phase 9-3: W3 (Axis Candidates)
# ==========================================

from .schema_w3 import (
    W3Record,
    CandidateToken,
    compute_analysis_id,
    compute_stats_hash,
    W3_VERSION,
    W3_ALGORITHM,
    EPSILON,
    DEFAULT_MIN_COUNT_FOR_W3,
    DEFAULT_TOP_K,
)

from .w3_calculator import (
    W3Calculator,
    compare_w3_records,
)

# ==========================================
# Phase 9-4: W4 (Structural Projection)
# ==========================================

from .schema_w4 import (
    W4Record,
    compute_w4_analysis_id,
    W4_VERSION,
    W4_ALGORITHM,
    W4_PROJECTION_NORM,
    DEFAULT_W4_OUTPUT_DIR,
)

from .w4_projector import (
    W4Projector,
    build_sscore_dict,
    compare_w4_records,
)
# ==========================================
# Phase 9-5: W5 (Canonical Structures)
# ==========================================
# W5 exports
from .schema_w5 import (
    W5Island,
    W5Structure,
    get_canonical_json,
    compute_canonical_hash,
    compare_w5_structures,
    W5_VERSION,
    W5_ALGORITHM,
    W5_VECTOR_POLICY,
)

from .w5_condensator import (
    W5Condensator,
    condense_batch,
)

__all__ = [
    # W1 Schema
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
    
    # W1 Aggregator
    "W1Aggregator",
    "compare_w1_stats",
    "format_source_id",
    "DEFAULT_STORAGE_PATH",
    
    # W2
    "W2Record",
    "W2GlobalStats",
    "ConditionEntry",
    "compute_condition_signature",
    "compute_record_id",
    "compute_time_bucket",
    "W2_VERSION",
    "SOURCE_TYPES",
    "LANGUAGE_PROFILES",
    "W2Aggregator",
    
    # W3
    "W3Record",
    "CandidateToken",
    "compute_analysis_id",
    "compute_stats_hash",
    "W3_VERSION",
    "W3_ALGORITHM",
    "EPSILON",
    "DEFAULT_MIN_COUNT_FOR_W3",
    "DEFAULT_TOP_K",
    "W3Calculator",
    "compare_w3_records",
    
    # W4
    "W4Record",
    "compute_w4_analysis_id",
    "W4_VERSION",
    "W4_ALGORITHM",
    "W4_PROJECTION_NORM",
    "DEFAULT_W4_OUTPUT_DIR",
    "W4Projector",
    "build_sscore_dict",
    "compare_w4_records",

    # W5
    "W5Island",
    "W5Structure",
    "get_canonical_json",
    "compute_canonical_hash",
    "compare_w5_structures",
    "W5_VERSION",
    "W5_ALGORITHM",
    "W5_VECTOR_POLICY",
    "W5Condensator",
    "condense_batch",
]

__version__ = "5.4.5-P9.5"