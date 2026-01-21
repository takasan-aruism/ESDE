"""
ESDE Phase 9: Statistics Package
================================
W1/W2/W3/W4 statistics components.

Exports:
  - Tokenizer: get_w1_tokenizer, W1Tokenizer
  - Normalizer: normalize_token, NORMALIZER_VERSION
  - W3: W3Record, CandidateToken
  - W4: W4Record, W4Projector

Spec: v5.4.4-P9.4
"""

# Tokenizer
from .tokenizer import get_w1_tokenizer, W1Tokenizer, TokenResult

# Normalizer
from .normalizer import normalize_token, is_valid_token, NORMALIZER_VERSION

# W3 Schema
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

# W4 Schema and Projector
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

__all__ = [
    # Tokenizer
    'get_w1_tokenizer',
    'W1Tokenizer',
    'TokenResult',
    
    # Normalizer
    'normalize_token',
    'is_valid_token',
    'NORMALIZER_VERSION',
    
    # W3
    'W3Record',
    'CandidateToken',
    'compute_analysis_id',
    'compute_stats_hash',
    'W3_VERSION',
    'W3_ALGORITHM',
    'EPSILON',
    'DEFAULT_MIN_COUNT_FOR_W3',
    'DEFAULT_TOP_K',
    
    # W4
    'W4Record',
    'W4Projector',
    'compute_w4_analysis_id',
    'build_sscore_dict',
    'compare_w4_records',
    'W4_VERSION',
    'W4_ALGORITHM',
    'W4_PROJECTION_NORM',
    'DEFAULT_W4_OUTPUT_DIR',
]
