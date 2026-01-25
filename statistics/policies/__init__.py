"""
ESDE Migration Phase 2: Condition Policies Package
===================================================

Policy-based condition signature generation for W2 Aggregator.

This package provides:
  - BaseConditionPolicy: Abstract base class
  - StandardConditionPolicy: Standard implementation for trace-based extraction

Design Philosophy:
  - Substrate層のContextRecordからW2のCondition Signatureを生成
  - Policy ID/Versionによる衝突防止
  - 型維持による決定論保証

Spec: Migration Phase 2 v0.2.1
"""

from .base import BaseConditionPolicy
from .standard import (
    StandardConditionPolicy,
    CANONICAL_SORT_KEYS,
    CANONICAL_ENSURE_ASCII,
    CANONICAL_SEPARATORS,
)

__all__ = [
    # Base
    "BaseConditionPolicy",
    
    # Standard
    "StandardConditionPolicy",
    
    # Constants (for testing/verification)
    "CANONICAL_SORT_KEYS",
    "CANONICAL_ENSURE_ASCII",
    "CANONICAL_SEPARATORS",
]

__version__ = "0.2.1"
