"""
ESDE Migration Phase 2: Base Condition Policy
==============================================

Abstract base class for condition signature policies.

Design Principles:
  - Substrate層のContextRecordからCondition Signatureを生成
  - "記述はするが、決定しない" (Describe, but do not decide)

P0 Requirements:
  - compute_signature() returns full SHA256 hex (64 chars, no truncation)
  - extract_factors() preserves original types (no str() conversion)

Spec: Migration Phase 2 v0.2.1
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

# Import will be available when running from esde/ directory
# Conditional import for type hints
try:
    from esde.substrate.schema import ContextRecord
except ImportError:
    # Fallback for when running tests or from different directory
    try:
        from substrate.schema import ContextRecord
    except ImportError:
        # Type stub for IDE support
        ContextRecord = Any  # type: ignore


class BaseConditionPolicy(ABC):
    """
    Abstract base class for condition signature policies.
    
    A policy defines how to extract condition factors from a ContextRecord
    and compute a deterministic signature for W2 aggregation.
    
    Invariants:
      - Signature is deterministic (same input → same output)
      - Signature includes policy_id and version (P0-1: collision prevention)
      - Types are preserved in factors (P0-2: no str() coercion)
      - Missing keys are explicitly tracked (P0-4: no silent defaults)
    
    Subclasses must implement:
      - compute_signature(record) → str (64 hex chars)
      - extract_factors(record) → Dict[str, Any]
    """
    
    # Required attributes (must be set by subclasses)
    policy_id: str
    version: str
    
    @abstractmethod
    def compute_signature(self, record: "ContextRecord") -> str:
        """
        Compute deterministic signature from ContextRecord.
        
        P0 Requirement: Must return full SHA256 hex digest (64 chars).
        Do NOT truncate the hash.
        
        The signature should include:
          - policy_id (P0-1: prevents collision across policies)
          - version (P1-3: enables future migrations)
          - factors extracted from traces
          - missing_keys list (P0-4: distinguishes empty from missing)
        
        Args:
            record: ContextRecord with traces
            
        Returns:
            64-character hexadecimal SHA256 digest
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_factors(self, record: "ContextRecord") -> Dict[str, Any]:
        """
        Extract condition factors from ContextRecord for debugging/W6.
        
        P0 Requirement: Must preserve original types.
        Do NOT convert values to strings.
        
        This method is for:
          - Debugging and observability
          - W6 export (human-readable factors)
          - NOT for signature computation (use compute_signature for that)
        
        Args:
            record: ContextRecord with traces
            
        Returns:
            Dict with original types preserved (int, float, bool, str, None)
        """
        raise NotImplementedError
