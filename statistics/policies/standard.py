"""
ESDE Migration Phase 2: Standard Condition Policy
==================================================

Standard policy for generating condition signatures from Substrate traces.

P0 Compliance:
  - P0-1: policy_id + version mixed into payload (collision prevention)
  - P0-2: Type preservation (no str() coercion)
  - P0-3: Canonical JSON unified with Substrate spec
  - P0-4: Missing keys explicitly listed

Spec: Migration Phase 2 v0.2.1 (Audit Fixed)
"""

import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseConditionPolicy

# Conditional import for ContextRecord
try:
    from esde.substrate.schema import ContextRecord
except ImportError:
    try:
        from substrate.schema import ContextRecord
    except ImportError:
        ContextRecord = Any  # type: ignore


# ==========================================
# Constants (Canonical JSON - Substrate Unified)
# ==========================================

# P0-3: These settings MUST match Substrate's canonical.py
CANONICAL_SORT_KEYS = True
CANONICAL_ENSURE_ASCII = False
CANONICAL_SEPARATORS = (',', ':')  # No spaces


# ==========================================
# Standard Condition Policy
# ==========================================

@dataclass
class StandardConditionPolicy(BaseConditionPolicy):
    """
    Standard policy that extracts condition factors from specified trace keys.
    
    This policy:
      1. Extracts values from ContextRecord.traces using target_keys
      2. Builds a payload with policy_id, version, factors, and missing keys
      3. Computes SHA256 of canonical JSON representation
    
    P0 Compliance:
      - P0-1: policy_id and version are included in hash input
      - P0-2: Types are preserved (int, float, bool, str, None)
      - P0-3: Canonical JSON matches Substrate spec exactly
      - P0-4: Missing keys are explicitly tracked in payload
    
    Example:
        policy = StandardConditionPolicy(
            policy_id="legacy_migration_v1",
            target_keys=["legacy:source_type", "legacy:language_profile"],
        )
        
        signature = policy.compute_signature(record)
        factors = policy.extract_factors(record)
    """
    
    policy_id: str
    target_keys: List[str]  # e.g., ["legacy:source_type", "legacy:language_profile"]
    version: str = "v1.0"   # P1-3: Policy versioning for future migrations
    
    def compute_signature(self, record: "ContextRecord") -> str:
        """
        Compute deterministic signature from ContextRecord.
        
        P0 Requirements:
          - Returns full SHA256 hex (64 chars)
          - Includes policy_id and version in hash input
          - Uses Canonical JSON (Substrate-unified)
        
        Args:
            record: ContextRecord with traces
            
        Returns:
            64-character hexadecimal SHA256 digest
        """
        # 1. Extract factors and missing keys (P0-4: separation)
        factors, missing_keys = self._extract_raw_factors(record)
        
        # 2. Build payload (P0-1: policy_id/version included)
        payload = {
            "policy_id": self.policy_id,
            "version": self.version,
            "factors": factors,       # P0-2: Types preserved
            "missing": missing_keys,  # P0-4: Explicit missing list
        }
        
        # 3. Canonical JSON (P0-3: Substrate spec unified)
        canonical = json.dumps(
            payload,
            sort_keys=CANONICAL_SORT_KEYS,
            ensure_ascii=CANONICAL_ENSURE_ASCII,
            separators=CANONICAL_SEPARATORS,
        )
        
        # 4. Full SHA256 hash (no truncation)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def extract_factors(self, record: "ContextRecord") -> Dict[str, Any]:
        """
        Extract factors for debugging/W6 output.
        
        P0-2 Compliance: Types are preserved (no str() conversion).
        
        Args:
            record: ContextRecord with traces
            
        Returns:
            Dict with original types preserved
        """
        factors, _ = self._extract_raw_factors(record)
        return factors
    
    def _extract_raw_factors(
        self, record: "ContextRecord"
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Internal method to extract factors and identify missing keys.
        
        P0-2: Types are preserved from traces (no str() coercion).
        P0-4: Missing keys are collected separately.
        
        Args:
            record: ContextRecord with traces
            
        Returns:
            Tuple of (factors dict, sorted missing keys list)
        """
        factors: Dict[str, Any] = {}
        missing: List[str] = []
        
        for key in self.target_keys:
            if key in record.traces:
                # P0-2: Preserve original type (int, float, bool, str, None)
                factors[key] = record.traces[key]
            else:
                missing.append(key)
        
        # Sort missing keys for determinism (input order independence)
        missing.sort()
        
        return factors, missing
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get policy metadata for debugging/logging.
        
        Returns:
            Dict with policy configuration
        """
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "target_keys": self.target_keys,
        }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("StandardConditionPolicy Test")
    print("=" * 60)
    
    # Mock ContextRecord for testing
    class MockContextRecord:
        def __init__(self, traces: Dict[str, Any]):
            self.traces = traces
    
    # Test 1: Basic signature computation
    print("\n[Test 1] Basic signature computation")
    
    policy = StandardConditionPolicy(
        policy_id="test_policy",
        target_keys=["legacy:source_type", "legacy:language_profile"],
        version="v1.0",
    )
    
    record = MockContextRecord(traces={
        "legacy:source_type": "news",
        "legacy:language_profile": "en",
    })
    
    sig = policy.compute_signature(record)
    print(f"  Signature: {sig}")
    print(f"  Length: {len(sig)}")
    
    assert len(sig) == 64, f"Expected 64 chars, got {len(sig)}"
    assert all(c in '0123456789abcdef' for c in sig), "Should be hex"
    print("  ✅ Full SHA256 (64 chars)")
    
    # Test 2: Determinism (same input = same output)
    print("\n[Test 2] Determinism")
    
    sig2 = policy.compute_signature(record)
    assert sig == sig2, "Same input should produce same signature"
    print("  ✅ Deterministic")
    
    # Test 3: Type preservation
    print("\n[Test 3] Type preservation")
    
    record_with_types = MockContextRecord(traces={
        "legacy:source_type": "news",
        "legacy:count": 42,
        "legacy:ratio": 0.75,
        "legacy:active": True,
    })
    
    policy_types = StandardConditionPolicy(
        policy_id="type_test",
        target_keys=["legacy:source_type", "legacy:count", "legacy:ratio", "legacy:active"],
    )
    
    factors = policy_types.extract_factors(record_with_types)
    print(f"  Factors: {factors}")
    print(f"  Types: {[(k, type(v).__name__) for k, v in factors.items()]}")
    
    assert isinstance(factors["legacy:count"], int), "int should be preserved"
    assert isinstance(factors["legacy:ratio"], float), "float should be preserved"
    assert isinstance(factors["legacy:active"], bool), "bool should be preserved"
    print("  ✅ Types preserved")
    
    # Test 4: Missing keys handling
    print("\n[Test 4] Missing keys handling")
    
    record_partial = MockContextRecord(traces={
        "legacy:source_type": "news",
        # legacy:language_profile is missing
    })
    
    factors_partial, missing = policy._extract_raw_factors(record_partial)
    print(f"  Factors: {factors_partial}")
    print(f"  Missing: {missing}")
    
    assert "legacy:language_profile" in missing, "Missing key should be tracked"
    assert "legacy:source_type" in factors_partial, "Present key should be in factors"
    print("  ✅ Missing keys tracked")
    
    # Test 5: Missing vs empty factors produce different signatures
    print("\n[Test 5] Missing vs empty distinction")
    
    record_empty = MockContextRecord(traces={})
    record_with_one = MockContextRecord(traces={
        "legacy:source_type": "unknown",
    })
    
    sig_empty = policy.compute_signature(record_empty)
    sig_with_one = policy.compute_signature(record_with_one)
    
    print(f"  Empty traces sig: {sig_empty[:16]}...")
    print(f"  One factor sig: {sig_with_one[:16]}...")
    
    assert sig_empty != sig_with_one, "Empty vs partial should differ"
    print("  ✅ Different signatures")
    
    # Test 6: Policy ID collision prevention
    print("\n[Test 6] Policy ID collision prevention")
    
    policy_a = StandardConditionPolicy(
        policy_id="policy_a",
        target_keys=["legacy:source_type"],
    )
    
    policy_b = StandardConditionPolicy(
        policy_id="policy_b",
        target_keys=["legacy:source_type"],
    )
    
    record_same = MockContextRecord(traces={
        "legacy:source_type": "news",
    })
    
    sig_a = policy_a.compute_signature(record_same)
    sig_b = policy_b.compute_signature(record_same)
    
    print(f"  Policy A sig: {sig_a[:16]}...")
    print(f"  Policy B sig: {sig_b[:16]}...")
    
    assert sig_a != sig_b, "Different policy_id should produce different signatures"
    print("  ✅ Policy ID prevents collision")
    
    # Test 7: Canonical JSON verification
    print("\n[Test 7] Canonical JSON verification")
    
    # Verify no spaces in JSON output
    test_payload = {
        "policy_id": "test",
        "version": "v1.0",
        "factors": {"key": "value"},
        "missing": [],
    }
    
    canonical = json.dumps(
        test_payload,
        sort_keys=CANONICAL_SORT_KEYS,
        ensure_ascii=CANONICAL_ENSURE_ASCII,
        separators=CANONICAL_SEPARATORS,
    )
    
    print(f"  Canonical JSON: {canonical}")
    assert ': ' not in canonical, "Should not have space after colon"
    assert ', ' not in canonical, "Should not have space after comma"
    print("  ✅ No spaces in separators")
    
    # Test 8: Missing keys sorting
    print("\n[Test 8] Missing keys sorting (input order independence)")
    
    policy_multi = StandardConditionPolicy(
        policy_id="multi",
        target_keys=["z:key", "a:key", "m:key"],
    )
    
    empty_record = MockContextRecord(traces={})
    _, missing_sorted = policy_multi._extract_raw_factors(empty_record)
    
    print(f"  target_keys order: {policy_multi.target_keys}")
    print(f"  missing_keys order: {missing_sorted}")
    
    assert missing_sorted == sorted(missing_sorted), "Missing keys should be sorted"
    print("  ✅ Missing keys are sorted")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
