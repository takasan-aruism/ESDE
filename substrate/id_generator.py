"""
ESDE Substrate Layer: Canonical ID Generator
=============================================

Deterministic ID generation for ContextRecord.

Philosophy: "Same inputs = Same ID, always."

GPT Audit Compliance:
  - GPT追記2: Canonical JSON仕様固定
  - INV-SUB-006: ID Determinism

Key Requirements:
  - No timestamp dependency
  - No random component
  - Reproducible across environments
  - Float precision fixed
  - Key ordering fixed

Spec: Substrate Layer v0.1.0
"""

import hashlib
import json
from typing import Any, Dict, Optional

from .traces import normalize_traces, FLOAT_PRECISION

# ==========================================
# Constants
# ==========================================

# ID generation version (for future migration)
ID_GENERATOR_VERSION = "v0.1.0"

# Hash truncation length (32 hex chars = 128 bits)
CONTEXT_ID_LENGTH = 32

# Canonical JSON settings (GPT追記2)
CANONICAL_JSON_SETTINGS = {
    "sort_keys": True,
    "separators": (",", ":"),  # No spaces
    "ensure_ascii": False,     # Allow Unicode
}

# ==========================================
# Canonical JSON Encoder
# ==========================================

class CanonicalJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for canonical (deterministic) output.
    
    GPT追記2: 浮動小数点の文字列表現を固定
    
    Rules:
      - Floats are rounded to FLOAT_PRECISION decimal places
      - Floats are represented without unnecessary trailing zeros
      - Keys are always sorted (handled by dumps)
      - No pretty printing
    """
    
    def encode(self, o: Any) -> str:
        """
        Override encode to handle the full object.
        """
        return super().encode(self._normalize_for_json(o))
    
    def _normalize_for_json(self, obj: Any) -> Any:
        """
        Recursively normalize values for canonical JSON.
        """
        if obj is None:
            return None
        
        if isinstance(obj, bool):
            return obj
        
        if isinstance(obj, int):
            return obj
        
        if isinstance(obj, float):
            # Round to fixed precision
            rounded = round(obj, FLOAT_PRECISION)
            # Convert to string representation then back to float
            # This ensures consistent representation
            return float(f"{rounded:.{FLOAT_PRECISION}g}")
        
        if isinstance(obj, str):
            return obj
        
        if isinstance(obj, dict):
            # Sort keys and normalize values
            return {k: self._normalize_for_json(v) for k, v in sorted(obj.items())}
        
        if isinstance(obj, (list, tuple)):
            return [self._normalize_for_json(item) for item in obj]
        
        # Fallback: convert to string
        return str(obj)


def canonical_json_dumps(obj: Any) -> str:
    """
    Serialize object to canonical JSON string.
    
    This function produces deterministic JSON output:
      - Keys are sorted
      - No whitespace
      - Floats have fixed precision
      - Unicode is preserved (ensure_ascii=False)
    
    Args:
        obj: Object to serialize
        
    Returns:
        Canonical JSON string
    """
    encoder = CanonicalJSONEncoder(**CANONICAL_JSON_SETTINGS)
    return encoder.encode(obj)


# ==========================================
# ID Generation
# ==========================================

def compute_context_id(
    retrieval_path: Optional[str],
    traces: Dict[str, Any],
    capture_version: str,
) -> str:
    """
    Compute deterministic context_id from canonical inputs.
    
    INV-SUB-006: ID depends ONLY on (retrieval_path, traces, capture_version).
    No timestamp, no random component.
    
    Algorithm:
      1. Normalize traces (sort keys, round floats) - P1-SUB-004: float丸めはここで先に行う
      2. Build canonical payload dict
      3. Serialize to canonical JSON
      4. SHA256 hash
      5. Truncate to CONTEXT_ID_LENGTH
    
    P1-SUB-004: Normalization Order
      - Float rounding happens FIRST in normalize_traces()
      - Then canonical JSON serialization (which also rounds, but values are already normalized)
      - This ensures environment-independent determinism
    
    Args:
        retrieval_path: URL, file path, or None
        traces: Trace dict (will be normalized)
        capture_version: Version of trace extraction logic
        
    Returns:
        32-character hex string (context_id)
        
    Example:
        >>> compute_context_id(
        ...     retrieval_path="https://example.com/article",
        ...     traces={"html:tag_count": 42, "text:char_count": 1000},
        ...     capture_version="v1.0"
        ... )
        'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
    """
    # Step 1: Normalize traces FIRST (P1-SUB-004: float rounding happens here)
    normalized_traces = normalize_traces(traces)
    
    # Step 2: Build canonical payload
    # Keys are explicitly sorted for determinism
    payload = {
        "capture_version": capture_version,
        "retrieval_path": retrieval_path,
        "traces": normalized_traces,
    }
    
    # Step 3: Serialize to canonical JSON
    canonical_str = canonical_json_dumps(payload)
    
    # Step 4: SHA256 hash
    hash_bytes = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    
    # Step 5: Truncate
    return hash_bytes[:CONTEXT_ID_LENGTH]


def verify_context_id(
    context_id: str,
    retrieval_path: Optional[str],
    traces: Dict[str, Any],
    capture_version: str,
) -> bool:
    """
    Verify that a context_id matches the expected inputs.
    
    Args:
        context_id: The ID to verify
        retrieval_path: Expected retrieval path
        traces: Expected traces
        capture_version: Expected capture version
        
    Returns:
        True if context_id matches, False otherwise
    """
    expected_id = compute_context_id(retrieval_path, traces, capture_version)
    return context_id == expected_id


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Substrate ID Generator Test")
    print("=" * 60)
    
    # Test 1: Determinism
    print("\n[Test 1] Determinism - same input = same output")
    traces1 = {"html:tag_count": 42, "text:char_count": 1000}
    
    id1 = compute_context_id("https://example.com", traces1, "v1.0")
    id2 = compute_context_id("https://example.com", traces1, "v1.0")
    
    print(f"  ID 1: {id1}")
    print(f"  ID 2: {id2}")
    assert id1 == id2, "IDs should match"
    print("  ✅ IDs match")
    
    # Test 2: Key order independence
    print("\n[Test 2] Key order independence")
    traces_a = {"html:tag_count": 42, "text:char_count": 1000}
    traces_b = {"text:char_count": 1000, "html:tag_count": 42}
    
    id_a = compute_context_id(None, traces_a, "v1.0")
    id_b = compute_context_id(None, traces_b, "v1.0")
    
    print(f"  traces_a order: {list(traces_a.keys())}")
    print(f"  traces_b order: {list(traces_b.keys())}")
    print(f"  ID A: {id_a}")
    print(f"  ID B: {id_b}")
    assert id_a == id_b, "IDs should match regardless of key order"
    print("  ✅ Key order doesn't affect ID")
    
    # Test 3: Float precision consistency
    print("\n[Test 3] Float precision consistency")
    traces_float1 = {"text:ratio": 0.1 + 0.2}  # Famous float issue
    traces_float2 = {"text:ratio": 0.3}
    
    id_f1 = compute_context_id(None, traces_float1, "v1.0")
    id_f2 = compute_context_id(None, traces_float2, "v1.0")
    
    print(f"  0.1 + 0.2 = {0.1 + 0.2}")
    print(f"  0.3       = {0.3}")
    print(f"  ID (0.1+0.2): {id_f1}")
    print(f"  ID (0.3):     {id_f2}")
    assert id_f1 == id_f2, "Float precision should be normalized"
    print("  ✅ Float precision normalized correctly")
    
    # Test 4: Different inputs = different IDs
    print("\n[Test 4] Different inputs = different IDs")
    id_diff1 = compute_context_id("https://a.com", traces1, "v1.0")
    id_diff2 = compute_context_id("https://b.com", traces1, "v1.0")
    
    print(f"  ID (a.com): {id_diff1}")
    print(f"  ID (b.com): {id_diff2}")
    assert id_diff1 != id_diff2, "Different paths should produce different IDs"
    print("  ✅ Different inputs produce different IDs")
    
    # Test 5: Canonical JSON output
    print("\n[Test 5] Canonical JSON output")
    payload = {
        "z_key": "last",
        "a_key": "first",
        "num": 3.14159265358979,
    }
    canonical = canonical_json_dumps(payload)
    print(f"  Input:  {payload}")
    print(f"  Output: {canonical}")
    assert '"a_key"' in canonical.split('"z_key"')[0], "Keys should be sorted"
    print("  ✅ Keys are sorted in output")
    
    # Test 6: Verification
    print("\n[Test 6] ID verification")
    traces = {"html:has_h1": True}
    path = "https://test.com"
    version = "v1.0"
    
    generated_id = compute_context_id(path, traces, version)
    is_valid = verify_context_id(generated_id, path, traces, version)
    is_invalid = verify_context_id("wrong_id", path, traces, version)
    
    print(f"  Generated ID: {generated_id}")
    print(f"  Valid check:   {is_valid}")
    print(f"  Invalid check: {is_invalid}")
    assert is_valid, "Valid ID should verify"
    assert not is_invalid, "Invalid ID should not verify"
    print("  ✅ Verification works correctly")
    
    # Test 7: ID length
    print("\n[Test 7] ID length")
    test_id = compute_context_id(None, {"a:b": 1}, "v1.0")
    print(f"  ID length: {len(test_id)} (expected: {CONTEXT_ID_LENGTH})")
    assert len(test_id) == CONTEXT_ID_LENGTH
    print("  ✅ ID length correct")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
