"""
ESDE Substrate Layer: Schema
============================

Core data structures for the Substrate (Context Fabric) layer.

Philosophy: "Describe, but do not decide."

Design Principles:
  - ContextRecord is IMMUTABLE (frozen=True)
  - No semantic interpretation
  - No trace_tags (removed per Gemini spec - tagging is W2's job)
  - Deterministic ID generation

Invariants:
  INV-SUB-001: Upper Read-Only
  INV-SUB-002: No Semantic Transform
  INV-SUB-003: Machine-Observable
  INV-SUB-004: No Inference
  INV-SUB-005: Append-Only
  INV-SUB-006: ID Determinism

Spec: Substrate Layer v0.1.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json

from .id_generator import compute_context_id, ID_GENERATOR_VERSION
from .traces import (
    normalize_traces,
    validate_traces,
    TRACE_NORMALIZER_VERSION,
)

# ==========================================
# Constants
# ==========================================

SUBSTRATE_SCHEMA_VERSION = "v0.1.0"

# Default capture version if not specified
DEFAULT_CAPTURE_VERSION = "v1.0"


# ==========================================
# ContextRecord (Frozen / Immutable)
# ==========================================

@dataclass(frozen=True)
class ContextRecord:
    """
    Substrate層の不変レコード。
    
    This is the core data structure of the Substrate layer.
    Once created, it cannot be modified (frozen=True).
    
    Invariants:
      INV-SUB-005: Append-only (no updates, no deletes)
      INV-SUB-006: context_id is deterministic
    
    Fields:
      Identity (Deterministic):
        - context_id: SHA256 hash of canonical(retrieval_path, traces, capture_version)
      
      Observation Facts (Used for ID generation):
        - retrieval_path: URL, file path, or None
        - capture_version: Version of trace extraction logic
        - traces: Schema-less observation values
      
      Metadata (NOT used for ID generation):
        - observed_at: When the observation occurred
        - created_at: When this record was created
    
    Note:
      trace_tags field was REMOVED per Gemini spec.
      Tag extraction is the responsibility of W2 Aggregator via Policy.
    """
    
    # === Identity (Deterministic) ===
    context_id: str
    
    # === Observation Facts (Canonical対象) ===
    retrieval_path: Optional[str]
    capture_version: str
    traces: Dict[str, Any]  # Normalized, sorted
    
    # === Metadata (Canonical対象外 - IDに影響しない) ===
    observed_at: str
    created_at: str
    
    # === Version Info ===
    schema_version: str = field(default=SUBSTRATE_SCHEMA_VERSION)
    
    def __post_init__(self):
        """
        Validate the record after creation.
        
        Note: Since frozen=True, we can't modify fields here.
        Validation must happen in the factory function.
        """
        # Validate traces (will raise if invalid)
        is_valid, errors = validate_traces(self.traces)
        if not is_valid:
            raise ValueError(f"Invalid traces: {errors}")
        
        # Verify context_id matches (defensive check)
        expected_id = compute_context_id(
            self.retrieval_path,
            self.traces,
            self.capture_version
        )
        if self.context_id != expected_id:
            raise ValueError(
                f"context_id mismatch: got {self.context_id}, expected {expected_id}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict representation (sorted keys for determinism)
        """
        return {
            "capture_version": self.capture_version,
            "context_id": self.context_id,
            "created_at": self.created_at,
            "observed_at": self.observed_at,
            "retrieval_path": self.retrieval_path,
            "schema_version": self.schema_version,
            "traces": dict(sorted(self.traces.items())),
        }
    
    def to_jsonl(self) -> str:
        """
        Convert to JSONL line (for Registry storage).
        
        INV-SUB-007: Canonical Export (sorted keys)
        
        Returns:
            Single-line JSON string
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextRecord":
        """
        Create ContextRecord from dictionary.
        
        Args:
            data: Dictionary with ContextRecord fields
            
        Returns:
            ContextRecord instance
        """
        return cls(
            context_id=data["context_id"],
            retrieval_path=data.get("retrieval_path"),
            capture_version=data["capture_version"],
            traces=data["traces"],
            observed_at=data["observed_at"],
            created_at=data["created_at"],
            schema_version=data.get("schema_version", SUBSTRATE_SCHEMA_VERSION),
        )
    
    @classmethod
    def from_jsonl(cls, line: str) -> "ContextRecord":
        """
        Create ContextRecord from JSONL line.
        
        Args:
            line: Single-line JSON string
            
        Returns:
            ContextRecord instance
        """
        return cls.from_dict(json.loads(line))


# ==========================================
# Factory Function
# ==========================================

def create_context_record(
    traces: Dict[str, Any],
    retrieval_path: Optional[str] = None,
    capture_version: str = DEFAULT_CAPTURE_VERSION,
    observed_at: Optional[str] = None,
) -> ContextRecord:
    """
    Factory function for creating ContextRecord.
    
    This is the RECOMMENDED way to create ContextRecord instances.
    It handles:
      - Trace normalization
      - Deterministic ID generation
      - Timestamp generation
    
    Args:
        traces: Raw traces dict (will be normalized)
        retrieval_path: URL, file path, or None
        capture_version: Version of trace extraction logic
        observed_at: When observation occurred (default: now)
        
    Returns:
        ContextRecord instance
        
    Example:
        >>> record = create_context_record(
        ...     traces={"html:tag_count": 42, "text:char_count": 1000},
        ...     retrieval_path="https://example.com/article",
        ...     capture_version="v1.0"
        ... )
        >>> record.context_id
        'a1b2c3d4...'
    """
    # Normalize traces
    normalized_traces = normalize_traces(traces)
    
    # Generate deterministic ID
    context_id = compute_context_id(
        retrieval_path,
        normalized_traces,
        capture_version
    )
    
    # Generate timestamps
    now = datetime.now(timezone.utc).isoformat()
    
    return ContextRecord(
        context_id=context_id,
        retrieval_path=retrieval_path,
        capture_version=capture_version,
        traces=normalized_traces,
        observed_at=observed_at or now,
        created_at=now,
    )


# ==========================================
# Migration Helper (Phase 1)
# ==========================================

def convert_source_meta_to_traces(
    source_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert legacy source_meta to Substrate traces.
    
    Migration Phase 1:
      - Prefix all legacy keys with "legacy:"
      - Preserve original values
    
    Args:
        source_meta: Legacy source_meta dict from ArticleRecord
        
    Returns:
        Traces dict with "legacy:" prefix
        
    Example:
        >>> source_meta = {"source_type": "news", "language_profile": "en"}
        >>> convert_source_meta_to_traces(source_meta)
        {"legacy:language_profile": "en", "legacy:source_type": "news"}
    """
    traces = {}
    
    for key, value in source_meta.items():
        # Skip None values
        if value is None:
            continue
        
        # Convert key to lowercase with legacy prefix
        trace_key = f"legacy:{key.lower()}"
        
        # Convert value to allowed type
        if isinstance(value, (str, int, float, bool)):
            traces[trace_key] = value
        else:
            # Convert complex types to string
            traces[trace_key] = str(value)
    
    return traces


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Substrate Schema Test")
    print("=" * 60)
    
    # Test 1: Basic creation via factory
    print("\n[Test 1] Basic creation via factory")
    record = create_context_record(
        traces={"html:tag_count": 42, "text:char_count": 1000},
        retrieval_path="https://example.com/article",
        capture_version="v1.0"
    )
    print(f"  context_id: {record.context_id}")
    print(f"  retrieval_path: {record.retrieval_path}")
    print(f"  traces: {record.traces}")
    print(f"  schema_version: {record.schema_version}")
    print("  ✅ Created successfully")
    
    # Test 2: Immutability (frozen=True)
    print("\n[Test 2] Immutability")
    try:
        record.context_id = "new_id"  # Should fail
        print("  ❌ Should have raised FrozenInstanceError")
    except Exception as e:
        print(f"  ✅ Correctly immutable: {type(e).__name__}")
    
    # Test 3: Serialization round-trip
    print("\n[Test 3] Serialization round-trip")
    jsonl = record.to_jsonl()
    restored = ContextRecord.from_jsonl(jsonl)
    print(f"  JSONL length: {len(jsonl)} chars")
    print(f"  Restored ID: {restored.context_id}")
    assert record.context_id == restored.context_id
    assert record.traces == restored.traces
    print("  ✅ Round-trip successful")
    
    # Test 4: Deterministic ID (same input = same ID)
    print("\n[Test 4] Deterministic ID")
    record1 = create_context_record(
        traces={"a:b": 1, "c:d": 2},
        retrieval_path=None,
        capture_version="v1.0"
    )
    record2 = create_context_record(
        traces={"c:d": 2, "a:b": 1},  # Different order
        retrieval_path=None,
        capture_version="v1.0"
    )
    print(f"  Record 1 ID: {record1.context_id}")
    print(f"  Record 2 ID: {record2.context_id}")
    assert record1.context_id == record2.context_id
    print("  ✅ IDs match (key order independent)")
    
    # Test 5: ID verification in __post_init__
    print("\n[Test 5] ID verification")
    try:
        bad_record = ContextRecord(
            context_id="wrong_id_on_purpose",
            retrieval_path=None,
            capture_version="v1.0",
            traces={"a:b": 1},
            observed_at="2024-01-01T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
        )
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: context_id mismatch")
    
    # Test 6: Legacy source_meta conversion
    print("\n[Test 6] Legacy source_meta conversion")
    source_meta = {
        "source_type": "news",
        "language_profile": "en",
        "custom_field": 123,
    }
    traces = convert_source_meta_to_traces(source_meta)
    print(f"  Input:  {source_meta}")
    print(f"  Output: {traces}")
    assert "legacy:source_type" in traces
    assert traces["legacy:source_type"] == "news"
    print("  ✅ Conversion successful")
    
    # Test 7: to_dict output
    print("\n[Test 7] to_dict output")
    d = record.to_dict()
    print(f"  Keys: {sorted(d.keys())}")
    assert "context_id" in d
    assert "traces" in d
    print("  ✅ to_dict works correctly")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
