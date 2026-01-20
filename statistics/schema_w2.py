"""
ESDE Phase 9-2: W2 Schema
=========================
W2 (Weak Conditional Statistics) data structures.

Design Principles:
  - W2 = W1 sliced by condition factors
  - Condition factors: source_type, language_profile, time_bucket
  - Lightweight records (no full surface_forms dict)
  - Signature-based condition hashing for sparsity management

Invariants:
  INV-W2-001 (Read-Only Condition):
      W2Aggregator reads condition factors from ArticleRecord.source_meta.
      It does NOT write, infer, or modify conditions.
      Missing factors are treated as "unknown".
  
  INV-W2-002 (Time Bucket Fixed):
      In v9.2, time_bucket is fixed to YYYY-MM (monthly).
      Weekly buckets are not introduced in this version.

Spec: v5.4.2-P9.2
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import hashlib
import math


# ==========================================
# Constants
# ==========================================

W2_VERSION = "v9.2.0"
SOURCE_META_VERSION = "v1"

# Condition factor allowed values
SOURCE_TYPES = {"news", "dialog", "paper", "social", "unknown"}
LANGUAGE_PROFILES = {"en", "ja", "mixed", "unknown"}

# Default values for missing factors
DEFAULT_SOURCE_TYPE = "unknown"
DEFAULT_LANGUAGE_PROFILE = "unknown"


# ==========================================
# Condition Entry (Registry)
# ==========================================

@dataclass
class ConditionEntry:
    """
    Registry entry for a condition signature.
    
    Maps signature hash to the original condition factors.
    Stored in w2_conditions.jsonl for lookup.
    
    INV-W2-003 (Denominator Ownership):
        - total_token_count: Sum of all tokens from ArticleRecords under this condition
        - total_doc_count: Count of distinct source_ids under this condition
        - W2Aggregator writes these. W3 NEVER writes.
    """
    signature: str                  # SHA256 hash (primary key)
    factors: Dict[str, str]         # Original factors (source_type, language_profile, time_bucket)
    first_seen: str = ""            # ISO8601 timestamp
    
    # --- P0-1: Denominator for probability calculation ---
    # INV-W2-003: These are written by W2Aggregator only
    total_token_count: int = 0      # Total tokens under this condition
    total_doc_count: int = 0        # Total distinct documents under this condition
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "factors": self.factors,
            "first_seen": self.first_seen,
            "total_token_count": self.total_token_count,
            "total_doc_count": self.total_doc_count,
        }
    
    def to_jsonl(self) -> str:
        """Convert to JSONL line (sorted keys for consistency)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Canonical dict for rebuild comparison.
        Includes denominators, excludes first_seen (may vary).
        """
        return {
            "signature": self.signature,
            "factors": dict(sorted(self.factors.items())),
            "total_token_count": self.total_token_count,
            "total_doc_count": self.total_doc_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConditionEntry':
        return cls(
            signature=data["signature"],
            factors=data["factors"],
            first_seen=data.get("first_seen", ""),
            total_token_count=data.get("total_token_count", 0),
            total_doc_count=data.get("total_doc_count", 0),
        )
    
    @classmethod
    def from_jsonl(cls, line: str) -> 'ConditionEntry':
        return cls.from_dict(json.loads(line))


# ==========================================
# Condition Signature Generator
# ==========================================

def compute_condition_signature(factors: Dict[str, str]) -> str:
    """
    Compute deterministic signature for condition factors.
    
    Uses Canonical JSON (sorted keys) to ensure order-independence.
    
    Args:
        factors: Condition factors dict
        
    Returns:
        SHA256 hash (first 32 chars for readability)
    """
    canonical = json.dumps(factors, sort_keys=True, ensure_ascii=False)
    full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return full_hash[:32]  # Truncated for readability


def compute_record_id(token_norm: str, condition_signature: str) -> str:
    """
    Compute deterministic record ID for W2Record.
    
    P0-3 Compliance: Uses Canonical JSON to avoid concatenation ambiguity.
    
    Args:
        token_norm: Normalized token
        condition_signature: Condition signature hash
        
    Returns:
        SHA256 hash (first 32 chars)
    """
    canonical = json.dumps(
        {"token_norm": token_norm, "cond": condition_signature},
        sort_keys=True,
        ensure_ascii=False
    )
    full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return full_hash[:32]


# ==========================================
# Time Bucket (INV-W2-002: Monthly Fixed)
# ==========================================

def compute_time_bucket(timestamp: str) -> str:
    """
    Compute time bucket from ISO8601 timestamp.
    
    INV-W2-002: Fixed to YYYY-MM (monthly) in v9.2.
    
    Args:
        timestamp: ISO8601 timestamp string
        
    Returns:
        Time bucket string (e.g., "2026-01")
    """
    try:
        # Parse ISO8601 timestamp
        if 'T' in timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(timestamp)
        
        return dt.strftime("%Y-%m")
    except (ValueError, AttributeError):
        # Fallback to current month if parsing fails
        return datetime.now(timezone.utc).strftime("%Y-%m")


# ==========================================
# W2 Record
# ==========================================

@dataclass
class W2Record:
    """
    W2: Conditional statistics for a (token, condition) pair.
    
    Lightweight schema - does NOT store full surface_forms dict.
    Links to W1 for detailed surface form data.
    """
    
    # --- Composite Key ---
    record_id: str                      # SHA256(CanonicalJSON({token_norm, cond}))
    token_norm: str                     # Normalized token (link to W1)
    condition_signature: str            # Link to ConditionRegistry
    
    # --- Conditional Stats ---
    count: int = 0                      # Occurrences under this condition
    doc_freq: int = 0                   # DF under this condition
    
    # --- Variance Metrics (Lightweight) ---
    # P1-2: entropy includes __OTHER__/__LONG__, top_form excludes them
    entropy: float = 0.0                # Surface form entropy under this condition
    top_surface_form: str = ""          # Most frequent surface form (excluding __OTHER__/__LONG__)
    
    # --- Meta ---
    normalizer_version: str = "v9.1.0"  # P1-4: Version tracking
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "token_norm": self.token_norm,
            "condition_signature": self.condition_signature,
            "count": self.count,
            "doc_freq": self.doc_freq,
            "entropy": round(self.entropy, 6),
            "top_surface_form": self.top_surface_form,
            "normalizer_version": self.normalizer_version,
            "updated_at": self.updated_at,
        }
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Canonical dict for rebuild comparison.
        Excludes updated_at (changes on each run).
        """
        return {
            "record_id": self.record_id,
            "token_norm": self.token_norm,
            "condition_signature": self.condition_signature,
            "count": self.count,
            "doc_freq": self.doc_freq,
            "entropy": round(self.entropy, 6),
            "top_surface_form": self.top_surface_form,
            "normalizer_version": self.normalizer_version,
            # updated_at excluded
        }
    
    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W2Record':
        return cls(
            record_id=data["record_id"],
            token_norm=data["token_norm"],
            condition_signature=data["condition_signature"],
            count=data.get("count", 0),
            doc_freq=data.get("doc_freq", 0),
            entropy=data.get("entropy", 0.0),
            top_surface_form=data.get("top_surface_form", ""),
            normalizer_version=data.get("normalizer_version", "v9.1.0"),
            updated_at=data.get("updated_at", ""),
        )
    
    @classmethod
    def from_jsonl(cls, line: str) -> 'W2Record':
        return cls.from_dict(json.loads(line))


# ==========================================
# W2 Global Stats Container
# ==========================================

@dataclass
class W2GlobalStats:
    """
    Container for all W2 records and condition registry.
    
    Storage:
      - Records: data/stats/w2_conditional_stats.jsonl
      - Conditions: data/stats/w2_conditions.jsonl
    """
    
    # (record_id -> W2Record)
    records: Dict[str, W2Record] = field(default_factory=dict)
    
    # (signature -> ConditionEntry)
    conditions: Dict[str, ConditionEntry] = field(default_factory=dict)
    
    # Meta
    total_records: int = 0
    total_conditions: int = 0
    w2_version: str = W2_VERSION
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
    
    def get_or_create_condition(self, factors: Dict[str, str]) -> str:
        """
        Get existing condition signature or create new entry.
        
        Returns:
            Condition signature
        """
        signature = compute_condition_signature(factors)
        
        if signature not in self.conditions:
            self.conditions[signature] = ConditionEntry(
                signature=signature,
                factors=factors,
            )
            self.total_conditions = len(self.conditions)
        
        return signature
    
    def get_or_create_record(self, token_norm: str, condition_signature: str) -> W2Record:
        """
        Get existing W2Record or create new one.
        
        Returns:
            W2Record instance
        """
        record_id = compute_record_id(token_norm, condition_signature)
        
        if record_id not in self.records:
            self.records[record_id] = W2Record(
                record_id=record_id,
                token_norm=token_norm,
                condition_signature=condition_signature,
            )
            self.total_records = len(self.records)
        
        return self.records[record_id]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary for logging (not full export)."""
        return {
            "total_records": self.total_records,
            "total_conditions": self.total_conditions,
            "w2_version": self.w2_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-2 W2 Schema Test")
    print("=" * 60)
    
    # Test 1: Condition signature (order-independence)
    print("\n[Test 1] Condition signature (order-independent)")
    
    factors1 = {"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"}
    factors2 = {"time_bucket": "2026-01", "source_type": "news", "language_profile": "en"}
    factors3 = {"language_profile": "en", "time_bucket": "2026-01", "source_type": "news"}
    
    sig1 = compute_condition_signature(factors1)
    sig2 = compute_condition_signature(factors2)
    sig3 = compute_condition_signature(factors3)
    
    print(f"  factors1: {factors1} -> {sig1}")
    print(f"  factors2: {factors2} -> {sig2}")
    print(f"  factors3: {factors3} -> {sig3}")
    
    assert sig1 == sig2 == sig3, "Signatures should be identical regardless of key order"
    print("  ✅ PASS: All signatures match")
    
    # Test 2: Record ID (P0-3 compliant)
    print("\n[Test 2] Record ID (Canonical JSON)")
    
    rid1 = compute_record_id("apple", sig1)
    rid2 = compute_record_id("apple", sig1)
    rid3 = compute_record_id("banana", sig1)
    
    print(f"  apple + sig1: {rid1}")
    print(f"  apple + sig1 (again): {rid2}")
    print(f"  banana + sig1: {rid3}")
    
    assert rid1 == rid2, "Same inputs should produce same ID"
    assert rid1 != rid3, "Different tokens should produce different IDs"
    print("  ✅ PASS")
    
    # Test 3: Time bucket (INV-W2-002)
    print("\n[Test 3] Time bucket (YYYY-MM fixed)")
    
    tb1 = compute_time_bucket("2026-01-20T10:30:00Z")
    tb2 = compute_time_bucket("2026-01-31T23:59:59+00:00")
    tb3 = compute_time_bucket("2026-02-01T00:00:00Z")
    
    print(f"  2026-01-20T10:30:00Z -> {tb1}")
    print(f"  2026-01-31T23:59:59 -> {tb2}")
    print(f"  2026-02-01T00:00:00Z -> {tb3}")
    
    assert tb1 == "2026-01", f"Expected 2026-01, got {tb1}"
    assert tb2 == "2026-01", f"Expected 2026-01, got {tb2}"
    assert tb3 == "2026-02", f"Expected 2026-02, got {tb3}"
    print("  ✅ PASS")
    
    # Test 4: ConditionEntry JSONL
    print("\n[Test 4] ConditionEntry JSONL serialization")
    
    entry = ConditionEntry(signature=sig1, factors=factors1)
    jsonl = entry.to_jsonl()
    restored = ConditionEntry.from_jsonl(jsonl)
    
    print(f"  Original: {entry.signature[:16]}...")
    print(f"  JSONL: {jsonl[:60]}...")
    print(f"  Restored: {restored.signature[:16]}...")
    
    assert entry.signature == restored.signature
    assert entry.factors == restored.factors
    print("  ✅ PASS")
    
    # Test 5: W2Record
    print("\n[Test 5] W2Record creation and serialization")
    
    record = W2Record(
        record_id=rid1,
        token_norm="apple",
        condition_signature=sig1,
        count=10,
        doc_freq=3,
        entropy=0.5,
        top_surface_form="Apple",
    )
    
    jsonl = record.to_jsonl()
    restored = W2Record.from_jsonl(jsonl)
    
    print(f"  token_norm: {record.token_norm}")
    print(f"  count: {record.count}, doc_freq: {record.doc_freq}")
    print(f"  Restored count: {restored.count}")
    
    assert record.count == restored.count
    assert record.token_norm == restored.token_norm
    print("  ✅ PASS")
    
    # Test 6: W2GlobalStats
    print("\n[Test 6] W2GlobalStats container")
    
    stats = W2GlobalStats()
    
    # Create condition
    sig = stats.get_or_create_condition({"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"})
    
    # Create record
    rec = stats.get_or_create_record("apple", sig)
    rec.count = 5
    
    print(f"  Conditions: {stats.total_conditions}")
    print(f"  Records: {stats.total_records}")
    print(f"  apple count: {rec.count}")
    
    assert stats.total_conditions == 1
    assert stats.total_records == 1
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")