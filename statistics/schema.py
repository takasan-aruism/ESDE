"""
ESDE Phase 9-1: Statistics Schema
=================================
W1 (Weak Cross-Sectional Statistics) data structures.

Design Principles:
  - W1 is a cache, not source of truth (INV-W1-001)
  - Can always be rebuilt from W0 observations
  - No semantic inference, just counting

Invariants:
  INV-W1-001 (Recalculability):
      W1 can always be regenerated from W0. Loss of W1 is not system corruption.
  
  INV-W1-002 (Input Authority):
      W1 input is ArticleRecord.raw_text sliced by ObservationEvent.segment_span.
      ObservationEvent.segment_text is cache only, not authoritative.

Spec: v5.4.1-P9.1
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import math


# ==========================================
# Constants
# ==========================================

AGGREGATION_VERSION = "v9.1.0"

# Surface forms limits (P0-4)
SURFACE_FORMS_LIMIT = 50
SURFACE_FORMS_OTHER_KEY = "__OTHER__"
SURFACE_FORMS_LONG_KEY = "__LONG__"
SURFACE_FORMS_LONG_THRESHOLD = 50  # tokens longer than this go to __LONG__


# ==========================================
# W1 Record
# ==========================================

@dataclass
class W1Record:
    """
    W1: Weak Cross-Sectional Statistics for a single normalized token.
    
    This is a CACHE derived from W0 observations.
    It can be rebuilt at any time from ObservationEvents.
    
    Primary Key: token_norm (normalized token)
    """
    
    # --- Identity ---
    token_norm: str                     # Primary Key (Normalized)
    
    # --- Global Stats ---
    total_count: int = 0                # Total observations
    document_frequency: int = 0         # Unique source_ids (P0-3: counted once per article)
    
    # --- Surface Variance (The Wobble) ---
    # Raw forms and their counts (limited to SURFACE_FORMS_LIMIT)
    # Keys: actual surface forms, "__OTHER__" for overflow, "__LONG__" for long tokens
    surface_forms: Dict[str, int] = field(default_factory=dict)
    
    # --- Time & Source (P1-3: ObservationEvent.timestamp is authoritative) ---
    first_seen_at: Optional[str] = None     # ISO8601 (earliest observation timestamp)
    last_seen_at: Optional[str] = None      # ISO8601 (latest observation timestamp)
    last_seen_source: Optional[str] = None  # source_id with prefix (P1-4: "A:uuid" or "D:uuid")
    
    # --- Volatility Metrics ---
    # Shannon entropy of surface_forms distribution (P1-2)
    entropy: float = 0.0
    
    # --- Meta ---
    aggregation_version: str = AGGREGATION_VERSION
    updated_at: str = ""                    # Last aggregation time (excluded from rebuild comparison)
    
    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def compute_entropy(self) -> float:
        """
        Compute Shannon entropy (log2) of surface_forms distribution.
        
        P1-2 Rules:
          - Base: log2
          - Returns 0 if total_count < 2
          - Normalized to bits
        """
        if self.total_count < 2:
            return 0.0
        
        counts = [v for k, v in self.surface_forms.items()]
        if not counts or len(counts) < 2:
            return 0.0
        
        total = sum(counts)
        if total <= 0:
            return 0.0
        
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return round(entropy, 6)
    
    def update_entropy(self):
        """Recompute and store entropy."""
        self.entropy = self.compute_entropy()
    
    def add_surface_form(self, surface: str):
        """
        Add a surface form observation, respecting limits (P0-4).
        
        Rules:
          - If len(surface) > LONG_THRESHOLD: count under __LONG__
          - If already in dict: increment
          - If dict has room (< LIMIT): add new entry
          - Else: increment __OTHER__
        """
        # Check for long token
        if len(surface) > SURFACE_FORMS_LONG_THRESHOLD:
            self.surface_forms[SURFACE_FORMS_LONG_KEY] = \
                self.surface_forms.get(SURFACE_FORMS_LONG_KEY, 0) + 1
            return
        
        # Already tracked
        if surface in self.surface_forms:
            self.surface_forms[surface] += 1
            return
        
        # Count current real forms (exclude __OTHER__ and __LONG__)
        real_forms = [k for k in self.surface_forms.keys() 
                      if k not in (SURFACE_FORMS_OTHER_KEY, SURFACE_FORMS_LONG_KEY)]
        
        if len(real_forms) < SURFACE_FORMS_LIMIT:
            # Room for new form
            self.surface_forms[surface] = 1
        else:
            # Overflow to __OTHER__
            self.surface_forms[SURFACE_FORMS_OTHER_KEY] = \
                self.surface_forms.get(SURFACE_FORMS_OTHER_KEY, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "token_norm": self.token_norm,
            "total_count": self.total_count,
            "document_frequency": self.document_frequency,
            "surface_forms": self.surface_forms,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "last_seen_source": self.last_seen_source,
            "entropy": self.entropy,
            "aggregation_version": self.aggregation_version,
            "updated_at": self.updated_at,
        }
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Convert to canonical dict for rebuild comparison (P1-5).
        
        Rules:
          - Exclude updated_at (changes on each rebuild)
          - Sort surface_forms keys
        """
        return {
            "token_norm": self.token_norm,
            "total_count": self.total_count,
            "document_frequency": self.document_frequency,
            "surface_forms": dict(sorted(self.surface_forms.items())),
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "last_seen_source": self.last_seen_source,
            "entropy": round(self.entropy, 6),  # Fixed precision
            "aggregation_version": self.aggregation_version,
            # updated_at excluded
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W1Record':
        """Create from dict."""
        return cls(
            token_norm=data["token_norm"],
            total_count=data.get("total_count", 0),
            document_frequency=data.get("document_frequency", 0),
            surface_forms=data.get("surface_forms", {}),
            first_seen_at=data.get("first_seen_at"),
            last_seen_at=data.get("last_seen_at"),
            last_seen_source=data.get("last_seen_source"),
            entropy=data.get("entropy", 0.0),
            aggregation_version=data.get("aggregation_version", AGGREGATION_VERSION),
            updated_at=data.get("updated_at", ""),
        )


# ==========================================
# W1 Global Stats Container
# ==========================================

@dataclass
class W1GlobalStats:
    """
    Container for all W1 records (global cross-article statistics).
    
    Storage: data/stats/w1_global_stats.json
    """
    
    records: Dict[str, W1Record] = field(default_factory=dict)  # token_norm -> W1Record
    
    # Meta
    total_tokens_processed: int = 0      # Valid tokens (after normalization)
    total_tokens_raw: int = 0            # Raw tokens (before filtering) - for W3 denominator
    total_articles_processed: int = 0
    aggregation_version: str = AGGREGATION_VERSION
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
    
    def get_or_create(self, token_norm: str) -> W1Record:
        """Get existing record or create new one."""
        if token_norm not in self.records:
            self.records[token_norm] = W1Record(token_norm=token_norm)
        return self.records[token_norm]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization (sorted keys for P1-5)."""
        return {
            "records": {k: v.to_dict() for k, v in sorted(self.records.items())},
            "total_tokens_processed": self.total_tokens_processed,
            "total_tokens_raw": self.total_tokens_raw,
            "total_articles_processed": self.total_articles_processed,
            "aggregation_version": self.aggregation_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string with sorted keys."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W1GlobalStats':
        """Create from dict."""
        stats = cls(
            total_tokens_processed=data.get("total_tokens_processed", 0),
            total_tokens_raw=data.get("total_tokens_raw", 0),
            total_articles_processed=data.get("total_articles_processed", 0),
            aggregation_version=data.get("aggregation_version", AGGREGATION_VERSION),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        for token_norm, record_data in data.get("records", {}).items():
            stats.records[token_norm] = W1Record.from_dict(record_data)
        return stats
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W1GlobalStats':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-1 Statistics Schema Test")
    print("=" * 60)
    
    # Test 1: W1Record creation and surface_forms limit
    print("\n[Test 1] W1Record surface_forms limit (P0-4)")
    record = W1Record(token_norm="test")
    
    # Add 55 unique forms (should hit limit at 50)
    for i in range(55):
        record.add_surface_form(f"Form{i}")
        record.total_count += 1
    
    real_forms = [k for k in record.surface_forms.keys() 
                  if k not in (SURFACE_FORMS_OTHER_KEY, SURFACE_FORMS_LONG_KEY)]
    print(f"  Real forms: {len(real_forms)} (limit: {SURFACE_FORMS_LIMIT})")
    print(f"  __OTHER__ count: {record.surface_forms.get(SURFACE_FORMS_OTHER_KEY, 0)}")
    
    assert len(real_forms) == SURFACE_FORMS_LIMIT, f"Expected {SURFACE_FORMS_LIMIT}, got {len(real_forms)}"
    assert record.surface_forms.get(SURFACE_FORMS_OTHER_KEY, 0) == 5, "Expected 5 in __OTHER__"
    print("  ✅ PASS")
    
    # Test 2: Long token handling
    print("\n[Test 2] Long token handling (__LONG__)")
    record2 = W1Record(token_norm="longtoken")
    long_token = "a" * 60  # > 50 chars
    record2.add_surface_form(long_token)
    record2.add_surface_form("short")
    
    print(f"  __LONG__ count: {record2.surface_forms.get(SURFACE_FORMS_LONG_KEY, 0)}")
    print(f"  'short' count: {record2.surface_forms.get('short', 0)}")
    
    assert record2.surface_forms.get(SURFACE_FORMS_LONG_KEY) == 1
    assert record2.surface_forms.get("short") == 1
    print("  ✅ PASS")
    
    # Test 3: Entropy calculation (P1-2)
    print("\n[Test 3] Entropy calculation (Shannon log2)")
    record3 = W1Record(token_norm="entropy_test")
    record3.surface_forms = {"Apple": 50, "apple": 50}  # Uniform distribution
    record3.total_count = 100
    
    entropy = record3.compute_entropy()
    print(f"  Uniform distribution (50/50): entropy = {entropy:.4f} bits")
    assert 0.99 < entropy < 1.01, f"Expected ~1.0 for uniform, got {entropy}"
    
    record3.surface_forms = {"Apple": 100}  # Single form
    record3.total_count = 100
    entropy = record3.compute_entropy()
    print(f"  Single form (100/0): entropy = {entropy:.4f} bits")
    assert entropy == 0.0, f"Expected 0.0 for single form, got {entropy}"
    
    record3.total_count = 1  # < 2 observations
    entropy = record3.compute_entropy()
    print(f"  total_count < 2: entropy = {entropy:.4f} bits")
    assert entropy == 0.0, "Expected 0.0 for count < 2"
    print("  ✅ PASS")
    
    # Test 4: Canonical dict (P1-5)
    print("\n[Test 4] Canonical dict (rebuild comparison)")
    record4 = W1Record(
        token_norm="canonical",
        total_count=10,
        surface_forms={"z_form": 5, "a_form": 5},
    )
    
    canonical = record4.to_canonical_dict()
    print(f"  Keys in surface_forms: {list(canonical['surface_forms'].keys())}")
    assert list(canonical["surface_forms"].keys()) == ["a_form", "z_form"], "Should be sorted"
    assert "updated_at" not in canonical, "updated_at should be excluded"
    print("  ✅ PASS")
    
    # Test 5: W1GlobalStats serialization
    print("\n[Test 5] W1GlobalStats JSON serialization")
    stats = W1GlobalStats()
    stats.records["apple"] = W1Record(token_norm="apple", total_count=10)
    stats.records["banana"] = W1Record(token_norm="banana", total_count=5)
    
    json_str = stats.to_json()
    restored = W1GlobalStats.from_json(json_str)
    
    print(f"  Original records: {list(stats.records.keys())}")
    print(f"  Restored records: {list(restored.records.keys())}")
    
    assert set(stats.records.keys()) == set(restored.records.keys())
    assert restored.records["apple"].total_count == 10
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")